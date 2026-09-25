"""Unified evaluation runner for SWE-Explore.

Supports both local (BM25, TF-IDF, Potion, RAG, SimpleRule, Oracle, Random)
and agentic (Claude Code, Cursor Agent) explorers.

Usage:
    python eval_runner.py --explorers bm25 tfidf --top-k 5,10,20 -o results/{explorer}/top{k}.jsonl
"""
from __future__ import annotations

import contextlib
import functools
import json
import os
import signal
import subprocess
import sys
import time
from threading import Event
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, List, Dict, NamedTuple

import typer
from rich.console import Console
from rich.table import Table

from eval import ExploreEvaluator
from explorers._cli_agent_base import set_log_level, sha256_file, sha256_json
from explorers._cli_process import cli_cancel_event, kill_active_cli_trees
from explorers.base import (
    ERROR,
    NOT_ATTEMPTED,
    SUCCESS,
    ExplorerFailure,
    ExplorerResult,
    classify_failure,
)
from explorers.parsing import (
    TokenUsage,
    register_litellm_usage_callback,
    usage_collector,
)

app = typer.Typer(rich_markup_mode="rich")
console = Console()


@contextlib.contextmanager
def _cli_cancellation():
    """Share cancellation with CLI calls; a second Ctrl+C exits immediately."""
    signals = [signal.SIGINT]
    if os.name == "posix":
        signals.extend((signal.SIGTERM, signal.SIGHUP))
    previous = {sig: signal.getsignal(sig) for sig in signals}
    seen = False
    cancel = Event()

    def _handler(signum, frame):
        nonlocal seen
        if seen or signum != signal.SIGINT:
            kill_active_cli_trees()
            os._exit(128 + signum)
        seen = True
        raise KeyboardInterrupt

    for sig in signals:
        signal.signal(sig, _handler)
    token = cli_cancel_event.set(cancel)
    try:
        yield cancel
    except BaseException:
        cancel.set()
        raise
    finally:
        cli_cancel_event.reset(token)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextlib.contextmanager
def _interruptible_pool(workers: int):
    """Drop queued work and stop active CLI trees when the pool is interrupted."""
    with _cli_cancellation() as cancel:
        pool = ThreadPoolExecutor(
            max_workers=workers, initializer=cli_cancel_event.set, initargs=(cancel,),
        )
        try:
            yield pool
        except BaseException:
            cancel.set()
            raise
        finally:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except BaseException:
                # SIGINT can also arrive during a normal shutdown.
                cancel.set()
                pool.shutdown(wait=True, cancel_futures=True)
                raise


METRICS = [
    "precision",
    "recall",
    "f1_score",
    "hit_file_rate",
    "noise_file_rate",
    "hit_region_rate",
    "noise_region_rate",
    "weighted_core_coverage",
    "context_efficiency",
    "optional_coverage",
    "ndcg_at_100",
    "ndcg_at_300",
    "ndcg_at_500",
    "recall_at_100",
    "recall_at_300",
    "recall_at_500",
    "first_useful_hit",
]

LOCAL_EXPLORERS = {
    "bm25",
    "codenib",
    "rag",
    "tfidf",
    "potion",
    "simple_rule",
    "oracle",
    "random",
    "embed",
    "swerank",
}
AGENTIC_EXPLORERS = {"claude_code", "cursor", "opencode", "deveco"}
CHUNKED_EXPLORERS = {"bm25", "tfidf", "potion", "embed", "swerank"}
ACADEMIC_EXPLORERS = {"autocr", "cosil", "locagent", "orcaloca", "mini_swe_agent", "awe_agent"}
ALL_EXPLORERS = LOCAL_EXPLORERS | AGENTIC_EXPLORERS | ACADEMIC_EXPLORERS
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


# ── helpers ─────────────────────────────────────────────────────────────

def _load_bench_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_issue_map(trajs_dir: Path) -> dict[str, str]:
    issue_map: dict[str, str] = {}
    # Sorted, so a duplicated instance gets the same issue text on every filesystem.
    for p in sorted(trajs_dir.rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        info = data.get("info") or {}
        iid = info.get("instance_id") or p.stem
        issue = info.get("issue") or ""
        if iid and issue and iid not in issue_map:
            issue_map[iid] = issue
    return issue_map


def _resolve_repo_dir(
    repo_dir_value: str | None,
    repos_root: Path | None,
    instance_id: str,
) -> Path | None:
    if repo_dir_value:
        p = Path(repo_dir_value)
        if not p.is_absolute() and repos_root is not None:
            rooted = repos_root / p
            if rooted.is_dir():
                return rooted
            if p.parts and p.parts[0] == repos_root.name:
                return repos_root.joinpath(*p.parts[1:])
            p = rooted
        return p
    if repos_root is None or "__" not in instance_id:
        return None
    org, rest = instance_id.split("__", 1)
    repo = rest.rsplit("-", 1)[0] if "-" in rest else rest
    for cand in [
        repos_root / instance_id,          # 优先：按 instance_id 命名的目录（新方案）
        repos_root / repo,                  # fallback：按 repo 名（旧方案，单 commit repo）
        repos_root / f"{org}__{repo}",
        repos_root / f"{org}-{repo}",
        repos_root / org,
    ]:
        if cand.is_dir():
            return cand
    return None


def _iter_gt_paths(gt: dict) -> Iterable[str]:
    for r in gt.get("read_core_regions") or []:
        path = r.get("path")
        if isinstance(path, str):
            yield path
    for regions in (gt.get("read_optional_regions_map") or {}).values():
        for r in regions:
            path = r.get("path")
            if isinstance(path, str):
                yield path


def _build_file_line_counts(
    records: list[dict], repos_root: Path | None,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for rec in records:
        iid = rec.get("instance_id", "")
        gt = rec.get("ground_truth") or {}
        repo_dir = _resolve_repo_dir(rec.get("repo_dir"), repos_root, iid)
        if not repo_dir or not repo_dir.is_dir():
            continue
        per: dict[str, int] = {}
        for rel in set(_iter_gt_paths(gt)):
            fpath = repo_dir / rel
            if fpath.is_file():
                try:
                    per[rel] = len(fpath.read_text(encoding="utf-8", errors="ignore").splitlines())
                except OSError:
                    pass
        if per:
            counts[iid] = per
    return counts


def _results_to_regions(results: list[ExplorerResult]) -> list[tuple[str, int, int]]:
    regions: list[tuple[str, int, int]] = []
    for res in results:
        for r in res.regions:
            regions.append((r.path, r.start, r.end))
    return regions


def _parse_top_k_list(value: str) -> list[int]:
    """Parse comma-separated top_k values like '5,10,20'."""
    return sorted(set(int(x.strip()) for x in value.split(",")))


def _format_output_path(template: str, explorer: str, k: int) -> Path:
    """Format output path template with {explorer} and {k} placeholders."""
    return Path(template.format(explorer=explorer, k=k))


def _load_existing_results(path: Path) -> list[dict]:
    """Load existing JSONL results for resume."""
    if not path.is_file():
        return []
    rows = []
    # Old cp1252 files or stray CLI bytes must not abort a resume.
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _dump_row(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False) + "\n"


def _case_ids(rows: list[dict], *, finished_only: bool = False) -> set[str]:
    """Non-empty instance_ids in `rows`; with `finished_only`, only cases that
    succeeded or predate recorded outcomes."""
    ids = set()
    for row in rows:
        iid = row.get("instance_id")
        if not (isinstance(iid, str) and iid):
            continue
        if finished_only and row.get("outcome") not in (None, SUCCESS):
            continue
        ids.add(iid)
    return ids


def _dedupe_rows(rows: list[dict], keep: set[str]) -> list[dict]:
    """One row per instance_id (the last one wins), restricted to `keep`."""
    latest: dict[str, dict] = {}
    for row in rows:
        iid = row.get("instance_id")
        if isinstance(iid, str) and iid in keep:
            latest[iid] = row
    return list(latest.values())


@contextlib.contextmanager
def _atomic_target(path: Path):
    """Yield a temporary path beside the symlink-resolved target, then replace it."""
    target = path.resolve()
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        yield tmp
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


def _rewrite_results(path: Path, rows: list[dict]) -> None:
    with _atomic_target(path) as tmp:
        with tmp.open("w", encoding="utf-8") as f:
            f.writelines(_dump_row(row) for row in rows)


def _append_row(fh, row: dict) -> None:
    fh.write(_dump_row(row))
    fh.flush()


class ResumeMismatch(RuntimeError):
    """The existing result files do not match the run being resumed."""


def _clear_outputs(output_jsonl: str, explorers: list[str], top_k_list: list[int]) -> None:
    """Empty every result file up front, so an interrupt cannot leave a
    previous run's rows for a later --resume to adopt."""
    for explorer in explorers:
        for k in top_k_list:
            path = _format_output_path(output_jsonl, explorer, k)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.open("w", encoding="utf-8").close()


@contextlib.contextmanager
def _resume_mismatch_exits():
    """Report an unresumable output layout as a CLI error, not a traceback."""
    try:
        yield
    except ResumeMismatch as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _load_resume_state(
    output_jsonl: str, explorer: str, top_k_list: list[int]
) -> tuple[dict[int, Path], dict[int, list[dict]]]:
    """Read an explorer's result files, rejecting ones it cannot prune."""
    out_paths = {k: _format_output_path(output_jsonl, explorer, k) for k in top_k_list}
    existing = {k: _load_existing_results(path) for k, path in out_paths.items()}
    # A row without an explorer predates the field and counts as ours.
    for k, rows in existing.items():
        foreign = sorted(
            {str(r["explorer"]) for r in rows if r.get("explorer") not in (None, explorer)}
        )
        if foreign:
            raise ResumeMismatch(
                f"cannot resume {explorer}: {out_paths[k]} also holds rows from "
                f"{', '.join(foreign)}. Resuming needs one file per explorer; add "
                f"{{explorer}} to --output, or rerun without --resume."
            )
    # A missing budget file means --top-k or --output changed; an empty one is
    # only an interrupted first case.
    missing = [path for path in out_paths.values() if not path.is_file()]
    if missing and any(existing.values()):
        raise ResumeMismatch(
            f"cannot resume {explorer}: no results at {missing[0]}, but other "
            f"top_k files already hold rows. The --top-k budgets or --output "
            f"path do not match the run being resumed; rerun without --resume "
            f"to start these files over."
        )
    return out_paths, existing


def _reconcile_resume_state(
    output_jsonl: str,
    explorer: str,
    top_k_list: list[int],
    *,
    retry_failed: bool = True,
    selected_ids: set[str] | None = None,
) -> tuple[set[str], dict[int, list[dict]], int]:
    """Prune an explorer's result files to the cases finished at every budget.

    A case an interrupt left in only some budget files, or (with
    `retry_failed`) one that failed, is dropped so it reruns. Only selected
    cases are retried: an unselected failed row would be deleted with nothing
    to replace it. Rewriting also repairs a half-written last line.

    Returns the resumed ids, the kept rows per budget and the retry count.
    """
    out_paths, existing = _load_resume_state(output_jsonl, explorer, top_k_list)
    all_ids = set.intersection(*(_case_ids(rows) for rows in existing.values()))
    retry: set[str] = set()
    if retry_failed:
        finished_ids = set.intersection(
            *(_case_ids(rows, finished_only=True) for rows in existing.values())
        )
        retry = all_ids - finished_ids
        if selected_ids is not None:
            retry &= selected_ids
    resumed_ids = all_ids - retry
    retried = len(retry)
    kept_per_k: dict[int, list[dict]] = {}
    for k, path in out_paths.items():
        kept_per_k[k] = _dedupe_rows(existing[k], resumed_ids)
        if path.is_file():
            _rewrite_results(path, kept_per_k[k])
    return resumed_ids, kept_per_k, retried


# ── run manifest (README "Run manifest") ────────────────────────────────

MANIFEST_SCHEMA = 1
# Resume refuses to mix rows across a change to any of these.
_MANIFEST_IDENTITY = ("explorer", "bench_sha256", "issues_sha256", "explorer_config")
# How the model was chosen, not which one ran.
_INFORMATIONAL_CONFIG_KEYS = frozenset({"model_source"})


def _manifest_path(results_path: Path) -> Path:
    return results_path.with_name(results_path.stem + ".manifest.json")


@functools.lru_cache(maxsize=None)
def _git_revision(path: Path) -> str | None:
    """HEAD of a git work tree rooted exactly at `path` (not an enclosing one), else None."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = proc.stdout.splitlines()
    if proc.returncode != 0 or len(lines) != 2:
        return None
    try:
        if Path(lines[0]).resolve() != path.resolve():
            return None
    except OSError:
        return None
    return lines[1]


def _row_explorer_config(config: dict) -> dict:
    """The config a row repeats; a CLI agent's hashes and version stay in the manifest."""
    if "cli" in config:
        return {key: config[key] for key in ("cli", "model") if key in config}
    return config


def _portable_path(value: str | Path) -> str:
    """An absolute path made shareable: relative to the cwd, else just its name."""
    path = Path(value)
    if not path.is_absolute():
        return str(value)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _build_manifests(
    explorer_configs: dict[str, dict], bench_path: Path, issue_map: dict[str, str]
) -> dict[str, dict]:
    recorded = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "bench_path": _portable_path(bench_path),
        "harness_revision": _git_revision(Path(__file__).resolve().parent),
    }
    bench_sha256 = sha256_file(bench_path)
    issues_sha256 = sha256_json(issue_map)
    # Round-tripped, so it compares equal to a manifest read back from disk.
    return json.loads(json.dumps({
        name: {
            "explorer": name,
            "bench_sha256": bench_sha256,
            "issues_sha256": issues_sha256,
            "explorer_config": config,
            "recorded": recorded,
        }
        for name, config in explorer_configs.items()
    }))


def _manifest_diff(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """Identity fields that differ, and those unknown (None) on one side.

    A probe that could not run records None, and one slow CLI startup must
    not refuse every later resume.
    """
    diffs: list[str] = []
    unknown: list[str] = []
    for key in _MANIFEST_IDENTITY:
        before, after = old.get(key), new.get(key)
        if isinstance(before, dict) and isinstance(after, dict):
            for sub in sorted(set(before) | set(after)):
                if sub in _INFORMATIONAL_CONFIG_KEYS:
                    continue
                if before.get(sub) == after.get(sub):
                    continue
                if before.get(sub) is None or after.get(sub) is None:
                    unknown.append(f"{key}.{sub}")
                    continue
                diffs.append(
                    f"{key}.{sub}: {before.get(sub)!r} -> {after.get(sub)!r}"
                )
        elif before != after:
            if before is None or after is None:
                unknown.append(key)
            else:
                diffs.append(f"{key}: {before!r} -> {after!r}")
    return diffs, unknown


def _read_sidecar(path: Path) -> tuple[object, dict]:
    """A sidecar's schema and `explorers` map; anything malformed reads as the default."""
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict):
        return MANIFEST_SCHEMA, {}
    entries = data.get("explorers")
    if not isinstance(entries, dict):
        entries = {}
    return data.get("schema", MANIFEST_SCHEMA), entries


def _write_sidecar(path: Path, entries: dict) -> None:
    data = {"schema": MANIFEST_SCHEMA, "explorers": entries}
    with _atomic_target(path) as tmp:
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def _check_resume(
    output_jsonl: str, explorers: list[str], manifests: dict[str, dict], top_k_list: list[int]
) -> list[str]:
    """Reject result files this run cannot continue, before any case runs.

    Returns warnings for what is adopted rather than refused: results with no
    manifest, and fields unknown on one side.
    """
    # Rows record no top_k, so pruning a shared file would delete another
    # budget's or explorer's rows.
    seen: dict[Path, tuple[str, int]] = {}
    for explorer in explorers:
        for k in top_k_list:
            path = _format_output_path(output_jsonl, explorer, k)
            if path in seen:
                other_explorer, other_k = seen[path]
                raise ResumeMismatch(
                    f"cannot resume: --output {output_jsonl!r} writes both "
                    f"({other_explorer}, top_k={other_k}) and ({explorer}, top_k={k}) "
                    f"to {path}. Resuming needs one file per explorer and budget; "
                    f"add {{explorer}}/{{k}} to --output, or rerun without --resume."
                )
            seen[path] = (explorer, k)

    warnings: list[str] = []

    def warn(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    for explorer, manifest in manifests.items():
        _load_resume_state(output_jsonl, explorer, top_k_list)
        for k in top_k_list:
            results = _format_output_path(output_jsonl, explorer, k)
            sidecar = _manifest_path(results)
            schema, entries = _read_sidecar(sidecar)
            if not isinstance(schema, int) or schema > MANIFEST_SCHEMA:
                raise ResumeMismatch(
                    f"cannot resume {explorer}: {sidecar} records "
                    f"manifest schema {schema!r}, and this harness understands "
                    f"{MANIFEST_SCHEMA}. Use a newer harness, write to a new "
                    f"--output, or rerun without --resume."
                )
            entry = entries.get(explorer)
            if not isinstance(entry, dict):
                if results.is_file() and results.stat().st_size:
                    warn(
                        f"{explorer}: the existing results record no run manifest; "
                        f"assuming they match this configuration"
                    )
                continue
            diffs, unknown = _manifest_diff(entry.get("manifest") or {}, manifest)
            if diffs:
                raise ResumeMismatch(
                    f"cannot resume {explorer}: {results} was produced by a different "
                    f"configuration ({'; '.join(diffs)}). Resuming would mix two "
                    f"experiments in one result; write to a new --output, or rerun "
                    f"without --resume to start these files over."
                )
            for field in unknown:
                warn(
                    f"{explorer}: {field} is unknown on one side — a probe that "
                    f"could not run, or a field that side does not record — so it "
                    f"was not compared"
                )
    return warnings


def _write_manifests(
    output_jsonl: str, manifests: dict[str, dict], top_k_list: list[int], *, fresh: bool
) -> None:
    """Record each explorer's manifest beside its result files; a resumed run
    keeps the entries it already has."""
    # Grouped, since explorers or budgets can share one sidecar.
    by_sidecar: dict[Path, dict[str, dict]] = {}
    for explorer, manifest in manifests.items():
        for k in top_k_list:
            sidecar = _manifest_path(_format_output_path(output_jsonl, explorer, k))
            by_sidecar.setdefault(sidecar, {})[explorer] = manifest
    for sidecar, wanted in by_sidecar.items():
        entries = {} if fresh else _read_sidecar(sidecar)[1]
        for explorer, manifest in wanted.items():
            if fresh or not isinstance(entries.get(explorer), dict):
                entries[explorer] = {"manifest": manifest, "summary": {}}
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        _write_sidecar(sidecar, entries)


def _write_summary(results_path: Path, explorer: str, k: int, summary: dict) -> None:
    sidecar = _manifest_path(results_path)
    entries = _read_sidecar(sidecar)[1]
    entry = entries.get(explorer)
    if not isinstance(entry, dict):
        return
    if not isinstance(entry.get("summary"), dict):
        entry["summary"] = {}
    entry["summary"][str(k)] = summary
    _write_sidecar(sidecar, entries)


class _Tally:
    """One explorer's scored rows per budget, resumed and new alike."""

    def __init__(self, top_k_list: list[int]) -> None:
        self.rows: dict[int, list[dict]] = {k: [] for k in top_k_list}
        self._totals = {k: {m: 0.0 for m in METRICS} for k in top_k_list}
        self.usage = TokenUsage()
        self.usage_cases = 0

    def add_row(self, k: int, row: dict) -> None:
        self.rows[k].append(row)
        for m in METRICS:
            self._totals[k][m] += (row.get("metrics") or {}).get(m, 0.0)

    def add_usage(self, usage: TokenUsage | None) -> None:
        # A zeroed usage would depress the per-case mean.
        if usage is not None and usage.has_any():
            self.usage.add(usage)
            self.usage_cases += 1

    def averages(self, k: int) -> dict[str, float]:
        n = len(self.rows[k])
        return {m: (self._totals[k][m] / n if n else 0.0) for m in METRICS}


def _summarize(tally: _Tally, k: int, cases: int) -> dict:
    """How the cases at one budget ended; `cases` is the run's selection, so
    the completion rate keeps its denominator across resumes."""
    rows = tally.rows[k]
    outcomes: dict[str, int] = {}
    for row in rows:
        outcome = row.get("outcome") or "unrecorded"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    attempted = len(rows)
    return {
        "cases": cases,
        "attempted": attempted,
        "not_attempted": max(cases - attempted, 0),
        "outcomes": dict(sorted(outcomes.items())),
        "completion_rate": outcomes.get(SUCCESS, 0) / cases if cases else None,
        "metrics": tally.averages(k),
        "token_usage": tally.usage.to_dict(),
        "token_usage_cases": tally.usage_cases,
    }


def _print_usage_table(name: str, totals: TokenUsage, cases: int) -> None:
    if not totals.has_any():
        console.print(f"  [dim]Token usage: no data reported by {name}[/dim]")
        return
    table = Table(title=f"{name} Token Usage", show_lines=False)
    table.add_column("Cases", justify="right")
    display = (
        ("Input", "input"),
        ("Output", "output"),
        ("Cache Read", "cache_read"),
        ("Cache Write", "cache_write"),
        ("Reasoning", "reasoning"),
        ("Total", "total"),
    )
    for label, _ in display:
        table.add_column(label, justify="right")
    values = totals.to_dict()
    table.add_row(str(cases), *[f"{values[key]:,}" for _, key in display])
    console.print(table)
    # Printed even when zero: 0 with sub-agents means the session query fell back.
    sub = totals.subagent or TokenUsage()
    console.print(
        f"  [dim]of which sub-agents: {sub.total:,} "
        f"(in {sub.input_tokens:,}, out {sub.output_tokens:,}, "
        f"reasoning {sub.reasoning_tokens:,})[/dim]"
    )


def _report_explorer(name: str, tally: _Tally, cases: int, output_jsonl: str | None) -> None:
    table = Table(title=f"{name} Results", show_lines=False)
    table.add_column("top_k", justify="right")
    table.add_column("Eval", justify="right")
    for metric in METRICS:
        table.add_column(metric, justify="right")
    summaries = {}
    for k, rows in tally.rows.items():
        summaries[k] = _summarize(tally, k, cases)
        avg = summaries[k]["metrics"]
        table.add_row(str(k), str(len(rows)), *[f"{avg[m]:.4f}" for m in METRICS])
    console.print(table)
    first = next(iter(summaries.values()))
    rate = first["completion_rate"]
    outcomes = ", ".join(f"{o}={n}" for o, n in first["outcomes"].items()) or "none"
    console.print(
        f"  Outcomes: {outcomes}; not attempted={first['not_attempted']}; "
        f"completion rate={'n/a' if rate is None else f'{rate:.1%}'}"
    )
    _print_usage_table(name, tally.usage, tally.usage_cases)
    if output_jsonl:
        for k, summary in summaries.items():
            _write_summary(_format_output_path(output_jsonl, name, k), name, k, summary)


class _CaseRun(NamedTuple):
    """One case as run: `preds` is None when it was never attempted (no repo)."""

    iid: str
    preds: list[tuple[str, int, int]] | None
    usage: TokenUsage | None
    seconds: float
    outcome: str
    error: str | None
    repo_revision: str | None


MAX_ERROR_CHARS = 4000


# ── main command ────────────────────────────────────────────────────────

@app.command()
def run(
    bench_path: Path = typer.Option(
        Path("bench.jsonl"),
        "--bench", "-b",
        help="Path to bench JSONL file",
    ),
    repos_root: Path | None = typer.Option(
        Path("repos"),
        "--repos", "-r",
        help="Repos root directory",
    ),
    trajs_dir: Path | None = typer.Option(
        Path("unify_trajs"),
        "--trajs-dir", "-t",
        help="Unified trajectories dir (for issue text)",
    ),
    issue_map_file: Path | None = typer.Option(
        None,
        "--issue-map",
        help="Pre-built issue map JSON file {instance_id: issue_text}，优先级高于 trajs_dir",
    ),
    explorers: List[str] = typer.Option(
        ["bm25"],
        "--explorers", "-e",
        help=f"Explorers to evaluate: {', '.join(sorted(ALL_EXPLORERS))}",
    ),
    top_k_str: str = typer.Option("5", "--top-k", "-k", help="Comma-separated top_k values, e.g. 5,10,20"),
    chunk_size: int = typer.Option(80, "--chunk-size", help="Chunk size (lines)"),
    chunk_overlap: int = typer.Option(20, "--chunk-overlap", help="Chunk overlap"),
    codenib_auto_index: bool = typer.Option(
        True,
        "--codenib-auto-index/--no-codenib-auto-index",
        help="Build or update the CodeNib views required by --codenib-policy.",
    ),
    codenib_rebuild: bool = typer.Option(
        False,
        "--codenib-rebuild/--no-codenib-rebuild",
        help="Force CodeNib to rebuild required views instead of reusing them.",
    ),
    codenib_policy: str = typer.Option(
        "bm25",
        "--codenib-policy",
        help=(
            "CodeNib policy: auto, bm25, dense, hybrid, hybrid_rerank, or graph. "
            "The published compatibility control uses bm25."
        ),
    ),
    codenib_planning_budget: str = typer.Option(
        "balanced",
        "--codenib-planning-budget",
        help="CodeNib planning budget: fast, balanced, or thorough.",
    ),
    codenib_retrieval_level: str = typer.Option(
        "l2",
        "--codenib-retrieval-level",
        help="CodeNib dense retrieval level: l0 or l2.",
    ),
    rag_endpoint: str | None = typer.Option(None, "--rag-endpoint"),
    rag_api_key: str | None = typer.Option(None, "--rag-api-key"),
    potion_model_path: str = typer.Option(
        "/tmp/potion-base-8M", "--potion-model-path",
    ),
    claude_model: str = typer.Option("sonnet", "--claude-model"),
    claude_timeout: int = typer.Option(600, "--claude-timeout"),
    cursor_api_key: str | None = typer.Option(None, "--cursor-api-key"),
    cursor_model: str | None = typer.Option(None, "--cursor-model"),
    # ── opencode ──
    opencode_bin: str = typer.Option("opencode", "--opencode-bin"),
    opencode_timeout: int = typer.Option(600, "--opencode-timeout"),
    opencode_config_dir: Path | None = typer.Option(
        None, "--opencode-config-dir",
        help="Directory containing OpenCode config such as opencode.json.",
    ),
    opencode_prompt_additions: str = typer.Option(
        "", "--opencode-prompt-additions",
        help="Extra instructions to append to the OpenCode prompt.",
    ),
    opencode_model: str = typer.Option(
        "", "--opencode-model",
        help="Model as provider/model, passed to every run as --model. Without "
        "it the model the resolved configuration names is pinned the same way.",
    ),
    opencode_agent: str = typer.Option(
        "", "--opencode-agent", help="Agent passed to every run as --agent.",
    ),
    # ── deveco ──
    deveco_bin: str = typer.Option("deveco", "--deveco-bin"),
    deveco_timeout: int = typer.Option(600, "--deveco-timeout"),
    deveco_config_dir: Path | None = typer.Option(
        None, "--deveco-config-dir",
        help="Directory containing DevEco Code config such as deveco.json.",
    ),
    deveco_prompt_additions: str = typer.Option(
        "", "--deveco-prompt-additions",
        help="Extra instructions to append to the DevEco Code prompt.",
    ),
    deveco_model: str = typer.Option(
        "", "--deveco-model",
        help="Model as provider/model, passed to every run as --model. Without "
        "it the model the resolved configuration names is pinned the same way.",
    ),
    deveco_agent: str = typer.Option(
        "", "--deveco-agent", help="Agent passed to every run as --agent.",
    ),
    deveco_skip_permissions: bool = typer.Option(
        True,
        "--deveco-skip-permissions/--no-deveco-skip-permissions",
        help=(
            "Pass --dangerously-skip-permissions so unattended runs never block "
            "on an approval prompt."
        ),
    ),
    # ── academic agents — all route through local LiteLLM proxy by default ──
    academic_model: str = typer.Option(
        "gpt-5.4", "--academic-model",
        help="Model name as exposed by the LiteLLM proxy.",
    ),
    academic_api_key: str = typer.Option(
        "", "--academic-api-key", envvar="ACADEMIC_API_KEY",
        help="API key for the LiteLLM proxy (the proxy's master_key).",
    ),
    academic_api_base: str = typer.Option(
        "http://127.0.0.1:4000/v1", "--academic-api-base", envvar="ACADEMIC_API_BASE",
        help="OpenAI-compatible base URL. Defaults to a local LiteLLM proxy.",
    ),
    academic_timeout: int = typer.Option(
        3600, "--academic-timeout",
        help="Per-instance wall-clock timeout (sec) for academic-agent subprocesses.",
    ),
    orcaloca_docker_image: str = typer.Option(
        "hejiaz/swe-agent:latest", "--orcaloca-docker-image",
    ),
    embed_preset: str | None = typer.Option(
        None, "--embed-preset",
        help="EmbedExplorer preset (e.g. bge-code-v1, jina-v4, text-embedding-3-large)",
    ),
    embed_backend: str = typer.Option("sentence_transformers", "--embed-backend"),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, "--embed-model"),
    embed_api_key: str | None = typer.Option(None, "--embed-api-key"),
    embed_api_base: str | None = typer.Option(None, "--embed-api-base"),
    swerank_embed_model: str = typer.Option(
        DEFAULT_EMBED_MODEL, "--swerank-embed-model",
    ),
    swerank_rerank_model: str = typer.Option(
        "gpt-5.4", "--swerank-rerank-model",
    ),
    swerank_api_key: str | None = typer.Option(
        None, "--swerank-api-key", envvar="SWERANK_API_KEY",
    ),
    swerank_api_base: str | None = typer.Option(
        None, "--swerank-api-base", envvar="SWERANK_API_BASE",
    ),
    workers: int = typer.Option(
        1, "--workers", "-w",
        help="Parallel workers (for all explorers)",
    ),
    log_level: str = typer.Option(
        "info", "--log",
        help="Console log level: info, debug, or trace. Debug logs CLI-agent "
        "launch/return; trace additionally dumps the agent output before "
        "parsing. Applies to shared-base CLI agents (opencode, deveco).",
    ),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    skip_missing_repo: bool = typer.Option(True, "--skip-missing-repo/--no-skip-missing-repo"),
    retry_failed: bool = typer.Option(
        True, "--retry-failed/--no-retry-failed",
        help="On --resume, run the cases whose rows record a failure again "
             "(replacing those rows) instead of keeping them as done.",
    ),
    no_line_counts: bool = typer.Option(False, "--no-line-counts"),
    skip_empty_core: bool = typer.Option(
        True, "--skip-empty-core/--no-skip-empty-core",
        help="Skip instances with empty read_core_regions (default: True)",
    ),
    output_jsonl: str | None = typer.Option(
        None, "--output", "-o",
        help="Save per-instance results to JSONL. Supports {explorer} and {k} "
        "placeholders; leaving one out points several explorers or budgets at a "
        "single file, which --resume cannot continue.",
    ),
    resume: bool = typer.Option(
        False, "--resume/--no-resume",
        help="Resume from existing output files, skipping instances already scored "
        "at every top_k. Needs one output file per explorer and budget.",
    ),
) -> None:
    """Run evaluation for one or more explorers."""
    bench_path = bench_path.resolve()
    if not bench_path.is_file():
        console.print(f"[red]bench not found: {bench_path}[/red]")
        raise typer.Exit(1)

    if repos_root is not None and not repos_root.is_dir():
        console.print("[yellow]repos dir missing; local explorers need it[/yellow]")
        repos_root = None

    top_k_list = _parse_top_k_list(top_k_str)
    max_top_k = max(top_k_list)
    console.print(f"[dim]top_k values: {top_k_list}[/dim]")

    # Set generic env vars for explorers that read MSWEA_* directly.
    import os
    os.environ.setdefault("DEFAULT_LLM_PROVIDER", "openai")
    os.environ.setdefault("MSWEA_API_KEY", os.environ.get("SWERANK_API_KEY", ""))
    os.environ.setdefault("MSWEA_AZURE_ENDPOINT", os.environ.get("LLM_API_BASE", ""))
    os.environ.setdefault("MSWEA_MODEL_NAME", os.environ.get("LLM_DEPLOYMENT", "gpt-5.4"))
    os.environ.setdefault("MSWEA_API_VERSION", "2024-12-01-preview")

    if log_level.lower() not in {"info", "debug", "trace"}:
        console.print(
            f"[red]Invalid --log level: {log_level} (info|debug|trace)[/red]"
        )
        raise typer.Exit(1)
    set_log_level(log_level)

    register_litellm_usage_callback()

    records = _load_bench_records(bench_path)
    if skip_empty_core:
        before = len(records)
        records = [
            r for r in records
            if (r.get("ground_truth") or {}).get("read_core_regions")
        ]
        console.print(f"[dim]Filtered to {len(records)}/{before} instances with non-empty core[/dim]")
    if limit:
        records = records[:limit]

    issue_map: dict[str, str] = {}
    if issue_map_file and issue_map_file.is_file():
        with open(issue_map_file, encoding="utf-8") as f:
            issue_map = json.load(f)
    elif trajs_dir and trajs_dir.is_dir():
        issue_map = _load_issue_map(trajs_dir)
    else:
        console.print("[yellow]trajs_dir missing; using problem_statement from bench[/yellow]")

    file_line_counts = {} if no_line_counts else _build_file_line_counts(records, repos_root)
    evaluator = ExploreEvaluator(bench_path, file_line_counts=file_line_counts)

    explorer_names = [x.strip().lower() for x in explorers]
    unknown = set(explorer_names) - ALL_EXPLORERS
    if unknown:
        console.print(f"[red]Unknown explorers: {unknown}[/red]")
        raise typer.Exit(1)
    if "codenib" in explorer_names:
        try:
            from codenib.agent import normalize_repository_explorer_policy

            codenib_policy = normalize_repository_explorer_policy(codenib_policy)
        except (ImportError, ValueError) as exc:
            console.print(f"[red]Invalid CodeNib setup or policy: {exc}[/red]")
            raise typer.Exit(1) from exc
        if codenib_planning_budget not in {"fast", "balanced", "thorough"}:
            console.print("[red]Invalid CodeNib planning budget[/red]")
            raise typer.Exit(1)
        if codenib_retrieval_level not in {"l0", "l2"}:
            console.print("[red]Invalid CodeNib retrieval level[/red]")
            raise typer.Exit(1)
        console.print(
            "[dim]CodeNib configuration: "
            f"policy={codenib_policy}, budget={codenib_planning_budget}, "
            f"level={codenib_retrieval_level}[/dim]"
        )

    # ── shared model caches (survive across instances) ──
    _potion_model = None
    _rag_st_model = None
    _embed_st_cache: dict = {}  # sentence-transformers model cache (shared across EmbedExplorer instances)
    _swerank_embedder = None

    # ── local explorer methods ──
    # Each returns a list of (path, start, end) regions using max_top_k.
    # We slice later for each actual top_k value.

    def _get_repo_dir(rec: dict) -> Path | None:
        iid = rec.get("instance_id", "")
        rd = _resolve_repo_dir(rec.get("repo_dir"), repos_root, iid)
        if rd and rd.is_dir():
            return rd
        return None

    def _case_repo_dir(rec: dict) -> Path | None:
        """The checkout for a case, None to skip it, or a failure with --no-skip-missing-repo."""
        rd = _get_repo_dir(rec)
        if rd is not None or skip_missing_repo:
            return rd
        # Portable: this becomes a row's `error`, and rows may be published.
        where = f" under {_portable_path(repos_root)}" if repos_root is not None else ""
        raise ExplorerFailure(ERROR, f"no checkout for {rec.get('instance_id', '')}{where}")

    def _get_issue(rec: dict) -> str:
        iid = rec.get("instance_id", "")
        issue = issue_map.get(iid, "")
        if not issue:
            issue = rec.get("problem_statement", "")
        return issue

    def bm25_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.bm25 import BM25Explorer as LineBM25Explorer
        rd = _case_repo_dir(rec)
        if rd is None:
            return None
        explorer = LineBM25Explorer(rd, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        results = explorer.explore(instance_id=rec["instance_id"], query=_get_issue(rec), top_k=max_top_k)
        return _results_to_regions(results)

    def codenib_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.codenib_explorer import CodeNibExplorer
        rd = _case_repo_dir(rec)
        if rd is None:
            return None
        with CodeNibExplorer(
            rd,
            auto_index=codenib_auto_index,
            rebuild=codenib_rebuild,
            policy=codenib_policy,
            planning_budget=codenib_planning_budget,
            retrieval_level=codenib_retrieval_level,
        ) as explorer:
            results = explorer.explore(
                instance_id=rec["instance_id"],
                query=_get_issue(rec),
                top_k=max_top_k,
            )
        return _results_to_regions(results)

    def rag_method(rec: dict) -> list[tuple[str, int, int]] | None:
        nonlocal _rag_st_model
        from explorers.rag import RAGExplorer
        rd = _case_repo_dir(rec)
        if rd is None:
            return None
        if _rag_st_model is None:
            from sentence_transformers import SentenceTransformer
            _rag_st_model = SentenceTransformer(
                embed_model or DEFAULT_EMBED_MODEL, trust_remote_code=True
            )
        explorer = RAGExplorer(rd, _model=_rag_st_model)
        results = explorer.explore(instance_id=rec["instance_id"], query=_get_issue(rec), top_k=max_top_k)
        return _results_to_regions(results)

    def tfidf_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.rag_tfidf import TFIDFExplorer
        rd = _case_repo_dir(rec)
        if rd is None:
            return None
        explorer = TFIDFExplorer(rd, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        results = explorer.explore(instance_id=rec["instance_id"], query=_get_issue(rec), top_k=max_top_k)
        return _results_to_regions(results)

    def potion_method(rec: dict) -> list[tuple[str, int, int]] | None:
        nonlocal _potion_model
        from explorers.rag_potion import PotionExplorer, _load_potion_model
        rd = _case_repo_dir(rec)
        if rd is None:
            return None
        if _potion_model is None:
            _potion_model = _load_potion_model(potion_model_path)
        explorer = PotionExplorer(
            rd, model_path=potion_model_path,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            _model=_potion_model,
        )
        results = explorer.explore(instance_id=rec["instance_id"], query=_get_issue(rec), top_k=max_top_k)
        return _results_to_regions(results)

    def simple_rule_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.baselines import SimpleRuleExplorer
        rd = _case_repo_dir(rec)
        if rd is None:
            return None
        explorer = SimpleRuleExplorer(rd)
        results = explorer.explore(instance_id=rec["instance_id"], query=_get_issue(rec), top_k=max_top_k)
        return _results_to_regions(results)

    def oracle_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.baselines import OracleExplorer
        explorer = OracleExplorer(bench_path)
        results = explorer.explore(instance_id=rec["instance_id"], query=_get_issue(rec), top_k=max_top_k)
        return _results_to_regions(results)

    def random_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.baselines import RandomExplorer
        explorer = RandomExplorer(bench_path)
        results = explorer.explore(instance_id=rec["instance_id"], query=_get_issue(rec), top_k=max_top_k)
        return _results_to_regions(results)

    def embed_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.rag_embed import EmbedExplorer
        rd = _case_repo_dir(rec)
        if rd is None:
            return None
        explorer = EmbedExplorer(
            rd,
            backend=embed_backend,
            model_name=embed_model,
            preset=embed_preset,
            api_key=embed_api_key,
            api_base=embed_api_base,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        results = explorer.explore(instance_id=rec["instance_id"], query=_get_issue(rec), top_k=max_top_k)
        return _results_to_regions(results)

    def swerank_method(rec: dict) -> list[tuple[str, int, int]] | None:
        nonlocal _swerank_embedder
        from explorers.swerank import SweRankExplorer
        rd = _case_repo_dir(rec)
        if rd is None:
            return None
        if _swerank_embedder is None:
            from sentence_transformers import SentenceTransformer
            _swerank_embedder = SentenceTransformer(swerank_embed_model, trust_remote_code=True)
        explorer = SweRankExplorer(
            rd,
            embed_model=swerank_embed_model,
            embedder=_swerank_embedder,
            rerank_model=swerank_rerank_model,
            api_key=swerank_api_key or "",
            api_base=swerank_api_base or "",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        results = explorer.explore(instance_id=rec["instance_id"], query=_get_issue(rec), top_k=max_top_k)
        return _results_to_regions(results)

    # ── agentic explorer methods ──
    def _agentic_method(
        rec: dict, make_explorer: Callable,
    ) -> list[tuple[str, int, int]] | None:
        rd = _case_repo_dir(rec)
        if rd is None:
            return None
        explorer = make_explorer(rd)
        results = explorer.explore(
            instance_id=rec["instance_id"], query=_get_issue(rec), top_k=max_top_k,
        )
        return _results_to_regions(results)

    # Resolved here so both Claude Code (Anthropic-protocol via LiteLLM) and
    # academic explorers (OpenAI-protocol via LiteLLM) share the same key.
    _academic_key = academic_api_key or os.environ.get("SWERANK_API_KEY", "") or "sk-swe-explore-local"

    def claude_code_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.claude_code import ClaudeCodeExplorer
        return _agentic_method(
            rec,
            lambda rd: ClaudeCodeExplorer(
                repo_root=rd, model=claude_model, timeout=claude_timeout,
                api_base=academic_api_base, api_key=_academic_key,
            ),
        )

    def cursor_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.cursor_agent import CursorAgentExplorer
        return _agentic_method(
            rec,
            lambda rd: CursorAgentExplorer(
                repo_root=rd,
                api_key=cursor_api_key or "",
                model=cursor_model or "",
            ),
        )

    def make_opencode(rd: Path, model: str = opencode_model, agent: str = opencode_agent):
        from explorers.opencode import OpenCodeExplorer
        return OpenCodeExplorer(
            repo_root=rd,
            bin_path=opencode_bin,
            timeout=opencode_timeout,
            config_dir=opencode_config_dir,
            prompt_additions=opencode_prompt_additions,
            model=model,
            agent=agent,
        )

    def make_deveco(rd: Path, model: str = deveco_model, agent: str = deveco_agent):
        from explorers.deveco import DevEcoExplorer
        return DevEcoExplorer(
            repo_root=rd,
            bin_path=deveco_bin,
            timeout=deveco_timeout,
            config_dir=deveco_config_dir,
            prompt_additions=deveco_prompt_additions,
            skip_permissions=deveco_skip_permissions,
            model=model,
            agent=agent,
        )

    CLI_AGENT_MAKERS = {"opencode": make_opencode, "deveco": make_deveco}
    CLI_AGENT_FLAG_AGENTS = {"opencode": opencode_agent, "deveco": deveco_agent}

    # Filled before the first case runs; every CLI run pins the model it names.
    explorer_configs: dict[str, dict] = {}

    def _cli_agent_method(name: str) -> Callable[[dict], list[tuple[str, int, int]] | None]:
        def method(rec: dict) -> list[tuple[str, int, int]] | None:
            config = explorer_configs[name]
            model = config["model"] or ""
            # Pin the agent only when its model was chosen; otherwise the CLI
            # keeps its own fallback.
            agent = config["agent"] if config.get("model_source") == "agent" else ""
            return _agentic_method(
                rec, lambda rd: CLI_AGENT_MAKERS[name](rd, model, agent or CLI_AGENT_FLAG_AGENTS[name])
            )
        return method

    opencode_method = _cli_agent_method("opencode")
    deveco_method = _cli_agent_method("deveco")

    # ── academic-agent methods ──

    def autocr_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.autocr_explorer import AutoCodeRoverExplorer
        return _agentic_method(
            rec,
            lambda rd: AutoCodeRoverExplorer(
                repo_root=rd, model=academic_model,
                api_key=_academic_key, api_base=academic_api_base,
                timeout=academic_timeout,
            ),
        )

    def cosil_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.cosil_explorer import CoSILExplorer
        return _agentic_method(
            rec,
            lambda rd: CoSILExplorer(
                repo_root=rd, model=academic_model,
                api_key=_academic_key, api_base=academic_api_base,
                timeout=academic_timeout,
            ),
        )

    def locagent_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.locagent_explorer import LocAgentExplorer
        # LocAgent uses litellm — needs "openai/" prefix to route to OpenAI-compatible proxy
        m = academic_model if academic_model.startswith("openai/") else f"openai/{academic_model}"
        return _agentic_method(
            rec,
            lambda rd: LocAgentExplorer(
                repo_root=rd, model=m,
                api_key=_academic_key, api_base=academic_api_base,
                timeout=academic_timeout,
            ),
        )

    def orcaloca_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.orcaloca_explorer import OrcaLocaExplorer
        # OrcaLoca uses llama_index.OpenAI which validates the model name against a
        # hard-coded list. Always advertise "gpt-4o" — the LiteLLM proxy maps it to
        # the same Azure deployment as gpt-5.4.
        m = "gpt-4o"
        return _agentic_method(
            rec,
            lambda rd: OrcaLocaExplorer(
                repo_root=rd, model=m,
                docker_image=orcaloca_docker_image,
                api_key=_academic_key, api_base=academic_api_base,
                timeout=academic_timeout,
            ),
        )

    def mini_swe_agent_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.mini_swe_agent_explorer import MiniSWEAgentExplorer
        # mini-swe-agent uses litellm — needs "openai/" prefix to route via the proxy.
        m = academic_model if "/" in academic_model else f"openai/{academic_model}"
        return _agentic_method(
            rec,
            lambda rd: MiniSWEAgentExplorer(
                repo_root=rd, model=m,
                api_key=_academic_key, api_base=academic_api_base,
                timeout=academic_timeout,
            ),
        )

    def awe_agent_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.awe_agent_explorer import AweAgentExplorer
        return _agentic_method(
            rec,
            lambda rd: AweAgentExplorer(
                repo_root=rd, model=academic_model,
                api_key=_academic_key, base_url=academic_api_base,
            ),
        )

    METHOD_MAP: dict[str, Callable] = {
        "bm25": bm25_method,
        "codenib": codenib_method,
        "rag": rag_method,
        "tfidf": tfidf_method,
        "potion": potion_method,
        "simple_rule": simple_rule_method,
        "oracle": oracle_method,
        "random": random_method,
        "embed": embed_method,
        "swerank": swerank_method,
        "claude_code": claude_code_method,
        "cursor": cursor_method,
        "opencode": opencode_method,
        "deveco": deveco_method,
        "autocr": autocr_method,
        "cosil": cosil_method,
        "locagent": locagent_method,
        "orcaloca": orcaloca_method,
        "mini_swe_agent": mini_swe_agent_method,
        "awe_agent": awe_agent_method,
    }

    # ── configuration each explorer runs with (the run manifest) ──
    def _explorer_config(name: str) -> dict:
        if name in CHUNKED_EXPLORERS:
            config = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            if name == "embed":
                config.update(
                    backend=embed_backend, model=embed_model, preset=embed_preset
                )
            elif name == "swerank":
                config.update(
                    embed_model=swerank_embed_model,
                    rerank_model=swerank_rerank_model,
                )
            elif name == "potion":
                config.update(model_path=_portable_path(potion_model_path))
            return config
        if name == "rag":
            return {"model": embed_model or DEFAULT_EMBED_MODEL}
        if name == "codenib":
            return {
                "policy": codenib_policy,
                "planning_budget": codenib_planning_budget,
                "retrieval_level": codenib_retrieval_level,
            }
        if name in CLI_AGENT_MAKERS:
            # Once per run: `debug config` takes seconds. The repo root is unused.
            return CLI_AGENT_MAKERS[name](Path.cwd()).describe()
        if name == "claude_code":
            return {"model": claude_model}
        if name == "cursor":
            return {"model": cursor_model or None}
        if name in ACADEMIC_EXPLORERS:
            return {"model": academic_model}
        return {}

    for name in explorer_names:
        try:
            explorer_configs[name] = _explorer_config(name)
        except FileNotFoundError as exc:
            console.print(f"[red]{name}: {exc}[/red]")
            raise typer.Exit(1) from exc
        if name in CLI_AGENT_MAKERS:
            cfg = explorer_configs[name]
            console.print(
                f"[dim]{name} configuration: model={cfg['model']} "
                f"({cfg['model_source']}), cli={cfg['cli_version']}[/dim]"
            )
            if cfg["resolved_config_sha256"] is None:
                console.print(
                    f"[yellow]{name}: the CLI did not answer `debug config`, so its "
                    f"resolved configuration is unknown and cannot be compared on a "
                    f"later --resume[/yellow]"
                )
            if not cfg["model"]:
                console.print(
                    f"[yellow]{name}: no model named by --{name}-model or the "
                    f"configuration; each run uses whatever the CLI defaults to[/yellow]"
                )
    manifests = _build_manifests(explorer_configs, bench_path, issue_map)

    # ── evaluation loop ──
    if output_jsonl:
        if resume:
            with _resume_mismatch_exits():
                for warning in _check_resume(output_jsonl, explorer_names, manifests, top_k_list):
                    console.print(f"[yellow]{warning}[/yellow]")
        else:
            _clear_outputs(output_jsonl, explorer_names, top_k_list)
        _write_manifests(output_jsonl, manifests, top_k_list, fresh=not resume)
    total_records = len(records)
    # After --limit and id filters; the completion rate and resume both use it.
    selected_ids = {r.get("instance_id") for r in records}

    for name in explorer_names:
        method = METHOD_MAP[name]

        # ── resume: load existing results and skip completed instances ──
        tally = _Tally(top_k_list)
        not_attempted = 0
        resumed_ids: set[str] = set()

        if resume and output_jsonl:
            with _resume_mismatch_exits():
                resumed_ids, kept_per_k, retried = _reconcile_resume_state(
                    output_jsonl, name, top_k_list,
                    retry_failed=retry_failed, selected_ids=selected_ids,
                )
            if retried:
                console.print(
                    f"  [yellow]{name}: {retried} case(s) that failed before will be "
                    f"run again; their rows are replaced (--no-retry-failed keeps "
                    f"them)[/yellow]"
                )
            # Only selected cases count; a narrowed run leaves other rows on disk.
            for k in top_k_list:
                for r in kept_per_k[k]:
                    if r.get("instance_id") in selected_ids:
                        tally.add_row(k, r)
            # Usage is per case: count it once, from the first budget.
            for r in tally.rows[top_k_list[0]]:
                tally.add_usage(TokenUsage.from_dict(r.get("token_usage")))

        remaining_records = [r for r in records if r.get("instance_id", "") not in resumed_ids]
        total_remaining = len(remaining_records)
        console.print(
            f"\n[bold cyan]▶ {name}[/bold cyan]  "
            f"({total_remaining} to run, {len(resumed_ids & selected_ids)} resumed, "
            f"top_k={top_k_list})"
        )
        if total_remaining == 0:
            console.print(f"  [dim]All instances already completed, skipping.[/dim]")
            _report_explorer(name, tally, total_records, output_jsonl)
            continue

        t0 = time.time()
        primary_k = top_k_list[0]
        console.print(
            f"  [dim]case log tuple = (prec, recall, f1, in, out, think, total); "
            f"aggr = metrics averaged, tokens summed over evaluated cases "
            f"@ top_k={primary_k}[/dim]"
        )

        def _eval_one(rec: dict) -> _CaseRun:
            iid = rec.get("instance_id", "")
            case_t0 = time.perf_counter()
            outcome, error = SUCCESS, None
            with usage_collector() as tracker:
                try:
                    preds = method(rec)
                    if preds is None:
                        outcome = NOT_ATTEMPTED
                except Exception as e:
                    # An unclassified crash needs its exception type.
                    outcome = classify_failure(e)
                    error = (
                        str(e) if isinstance(e, ExplorerFailure)
                        else f"{type(e).__name__}: {e}"
                    )
                    preds = []
                    sys.stderr.write(f"\n  [ERROR] {name} {iid} ({outcome}): {e}\n")
            repo_dir = _get_repo_dir(rec)
            return _CaseRun(
                iid,
                preds,
                tracker if tracker.has_any() else None,
                time.perf_counter() - case_t0,
                outcome,
                error[:MAX_ERROR_CHARS] if error else None,
                _git_revision(repo_dir) if repo_dir is not None else None,
            )

        def _score_instance(iid: str, preds: list[tuple[str, int, int]]) -> dict[int, dict[str, float]]:
            """Evaluate one instance at all top_k values. Returns {k: {metric: score}}."""
            bench_gt = evaluator.bench_data_dict[iid]["ground_truth"]
            per_file_lines = file_line_counts.get(iid, {})
            result_per_k: dict[int, dict[str, float]] = {}
            for k in top_k_list:
                sliced = preds[:k]
                evaluator._current_instance_id = iid
                evaluator._current_file_line_counts = per_file_lines
                scores = {}
                for metric in METRICS:
                    scores[metric] = getattr(evaluator, f"evaluate_{metric}")(sliced, bench_gt)
                result_per_k[k] = scores
            return result_per_k

        # Budgets sharing one file share one handle.
        out_files: dict[int, object] = {}
        out_handles = contextlib.ExitStack()
        if output_jsonl:
            with contextlib.ExitStack() as opening:
                by_path: dict[Path, object] = {}
                for k in top_k_list:
                    out_path = _format_output_path(output_jsonl, name, k)
                    if out_path not in by_path:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        by_path[out_path] = opening.enter_context(
                            out_path.open("a", encoding="utf-8")
                        )
                    out_files[k] = by_path[out_path]
                out_handles = opening.pop_all()

        row_config = _row_explorer_config(explorer_configs[name])

        def _record_result(case: _CaseRun) -> tuple[float, float, float]:
            """Score and write one attempted case; a failure scores as an empty answer."""
            iid, preds, usage = case.iid, case.preds, case.usage
            scores_per_k = _score_instance(iid, preds)
            row_usage = usage.to_dict() if usage is not None else None
            for k in top_k_list:
                sliced = preds[:k]
                row = {
                    "instance_id": iid,
                    "explorer": name,
                    "outcome": case.outcome,
                    "error": case.error,
                    "regions": [{"path": p, "start": s, "end": e} for p, s, e in sliced],
                    "metrics": scores_per_k[k],
                    "num_regions": min(len(preds), k),
                    "token_usage": row_usage,
                    "repo_revision": case.repo_revision,
                }
                if row_config:
                    row["explorer_config"] = row_config
                tally.add_row(k, row)
                if k in out_files:
                    _append_row(out_files[k], row)
            tally.add_usage(usage)
            primary_scores = scores_per_k[primary_k]
            return (
                primary_scores["precision"],
                primary_scores["recall"],
                primary_scores["f1_score"],
            )

        def _log_case(
            iid: str,
            done: int,
            case_scores: tuple[float, float, float],
            usage: TokenUsage | None,
            case_dt: float,
            elapsed: float,
            eta: float,
        ) -> None:
            """Per-case progress log: case tuple, aggregates, timings."""
            case_u = usage if usage is not None else TokenUsage()
            case_vals = (
                *case_scores,
                case_u.input_tokens,
                case_u.output_tokens,
                case_u.reasoning_tokens,
                case_u.total,
            )
            avg = tally.averages(primary_k)
            aggr_vals = (
                avg["precision"],
                avg["recall"],
                avg["f1_score"],
                tally.usage.input_tokens,
                tally.usage.output_tokens,
                tally.usage.reasoning_tokens,
                tally.usage.total,
            )
            labels = ("prec", "recall", "f1", "in", "out", "think", "total")

            def fmt(vals: tuple) -> str:
                parts = [
                    f"{labels[i]}: {v:.3f}" if i < 3 else f"{labels[i]}: {v:,.0f}"
                    for i, v in enumerate(vals)
                ]
                return "(" + ", ".join(parts) + ")"

            now = time.strftime("%H:%M:%S")
            sys.stderr.write(
                f"\n  [{name} {now}] case {done}/{total_remaining} {iid}  "
                f"case={fmt(case_vals)}  aggr={fmt(aggr_vals)}  "
                f"time={case_dt:.0f}s elapsed={elapsed:.0f}s ETA={eta:.0f}s\n"
            )
            sys.stderr.flush()

        done = 0
        interrupted = False

        def _consume(case: _CaseRun) -> None:
            """Report and record one finished case, however it was run."""
            nonlocal done, not_attempted
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_remaining - done) / rate if rate > 0 else 0
            sys.stderr.write(
                f"\r  [{name}] {done}/{total_remaining}  "
                f"{rate:.1f} it/s  ETA {eta:.0f}s  "
            )
            sys.stderr.flush()
            if case.preds is None:
                not_attempted += 1
                return
            case_scores = _record_result(case)
            _log_case(case.iid, done, case_scores, case.usage, case.seconds, elapsed, eta)

        try:
            if workers > 1:
                with _interruptible_pool(workers) as pool:
                    futures = [pool.submit(_eval_one, rec) for rec in remaining_records]
                    for fut in as_completed(futures):
                        _consume(fut.result())
            else:
                with _cli_cancellation():
                    for rec in remaining_records:
                        _consume(_eval_one(rec))
        except KeyboardInterrupt:
            interrupted = True
        finally:
            out_handles.close()

        if interrupted:
            print(file=sys.stderr)
            hint = (
                "results flushed, rerun with --resume."
                if out_files else "no output file, nothing was saved."
            )
            console.print(
                f"  [yellow]Interrupted after {done}/{total_remaining} instances; "
                f"{hint}[/yellow]"
            )
            raise typer.Exit(130)

        sys.stderr.write("\n")
        total_elapsed = time.time() - t0
        console.print(
            f"  [dim]{name} done in {total_elapsed:.0f}s  "
            f"(eval={len(tally.rows[primary_k])}, not attempted={not_attempted}, "
            f"resumed={len(resumed_ids & selected_ids)})[/dim]"
        )
        _report_explorer(name, tally, total_records, output_jsonl)

        # ── save per top_k (already written incrementally; just log) ──
        if output_jsonl:
            for k in top_k_list:
                out_path = _format_output_path(output_jsonl, name, k)
                console.print(f"  [green]Saved {out_path} ({len(tally.rows[k])} records)[/green]")


if __name__ == "__main__":
    app()
