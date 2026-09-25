"""Unified evaluation runner for SWE-Explore.

Supports both local (BM25, TF-IDF, Potion, RAG, SimpleRule, Oracle, Random)
and agentic (Claude Code, Cursor Agent) explorers.

Usage:
    python eval_runner.py --explorers bm25 tfidf --top-k 5,10,20 -o results/{explorer}/top{k}.jsonl
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
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
from explorers._cli_agent_base import set_log_level
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
                # Set the event outside the signal handler, including when
                # SIGINT arrives during an otherwise normal pool shutdown.
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
#: Explorers the runner hands --chunk-size / --chunk-overlap to.
CHUNKED_EXPLORERS = {"bm25", "tfidf", "potion", "embed", "swerank"}
ACADEMIC_EXPLORERS = {"autocr", "cosil", "locagent", "orcaloca", "mini_swe_agent", "awe_agent"}
ALL_EXPLORERS = LOCAL_EXPLORERS | AGENTIC_EXPLORERS | ACADEMIC_EXPLORERS
#: The sentence-transformers model `rag`, `embed` and `swerank` fall back to.
#: Named once so the manifest records the model a run actually loaded.
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
    # Sorted: two trajectory files can carry the same instance, the first one
    # read wins, and `rglob` alone would let the filesystem pick the query the
    # run is given — and with it `issues_sha256`.
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
    # A file written before results were UTF-8 everywhere (a Windows run wrote
    # cp1252) or holding bytes from an agent CLI must not abort the resume:
    # replace what cannot be decoded and let json.loads keep or skip the line.
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
    """Serialize one result row as a JSONL line."""
    return json.dumps(row, ensure_ascii=False) + "\n"


def _case_ids(rows: list[dict], *, finished_only: bool = False) -> set[str]:
    """The instance_ids in `rows`, skipping rows that identify no case.

    A missing, null or empty id can be matched against no bench record and
    re-run for none either, and an empty one would stand in for every bench
    record that has no id of its own.

    With `finished_only`, a row that records a failure does not count as a
    case that is done: a timeout or a rate limit is a reason to run it again,
    not a verdict. A row written before outcomes existed records none and is
    taken as finished, since there is nothing to say it is not.
    """
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
    """Yield a temporary path, then move it over `path` atomically.

    Written through to the real file: an output path symlinked to shared
    storage must keep pointing there, and the temporary file has to land on
    the target's filesystem for the replace to be atomic.
    """
    target = path.resolve()
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        yield tmp
        tmp.replace(target)
    finally:
        # An interrupt or a failed write leaves the half-written file behind,
        # where the next run would find it beside the results it is not.
        tmp.unlink(missing_ok=True)


def _rewrite_results(path: Path, rows: list[dict]) -> None:
    """Replace a JSONL result file with `rows`, atomically."""
    with _atomic_target(path) as tmp:
        with tmp.open("w", encoding="utf-8") as f:
            f.writelines(_dump_row(row) for row in rows)


def _append_row(fh, row: dict) -> None:
    """Append one result row to an already-open JSONL file and flush it."""
    fh.write(_dump_row(row))
    fh.flush()


class ResumeMismatch(RuntimeError):
    """The existing result files do not match the run being resumed."""


def _clear_outputs(output_jsonl: str, explorers: list[str], top_k_list: list[int]) -> None:
    """Empty every result file a fresh run will write, before it writes any.

    Clearing them lazily, as each explorer starts, would let an interrupt leave
    a later explorer's file holding a previous run's rows — which a subsequent
    --resume would then adopt as this run's work.
    """
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
    """Read an explorer's result files, rejecting a layout it cannot prune.

    `_check_resume` has already validated the output template, so every budget
    here has a file of its own.
    """
    out_paths = {k: _format_output_path(output_jsonl, explorer, k) for k in top_k_list}
    existing = {k: _load_existing_results(path) for k, path in out_paths.items()}
    # Rows record the explorer that produced them. A file holding another
    # explorer's rows — an earlier run with a different --explorers set and no
    # {explorer} in --output — cannot be pruned as this one's: those rows would
    # be scored as ours and then rewritten away. A row with no explorer
    # recorded predates the field and is taken as ours.
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
    # A budget with no file at all has finished no cases, so pruning against it
    # would discard every row the other budgets hold. That means the --top-k
    # set or the --output path changed, not that a write was interrupted.
    # Existing empty files are valid: a kill during the first case can leave
    # later budgets empty. Resume must rerun that incomplete case.
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

    Results for one case are written to one file per top_k budget in sequence,
    so an interrupt in that gap can leave the case in some files but not
    others. Only cases present in ALL of them count as resumed; the rest are
    dropped from disk here, so re-running them replaces their rows instead of
    appending a second set and scoring the case twice. Each existing file is
    rewritten whole, which also repairs a line left half-written by a hard kill.

    With `retry_failed`, a case whose row records a failure is dropped the
    same way and run again. Before outcomes were recorded a failed case wrote
    no row at all and a resume simply retried it; now that every attempt
    leaves a row, keeping it would turn one rate limit into a permanent zero.

    Only a case this run selected is retried: dropping a failed row that
    `--limit` or `--instance-ids` excludes would delete a result — its error
    and the tokens it spent — that nothing in this run is going to replace.
    A narrowed run reads less of the file, and it destroys nothing.

    Returns the resumed instance_ids, the surviving rows per top_k, and how
    many previously failed cases are being run again.
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


# ── run manifest ────────────────────────────────────────────────────────
#
# Every result file `X.jsonl` gets a sidecar `X.manifest.json` recording, per
# explorer, the configuration that produced its rows and a summary of how the
# cases ended. See README "Run manifest".

MANIFEST_SCHEMA = 1
#: Manifest fields that identify the experiment; resume refuses to mix rows
#: across a change to any of them. `recorded` is informational.
_MANIFEST_IDENTITY = ("explorer", "bench_sha256", "issues_sha256", "explorer_config")
#: `explorer_config` keys that say how a value was chosen rather than what ran.
#: Pinning the model the configuration had already chosen produces the same
#: argv, so `model_source: config -> flag` is not a different experiment.
_INFORMATIONAL_CONFIG_KEYS = frozenset({"model_source"})


def _manifest_path(results_path: Path) -> Path:
    return results_path.with_name(results_path.stem + ".manifest.json")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@functools.lru_cache(maxsize=None)
def _git_revision(path: Path) -> str | None:
    """HEAD of the git work tree rooted exactly at `path`, else None.

    Only a work tree rooted there counts: a snapshot extracted from a tarball
    inside some other checkout must not report that checkout's HEAD.

    Cached: a checkout holds every case of one repository and nothing moves
    its HEAD during a run, so this forks `git` once per directory rather than
    once per case per explorer.
    """
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


def _json_sha256(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _row_explorer_config(config: dict) -> dict:
    """What a result row repeats from the explorer's configuration.

    A CLI agent's block is four hashes, a version and an MCP map, identical
    in every row at every budget; the manifest beside the file records it
    once. The row keeps the model, which is what a row is read for. Every
    other explorer's block is a handful of settings and is kept whole.
    """
    if "cli" in config:
        return {key: config[key] for key in ("cli", "model") if key in config}
    return config


def _portable_path(value: str | Path) -> str:
    """`value` with this machine left out of it.

    A manifest is read on another checkout and another machine, and an
    absolute path carries a username and a directory layout into a file that
    is meant to be shareable. A path inside the working directory is recorded
    relative to it; anything else keeps its name only. A value that is not a
    path at all — a model id such as `minishlab/potion-base-8M` — is already
    relative and is left exactly as it is.
    """
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
    """One manifest per explorer. The bench is hashed once, not once each."""
    recorded = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        # Informational: `bench_sha256` is what identifies the bench.
        "bench_path": _portable_path(bench_path),
        "harness_revision": _git_revision(Path(__file__).resolve().parent),
    }
    bench_sha256 = _file_sha256(bench_path)
    # The issue text is every explorer's query, so a changed issue map is a
    # changed experiment even when the bench and the explorer are identical.
    issues_sha256 = _json_sha256(issue_map)
    # Compare what a JSON round trip gives back, not Python tuples and the like.
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
    """The identity fields that differ between two manifests, readably.

    Returns the differences and the fields that could not be compared. A
    `None` is what a probe writes when it could not run — a CLI that took
    longer than its timeout to answer `--version` on a loaded machine — and
    it means *unknown*, not *absent*. Refusing a resume over one slow startup
    would make the manifest a liability rather than a safeguard, so an
    unknown on either side is reported and stepped over.
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
                # Qualified, so a bare `model:` says which block it is in.
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
    """A sidecar's schema and `explorers` map; a missing or malformed part
    reads as the default, since sidecars can be hand-edited."""
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

    Checking as each explorer starts would surface a later explorer's problem
    hours of API spend into the run. Returns what to warn about rather than
    refuse over: results written before manifests existed are adopted as this
    configuration's, and a field one side does not know is stepped over.
    """
    # A result row records no top_k, and resume rewrites the files it prunes,
    # so two budgets (or two explorers) sharing a file would each load the
    # other's rows as their own and then delete them.
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
        # Every budget repeats the same finding.
        if message not in warnings:
            warnings.append(message)

    for explorer, manifest in manifests.items():
        _load_resume_state(output_jsonl, explorer, top_k_list)
        for k in top_k_list:
            results = _format_output_path(output_jsonl, explorer, k)
            sidecar = _manifest_path(results)
            schema, entries = _read_sidecar(sidecar)
            # A newer harness may record fields this one cannot compare, so
            # silently continuing its run could mix configurations exactly as
            # this check exists to prevent.
            if not isinstance(schema, int) or schema > MANIFEST_SCHEMA:
                raise ResumeMismatch(
                    f"cannot resume {explorer}: {sidecar} records "
                    f"manifest schema {schema!r}, and this harness understands "
                    f"{MANIFEST_SCHEMA}. Use a newer harness, write to a new "
                    f"--output, or rerun without --resume."
                )
            entry = entries.get(explorer)
            if not isinstance(entry, dict):
                # Only whether the file holds anything matters here, so ask
                # the filesystem rather than parsing every row back.
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
                    f"could not run records no value — so it was not compared"
                )
    return warnings


def _write_manifests(
    output_jsonl: str, manifests: dict[str, dict], top_k_list: list[int], *, fresh: bool
) -> None:
    """Record each explorer's manifest beside its result files.

    A fresh run starts every sidecar over; a resumed one keeps the entries it
    has already checked, summaries included, and adds the missing ones.
    """
    # Grouped, because an --output template without {explorer} or {k} maps
    # several of them onto one sidecar, and a fresh write must not reset the
    # file between two explorers that share it.
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


def _summarize(
    rows: list[dict],
    averages: dict[str, float],
    cases: int,
    usage: TokenUsage,
    usage_cases: int,
) -> dict:
    """How the cases at one budget ended, for the manifest and the console.

    Every attempted case has a row, failures included, so `attempted` is the
    row count and `averages` — the accumulators the results table is printed
    from, passed in so the table and the manifest cannot disagree — is over
    all of them. Rows written before outcomes were recorded count as
    `unrecorded`.

    `cases` is the number of cases this run selected, after `--limit` and
    `--instance-ids`, so the cases that were never attempted are what is left
    over. Counting this run's skips instead would restart at zero on every
    `--resume` and shrink the completion rate's denominator.
    """
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
        "metrics": dict(averages),
        "token_usage": usage.to_dict(),
        "token_usage_cases": usage_cases,
    }


def _print_usage_table(name: str, totals: TokenUsage, cases: int) -> None:
    """Print the per-explorer total token usage table (in/out/cache/reasoning)."""
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
    # Printed even when zero: a run that used sub-agents but reports 0 here means
    # the session-store query fell back to the stdout count.
    sub = totals.subagent or TokenUsage()
    console.print(
        f"  [dim]of which sub-agents: {sub.total:,} "
        f"(in {sub.input_tokens:,}, out {sub.output_tokens:,}, "
        f"reasoning {sub.reasoning_tokens:,})[/dim]"
    )


class _CaseRun(NamedTuple):
    """One case as run: `preds` is None when it was never attempted (no repo)."""

    iid: str
    preds: list[tuple[str, int, int]] | None
    usage: TokenUsage | None
    seconds: float
    outcome: str
    error: str | None
    repo_revision: str | None


#: A failure's message is kept in its row, cut to this length.
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

    # Capture token usage from in-process LLM calls (litellm-based agents).
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
        """The checkout for a case, or None when the case may be skipped.

        Without `--skip-missing-repo` a missing checkout is a failure like any
        other: scored as an empty answer, and recorded with its reason rather
        than as an empty success.
        """
        rd = _get_repo_dir(rec)
        if rd is not None or skip_missing_repo:
            return rd
        # `_portable_path`, because this message becomes a row's `error`: an
        # absolute path would carry a username into a published result file.
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

    #: The CLI-agent explorers, by name. They share their whole run-time
    #: wiring — `describe()` for the manifest, `--model`/`--agent` on every
    #: run — so naming the pair once is all a third one should have to add.
    CLI_AGENT_MAKERS = {"opencode": make_opencode, "deveco": make_deveco}
    #: What --opencode-agent / --deveco-agent named, if anything.
    CLI_AGENT_FLAG_AGENTS = {"opencode": opencode_agent, "deveco": deveco_agent}

    # Filled before the first case runs (see explorer_configs below). A CLI
    # agent's model comes from there, so every run names it with --model,
    # including one the configuration chose.
    explorer_configs: dict[str, dict] = {}

    def _cli_agent_method(name: str) -> Callable[[dict], list[tuple[str, int, int]] | None]:
        def method(rec: dict) -> list[tuple[str, int, int]] | None:
            config = explorer_configs[name]
            model = config["model"] or ""
            # When the model came from an agent, name that agent on the
            # command line as well. The CLI would otherwise be free to run a
            # different one — the fallback agent is a documented default, not
            # a promise — and the model would land on an agent that never
            # chose it. Nothing is pinned when the model came from elsewhere,
            # so a CLI whose fallback we have not verified keeps its own.
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
        # Chunking decides what a retrieval explorer can return at all, so it
        # identifies the experiment as much as a model does for an agent —
        # and where a model scores those chunks, it identifies it just as
        # much, so both go in.
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
            # Whole files, no chunking — the embedding model is the setting.
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
            servers = cfg["mcp_servers"]
            console.print(
                f"[dim]{name} configuration: model={cfg['model']} "
                f"({cfg['model_source']}), cli={cfg['cli_version']}, "
                f"mcp={'unknown' if servers is None else sorted(servers) or 'none'}[/dim]"
            )
            if servers is None:
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
    #: The cases this run is about, after --limit and any id filter. Both the
    #: completion rate's denominator and the rows resume counts come from it,
    #: so the two cannot disagree.
    selected_ids = {r.get("instance_id") for r in records}

    for name in explorer_names:
        method = METHOD_MAP[name]

        # ── resume: load existing results and skip completed instances ──
        per_k_totals: dict[int, dict[str, float]] = {k: {m: 0.0 for m in METRICS} for k in top_k_list}
        per_k_evaluated: dict[int, int] = {k: 0 for k in top_k_list}
        per_k_results: dict[int, list[dict]] = {k: [] for k in top_k_list}
        usage_totals = TokenUsage()
        usage_cases = 0
        # Cases with no repository checkout: never run, so never scored.
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
            # Pre-load the surviving rows into the accumulators, but only for
            # the cases this run selected: --limit or --instance-ids can name
            # fewer than the file holds, and a total counting rows the
            # selection excludes reports on an experiment nobody asked for
            # (and a completion rate above 100%). The rows themselves stay in
            # the file — a narrowed selection reads less, it destroys nothing.
            for k in top_k_list:
                for r in kept_per_k[k]:
                    if r.get("instance_id") not in selected_ids:
                        continue
                    per_k_results[k].append(r)
                    per_k_evaluated[k] += 1
                    for m in METRICS:
                        per_k_totals[k][m] += (r.get("metrics") or {}).get(m, 0.0)
            # Token usage is per-case, not per-budget: count each resumed
            # instance once, from its row in the first top_k file.
            for r in per_k_results[top_k_list[0]]:
                tu = TokenUsage.from_dict(r.get("token_usage"))
                # `has_any`, as a fresh run counts it: `from_dict` returns a
                # zeroed object for a row whose tool reported nothing, and
                # counting it would depress the per-case mean on resume only.
                if tu is not None and tu.has_any():
                    usage_totals.add(tu)
                    usage_cases += 1

        def _report_explorer() -> None:
            """Summary table, outcomes and usage; the summary also goes to the manifest."""
            table = Table(title=f"{name} Results", show_lines=False)
            table.add_column("top_k", justify="right")
            table.add_column("Eval", justify="right")
            for metric in METRICS:
                table.add_column(metric, justify="right")
            summaries = {}
            for k in top_k_list:
                ev = per_k_evaluated[k]
                avg = {m: (per_k_totals[k][m] / ev if ev else 0.0) for m in METRICS}
                table.add_row(str(k), str(ev), *[f"{avg[m]:.4f}" for m in METRICS])
                summaries[k] = _summarize(
                    per_k_results[k], avg, total_records, usage_totals, usage_cases
                )
            console.print(table)
            first = summaries[top_k_list[0]]
            rate = first["completion_rate"]
            outcomes = ", ".join(f"{o}={n}" for o, n in first["outcomes"].items()) or "none"
            console.print(
                f"  Outcomes: {outcomes}; not attempted={first['not_attempted']}; "
                f"completion rate={'n/a' if rate is None else f'{rate:.1%}'}"
            )
            _print_usage_table(name, usage_totals, usage_cases)
            if output_jsonl:
                for k, summary in summaries.items():
                    _write_summary(
                        _format_output_path(output_jsonl, name, k), name, k, summary
                    )

        remaining_records = [r for r in records if r.get("instance_id", "") not in resumed_ids]
        total_remaining = len(remaining_records)
        console.print(
            f"\n[bold cyan]▶ {name}[/bold cyan]  "
            f"({total_remaining} to run, {len(resumed_ids & selected_ids)} resumed, "
            f"top_k={top_k_list})"
        )
        if total_remaining == 0:
            console.print(f"  [dim]All instances already completed, skipping.[/dim]")
            _report_explorer()
            continue

        t0 = time.time()
        primary_k = top_k_list[0]
        console.print(
            f"  [dim]case log tuple = (prec, recall, f1, in, out, think, total); "
            f"aggr = metrics averaged, tokens summed over evaluated cases "
            f"@ top_k={primary_k}[/dim]"
        )

        def _eval_one(rec: dict) -> _CaseRun:
            """Run one instance. A failure is a result too: see `_CaseRun`."""
            iid = rec.get("instance_id", "")
            case_t0 = time.perf_counter()
            outcome, error = SUCCESS, None
            # The collector is entered before the try, so the handler can
            # always read what it collected, however early the case failed.
            with usage_collector() as tracker:
                try:
                    preds = method(rec)
                    if preds is None:
                        # No checkout, so nothing ran: never scored, and not a
                        # success either.
                        outcome = NOT_ATTEMPTED
                except Exception as e:
                    # A classified failure already says what it was, in a
                    # sentence written to be read in a result file; anything
                    # else is a crash, where the exception type is the news.
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

        # Open output files for incremental append: a fresh run cleared them
        # above, and --resume continues what is on disk. An --output template
        # without {k} points several budgets at one file, so they share a
        # single handle rather than overwriting each other.
        out_files: dict[int, object] = {}
        out_handles = contextlib.ExitStack()
        if output_jsonl:
            # Opening is staged: a budget file that cannot be opened (no
            # permission, no space) closes the handles already opened for the
            # earlier budgets instead of leaking them.
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
            """Score and write one attempted case, whatever its outcome.

            A failed case is scored as an empty answer, exactly like a run that
            answered with nothing, so a failure never costs less than a bad
            answer (README "Case outcomes").
            """
            nonlocal usage_cases
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
                for m in METRICS:
                    per_k_totals[k][m] += scores_per_k[k][m]
                per_k_evaluated[k] += 1
                per_k_results[k].append(row)
                if k in out_files:
                    _append_row(out_files[k], row)
            # Token usage is per-case, not per-top_k: accumulate once.
            if usage is not None:
                usage_totals.add(usage)
                usage_cases += 1
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
            n = per_k_evaluated[primary_k]
            aggr_vals = (
                per_k_totals[primary_k]["precision"] / n,
                per_k_totals[primary_k]["recall"] / n,
                per_k_totals[primary_k]["f1_score"] / n,
                usage_totals.input_tokens,
                usage_totals.output_tokens,
                usage_totals.reasoning_tokens,
                usage_totals.total,
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
            # Close output files (budgets may share one handle)
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
            f"(eval={per_k_evaluated[top_k_list[0]]}, not attempted={not_attempted}, "
            f"resumed={len(resumed_ids & selected_ids)})[/dim]"
        )
        _report_explorer()

        # ── save per top_k (already written incrementally; just log) ──
        if output_jsonl:
            for k in top_k_list:
                out_path = _format_output_path(output_jsonl, name, k)
                console.print(f"  [green]Saved {out_path} ({per_k_evaluated[k]} records)[/green]")


if __name__ == "__main__":
    app()
