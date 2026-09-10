"""Unified evaluation runner for SWE-Explore.

Supports both local (BM25, TF-IDF, Potion, RAG, SimpleRule, Oracle, Random)
and agentic (Claude Code, Cursor Agent) explorers.

Usage:
    python eval_runner.py --explorers bm25 tfidf --top-k 5,10,20 -o results/{explorer}/top{k}.jsonl
"""
from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, List, Dict

import typer
from rich.console import Console
from rich.table import Table

from eval import ExploreEvaluator
from explorers._cli_agent_base import set_log_level
from explorers.base import ExplorerResult
from explorers.parsing import (
    TokenUsage,
    register_litellm_usage_callback,
    usage_collector,
)

app = typer.Typer(rich_markup_mode="rich")
console = Console()


@contextlib.contextmanager
def _interruptible_pool(workers: int):
    """A pool that drops its queued backlog on Ctrl+C, second Ctrl+C exits at once."""
    previous = signal.getsignal(signal.SIGINT)
    seen = False

    def _handler(signum, frame):
        nonlocal seen
        if seen:
            os._exit(130)
        seen = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        yield pool
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        signal.signal(signal.SIGINT, previous)


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
ACADEMIC_EXPLORERS = {"autocr", "cosil", "locagent", "orcaloca", "mini_swe_agent", "awe_agent"}
ALL_EXPLORERS = LOCAL_EXPLORERS | AGENTIC_EXPLORERS | ACADEMIC_EXPLORERS


# ── helpers ─────────────────────────────────────────────────────────────

def _load_bench_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_issue_map(trajs_dir: Path) -> dict[str, str]:
    issue_map: dict[str, str] = {}
    for p in trajs_dir.rglob("*.json"):
        try:
            data = json.loads(p.read_text())
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
                    per[rel] = len(fpath.read_text(errors="ignore").splitlines())
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
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


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
    embed_model: str = typer.Option("BAAI/bge-small-en-v1.5", "--embed-model"),
    embed_api_key: str | None = typer.Option(None, "--embed-api-key"),
    embed_api_base: str | None = typer.Option(None, "--embed-api-base"),
    swerank_embed_model: str = typer.Option(
        "BAAI/bge-small-en-v1.5", "--swerank-embed-model",
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
        "launch/return; trace additionally dumps the agent output before parsing.",
    ),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    skip_missing_repo: bool = typer.Option(True, "--skip-missing-repo/--no-skip-missing-repo"),
    no_line_counts: bool = typer.Option(False, "--no-line-counts"),
    skip_empty_core: bool = typer.Option(
        True, "--skip-empty-core/--no-skip-empty-core",
        help="Skip instances with empty read_core_regions (default: True)",
    ),
    output_jsonl: str | None = typer.Option(
        None, "--output", "-o",
        help="Save per-instance results to JSONL. Supports {explorer} and {k} placeholders.",
    ),
    resume: bool = typer.Option(
        False, "--resume/--no-resume",
        help="Resume from existing output files, skipping already-evaluated instances.",
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
        with open(issue_map_file) as f:
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

    def _get_issue(rec: dict) -> str:
        iid = rec.get("instance_id", "")
        issue = issue_map.get(iid, "")
        if not issue:
            issue = rec.get("problem_statement", "")
        return issue

    def bm25_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.bm25 import BM25Explorer as LineBM25Explorer
        rd = _get_repo_dir(rec)
        if rd is None:
            return None if skip_missing_repo else []
        explorer = LineBM25Explorer(rd, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        results = explorer.explore(instance_id=rec["instance_id"], query=_get_issue(rec), top_k=max_top_k)
        return _results_to_regions(results)

    def codenib_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.codenib_explorer import CodeNibExplorer
        rd = _get_repo_dir(rec)
        if rd is None:
            return None if skip_missing_repo else []
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
        rd = _get_repo_dir(rec)
        if rd is None:
            return None if skip_missing_repo else []
        if _rag_st_model is None:
            from sentence_transformers import SentenceTransformer
            _rag_st_model = SentenceTransformer(embed_model or "BAAI/bge-small-en-v1.5", trust_remote_code=True)
        explorer = RAGExplorer(rd, _model=_rag_st_model)
        results = explorer.explore(instance_id=rec["instance_id"], query=_get_issue(rec), top_k=max_top_k)
        return _results_to_regions(results)

    def tfidf_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.rag_tfidf import TFIDFExplorer
        rd = _get_repo_dir(rec)
        if rd is None:
            return None if skip_missing_repo else []
        explorer = TFIDFExplorer(rd, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        results = explorer.explore(instance_id=rec["instance_id"], query=_get_issue(rec), top_k=max_top_k)
        return _results_to_regions(results)

    def potion_method(rec: dict) -> list[tuple[str, int, int]] | None:
        nonlocal _potion_model
        from explorers.rag_potion import PotionExplorer, _load_potion_model
        rd = _get_repo_dir(rec)
        if rd is None:
            return None if skip_missing_repo else []
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
        rd = _get_repo_dir(rec)
        if rd is None:
            return None if skip_missing_repo else []
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
        rd = _get_repo_dir(rec)
        if rd is None:
            return None if skip_missing_repo else []
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
        rd = _get_repo_dir(rec)
        if rd is None:
            return None if skip_missing_repo else []
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
        rd = _get_repo_dir(rec)
        if rd is None:
            return None if skip_missing_repo else []
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

    def opencode_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.opencode import OpenCodeExplorer
        return _agentic_method(
            rec,
            lambda rd: OpenCodeExplorer(
                repo_root=rd,
                bin_path=opencode_bin,
                timeout=opencode_timeout,
                config_dir=opencode_config_dir,
                prompt_additions=opencode_prompt_additions,
            ),
        )

    def deveco_method(rec: dict) -> list[tuple[str, int, int]] | None:
        from explorers.deveco import DevEcoExplorer
        return _agentic_method(
            rec,
            lambda rd: DevEcoExplorer(
                repo_root=rd,
                bin_path=deveco_bin,
                timeout=deveco_timeout,
                config_dir=deveco_config_dir,
                prompt_additions=deveco_prompt_additions,
                skip_permissions=deveco_skip_permissions,
            ),
        )

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

    # ── evaluation loop ──
    total_records = len(records)

    for name in explorer_names:
        method = METHOD_MAP[name]

        # ── resume: load existing results and skip completed instances ──
        per_k_totals: dict[int, dict[str, float]] = {k: {m: 0.0 for m in METRICS} for k in top_k_list}
        per_k_evaluated: dict[int, int] = {k: 0 for k in top_k_list}
        per_k_results: dict[int, list[dict]] = {k: [] for k in top_k_list}
        usage_totals = TokenUsage()
        usage_cases = 0
        resumed_usage_ids: set[str] = set()
        skipped = 0
        resumed_ids: set[str] = set()

        if resume and output_jsonl:
            # Find instance_ids completed in ALL top_k files
            per_k_ids: list[set[str]] = []
            for k in top_k_list:
                out_path = _format_output_path(output_jsonl, name, k)
                existing = _load_existing_results(out_path)
                ids = {r["instance_id"] for r in existing}
                per_k_ids.append(ids)
                # Pre-load into accumulators
                for r in existing:
                    per_k_results[k].append(r)
                    per_k_evaluated[k] += 1
                    for m in METRICS:
                        per_k_totals[k][m] += r["metrics"].get(m, 0.0)
                    # Token usage is per-case: count each instance only once
                    # even when it appears in several top_k files.
                    iid = r.get("instance_id", "")
                    tu = TokenUsage.from_dict(r.get("token_usage"))
                    if tu is not None and iid not in resumed_usage_ids:
                        resumed_usage_ids.add(iid)
                        usage_totals.add(tu)
                        usage_cases += 1
            if per_k_ids:
                resumed_ids = per_k_ids[0]
                for s in per_k_ids[1:]:
                    resumed_ids &= s

        remaining_records = [r for r in records if r.get("instance_id", "") not in resumed_ids]
        total_remaining = len(remaining_records)
        console.print(
            f"\n[bold cyan]▶ {name}[/bold cyan]  "
            f"({total_remaining} to run, {len(resumed_ids)} resumed, top_k={top_k_list})"
        )
        if total_remaining == 0:
            console.print(f"  [dim]All instances already completed, skipping.[/dim]")
            # Still print table and save
            table = Table(title=f"{name} Results", show_lines=False)
            table.add_column("top_k", justify="right")
            table.add_column("Eval", justify="right")
            for metric in METRICS:
                table.add_column(metric, justify="right")
            for k in top_k_list:
                ev = per_k_evaluated[k]
                avg = {m: (per_k_totals[k][m] / ev if ev else 0.0) for m in METRICS}
                table.add_row(str(k), str(ev), *[f"{avg[m]:.4f}" for m in METRICS])
            console.print(table)
            _print_usage_table(name, usage_totals, usage_cases)
            continue

        t0 = time.time()
        primary_k = top_k_list[0]
        console.print(
            f"  [dim]case log tuple = (prec, recall, f1, in, out, think, total); "
            f"second tuple = cumulative sum over evaluated cases @ top_k={primary_k}[/dim]"
        )

        def _eval_one(
            rec: dict,
        ) -> tuple[str, list[tuple[str, int, int]] | None, TokenUsage | None, float]:
            """Run one instance: (instance_id, regions_at_max_k, token_usage, case_seconds)."""
            iid = rec.get("instance_id", "")
            case_t0 = time.perf_counter()
            try:
                with usage_collector() as tracker:
                    preds = method(rec)
            except Exception as e:
                sys.stderr.write(f"\n  [ERROR] {name} {iid}: {e}\n")
                return iid, None, None, time.perf_counter() - case_t0
            return (
                iid,
                preds,
                (tracker if tracker.has_any() else None),
                time.perf_counter() - case_t0,
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

        # Open output files for incremental append
        out_files: dict[int, object] = {}
        if output_jsonl:
            for k in top_k_list:
                out_path = _format_output_path(output_jsonl, name, k)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_files[k] = out_path.open("a")

        def _record_result(
            iid: str,
            preds: list[tuple[str, int, int]],
            usage: TokenUsage | None,
        ) -> tuple[float, float, float]:
            nonlocal usage_cases
            scores_per_k = _score_instance(iid, preds)
            row_usage = usage.to_dict() if usage is not None else None
            for k in top_k_list:
                sliced = preds[:k]
                row = {
                    "instance_id": iid,
                    "explorer": name,
                    "regions": [{"path": p, "start": s, "end": e} for p, s, e in sliced],
                    "metrics": scores_per_k[k],
                    "num_regions": min(len(preds), k),
                    "token_usage": row_usage,
                }
                if name == "codenib":
                    row["explorer_config"] = {
                        "policy": codenib_policy,
                        "planning_budget": codenib_planning_budget,
                        "retrieval_level": codenib_retrieval_level,
                    }
                for m in METRICS:
                    per_k_totals[k][m] += scores_per_k[k][m]
                per_k_evaluated[k] += 1
                per_k_results[k].append(row)
                if k in out_files:
                    out_files[k].write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_files[k].flush()
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
            """Per-case progress log: case tuple, cumulative sums, timings."""
            case_u = usage if usage is not None else TokenUsage()
            case_vals = (
                *case_scores,
                case_u.input_tokens,
                case_u.output_tokens,
                case_u.reasoning_tokens,
                case_u.total,
            )
            sum_vals = (
                per_k_totals[primary_k]["precision"],
                per_k_totals[primary_k]["recall"],
                per_k_totals[primary_k]["f1_score"],
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
                f"case={fmt(case_vals)}  sum={fmt(sum_vals)}  "
                f"time={case_dt:.0f}s elapsed={elapsed:.0f}s ETA={eta:.0f}s\n"
            )
            sys.stderr.flush()

        done = 0
        interrupted = False
        try:
            if workers > 1:
                with _interruptible_pool(workers) as pool:
                    futures = {pool.submit(_eval_one, rec): rec for rec in remaining_records}
                    for fut in as_completed(futures):
                        iid, preds, usage, case_dt = fut.result()
                        done += 1
                        elapsed = time.time() - t0
                        rate = done / elapsed if elapsed > 0 else 0
                        eta = (total_remaining - done) / rate if rate > 0 else 0
                        sys.stderr.write(
                            f"\r  [{name}] {done}/{total_remaining}  "
                            f"{rate:.1f} it/s  ETA {eta:.0f}s  "
                        )
                        sys.stderr.flush()
                        if preds is None:
                            skipped += 1
                            continue
                        case_scores = _record_result(iid, preds, usage)
                        _log_case(iid, done, case_scores, usage, case_dt, elapsed, eta)
            else:
                for rec in remaining_records:
                    iid, preds, usage, case_dt = _eval_one(rec)
                    done += 1
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total_remaining - done) / rate if rate > 0 else 0
                    sys.stderr.write(
                        f"\r  [{name}] {done}/{total_remaining}  "
                        f"{rate:.1f} it/s  ETA {eta:.0f}s  "
                    )
                    sys.stderr.flush()
                    if preds is None:
                        skipped += 1
                        continue
                    case_scores = _record_result(iid, preds, usage)
                    _log_case(iid, done, case_scores, usage, case_dt, elapsed, eta)
        except KeyboardInterrupt:
            interrupted = True

        # Close output files
        for fh in out_files.values():
            fh.close()

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
            f"(eval={per_k_evaluated[top_k_list[0]]}, skip={skipped}, resumed={len(resumed_ids)})[/dim]"
        )

        # ── per-explorer summary table ──
        table = Table(title=f"{name} Results", show_lines=False)
        table.add_column("top_k", justify="right")
        table.add_column("Eval", justify="right")
        for metric in METRICS:
            table.add_column(metric, justify="right")
        for k in top_k_list:
            ev = per_k_evaluated[k]
            avg = {m: (per_k_totals[k][m] / ev if ev else 0.0) for m in METRICS}
            table.add_row(
                str(k),
                str(ev),
                *[f"{avg[m]:.4f}" for m in METRICS],
            )
        console.print(table)
        _print_usage_table(name, usage_totals, usage_cases)

        # ── save per top_k (already written incrementally; just log) ──
        if output_jsonl:
            for k in top_k_list:
                out_path = _format_output_path(output_jsonl, name, k)
                console.print(f"  [green]Saved {out_path} ({per_k_evaluated[k]} records)[/green]")


if __name__ == "__main__":
    app()
