"""从 bench.jsonl + repos 目录提取实际代码内容，合并 problem_statement，输出 contexts.jsonl。

用法:
    uv run python -m quality.build_context \
        --bench bench.jsonl \
        --repos-dir repos \
        --output quality/outputs/contexts.jsonl

输出格式 (每行一个 JSON):
    {
        "instance_id": "...",
        "problem_statement": "...",
        "core_context": [{"path": "...", "start": N, "end": N, "content": "..."}],
        "optional_context": [{"path": "...", "start": N, "end": N, "content": "..."}],
        "modified_files": ["..."],
        "repo_dir": "...",
        "meta": {...}
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()

# ---------------------------------------------------------------------------
# problem_statement 加载
# ---------------------------------------------------------------------------

def _load_problem_statements() -> dict[str, str]:
    """从 HuggingFace 加载 problem_statement，合并 SWE-bench + nebius/SWE-rebench。"""
    from datasets import load_dataset

    ps: dict[str, str] = {}

    console.log("Loading princeton-nlp/SWE-bench (all splits)...")
    for split in ("train", "test", "dev"):
        try:
            ds = load_dataset("princeton-nlp/SWE-bench", split=split)
            for row in ds:
                ps[row["instance_id"]] = row["problem_statement"]
        except Exception:
            pass

    console.log("Loading nebius/SWE-rebench...")
    try:
        ds = load_dataset("nebius/SWE-rebench", split="test")
        for row in ds:
            iid = row["instance_id"]
            if iid not in ps:
                ps[iid] = row["problem_statement"]
    except Exception:
        pass

    console.log(f"Loaded problem_statement for {len(ps)} instances")
    return ps


# ---------------------------------------------------------------------------
# 代码读取
# ---------------------------------------------------------------------------

def _read_lines(file_path: Path, start: int, end: int) -> tuple[str, int]:
    """读取文件 [start, end] 行 (1-based 闭区间)，返回 (内容, 实际总行数)。

    end=-1 表示读到文件末尾；start<0 表示距末尾偏移。
    """
    try:
        all_lines = file_path.read_text(errors="replace").splitlines(keepends=True)
    except (OSError, UnicodeDecodeError):
        return "", 0

    total = len(all_lines)
    if total == 0:
        return "", 0

    s = (total + start + 1) if start < 0 else start
    e = total if end == -1 else end

    s = max(1, min(s, total))
    e = max(1, min(e, total))

    if s > e:
        return "", total

    selected = all_lines[s - 1 : e]
    return "".join(selected), total


def _extract_regions(
    regions: list[dict[str, Any]],
    repo_dir: Path,
) -> list[dict[str, Any]]:
    """将 region 列表转为带实际代码内容的列表。"""
    results: list[dict[str, Any]] = []
    for r in regions:
        path = r["path"]
        start = r["start"]
        end = r["end"]
        file_path = repo_dir / path
        content, _ = _read_lines(file_path, start, end)
        if not content:
            continue
        results.append({
            "path": path,
            "start": start,
            "end": end,
            "content": content,
        })
    return results


def _get_optional_regions(gt: dict[str, Any]) -> list[dict[str, Any]]:
    """合并所有 model 的 optional regions (去重)。"""
    opt_map = gt.get("read_optional_regions_map") or {}
    seen: set[tuple[str, int, int]] = set()
    out: list[dict[str, Any]] = []
    for regions in opt_map.values():
        for r in regions:
            key = (r["path"], r["start"], r["end"])
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_contexts(
    bench_path: Path,
    repos_dir: Path,
    output_path: Path,
    problem_statements: dict[str, str] | None = None,
) -> int:
    """构建 contexts.jsonl，返回成功写入的条数。"""
    with open(bench_path) as f:
        bench_items = [json.loads(line) for line in f]

    bench_items = [b for b in bench_items if b["ground_truth"].get("read_core_regions")]
    console.log(f"Bench items with core_regions: {len(bench_items)}")

    if problem_statements is None:
        problem_statements = _load_problem_statements()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_no_ps = 0
    skipped_no_repo = 0
    skipped_empty = 0

    with open(output_path, "w") as out_f, Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building contexts", total=len(bench_items))

        for item in bench_items:
            iid = item["instance_id"]
            gt = item["ground_truth"]

            ps = problem_statements.get(iid, "")
            if not ps:
                skipped_no_ps += 1
                progress.advance(task)
                continue

            repo_dir_str = item.get("repo_dir") or ""
            repo_dir = Path(repo_dir_str) if repo_dir_str else None
            if repo_dir is None or not repo_dir.exists():
                skipped_no_repo += 1
                progress.advance(task)
                continue

            core_ctx = _extract_regions(gt["read_core_regions"], repo_dir)
            optional_raw = _get_optional_regions(gt)
            optional_ctx = _extract_regions(optional_raw, repo_dir)

            if not core_ctx:
                skipped_empty += 1
                progress.advance(task)
                continue

            record = {
                "instance_id": iid,
                "problem_statement": ps,
                "core_context": core_ctx,
                "optional_context": optional_ctx,
                "modified_files": gt.get("modified_core_files") or [],
                "repo_dir": repo_dir_str,
                "meta": item.get("meta") or {},
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            progress.advance(task)

    console.print(f"\n[green]Written: {written}[/green]")
    if skipped_no_ps:
        console.print(f"[yellow]Skipped (no problem_statement): {skipped_no_ps}[/yellow]")
    if skipped_no_repo:
        console.print(f"[yellow]Skipped (repo dir missing): {skipped_no_repo}[/yellow]")
    if skipped_empty:
        console.print(f"[yellow]Skipped (empty core content): {skipped_empty}[/yellow]")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(
    bench: Path = typer.Option("bench.jsonl", "--bench", "-b", help="bench.jsonl 路径"),
    repos_dir: Path = typer.Option("repos", "--repos-dir", "-r", help="repos 根目录"),
    output: Path = typer.Option(
        "quality/outputs/contexts.jsonl", "--output", "-o", help="输出路径"
    ),
):
    """从 bench.jsonl + repos 提取代码内容，输出 contexts.jsonl。"""
    build_contexts(bench, repos_dir, output)


if __name__ == "__main__":
    typer.run(main)
