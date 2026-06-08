"""读取 bench.jsonl 中的核心上下文，调用 LLM 单次生成 patch，输出 SWE-bench predictions 格式。

用法:
    uv run python -m quality.gen_patches \
        --bench bench.jsonl \
        --mode core_only \
        --dataset swe_bench_verified \
        --model gpt-5-mini-2025-08-07 \
        --api-key sk-xxxx \
        --concurrency 20 \
        --nums 10

选项:
    --bench, -b        bench.jsonl 路径 (默认 bench.jsonl)
    --output, -o       输出路径 (默认 quality/outputs/predictions_{mode}.jsonl)
    --mode, -m         实验模式: core_only / core_optional / no_context
    --dataset, -d      数据集过滤: all / swe_bench_verified / swe_rebench / swe_bench
    --model            LLM 模型名称
    --api-key, -k      API key
    --base-url         API base URL
    --concurrency, -j  并发请求数 (默认 20)
    --temperature, -t  采样温度 (默认 0.0)
    --max-tokens       最大生成 token 数 (默认 32768)
    --nums, -n         只跑前 N 个 instance，不指定则跑全部
    --log-dir          日志目录 (默认 logs/gen_patches)
    --no-log           禁用日志记录

三种模式:
    core_only     — 只提供 core context
    core_optional — 提供 core + optional context
    no_context    — 不提供任何代码上下文

输出格式 (SWE-bench predictions):
    {"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()

# ---------------------------------------------------------------------------
# API 配置
# ---------------------------------------------------------------------------

from quality.models import DEFAULT_API_KEY, DEFAULT_BASE_URL

# ---------------------------------------------------------------------------
# 数据集过滤
# ---------------------------------------------------------------------------

DATASET_CHOICES: dict[str, str | None] = {
    "all": None,
    "swe_bench_verified": "princeton-nlp/SWE-bench_Verified",
    "swe_rebench": "nebius/SWE-rebench",
    "swe_bench": "princeton-nlp/SWE-bench",
}


def _load_dataset_instance_ids(dataset_key: str) -> set[str] | None:
    """加载指定数据集的 instance_id 集合。返回 None 表示不过滤。"""
    if dataset_key == "all":
        return None
    hf_name = DATASET_CHOICES[dataset_key]
    from datasets import load_dataset

    console.log(f"Loading instance IDs from {hf_name}...")
    ds = load_dataset(hf_name, split="test")
    ids = set(ds["instance_id"])
    console.log(f"Loaded {len(ids)} instance IDs from {hf_name}")
    return ids


# ---------------------------------------------------------------------------
# 代码读取 (与 build_context 相同逻辑，但直接从 bench.jsonl + repos 读取)
# ---------------------------------------------------------------------------

def _read_lines(file_path: Path, start: int, end: int) -> str:
    """读取文件 [start, end] 行 (1-based 闭区间)。"""
    try:
        all_lines = file_path.read_text(errors="replace").splitlines(keepends=True)
    except (OSError, UnicodeDecodeError):
        return ""
    total = len(all_lines)
    if total == 0:
        return ""
    s = (total + start + 1) if start < 0 else start
    e = total if end == -1 else end
    s = max(1, min(s, total))
    e = max(1, min(e, total))
    if s > e:
        return ""
    return "".join(all_lines[s - 1 : e])


def _extract_context_from_bench(
    item: dict[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    """从 bench item 直接提取代码上下文 (repo_dir 从 item 中读取)。"""
    gt = item["ground_truth"]
    repo_dir_str = item.get("repo_dir") or ""
    repo_dir = Path(repo_dir_str) if repo_dir_str else None
    if repo_dir is None or not repo_dir.exists():
        return []

    regions: list[dict[str, Any]] = []

    # core regions
    if mode in ("core_only", "core_optional"):
        for r in gt.get("read_core_regions") or []:
            content = _read_lines(repo_dir / r["path"], r["start"], r["end"])
            if content:
                regions.append({"path": r["path"], "start": r["start"], "end": r["end"], "content": content})

    # optional regions
    if mode == "core_optional":
        opt_map = gt.get("read_optional_regions_map") or {}
        seen: set[tuple[str, int, int]] = set()
        for model_regions in opt_map.values():
            for r in model_regions:
                key = (r["path"], r["start"], r["end"])
                if key not in seen:
                    seen.add(key)
                    content = _read_lines(repo_dir / r["path"], r["start"], r["end"])
                    if content:
                        regions.append({"path": r["path"], "start": r["start"], "end": r["end"], "content": content})

    return regions


def _extract_regions_metadata(
    item: dict[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    """从 bench item 提取 region 元数据 (path/start/end)，不读取文件内容。

    与 _extract_context_from_bench 不同，此函数不依赖本地 repo_dir，
    返回的 region 不含 content 字段。适用于需要从 Docker 容器读取内容的场景。
    """
    gt = item["ground_truth"]
    regions: list[dict[str, Any]] = []

    # core regions
    if mode in ("core_only", "core_optional"):
        for r in gt.get("read_core_regions") or []:
            regions.append({"path": r["path"], "start": r["start"], "end": r["end"]})

    # optional regions
    if mode == "core_optional":
        opt_map = gt.get("read_optional_regions_map") or {}
        seen: set[tuple[str, int, int]] = set()
        for model_regions in opt_map.values():
            for r in model_regions:
                key = (r["path"], r["start"], r["end"])
                if key not in seen:
                    seen.add(key)
                    regions.append({"path": r["path"], "start": r["start"], "end": r["end"]})

    return regions


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert software engineer. You will be given a GitHub issue description \
and relevant code context from the repository. Your task is to generate a patch \
(unified diff format) that fixes the issue.

Rules:
- Output ONLY the patch, nothing else.
- Use standard git diff format: each file starts with "diff --git a/<path> b/<path>".
- Use "--- a/<path>" and "+++ b/<path>" for file headers.
- Include proper @@ hunk headers with correct line numbers.
- Use the correct file paths as shown in the context.
- Make minimal changes to fix the issue.
- Do not add unrelated changes.

Example format:
diff --git a/src/example.py b/src/example.py
--- a/src/example.py
+++ b/src/example.py
@@ -10,4 +10,5 @@ def example():
     x = 1
     y = 2
-    return x
+    z = 3
+    return x + y + z
"""

SYSTEM_PROMPT_NO_CONTEXT = """\
You are an expert software engineer. You will be given a GitHub issue description \
from a Python repository. Your task is to generate a patch (unified diff format) \
that fixes the issue.

Rules:
- Output ONLY the unified diff (starting with --- and +++), nothing else.
- Infer the file paths and code structure from the issue description.
- Make minimal changes to fix the issue.
- Do not add unrelated changes.
"""


def _format_context_block(regions: list[dict[str, Any]]) -> str:
    """将 regions 列表格式化为可读的代码块。"""
    parts: list[str] = []
    for r in regions:
        header = f"--- {r['path']} (lines {r['start']}-{r['end']}) ---"
        parts.append(header)
        parts.append(r["content"].rstrip())
        parts.append("")
    return "\n".join(parts)


def build_user_prompt(
    problem_statement: str,
    modified_files: list[str],
    context_regions: list[dict[str, Any]],
    mode: str,
) -> str:
    """根据模式构造 user prompt。"""
    sections: list[str] = []
    sections.append(f"## Issue Description\n\n{problem_statement}")

    if modified_files:
        sections.append("## Files that need to be modified\n\n" + "\n".join(f"- {f}" for f in modified_files))

    if mode in ("core_only", "core_optional") and context_regions:
        ctx = _format_context_block(context_regions)
        sections.append(f"## Relevant Code Context\n\n{ctx}")

    sections.append(
        "## Task\n\n"
        "Generate a unified diff patch that fixes the issue above. "
        "Output ONLY the diff, no explanation."
    )
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# LLM 调用 & diff 提取
# ---------------------------------------------------------------------------

def _extract_diff(raw: str) -> str:
    """从 LLM 输出中提取 diff 内容。处理 markdown 代码块包裹的情况。"""
    def _trim_to_diff_start(text: str) -> str:
        match = re.search(r"(?m)^(diff --git |--- |\+\+\+ )", text)
        if match:
            return text[match.start():].strip()
        return text.strip()

    text = raw.strip()

    if "```" in text:
        blocks: list[str] = []
        in_block = False
        current: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```") and not in_block:
                in_block = True
                current = []
                continue
            if stripped == "```" and in_block:
                in_block = False
                blocks.append("\n".join(current))
                continue
            if in_block:
                current.append(line)
        if blocks:
            for b in blocks:
                if "---" in b and "+++" in b:
                    return _trim_to_diff_start(b)
            return _trim_to_diff_start(blocks[0])

    return _trim_to_diff_start(text)


def _fix_patch_format(patch: str) -> str:
    """修复 LLM 生成的 patch 中常见的格式问题。

    处理:
    - 去除行尾多余空格
    - 空行补 ' ' 前缀（上下文行）
    - 缺少前缀的行补 ' '
    - 重新计算 hunk header 中的行数
    """
    if not patch.strip():
        return patch

    lines = patch.splitlines()

    # ---- Phase 1: 基础格式修复 ----
    fixed: list[str] = []
    in_hunk = False

    for idx, line in enumerate(lines):
        rs = line.rstrip()

        if rs.startswith("diff --git"):
            in_hunk = False
            fixed.append(rs)
            continue

        # 检测 --- / +++ 文件头配对
        if (rs.startswith("--- ")
                and idx + 1 < len(lines)
                and lines[idx + 1].rstrip().startswith("+++ ")):
            in_hunk = False
            fixed.append(rs)
            continue
        if (rs.startswith("+++ ")
                and idx > 0
                and lines[idx - 1].rstrip().startswith("--- ")):
            in_hunk = False
            fixed.append(rs)
            continue

        if rs.startswith("@@") and "@@" in rs[2:]:
            in_hunk = True
            fixed.append(rs)
            continue

        if not in_hunk:
            fixed.append(rs)
            continue

        # hunk 内部
        if rs == "":
            fixed.append(" ")              # 空行 → 上下文行
        elif rs[0] in (" ", "+", "-", "\\"):
            fixed.append(rs)
        else:
            fixed.append(" " + rs)         # 缺少前缀 → 上下文行

    # ---- Phase 2: 重算 hunk header 行数 ----
    hunk_re = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)")
    result_lines: list[str] = []
    i = 0

    while i < len(fixed):
        if not fixed[i].startswith("@@") or not hunk_re.match(fixed[i]):
            result_lines.append(fixed[i])
            i += 1
            continue

        hunk_hdr_idx = len(result_lines)
        result_lines.append(fixed[i])
        i += 1

        old_count = 0
        new_count = 0
        while i < len(fixed):
            ln = fixed[i]
            if ln.startswith("@@") or ln.startswith("diff --git"):
                break
            # 检测新的文件头配对
            if (ln.startswith("--- ")
                    and i + 1 < len(fixed)
                    and fixed[i + 1].startswith("+++ ")):
                break

            ch = ln[0] if ln else " "
            if ch == "-":
                old_count += 1
            elif ch == "+":
                new_count += 1
            elif ch == " ":
                old_count += 1
                new_count += 1
            # "\" 行 (No newline at end of file) 不计入行数
            result_lines.append(ln)
            i += 1

        m = hunk_re.match(result_lines[hunk_hdr_idx])
        if m:
            result_lines[hunk_hdr_idx] = (
                f"@@ -{m.group(1)},{old_count} +{m.group(2)},{new_count} @@{m.group(3)}"
            )

    text = "\n".join(result_lines)
    if not text.endswith("\n"):
        text += "\n"
    return text


async def _call_llm(
    client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """单次调用 OpenAI-compatible API，返回 content 和 thinking。"""
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=300,
    )
    msg = resp.choices[0].message
    content = msg.content or ""
    # 尝试获取 thinking / reasoning 内容 (部分模型如 o1/deepseek-r1 会返回)
    thinking = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    return {"content": content, "thinking": thinking}


# ---------------------------------------------------------------------------
# 批量生成
# ---------------------------------------------------------------------------

async def generate_patches(
    bench_path: Path,
    problem_statements: dict[str, str],
    output_path: Path,
    mode: str,
    model: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    concurrency: int = 20,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    nums: int | None = None,
    dataset_ids: set[str] | None = None,
    log_dir: Path | None = None,
) -> int:
    """批量生成 patch，返回成功数。nums 不为 None 时只取前 nums 条。"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # 加载 bench
    items: list[dict[str, Any]] = []
    with open(bench_path) as f:
        for line in f:
            item = json.loads(line)
            if item["ground_truth"].get("read_core_regions"):
                items.append(item)
    console.log(f"Loaded {len(items)} instances with core_regions from {bench_path}")

    # 按数据集过滤
    if dataset_ids is not None:
        items = [it for it in items if it["instance_id"] in dataset_ids]
        console.log(f"Filtered to {len(items)} instances in selected dataset")

    if nums is not None:
        items = items[:nums]
        console.log(f"Limited to first {nums} instances")

    # 断点续跑
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        console.log(f"Already completed: {len(done_ids)}, skipping")

    pending = [it for it in items if it["instance_id"] not in done_ids]
    # 过滤没有 problem_statement 的
    pending = [it for it in pending if it["instance_id"] in problem_statements]
    console.log(f"Pending: {len(pending)}")

    if not pending:
        return 0

    system = SYSTEM_PROMPT_NO_CONTEXT if mode == "no_context" else SYSTEM_PROMPT
    model_label = f"{model}__quality_{mode}"

    sem = asyncio.Semaphore(concurrency)
    written = 0
    errors = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output_path, "a")

    # 日志文件
    log_f = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"gen_patches_{mode}_{ts}.jsonl"
        log_f = open(log_path, "a")
        console.log(f"Logging to {log_path}")

    async def process_one(item: dict[str, Any]) -> None:
        nonlocal written, errors
        iid = item["instance_id"]
        ps = problem_statements[iid]
        modified = item["ground_truth"].get("modified_core_files") or []
        context_regions = _extract_context_from_bench(item, mode)
        user_prompt = build_user_prompt(ps, modified, context_regions, mode)

        async with sem:
            try:
                llm_result = await _call_llm(
                    client, model, system, user_prompt,
                    temperature=temperature, max_tokens=max_tokens,
                )
                raw = llm_result["content"]
                extracted = _extract_diff(raw)
                patch = _fix_patch_format(extracted)
                result = {
                    "instance_id": iid,
                    "model_name_or_path": model_label,
                    "model_patch": patch,
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                written += 1
                # 写入日志
                if log_f is not None:
                    log_entry = {
                        "instance_id": iid,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "thinking": llm_result["thinking"],
                        "raw_response": raw,
                        "extracted_patch": extracted,
                        "fixed_patch": patch,
                    }
                    log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    log_f.flush()
            except Exception as e:
                errors += 1
                console.print(f"[red]Error {iid}: {e}[/red]")
                if log_f is not None:
                    log_entry = {
                        "instance_id": iid,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "error": str(e),
                    }
                    log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    log_f.flush()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Generating patches ({mode})", total=len(pending))

        batch_size = concurrency * 2
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            tasks = [process_one(it) for it in batch]
            await asyncio.gather(*tasks)
            progress.advance(task, len(batch))

    out_f.close()
    if log_f is not None:
        log_f.close()
    console.print(f"\n[green]Written: {written}[/green]")
    if errors:
        console.print(f"[red]Errors: {errors}[/red]")
    return written


# ---------------------------------------------------------------------------
# problem_statement 加载 (复用 build_context 的逻辑)
# ---------------------------------------------------------------------------

def _load_problem_statements() -> dict[str, str]:
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
            if row["instance_id"] not in ps:
                ps[row["instance_id"]] = row["problem_statement"]
    except Exception:
        pass

    console.log(f"Loaded problem_statement for {len(ps)} instances")
    return ps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(
    bench: Path = typer.Option("bench.jsonl", "--bench", "-b", help="bench.jsonl 路径"),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="输出路径，默认 quality/outputs/predictions_{mode}.jsonl",
    ),
    mode: str = typer.Option(
        "core_only", "--mode", "-m",
        help="实验模式: core_only / core_optional / no_context",
    ),
    model: str = typer.Option(
        "deepseek-chat", "--model",
        help="LLM 模型名称",
    ),
    api_key: str = typer.Option(
        DEFAULT_API_KEY, "--api-key", "-k",
        help="API key",
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL, "--base-url",
        help="API base URL",
    ),
    concurrency: int = typer.Option(
        20, "--concurrency", "-j",
        help="并发请求数",
    ),
    temperature: float = typer.Option(0.0, "--temperature", "-t"),
    max_tokens: int = typer.Option(32768, "--max-tokens"),
    nums: int = typer.Option(
        None, "--nums", "-n",
        help="只跑前 N 个 instance，不指定则跑全部",
    ),
    dataset: str = typer.Option(
        "all", "--dataset", "-d",
        help="数据集过滤: all / swe_bench_verified / swe_rebench / swe_bench",
    ),
    log_dir: Path = typer.Option(
        "logs/gen_patches", "--log-dir",
        help="日志目录，保存 LLM 原始返回内容 (含 thinking)",
    ),
    no_log: bool = typer.Option(
        False, "--no-log",
        help="禁用日志记录",
    ),
):
    """读取 bench 核心上下文，调用 LLM 生成 patch。"""
    if mode not in ("core_only", "core_optional", "no_context"):
        console.print(f"[red]Invalid mode: {mode}[/red]")
        raise typer.Exit(1)

    if dataset not in DATASET_CHOICES:
        console.print(f"[red]Invalid dataset: {dataset}. Choose from: {', '.join(DATASET_CHOICES)}[/red]")
        raise typer.Exit(1)

    if output is None:
        output = Path(f"quality/outputs/predictions_{mode}.jsonl")

    dataset_ids = _load_dataset_instance_ids(dataset)
    ps = _load_problem_statements()

    asyncio.run(generate_patches(
        bench, ps, output, mode, model,
        api_key=api_key, base_url=base_url,
        concurrency=concurrency,
        temperature=temperature, max_tokens=max_tokens,
        nums=nums,
        dataset_ids=dataset_ids,
        log_dir=None if no_log else log_dir,
    ))


if __name__ == "__main__":
    typer.run(main)
