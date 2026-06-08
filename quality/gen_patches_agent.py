"""使用 agent 框架生成 patch，将 core context 作为初始上下文，禁止 agent 读取新代码。

用法:
    uv run python -m quality.gen_patches_agent \
        --bench bench.jsonl \
        --mode core_only \
        --dataset swe_bench_verified \
        --model gpt-5-mini-2025-08-07 \
        --api-key sk-xxxx \
        --concurrency 5 \
        --nums 10

选项:
    --bench, -b          bench.jsonl 路径 (默认 bench.jsonl)
    --output, -o         输出路径 (默认 quality/outputs/predictions_{mode}_agent.jsonl)
    --mode, -m           实验模式: core_only / core_optional / no_context
    --dataset, -d        数据集过滤: all / swe_bench_verified / swe_rebench / swe_bench
    --model              LLM 模型名称
    --api-key, -k        API key
    --base-url           API base URL
    --concurrency, -j    并发请求数 (默认 5)
    --temperature, -t    采样温度 (默认 0.0)
    --max-tokens         最大生成 token 数 (默认 32768)
    --max-iterations     agent 最大迭代次数 (默认 10)
    --nums, -n           只跑前 N 个 instance，不指定则跑全部
    --log-dir            日志目录 (默认 logs/gen_patches_agent)
    --no-log             禁用日志记录
    --max-context-lines  最大 context 行数，超过则截断（优先保留 modified_files）

输出格式 (SWE-bench predictions):
    {"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from quality.gen_patches import (
    DEFAULT_BASE_URL,
    DEFAULT_API_KEY,
    DATASET_CHOICES,
    _extract_context_from_bench,
    _extract_diff,
    _fix_patch_format,
    _format_context_block,
    _load_dataset_instance_ids,
    _load_problem_statements,
)

console = Console()

# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are an expert software engineer. You are given a GitHub issue and all the \
relevant code context needed to fix it. Your goal is to produce a minimal \
unified diff patch that resolves the issue.

IMPORTANT CONSTRAINTS:
- All necessary code has already been provided to you. Do NOT request additional files.
- Use the "think" tool to reason step-by-step about the root cause and your fix.
- When ready, use the "submit_patch" tool to submit your final patch.
- Make minimal changes. Do not add unrelated modifications.

PATCH FORMAT REQUIREMENTS (strictly enforced):
Your patch MUST follow the standard git unified diff format exactly. \
Patches that do not follow this format will be REJECTED.

1. Each modified file starts with: diff --git a/<path> b/<path>
2. Followed by: --- a/<path>
3. Followed by: +++ b/<path>
4. Each hunk MUST have a complete header with line numbers:
   @@ -<old_start>,<old_count> +<new_start>,<new_count> @@
   - <old_start>: starting line number in the original file
   - <old_count>: number of lines from the original (context + removed)
   - <new_start>: starting line number in the modified file
   - <new_count>: number of lines in the result (context + added)
   NEVER write a bare "@@" without line numbers. This will cause the patch to fail.
5. Context lines (unchanged) start with a space " "
6. Removed lines start with "-"
7. Added lines start with "+"
8. Include 3 lines of context before and after each change when possible.

Use the line numbers shown in the code context to compute correct hunk headers. \
For example, if the code context says "lines 100-120" and you modify line 110, \
your hunk header should be approximately @@ -107,7 +107,8 @@ (3 lines context \
before line 110).

Example:
diff --git a/src/utils.py b/src/utils.py
--- a/src/utils.py
+++ b/src/utils.py
@@ -10,7 +10,8 @@ def process(data):
     items = data.get("items", [])
     for item in items:
         if item.is_valid():
-            result.append(item.value)
+            transformed = item.transform()
+            result.append(transformed)
     return result
"""

AGENT_SYSTEM_PROMPT_NO_CONTEXT = """\
You are an expert software engineer. You are given a GitHub issue from a Python \
repository. Your goal is to produce a minimal unified diff patch that resolves \
the issue.

IMPORTANT CONSTRAINTS:
- Use the "think" tool to reason step-by-step about the root cause and your fix.
- When ready, use the "submit_patch" tool to submit your final patch.
- Infer file paths and code structure from the issue description.
- Make minimal changes. Do not add unrelated modifications.

PATCH FORMAT REQUIREMENTS (strictly enforced):
Your patch MUST follow the standard git unified diff format exactly. \
Patches that do not follow this format will be REJECTED.

1. Each modified file starts with: diff --git a/<path> b/<path>
2. Followed by: --- a/<path>
3. Followed by: +++ b/<path>
4. Each hunk MUST have a complete header with line numbers:
   @@ -<old_start>,<old_count> +<new_start>,<new_count> @@
   NEVER write a bare "@@" without line numbers. This will cause the patch to fail.
5. Context lines start with " ", removed with "-", added with "+"
6. Include 3 lines of context before and after each change.

Example:
diff --git a/src/utils.py b/src/utils.py
--- a/src/utils.py
+++ b/src/utils.py
@@ -10,7 +10,8 @@ def process(data):
     items = data.get("items", [])
     for item in items:
         if item.is_valid():
-            result.append(item.value)
+            transformed = item.transform()
+            result.append(transformed)
     return result
"""


# ---------------------------------------------------------------------------
# User message 构造
# ---------------------------------------------------------------------------

def _truncate_context(
    context_regions: list[dict[str, Any]],
    modified_files: list[str],
    max_lines: int,
) -> list[dict[str, Any]]:
    """截断过大的 context，优先保留 modified_files 中的区域。"""
    total_lines = sum(r.get("end", 0) - r.get("start", 0) + 1 for r in context_regions if r.get("end", -1) != -1)
    if total_lines <= max_lines:
        return context_regions

    # 分组：modified vs others
    modified_regions = [r for r in context_regions if r["path"] in modified_files]
    other_regions = [r for r in context_regions if r["path"] not in modified_files]

    # 优先保留 modified，剩余空间分配给 others
    modified_lines = sum(r.get("end", 0) - r.get("start", 0) + 1 for r in modified_regions if r.get("end", -1) != -1)
    remaining = max(0, max_lines - modified_lines)

    if remaining == 0:
        return modified_regions[:int(max_lines * 0.8)]  # 极端情况：只保留部分 modified

    # 按比例截断 other_regions
    other_lines = sum(r.get("end", 0) - r.get("start", 0) + 1 for r in other_regions if r.get("end", -1) != -1)
    if other_lines <= remaining:
        return modified_regions + other_regions

    ratio = remaining / other_lines
    truncated_others = []
    for r in other_regions:
        lines = r.get("end", 0) - r.get("start", 0) + 1 if r.get("end", -1) != -1 else 0
        if lines * ratio >= 10:  # 至少保留 10 行
            truncated_others.append(r)

    return modified_regions + truncated_others


def _build_agent_user_message(
    problem_statement: str,
    modified_files: list[str],
    context_regions: list[dict[str, Any]],
    mode: str,
    max_context_lines: int | None = None,
) -> str:
    """构造传给 agent 的初始 user message。"""
    sections: list[str] = []
    sections.append(f"## Issue Description\n\n{problem_statement}")

    if modified_files:
        sections.append(
            "## Files that need to be modified\n\n"
            + "\n".join(f"- {f}" for f in modified_files)
        )

    if mode in ("core_only", "core_optional") and context_regions:
        if max_context_lines:
            context_regions = _truncate_context(context_regions, modified_files, max_context_lines)
        ctx = _format_context_block(context_regions)
        sections.append(f"## Relevant Code Context\n\n{ctx}")

    sections.append(
        "## Task\n\n"
        "Analyze the issue and the provided code context. "
        "Use the think tool to reason about the root cause, then use submit_patch "
        "to submit a unified diff patch that fixes the issue."
    )
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Agent 构建与执行
# ---------------------------------------------------------------------------

def _create_agent(
    model_name: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    max_iterations: int,
    system_prompt: str,
):
    """创建 langgraph ReAct agent。"""
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_agent

    from quality.tools import SubmitPatchTool, ThinkTool

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    submit_tool = SubmitPatchTool()
    think_tool = ThinkTool()

    agent = create_agent(
        model=llm,
        tools=[think_tool, submit_tool],
        system_prompt=system_prompt,
    )

    return agent, submit_tool, think_tool


def _extract_trajectory(messages: list) -> list[dict[str, Any]]:
    """从 langgraph 返回的 messages 中提取 trajectory。"""
    trajectory: list[dict[str, Any]] = []
    step = 0
    for msg in messages:
        msg_type = type(msg).__name__
        if msg_type == "AIMessage":
            # tool calls
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    step += 1
                    trajectory.append({
                        "step": step,
                        "action": tc.get("name", "unknown"),
                        "input": tc.get("args", {}),
                    })
            elif msg.content:
                step += 1
                trajectory.append({
                    "step": step,
                    "action": "message",
                    "content": msg.content[:2000],
                })
        elif msg_type == "ToolMessage":
            if trajectory:
                trajectory[-1]["output"] = (msg.content or "")[:2000]
    return trajectory


def _run_agent_sync(
    agent,
    submit_tool,
    user_message: str,
    max_iterations: int,
) -> dict[str, Any]:
    """同步运行 agent，返回 patch + trajectory。"""
    from langgraph.errors import GraphRecursionError

    # 绑定 result store
    store: dict[str, Any] = {}
    submit_tool.bind_store(store)

    messages = []
    try:
        result = agent.invoke(
            {"messages": [("user", user_message)]},
            config={"recursion_limit": max_iterations * 2 + 5},
        )
        messages = result.get("messages", [])
    except GraphRecursionError:
        # agent 达到 recursion limit，尝试从已有 messages 中提取
        pass

    trajectory = _extract_trajectory(messages)

    patch = store.get("patch")
    termination = "tool_called" if patch else "max_iterations"

    # fallback: 如果 agent 没调用 submit_patch，尝试从最后的 AI message 提取
    if not patch:
        for msg in reversed(messages):
            if type(msg).__name__ == "AIMessage" and msg.content:
                candidate = _extract_diff(msg.content)
                if "---" in candidate and "+++" in candidate:
                    patch = candidate
                    termination = "fallback_extract"
                    break

    return {
        "patch": patch or "",
        "trajectory": trajectory,
        "num_steps": len(trajectory),
        "termination_reason": termination,
    }


# ---------------------------------------------------------------------------
# 批量生成
# ---------------------------------------------------------------------------

async def generate_patches_agent(
    bench_path: Path,
    problem_statements: dict[str, str],
    output_path: Path,
    mode: str,
    model: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    concurrency: int = 5,
    temperature: float = 0.0,
    max_tokens: int = 32768,
    max_iterations: int = 10,
    nums: int | None = None,
    dataset_ids: set[str] | None = None,
    log_dir: Path | None = None,
    max_context_lines: int | None = None,
) -> int:
    """批量使用 agent 生成 patch，返回成功数。"""

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
    pending = [it for it in pending if it["instance_id"] in problem_statements]
    console.log(f"Pending: {len(pending)}")

    if not pending:
        return 0

    system_prompt = AGENT_SYSTEM_PROMPT_NO_CONTEXT if mode == "no_context" else AGENT_SYSTEM_PROMPT
    model_label = f"{model}__quality_{mode}_agent"

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
        log_path = log_dir / f"gen_patches_agent_{mode}_{ts}.jsonl"
        log_f = open(log_path, "a")
        console.log(f"Logging to {log_path}")

    async def process_one(item: dict[str, Any]) -> None:
        nonlocal written, errors
        iid = item["instance_id"]
        ps = problem_statements[iid]
        modified = item["ground_truth"].get("modified_core_files") or []
        context_regions = _extract_context_from_bench(item, mode)
        user_message = _build_agent_user_message(ps, modified, context_regions, mode, max_context_lines)

        async with sem:
            try:
                # 每个 instance 创建独立的 agent（独立的 submit_tool store）
                agent, submit_tool, _ = _create_agent(
                    model, api_key, base_url, temperature, max_tokens,
                    max_iterations, system_prompt,
                )

                agent_result = await asyncio.to_thread(
                    _run_agent_sync, agent, submit_tool, user_message, max_iterations,
                )

                raw_patch = agent_result["patch"]
                patch = _fix_patch_format(_extract_diff(raw_patch)) if raw_patch else ""

                result = {
                    "instance_id": iid,
                    "model_name_or_path": model_label,
                    "model_patch": patch,
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                written += 1

                if log_f is not None:
                    log_entry = {
                        "instance_id": iid,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "num_steps": agent_result["num_steps"],
                        "termination_reason": agent_result["termination_reason"],
                        "trajectory": agent_result["trajectory"],
                        "raw_patch": raw_patch,
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
        task = progress.add_task(f"Agent generating patches ({mode})", total=len(pending))

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
# CLI
# ---------------------------------------------------------------------------

def main(
    bench: Path = typer.Option("bench.jsonl", "--bench", "-b", help="bench.jsonl 路径"),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="输出路径，默认 quality/outputs/predictions_{mode}_agent.jsonl",
    ),
    mode: str = typer.Option(
        "core_only", "--mode", "-m",
        help="实验模式: core_only / core_optional / no_context",
    ),
    model: str = typer.Option(
        "gpt-5-mini-2025-08-07", "--model",
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
        5, "--concurrency", "-j",
        help="并发请求数 (agent 更慢，建议低并发)",
    ),
    temperature: float = typer.Option(0.0, "--temperature", "-t"),
    max_tokens: int = typer.Option(32768, "--max-tokens"),
    max_iterations: int = typer.Option(
        10, "--max-iterations",
        help="agent 最大迭代次数",
    ),
    nums: int = typer.Option(
        None, "--nums", "-n",
        help="只跑前 N 个 instance，不指定则跑全部",
    ),
    dataset: str = typer.Option(
        "all", "--dataset", "-d",
        help="数据集过滤: all / swe_bench_verified / swe_rebench / swe_bench",
    ),
    log_dir: Path = typer.Option(
        "logs/gen_patches_agent", "--log-dir",
        help="日志目录，保存 agent trajectory",
    ),
    no_log: bool = typer.Option(
        False, "--no-log",
        help="禁用日志记录",
    ),
    max_context_lines: int = typer.Option(
        None, "--max-context-lines",
        help="最大 context 行数，超过则截断（优先保留 modified_files）",
    ),
):
    """使用 agent 框架生成 patch (core context 作为初始上下文，禁止读取新代码)。"""
    if mode not in ("core_only", "core_optional", "no_context"):
        console.print(f"[red]Invalid mode: {mode}[/red]")
        raise typer.Exit(1)

    if dataset not in DATASET_CHOICES:
        console.print(f"[red]Invalid dataset: {dataset}. Choose from: {', '.join(DATASET_CHOICES)}[/red]")
        raise typer.Exit(1)

    if output is None:
        output = Path(f"quality/outputs/predictions_{mode}_agent.jsonl")

    dataset_ids = _load_dataset_instance_ids(dataset)
    ps = _load_problem_statements()

    asyncio.run(generate_patches_agent(
        bench, ps, output, mode, model,
        api_key=api_key, base_url=base_url,
        concurrency=concurrency,
        temperature=temperature, max_tokens=max_tokens,
        max_iterations=max_iterations,
        nums=nums,
        dataset_ids=dataset_ids,
        log_dir=None if no_log else log_dir,
        max_context_lines=max_context_lines,
    ))


if __name__ == "__main__":
    typer.run(main)
