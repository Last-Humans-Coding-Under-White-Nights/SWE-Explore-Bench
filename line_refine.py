from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from bench_build import (
    _get_message_content,
    _merge_intervals,
    _resolve_repo_dir,
    load_trajectory_data,
)
from models import (
    ChatMessage,
    agent_input_to_messages,
    get_default_openrouter_client,
    messages_to_agent_input,
)


console = Console()
app = typer.Typer(rich_markup_mode="rich")


@dataclass
class LineRegion:
    path: str
    start: int
    end: int


def _load_bench_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _group_regions_by_file(
    regions: list[LineRegion],
) -> dict[str, list[tuple[int, int]]]:
    per_file: dict[str, list[tuple[int, int]]] = {}
    for r in regions:
        per_file.setdefault(r.path, []).append((r.start, r.end))
    return {p: _merge_intervals(ivals) for p, ivals in per_file.items()}


def _flatten_regions(per_file: dict[str, list[tuple[int, int]]]) -> list[LineRegion]:
    out: list[LineRegion] = []
    for path, ivals in per_file.items():
        for start, end in ivals:
            out.append(LineRegion(path=path, start=start, end=end))
    return out


def _load_context_map(
    path: Path | None,
) -> dict[str, dict[str, str]]:
    """从可选的 JSONL 文件中加载 instance 上下文信息。

    约定一行一个 JSON 对象，至少包含：
    - instance_id: str
    - issue: str（Issue 描述）
    - conversation: str（最近若干步对话的串联文本）
    该文件由上游轨迹处理脚本生成，这里只做简单关联，不做格式推断。
    """
    if path is None:
        return {}

    context_map: dict[str, dict[str, str]] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instance_id = obj.get("instance_id")
            if not isinstance(instance_id, str):
                continue
            issue = obj.get("issue") or ""
            conversation = obj.get("conversation") or ""
            context_map[instance_id] = {
                "issue": str(issue),
                "conversation": str(conversation),
            }
    return context_map


def _find_trajectory_path(trajs_dir: Path, instance_id: str) -> Path | None:
    """在 trajs_dir 下按 instance_id 查找任意一条轨迹文件（*/instance_id/instance_id.json 或 .traj.json）。"""
    if not trajs_dir.is_dir() or not instance_id:
        return None
    for model_dir in trajs_dir.iterdir():
        if not model_dir.is_dir():
            continue
        inst_dir = model_dir / instance_id
        if not inst_dir.is_dir():
            continue
        for name in (f"{instance_id}.json", f"{instance_id}.traj.json"):
            p = inst_dir / name
            if p.is_file():
                return p
    return None


# 缓存已加载的轨迹数据，避免重复读取
_trajectory_cache: dict[str, dict[str, Any]] = {}


def _load_trajectory_cached(traj_path: str) -> dict[str, Any] | None:
    """加载轨迹数据，带缓存。"""
    if traj_path in _trajectory_cache:
        return _trajectory_cache[traj_path]
    path = Path(traj_path)
    if not path.is_file():
        return None
    try:
        data = load_trajectory_data(path)
        _trajectory_cache[traj_path] = data
        return data
    except Exception:
        return None


def _load_context_from_step_info(
    step_info_list: list[dict[str, Any]],
    k: int = 4,
) -> dict[str, str] | None:
    """
    从 read_step_info 中加载上下文。
    step_info_list: [{traj_path, step_idx, start, end}, ...]
    k: 在 step_idx 前后各取 k 条消息作为上下文
    返回 {issue, conversation}
    """
    if not step_info_list:
        return None

    # 使用第一个 step_info 来获取 issue 和基础上下文
    first_info = step_info_list[0]
    traj_path = first_info.get("traj_path", "")
    step_idx = first_info.get("step_idx", 0)

    data = _load_trajectory_cached(traj_path)
    if not data:
        return None

    info = data.get("info") or {}
    messages = data.get("messages") or []
    issue = info.get("issue") or ""

    # 收集所有 step_idx 并取其周围的消息
    all_step_indices = {si.get("step_idx", 0) for si in step_info_list}

    # 构建需要包含的消息索引集合
    include_indices: set[int] = set()
    for idx in all_step_indices:
        for i in range(max(0, idx - k), min(len(messages), idx + k + 1)):
            include_indices.add(i)

    # 按顺序提取消息
    parts: list[str] = []
    for i in sorted(include_indices):
        if i >= len(messages):
            continue
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = _get_message_content(msg)
        if content and role in ("user", "assistant", "tool"):
            # 标记是否是读取步骤
            marker = " [READ STEP]" if i in all_step_indices else ""
            parts.append(f"[Step {i}] [{role}]{marker}\n{content[:2000]}")

    conversation = "\n\n".join(parts) if parts else ""
    return {"issue": str(issue), "conversation": conversation}


def _load_context_from_trajectory(
    trajs_dir: Path | None,
    instance_id: str,
    last_n_messages: int = 8,
) -> dict[str, str] | None:
    """从轨迹文件中抽取 issue 与最近对话片段。无 trajs_dir 或未找到轨迹时返回 None。"""
    if not trajs_dir or not instance_id:
        return None
    path = _find_trajectory_path(trajs_dir, instance_id)
    if not path:
        return None
    try:
        data = load_trajectory_data(path)
    except Exception:
        return None
    info = data.get("info") or {}
    messages = data.get("messages") or []
    issue = info.get("issue") or ""
    # 最近若干条消息拼接为 conversation
    recent = messages[-last_n_messages:] if len(messages) > last_n_messages else messages
    parts: list[str] = []
    for msg in recent:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = _get_message_content(msg)
        if content and role in ("user", "assistant"):
            parts.append(f"[{role}]\n{content[:2000]}")
    conversation = "\n\n".join(parts) if parts else ""
    return {"issue": str(issue), "conversation": conversation}


def _read_file_lines(path: Path) -> list[str]:
    """读取文件行，跳过二进制文件。"""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except (OSError, UnicodeDecodeError):
        return []


def _build_source_snippet(
    file_path: Path,
    intervals: list[tuple[int, int]],
    margin: int = 3,
    max_total_lines: int = 400,
) -> tuple[str, int]:
    """根据若干粗修区间，从源码构建上下文片段。

    返回 (snippet_text, total_lines)，其中 total_lines 为整个文件行数。
    """
    lines = _read_file_lines(file_path)
    n = len(lines)
    if not intervals:
        return "", n

    # 合并扩展 margin 后的区间，再整体裁剪；end=-1 表示到文件末尾
    expanded: list[tuple[int, int]] = []
    for s, e in intervals:
        start = max(1, s - margin)
        end_val = n if e < 0 else min(n, e + margin)
        expanded.append((start, end_val))
    expanded = _merge_intervals(expanded)

    # 若整体行数不大于 max_total_lines，直接返回全文件，便于模型考虑 AST 边界
    if n <= max_total_lines:
        snippet = "".join(lines)
        return snippet, n

    # 否则仅拼接相关区间附近的代码
    snippets: list[str] = []
    for start, end in expanded:
        start_idx = max(0, start - 1)
        end_idx = min(n, end)
        snippets.append(f"# L{start}-{end}\n")
        snippets.extend(lines[start_idx:end_idx])
        snippets.append("\n")

    # 简单截断，避免过长
    snippet_text = "".join(snippets)
    snippet_lines = snippet_text.splitlines()
    if len(snippet_lines) > max_total_lines:
        snippet_text = "\n".join(snippet_lines[:max_total_lines])
    return snippet_text, n


def _parse_range_list(text: str) -> list[tuple[int, int]]:
    """解析类似 `[5-7, 13-19]` 或 `5-7` 的行号区间列表。

    若解析失败，返回空列表，让上层决定是否回退到粗修结果。
    """
    raw = text.strip()
    if not raw:
        return []
    # 去掉前后括号/方括号
    if raw[0] in "[(" and raw[-1] in "])":
        raw = raw[1:-1]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    result: list[tuple[int, int]] = []
    for part in parts:
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            try:
                a = int(a_str)
                b = int(b_str)
            except ValueError:
                continue
            if a > 0 and b > 0:
                if a > b:
                    a, b = b, a
                result.append((a, b))
        else:
            try:
                x = int(part)
            except ValueError:
                continue
            if x > 0:
                result.append((x, x))
    return result


def _build_refine_messages(
    *,
    instance_id: str,
    repo_name: str,
    rel_path: str,
    coarse_intervals: list[tuple[int, int]],
    source_snippet: str,
    file_total_lines: int,
    issue: str | None,
    conversation: str | None,
    target_type: str,
    target_model: str | None,
) -> list[ChatMessage]:
    """构建精修调用使用的消息列表。"""
    system = (
        "你是一个代码阅读助手，负责为 SWE 任务确定“最小必要上下文”。\n"
        "给定：\n"
        "1. 某个实例的 Issue / 需求描述；\n"
        "2. 最近若干步对话（如果有）；\n"
        "3. 某个源文件的部分或全部内容；\n"
        "4. 由工具推断出的粗糙行区间。\n\n"
        "请你在理解代码语义和 AST / 函数 / 类边界的前提下，判断“解决该 Issue 至少需要看的最小行号区间集合”。\n"
        "输出要求：\n"
        "- 只输出行号区间，不要任何解释；\n"
        "- 使用闭区间，格式形如 `[5-7, 13-19]` 或 `[42-42]`；\n"
        "- 如果粗修区间包含明显无关的代码，请剔除；如需要扩展到完整函数，请适度扩展；\n"
        "- 行号以给定源码片段的实际行号为准（L1 从整文件第一行开始）。"
    )

    coarse_desc = ", ".join(f"{s}-{e}" for s, e in coarse_intervals) or "（无）"
    if target_type == "optional":
        target_desc = f"optional 读取上下文（model={target_model or 'unknown'}）"
    else:
        target_desc = "core 读取上下文"
    user_parts: list[str] = [
        f"Instance: {instance_id}",
        f"Repo: {repo_name}",
        f"File: {rel_path} （总行数约为 {file_total_lines} 行）",
        f"Refine Target: {target_desc}",
        f"粗修后的候选行区间: {coarse_desc}",
    ]
    if issue:
        user_parts.append(f"Issue 描述：\n{issue}")
    if conversation:
        user_parts.append(f"最近对话片段（可选）：\n{conversation}")
    user_parts.append("以下是该文件的相关源码片段：")
    user_parts.append("```code")
    user_parts.append(source_snippet)
    user_parts.append("```")
    user_parts.append(
        "请根据以上信息，只输出你认为“最小必要”的行号区间列表，格式严格为如 `[5-7,13-19]` 的一行文本。"
    )

    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content="\n\n".join(user_parts)),
    ]


def _regions_from_json(regions: list[dict[str, Any]]) -> list[LineRegion]:
    out: list[LineRegion] = []
    for r in regions:
        out.append(
            LineRegion(
                path=str(r["path"]),
                start=int(r["start"]),
                end=int(r["end"]),
            )
        )
    return out


def _per_file_to_region_dicts(
    per_file: dict[str, list[tuple[int, int]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(per_file):
        for start, end in _merge_intervals(per_file[path]):
            out.append({"path": path, "start": start, "end": end})
    return out


def _resolve_record_repo_dir(record: dict[str, Any], repos_root: Path) -> Path | None:
    instance_id = record.get("instance_id", "")
    record_repo_dir = record.get("repo_dir")
    if record_repo_dir:
        repo_dir = Path(record_repo_dir)
        if repo_dir.is_dir():
            return repo_dir
    return _resolve_repo_dir(repos_root, instance_id)


def _iter_refine_groups(
    record: dict[str, Any],
    target: str = "core",
) -> list[dict[str, Any]]:
    """按 target 返回需要精修的 region 分组。"""
    gt = record.get("ground_truth", {})
    groups: list[dict[str, Any]] = []

    if target == "core":
        groups.append(
            {
                "target_type": "core",
                "target_model": None,
                "regions": _regions_from_json(gt.get("read_core_regions", [])),
            }
        )
        return groups

    for model_name in sorted((gt.get("read_optional_regions_map") or {}).keys()):
        groups.append(
            {
                "target_type": "optional",
                "target_model": model_name,
                "regions": _regions_from_json(
                    (gt.get("read_optional_regions_map") or {}).get(model_name, [])
                ),
            }
        )
    return groups


def _build_refine_call_bundle(
    record: dict[str, Any],
    repos_root: Path,
    ctx: dict[str, str] | None,
    context_k: int = 4,
    target: str = "core",
) -> tuple[dict[str, Any], int]:
    """为单条记录构造可重放的 LLM 调用包。"""
    instance_id = record.get("instance_id", "")
    repo_name = instance_id.split("__")[0] if instance_id else ""
    repo_dir = _resolve_record_repo_dir(record, repos_root)
    read_step_info = record.get("read_step_info", {})
    calls: list[dict[str, Any]] = []
    total_input_chars = 0
    for group in _iter_refine_groups(
        record,
        target=target,
    ):
        per_file = _group_regions_by_file(group["regions"])
        for rel_path, intervals in per_file.items():
            file_step_info = read_step_info.get(rel_path, [])
            if file_step_info:
                file_ctx = _load_context_from_step_info(file_step_info, k=context_k)
            else:
                file_ctx = ctx
            issue = (file_ctx or {}).get("issue")
            conversation = (file_ctx or {}).get("conversation")

            file_path = (
                (repo_dir / rel_path)
                if repo_dir
                else (repos_root / repo_name / rel_path)
            )
            if not file_path.is_file():
                continue

            source_snippet, total_lines = _build_source_snippet(file_path, intervals)
            if not source_snippet.strip():
                continue

            messages = _build_refine_messages(
                instance_id=instance_id,
                repo_name=repo_name,
                rel_path=rel_path,
                coarse_intervals=intervals,
                source_snippet=source_snippet,
                file_total_lines=total_lines,
                issue=issue,
                conversation=conversation,
                target_type=group["target_type"],
                target_model=group["target_model"],
            )

            agent_input = messages_to_agent_input(messages)
            total_input_chars += len(agent_input)
            calls.append(
                {
                    "path": rel_path,
                    "target_type": group["target_type"],
                    "target_model": group["target_model"],
                    "coarse_intervals": [[s, e] for s, e in intervals],
                    "file_total_lines": total_lines,
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in messages
                    ],
                    "agent_input": agent_input,
                }
            )

    bundle = {
        "schema_version": 1,
        "kind": "line_refine_calls",
        "instance_id": instance_id,
        "target": target,
        "record": json.loads(json.dumps(record, ensure_ascii=False)),
        "calls": calls,
        "stats": {
            "input_chars": total_input_chars,
            "estimated_output_chars": len(calls) * 50,
            "num_calls": len(calls),
        },
    }
    return bundle, total_input_chars


def _extract_call_messages(call: dict[str, Any]) -> list[ChatMessage]:
    raw_messages = call.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        out: list[ChatMessage] = []
        for message in raw_messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user")
            content = str(message.get("content") or "")
            out.append(ChatMessage(role=role, content=content))
        if out:
            return out
    return agent_input_to_messages(str(call.get("agent_input") or ""))


def _invoke_refine_call(client: Any, call: dict[str, Any]) -> str:
    messages = _extract_call_messages(call)
    return client.invoke(messages)


def _normalize_refined_intervals(
    parsed: list[tuple[int, int]],
    coarse_intervals: list[tuple[int, int]],
    total_lines: int,
) -> list[tuple[int, int]]:
    if not parsed:
        return coarse_intervals

    clipped: list[tuple[int, int]] = []
    for start, end in parsed:
        if start > total_lines:
            continue
        start = max(1, start)
        end = min(total_lines, end)
        if start <= end:
            clipped.append((start, end))
    if not clipped:
        return coarse_intervals
    return _merge_intervals(clipped)


def _apply_refine_reply_to_record(
    record: dict[str, Any],
    target: str,
    call: dict[str, Any],
    reply: str,
) -> dict[str, Any]:
    """将单次 call 的回复应用到当前 record，返回更新后的 bench-like record。"""
    record = json.loads(json.dumps(record, ensure_ascii=False))
    gt = dict(record.get("ground_truth", {}))

    coarse_intervals = [
        (int(start), int(end))
        for start, end in call.get("coarse_intervals", [])
    ]
    refined_intervals = _normalize_refined_intervals(
        _parse_range_list(reply),
        coarse_intervals,
        int(call.get("file_total_lines") or 0),
    )
    path = str(call.get("path") or "")

    if target == "core":
        per_file = _group_regions_by_file(
            _regions_from_json(gt.get("read_core_regions", []))
        )
        per_file[path] = refined_intervals
        gt["read_core_regions"] = _per_file_to_region_dicts(per_file)
    else:
        model_name = str(call.get("target_model") or "")
        original_opt_map = gt.get("read_optional_regions_map") or {}
        per_model: dict[str, dict[str, list[tuple[int, int]]]] = {}
        for name, regions in original_opt_map.items():
            per_model[name] = _group_regions_by_file(_regions_from_json(regions))

        per_model.setdefault(model_name, {})[path] = refined_intervals
        gt["read_optional_regions_map"] = {
            name: _per_file_to_region_dicts(per_model.get(name, {}))
            for name in sorted(per_model)
        }
        gt["read_optional_files_map"] = {
            name: sorted(
                file_path
                for file_path, intervals in per_model.get(name, {}).items()
                if intervals
            )
            for name in sorted(per_model)
        }

    record["ground_truth"] = gt
    return record


def _execute_refine_call_bundle(
    bundle: dict[str, Any],
    client: Any,
) -> tuple[dict[str, Any], int, int]:
    """执行 call bundle，并返回与 bench.jsonl 同结构的精修结果。"""
    record = json.loads(json.dumps(bundle["record"], ensure_ascii=False))
    target = str(bundle.get("target") or "core")

    total_input_chars = 0
    total_output_chars = 0
    calls = bundle.get("calls", [])

    for call in calls:
        input_str = str(call.get("agent_input") or "")
        total_input_chars += len(input_str)
        reply = _invoke_refine_call(client, call)
        total_output_chars += len(reply)
        record = _apply_refine_reply_to_record(record, target, call, reply)

    return record, total_input_chars, total_output_chars


def _coarse_fix_single_record(
    record: dict[str, Any],
    repos_root: Path,
) -> dict[str, Any]:
    """对单条 bench 记录做粗修：

    - 将 end == -1 替换为真实文件末行
    - 裁剪越界区间；丢弃完全越界的区间
    - 合并同一文件中相邻/重叠区间
    """
    instance_id = record.get("instance_id", "")
    repo_name = instance_id.split("__")[0] if instance_id else ""

    core_regions_raw = [
        LineRegion(
            path=r["path"],
            start=int(r["start"]),
            end=int(r["end"]),
        )
        for r in record.get("ground_truth", {}).get("read_core_regions", [])
    ]

    fixed: list[LineRegion] = []
    repo_dir = _resolve_repo_dir(repos_root, instance_id)
    for region in core_regions_raw:
        file_path = (repo_dir / region.path) if repo_dir else (repos_root / repo_name / region.path)
        if not file_path.is_file():
            # 若本地没有对应文件，保守起见原样保留
            fixed.append(region)
            continue

        try:
            n_lines = sum(1 for _ in file_path.open("r", encoding="utf-8", errors="ignore"))
        except (OSError, UnicodeDecodeError):
            fixed.append(region)
            continue

        start = max(1, region.start)
        end = n_lines if region.end < 0 or region.end > n_lines else region.end

        if start > n_lines:
            # 完全越界，丢弃
            continue
        if start > end:
            start, end = end, start

        fixed.append(LineRegion(path=region.path, start=start, end=end))

    per_file = _group_regions_by_file(fixed)
    merged = _flatten_regions(per_file)

    record = dict(record)
    gt = dict(record.get("ground_truth", {}))
    gt["read_core_regions"] = [
        {"path": r.path, "start": r.start, "end": r.end} for r in merged
    ]
    record["ground_truth"] = gt
    return record


@app.command()
def coarse(
    bench_path: Path = typer.Argument(
        Path("bench.jsonl"),
        exists=True,
        readable=True,
        help="由 bench_build.py 生成的原始 bench.jsonl",
    ),
    repos_root: Path = typer.Argument(
        Path("repos"),
        exists=True,
        file_okay=False,
        help="包含各个原始仓库的根目录，例如 repos/django",
    ),
    output: Path = typer.Option(
        Path("bench.coarse.jsonl"),
        "--output",
        "-o",
        help="粗修后的 bench 输出路径",
    ),
) -> None:
    """对 bench_build 结果进行“粗修”：修正 -1 占位符与明显越界区间。"""
    records = list(_load_bench_records(bench_path))
    with (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress,
        output.open("w") as fout,
    ):
        task = progress.add_task("Coarse refining...", total=len(records))
        for rec in records:
            fixed = _coarse_fix_single_record(rec, repos_root=repos_root)
            fout.write(json.dumps(fixed, ensure_ascii=False) + "\n")
            progress.advance(task)

    console.print(
        f"[green]Wrote coarse-refined benchmark to [bold]{output}[/bold][/green]"
    )


@app.command()
def refine(
    bench_path: Path = typer.Argument(
        Path("bench.jsonl"),
        exists=True,
        readable=True,
        help="由 bench_build.py 生成的 bench.jsonl",
    ),
    repos_root: Path = typer.Argument(
        Path("repos"),
        exists=True,
        file_okay=False,
        help="包含各个原始仓库的根目录，例如 repos/django",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="非 dry-run 时为精修后的 bench 输出路径；dry-run 时可选，用于保存可重放的 LLM call JSONL",
    ),
    target: str = typer.Option(
        "core",
        "--target",
        help="精修目标：core 或 optional",
    ),
    context_k: int = typer.Option(
        4,
        "--context-k",
        "-k",
        help="在每个读取步骤前后各取 k 条消息作为上下文",
    ),
    context_path: Path | None = typer.Option(
        None,
        "--context",
        "-c",
        help="可选：包含 issue / conversation 的 JSONL 文件，按 instance_id 关联（已废弃，现在自动从 read_step_info 加载）",
    ),
    trajs_dir: Path | None = typer.Option(
        None,
        "--trajs-dir",
        "-t",
        help="可选：轨迹根目录；当 bench 中无 read_step_info 时，从该目录下对应轨迹中抽取 issue 与对话片段",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="只统计输入输出tokens和成本，不实际调用LLM",
    ),
    input_cost: float = typer.Option(
        3.0,
        "--input-cost",
        help="输入成本，单位$/M tokens",
    ),
    output_cost: float = typer.Option(
        15.0,
        "--output-cost",
        help="输出成本，单位$/M tokens",
    ),
) -> None:
    """使用 LLM 对行级上下文做"精修"，得到更接近最小必要上下文的行号范围。

    现在自动从 bench.jsonl 中的 read_step_info 加载每个文件读取时的上下文。
    可通过 --context-k 控制在读取步骤前后各取多少条消息作为上下文。
    模型由环境变量决定（DEFAULT_LLM_PROVIDER / MSWEA_* / OPENAI_*），直接使用封装好的 default agent。
    """
    if target not in ("core", "optional"):
        console.print(f"[red]target 只能是 core 或 optional，当前为: {target}[/red]")
        raise typer.Exit(1)

    if not dry_run and output is None:
        output = Path(
            "bench.refined.jsonl"
            if target == "core"
            else "bench.optional.refined.jsonl"
        )

    client = None if dry_run else get_default_openrouter_client()

    trajs_root = trajs_dir.resolve() if trajs_dir else None
    if trajs_root is not None and not trajs_root.is_dir():
        console.print(f"[red]trajs-dir 不存在或不是目录: {trajs_root}[/red]")
        raise typer.Exit(1)

    context_map = _load_context_map(context_path)
    records = list(_load_bench_records(bench_path))

    total_input_chars = 0
    total_output_chars = 0

    mode_desc = (
        f"Dry-run ({target}, preparing calls)"
        if dry_run
        else f"LLM refining ({target})"
    )

    if dry_run:
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(mode_desc + "...", total=len(records))
            fout = output.open("w") if output is not None else None
            try:
                for rec in records:
                    instance_id = rec.get("instance_id", "")
                    ctx = context_map.get(instance_id) or (
                        _load_context_from_trajectory(trajs_root, instance_id)
                        if trajs_root
                        else None
                    )
                    bundle, in_chars = _build_refine_call_bundle(
                        rec,
                        repos_root=repos_root,
                        ctx=ctx,
                        context_k=context_k,
                        target=target,
                    )
                    total_input_chars += in_chars
                    total_output_chars += int(
                        (bundle.get("stats") or {}).get("estimated_output_chars") or 0
                    )
                    if fout is not None:
                        fout.write(json.dumps(bundle, ensure_ascii=False) + "\n")
                    progress.advance(task)
            finally:
                if fout is not None:
                    fout.close()

        if output is not None:
            console.print(
                f"[green]Wrote refine call bundles to [bold]{output}[/bold][/green]"
            )
    else:
        with (
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress,
            output.open("w") as fout,
        ):
            task = progress.add_task(mode_desc + "...", total=len(records))
            for rec in records:
                instance_id = rec.get("instance_id", "")
                ctx = context_map.get(instance_id) or (
                    _load_context_from_trajectory(trajs_root, instance_id)
                    if trajs_root
                    else None
                )
                bundle, _ = _build_refine_call_bundle(
                    rec,
                    repos_root=repos_root,
                    ctx=ctx,
                    context_k=context_k,
                    target=target,
                )
                refined, in_chars, out_chars = _execute_refine_call_bundle(
                    bundle,
                    client=client,
                )
                total_input_chars += in_chars
                total_output_chars += out_chars
                fout.write(json.dumps(refined, ensure_ascii=False) + "\n")
                progress.advance(task)

        console.print(
            f"[green]Wrote LLM-refined benchmark to [bold]{output}[/bold][/green]"
        )

    # 统计和成本估算
    # 使用字符数作为 tokens 估算（通常 1 token ≈ 4 chars for English, ≈ 2 chars for Chinese）
    # 这里采用保守估算：1 token ≈ 2.5 chars
    estimated_input_tokens = total_input_chars / 2.5
    estimated_output_tokens = total_output_chars / 2.5

    input_cost_total = (estimated_input_tokens / 1_000_000) * input_cost
    output_cost_total = (estimated_output_tokens / 1_000_000) * output_cost
    total_cost = input_cost_total + output_cost_total

    console.print("\n[bold cyan]Token & Cost Statistics:[/bold cyan]")
    console.print(f"  Total records: [yellow]{len(records)}[/yellow]")
    console.print(f"  Input chars: [yellow]{total_input_chars:,}[/yellow]")
    console.print(f"  Output chars: [yellow]{total_output_chars:,}[/yellow]")
    console.print(
        f"  Estimated input tokens: [yellow]{estimated_input_tokens:,.0f}[/yellow]"
    )
    console.print(
        f"  Estimated output tokens: [yellow]{estimated_output_tokens:,.0f}[/yellow]"
    )
    console.print(
        f"  Input cost (${input_cost}/M): [green]${input_cost_total:.4f}[/green]"
    )
    console.print(
        f"  Output cost (${output_cost}/M): [green]${output_cost_total:.4f}[/green]"
    )
    console.print(
        f"  [bold]Total estimated cost: [green]${total_cost:.4f}[/green][/bold]"
    )

    if dry_run:
        console.print(
            "\n[yellow]Note: This was a dry-run. No actual LLM calls were made.[/yellow]"
        )


if __name__ == "__main__":
    app()
