import json
import re
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm
from logging import getLogger
from rich.logging import RichHandler
import logging

# 仅支持统一轨迹格式 (info + traj)，下游统一用 messages = traj 的形态

logger = getLogger(__name__)
handler = RichHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = typer.Typer(rich_markup_mode="rich")
console = Console()

# Bash code block pattern
_BASH_BLOCK = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)

# 行级读取：(文件路径, 起始行, 结束行)。结束行 -1 表示「到文件末尾」，评估时可替换为 path 的真实结尾行数
ReadRegion = tuple[str, int | None, int]
# 带步骤索引的读取区域：(文件路径, 起始行, 结束行, 轨迹步骤索引)
ReadRegionWithStep = tuple[str, int | None, int, int]

# 用于识别“像文件路径”的扩展名
_PATH_EXTS = (
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".kt",
    ".kts",
    ".scala",
    ".swift",
    ".cs",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".hh",
    ".hxx",
    ".m",
    ".mm",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sh",
    ".bash",
    ".zsh",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".less",
    ".vue",
    ".svelte",
    ".xml",
    ".toml",
    ".cfg",
    ".ini",
    ".gradle",
    ".sql",
    ".lua",
    ".pl",
    ".pm",
    ".ex",
    ".exs",
    ".erl",
    ".clj",
    ".dart",
    ".tf",
    ".proto",
)


def _is_likely_path(word: str) -> bool:
    w = word.strip("'\"")
    if w.startswith(("/path/to/", "/example/")):
        return False
    return "/" in w or any(w.lower().endswith(ext) for ext in _PATH_EXTS)


def _is_file_content_read_command(command: str) -> bool:
    """是否为准入读文件内容的命令：cat, head, tail, nl, less, more, sed -n 等。"""
    c = command.strip().lower()
    if not c:
        return False
    parts = c.split()
    cmd = parts[0]
    if cmd in ("cat", "head", "tail", "nl", "less", "more"):
        return True
    if cmd == "sed" and len(parts) >= 2 and "-n" in parts[:3]:
        return True
    return False


def _extract_paths_from_bash_command(command: str) -> list[str]:
    """从 cat/head/tail/sed 等命令中提取文件路径参数。"""
    paths: list[str] = []
    words = command.split()
    i = 0
    while i < len(words):
        w = words[i]
        if w in ("-n", "-c", "-q", "-v") and i + 1 < len(words):
            i += 2
            continue
        if w.startswith("-n") and len(w) > 2:
            i += 1
            continue
        if w.startswith("-"):
            i += 1
            continue
        cleaned = w.strip("'\"")
        if _is_likely_path(cleaned):
            paths.append(cleaned)
        i += 1
    return paths


def _parse_line_range_from_content_read_command(
    command: str,
) -> tuple[int, int]:
    """
    从 cat/head/tail/sed 等命令解析行范围。
    head -n 20 -> (1, 20); tail -n 8 -> (-8, -1) 表示末尾 8 行（start 负值表示距文件末尾的偏移，评估时可替换）；cat/整文件 -> (1, -1)。
    """
    c = command.strip()
    parts = c.split()
    if not parts:
        return (1, -1)
    cmd = parts[0].lower()
    if cmd == "cat" or cmd in ("nl", "less", "more"):
        return (1, -1)
    if cmd == "head":
        n_val: int | None = None
        for i, w in enumerate(parts):
            if w == "-n" and i + 1 < len(parts):
                try:
                    n_val = int(parts[i + 1])
                    break
                except ValueError:
                    pass
            if w.startswith("-n") and len(w) > 2:
                try:
                    n_val = int(w[2:])
                    break
                except ValueError:
                    pass
        if n_val is not None and n_val > 0:
            return (1, n_val)
        return (1, -1)
    if cmd == "tail":
        n_val = None
        for i, w in enumerate(parts):
            if w == "-n" and i + 1 < len(parts):
                try:
                    n_val = int(parts[i + 1])
                    break
                except ValueError:
                    pass
            if w.startswith("-n") and len(w) > 2:
                try:
                    n_val = int(w[2:])
                    break
                except ValueError:
                    pass
        if n_val is not None and n_val > 0:
            return (-n_val, -1)
        return (1, -1)
    if cmd == "sed":
        for w in parts:
            if "p" in w and ("'" in w or '"' in w):
                m = re.search(r"['\"](\d+)(?:,(\d+))?p?['\"]", w)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else start
                    return (start, end)
                m = re.search(r"['\"](\d+)p['\"]", w)
                if m:
                    line = int(m.group(1))
                    return (line, line)
        return (1, -1)
    logger.warning(f"Unknown command: {command}")
    return (1, -1)


def _extract_paths_from_grep_command(command: str) -> list[str]:
    """从 grep 命令中提取文件/目录路径。"""
    if not command.strip().lower().startswith("grep"):
        return []
    words = command.split()
    paths: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w.startswith("-"):
            i += 1
            continue
        cleaned = w.strip("'\"")
        if _is_likely_path(cleaned) or "/" in cleaned:
            paths.append(cleaned)
        i += 1
    return paths


def _parse_grep_n_output(content: str) -> list[ReadRegion]:
    """
    从 grep -n 的命令输出中解析 (path, line, line)。
    常见格式：path:line_num:... 或 path-line_num-...
    """
    regions: list[ReadRegion] = []
    seen: set[tuple[str, int]] = set()
    for line in content.splitlines():
        m = re.match(r"^([^:]+):(\d+)[:\-]", line)
        if m:
            path = m.group(1).strip()
            ln = int(m.group(2))
            if (path, ln) not in seen:
                seen.add((path, ln))
                regions.append((path, ln, ln))
    return regions


def _resolve_repo_dir(repos_root: Path, instance_id: str) -> Path | None:
    """
    解析 instance_id 对应的本地仓库目录。
    instance_id 格式: org__repo-issue (如 lincolnloop__goodconf-49)
    尝试多种命名模式匹配 repos_root 下的目录。
    """
    if not instance_id or not repos_root.is_dir():
        return None

    # 从 instance_id 解析 org 和 repo
    # lincolnloop__goodconf-49 -> org=lincolnloop, repo=goodconf
    if "__" not in instance_id:
        return None
    org, rest = instance_id.split("__", 1)
    # rest = goodconf-49, 去掉 issue 后缀得到 repo
    repo = rest.rsplit("-", 1)[0] if "-" in rest else rest

    # 尝试多种目录命名模式
    candidates = [
        repos_root
        / instance_id,  # lincolnloop__goodconf-49 (fetch_repos 按 instance_id 命名)
        repos_root / repo,  # goodconf
        repos_root / f"{org}__{repo}",  # lincolnloop__goodconf
        repos_root / f"{org}-{repo}",  # lincolnloop-goodconf
        repos_root / org,  # lincolnloop (for repos like django__django)
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return None


def if_path_exists_original(
    path: str,
    repo_name: str,
    *,
    repo_dir: Path | None = None,
    norm_path: str | None = None,
) -> bool:
    """Check if path exists in original repo. When repo_dir and norm_path are given, check on disk."""
    if not path:
        return False
    if "http" in path:
        return False
    special_chars = "\")'`][,<>?*|&^%$#@!~\\{\\};"
    if any(c in path for c in special_chars):
        return False
    # Filter temp/reproduce files by first path component or filename
    first_comp = path.split("/")[0]
    basename = path.rsplit("/", 1)[-1] if "/" in path else path
    temp_prefixes = ("reproduce", "tmp", "temp", "debug_", "test_reproduce")
    if first_comp.startswith(temp_prefixes) or basename.startswith(temp_prefixes):
        return False
    # If repo_dir is given, check on disk
    if repo_dir is not None and norm_path is not None:
        full = repo_dir / norm_path
        return full.is_file()
    # For paths without . (like directories), only allow known directory names
    if "." not in path:
        return first_comp in ["test", "tests", "src", "lib", "docs", "examples"]
    # Accept paths with valid source file extensions
    if any(basename.endswith(ext) for ext in _PATH_EXTS):
        return True
    # Legacy checks
    if repo_name in path.split("/"):
        return True
    if first_comp in ["test", "tests", "src", "lib", "docs", "examples"]:
        return True
    return False


def _get_message_content(msg: dict) -> str:
    """从单条 message 中取出纯文本 content。"""
    content = msg.get("content", "")
    if isinstance(content, list):
        return " ".join(str(x) for x in content if isinstance(x, str))
    if not isinstance(content, str):
        return str(content) if content else ""
    return content


# ==================== Read Region Extraction Strategies ====================


def _get_tool_call_args(tool_call: dict) -> dict[str, Any]:
    """Extract arguments from a tool call, handling both dict and JSON string formats."""
    func = tool_call.get("function", {})
    args = func.get("arguments", {})
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


class MiniSweStyleReadParser:
    """
    Parser for MiniSwe-style trajectories that use bash code blocks in content.
    Extracts read regions from ```bash...``` blocks with cat/head/tail/grep etc.
    """

    @staticmethod
    def parse_bash_command(command: str, next_content: str = "") -> list[ReadRegion]:
        """Parse a bash command string to extract read regions.

        Supports chained commands like `cd X && cat foo` or `a; b; c` by
        splitting on top-level `&&`, `||`, `;` and recursing.
        """
        regions: list[ReadRegion] = []
        if not command:
            return regions

        # Split chained commands; only attach next_content to the LAST segment
        # (the next message's content corresponds to the final command's stdout).
        segments = re.split(r"\s*(?:&&|\|\||;)\s*", command.strip())
        segments = [s for s in segments if s]
        if len(segments) > 1:
            for idx, seg in enumerate(segments):
                nc = next_content if idx == len(segments) - 1 else ""
                regions.extend(
                    MiniSweStyleReadParser.parse_bash_command(seg, nc)
                )
            return regions

        cmd = command.strip()
        if _is_file_content_read_command(cmd):
            paths = _extract_paths_from_bash_command(cmd)
            start, end = _parse_line_range_from_content_read_command(cmd)
            for p in paths:
                regions.append((p, start, end))
        elif cmd.lower().startswith("grep") or cmd.lower().startswith("rg "):
            paths = _extract_paths_from_grep_command(cmd)
            line_regions = _parse_grep_n_output(next_content) if next_content else []
            if line_regions:
                regions.extend(line_regions)
            else:
                for p in paths:
                    regions.append((p, 1, -1))
        return regions

    @staticmethod
    def extract_from_content(content: str, next_content: str = "") -> list[ReadRegion]:
        """Extract read regions from ```bash...``` code blocks in message content."""
        match = _BASH_BLOCK.search(content)
        if not match:
            return []
        command = match.group(1).strip()
        return MiniSweStyleReadParser.parse_bash_command(command, next_content)


class OpenhandsStyleReadParser:
    """
    Parser for OpenHands-style trajectories that use tool_calls.
    Extracts read regions from:
    - execute_bash: parse command string for read commands
    - str_replace_editor with command="view": extract path and view_range
    """

    @staticmethod
    def extract_from_tool_calls(
        tool_calls: list[dict], next_content: str = ""
    ) -> list[ReadRegion]:
        """Extract read regions from tool_calls array."""
        regions: list[ReadRegion] = []
        if not tool_calls:
            return regions

        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = _get_tool_call_args(tc)

            if name in ("execute_bash", "bash"):
                # Parse bash command using MiniSweStyleReadParser
                command = args.get("command", "")
                if command:
                    regions.extend(
                        MiniSweStyleReadParser.parse_bash_command(command, next_content)
                    )

            elif name == "str_replace_editor":
                cmd = args.get("command", "")
                path = args.get("path", "")
                if cmd == "view" and path:
                    view_range = args.get("view_range")
                    if isinstance(view_range, list) and len(view_range) >= 2:
                        start, end = int(view_range[0]), int(view_range[1])
                    else:
                        start, end = 1, -1  # Full file
                    regions.append((path, start, end))

        return regions


# Legacy function wrappers for backward compatibility
def _extract_read_regions_from_bash_content(
    content: str, next_content: str = ""
) -> list[ReadRegion]:
    """Strategy 1: Extract from ```bash...``` blocks (MiniSwe style)."""
    return MiniSweStyleReadParser.extract_from_content(content, next_content)


def _extract_read_regions_from_tool_calls(
    tool_calls: list[dict], next_content: str = ""
) -> list[ReadRegion]:
    """Strategy 2: Extract from tool_calls (OpenHands style)."""
    return OpenhandsStyleReadParser.extract_from_tool_calls(tool_calls, next_content)


# ==================== Modified Files Extraction Strategies ====================


def _extract_modified_files_from_tool_calls(messages: list[dict]) -> set[str]:
    """
    Extract modified file paths from str_replace_editor tool calls with command="str_replace".
    Complements diff-based parsing for trajectories without submission diff.
    """
    paths: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            if name != "str_replace_editor":
                continue
            args = _get_tool_call_args(tc)
            cmd = args.get("command", "")
            path = args.get("path", "")
            if cmd == "str_replace" and path:
                paths.add(path)
    return paths


def _detect_repo_path_from_trajectory(data: dict[str, Any]) -> str | None:
    """
    Try to detect repo_path from trajectory tool calls when config.environment.cwd is missing.
    Looks for /workspace/org__repo__version/ pattern in str_replace_editor paths.
    """
    messages = data.get("messages", [])
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            if name != "str_replace_editor":
                continue
            args = _get_tool_call_args(tc)
            path = args.get("path", "")
            # Match /workspace/org__repo__version/ pattern
            if path.startswith("/workspace/"):
                parts = path.split("/", 3)
                if len(parts) >= 3:
                    # Return /workspace/org__repo__version
                    return f"/workspace/{parts[2]}"
    return None


def load_trajectory_data(path: Path) -> dict[str, Any]:
    """
    仅加载统一轨迹格式 (info + traj)。返回形态为 {"info": ..., "messages": traj} 供下游复用。
    无 traj 或解析失败时返回 messages=[]。
    """
    with open(path, "r") as f:
        data = json.load(f)
    if "traj" not in data:
        logger.warning("Unified format requires 'traj' field: %s", path)
        return {"info": data.get("info", {}), "messages": []}
    info = data.get("info", {})
    traj = data.get("traj", [])
    # 下游期望 messages 列表，每项为 dict (role, content, tool_calls 等)
    messages = [dict(m) for m in traj if isinstance(m, dict)]
    return {"info": info, "messages": messages}


def extract_read_regions_from_trajectory_data(data: dict[str, Any]) -> list[ReadRegion]:
    """
    从轨迹中提取行级读取区域 (path, start_line, end_line)。
    使用多种策略:
    1. ```bash...``` 代码块中的 cat/head/tail/grep 等命令
    2. execute_bash tool_call 中的读取命令
    3. str_replace_editor tool_call 中的 view 命令
    """
    regions_with_step = extract_read_regions_with_step(data)
    # 去掉 step_idx，返回原始格式以保持向后兼容
    return [(path, start, end) for path, start, end, _ in regions_with_step]


def extract_read_regions_with_step(data: dict[str, Any]) -> list[ReadRegionWithStep]:
    """
    从轨迹中提取带步骤索引的行级读取区域 (path, start_line, end_line, step_idx)。
    step_idx 是该读取操作在轨迹 messages 列表中的位置索引。
    """
    regions: list[ReadRegionWithStep] = []
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return regions

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue

        # 获取下一条消息内容 (用于解析 grep -n 输出)
        next_content = ""
        if idx + 1 < len(messages):
            next_msg = messages[idx + 1]
            if isinstance(next_msg, dict):
                next_content = _get_message_content(next_msg)

        # Strategy 1: Extract from ```bash...``` blocks in content
        content = _get_message_content(msg)
        for path, start, end in _extract_read_regions_from_bash_content(
            content, next_content
        ):
            regions.append((path, start, end, idx))

        # Strategy 2: Extract from tool_calls (execute_bash, str_replace_editor view)
        tool_calls = msg.get("tool_calls") or []
        for path, start, end in _extract_read_regions_from_tool_calls(
            tool_calls, next_content
        ):
            regions.append((path, start, end, idx))

    return regions


def parse_modified_files_from_submission(submission: str) -> list[str]:
    """Parse modified file paths from submission diff (diff --git a/X b/Y and +++ b/X)."""
    if not submission:
        return []
    paths: set[str] = set()
    for m in re.finditer(r"^diff --git a/(.+?) b/", submission, re.MULTILINE):
        paths.add(m.group(1).strip())
    for m in re.finditer(r"^\+\+\+ b/(.+?)$", submission, re.MULTILINE):
        paths.add(m.group(1).strip())
    return sorted(paths)


def discover_trajectory_groups(
    trajs_dir: Path,
    model_filter: re.Pattern[str] | None,
    instance_filter: re.Pattern[str] | None,
    repo_filter: re.Pattern[str] | None = None,
) -> dict[str, list[Path]]:
    """
    递归发现 trajs_dir 下所有 .json，按文件内 info.instance_id 分组。
    model_filter: 对 info.model 做正则；instance_filter: 对 instance_id 做正则；repo_filter: 对 info.repo 做正则。
    """
    groups: dict[str, list[Path]] = {}
    if not trajs_dir.is_dir():
        return groups

    for traj_file in tqdm(trajs_dir.rglob("*.json"), desc="Discovering trajectories"):
        if not traj_file.is_file():
            continue
        try:
            data = load_trajectory_data(traj_file)
        except Exception:
            continue
        info = data.get("info") or {}
        instance_id = info.get("instance_id") or traj_file.stem.removesuffix(".traj")
        if not isinstance(instance_id, str):
            continue
        if instance_filter is not None and not instance_filter.search(instance_id):
            continue
        if model_filter is not None and not model_filter.search(
            str(info.get("model", ""))
        ):
            continue
        if repo_filter is not None and not repo_filter.search(
            str(info.get("repo", ""))
        ):
            continue
        groups.setdefault(instance_id, []).append(traj_file)

    return groups


def _normalize_path(p: str, repo_path: str) -> str:
    """Normalize to repo-relative path: strip repo prefix and leading / and ./."""
    if not p:
        return p
    p = p.removeprefix("open('").removesuffix("')")
    p = p.removeprefix("`").removesuffix("`")
    strip_chars = "\")'`,"
    p = p.rstrip(strip_chars)

    # Try direct repo_path prefix
    if p.startswith(repo_path):
        p = p.removeprefix(repo_path).lstrip("/").removeprefix("./")
        return p

    # Handle /workspace/org__repo__version/ style paths (OpenHands format)
    # e.g. /workspace/lux-org__lux__0.4/lux/file.py -> lux/file.py
    if p.startswith("/workspace/"):
        parts = p.split("/", 3)  # ['', 'workspace', 'org__repo__version', 'rest']
        if len(parts) >= 4:
            p = parts[3]  # repo-relative path
            return p
        elif len(parts) == 3:
            return ""  # Just /workspace/repo_dir with no subpath

    # Fallback: strip repo_path and clean up
    p = p.removeprefix(repo_path).lstrip("/").removeprefix("./")
    return p


# 用于过滤临时/调试文件的模式
_TEMP_FILE_PATTERNS = (
    "reproduce_",
    "test_reproduce",
    "debug_",
    "tmp_",
    "temp_",
    "__pycache__",
)


def _is_temp_file(path: str) -> bool:
    """Check if path looks like a temporary/debugging file that should be filtered."""
    basename = path.rsplit("/", 1)[-1] if "/" in path else path
    return any(basename.startswith(pat) for pat in _TEMP_FILE_PATTERNS)


# 行级区间：内部运算用 (1, MAX_LINE) 表示「到文件末尾」；输出时用 -1 表示，评估时可替换为真实结尾行数
MAX_LINE = 10**6
END_OF_FILE = -1


def _is_tail_interval(interval: tuple[int, int]) -> bool:
    """start 为负或 end 为 -1 表示「末尾 N 行」等，不参与合并/求交。"""
    a, b = interval
    return a < 0 or b < 0


def _to_concrete_interval(start: int | None, end: int | None) -> tuple[int, int]:
    """(1, -1) 等转为内部区间；tail 型 (负 start, -1) 原样返回不转换。"""
    if start is not None and start < 0:
        return (start, -1 if end is None or end == END_OF_FILE else end)
    if start is None and (end is None or end == END_OF_FILE):
        return (1, MAX_LINE)
    s = 1 if start is None else max(1, start)
    e = MAX_LINE if (end is None or end == END_OF_FILE) else max(s, end)
    return (s, e)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并重叠/相邻的区间。tail 型区间（start<0 或 end<0）保留不合并。"""
    if not intervals:
        return []
    concrete = [x for x in intervals if not _is_tail_interval(x)]
    tail = [x for x in intervals if _is_tail_interval(x)]
    if not concrete:
        return tail
    sorted_i = sorted(concrete, key=lambda x: (x[0], x[1]))
    out: list[tuple[int, int]] = [sorted_i[0]]
    for a, b in sorted_i[1:]:
        lo, hi = out[-1]
        if a <= hi + 1:
            out[-1] = (lo, max(hi, b))
        else:
            out.append((a, b))
    out.extend(tail)
    return out


def _intersect_two_intervals(
    a: tuple[int, int], b: tuple[int, int]
) -> tuple[int, int] | None:
    """两个闭区间的交集。tail 型区间不参与求交，返回 None。"""
    if _is_tail_interval(a) or _is_tail_interval(b):
        return None
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    if s <= e:
        return (s, e)
    return None


def _intersect_intervals(
    list_a: list[tuple[int, int]], list_b: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """两个区间集合的交集（每个集合已合并）。"""
    out: list[tuple[int, int]] = []
    for ia in list_a:
        for ib in list_b:
            inter = _intersect_two_intervals(ia, ib)
            if inter:
                out.append(inter)
    return _merge_intervals(out)


def _interval_set_minus(
    full: list[tuple[int, int]], to_remove: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """从 full 的区间中挖掉 to_remove 的区间。tail 型区间不参与减法，原样保留。"""
    if not full:
        return []
    if not to_remove:
        return full[:]
    result: list[tuple[int, int]] = []
    for fs, fe in full:
        if _is_tail_interval((fs, fe)):
            result.append((fs, fe))
            continue
        current = [(fs, fe)]
        for rs, r_end in to_remove:
            if _is_tail_interval((rs, r_end)):
                continue
            next_current: list[tuple[int, int]] = []
            for cs, ce in current:
                if r_end < cs or rs > ce:
                    next_current.append((cs, ce))
                else:
                    if cs < rs:
                        next_current.append((cs, rs - 1))
                    if r_end < ce:
                        next_current.append((r_end + 1, ce))
            current = _merge_intervals(next_current)
        result.extend(current)
    return _merge_intervals(result)


def _regions_to_file_intervals(
    regions: list[ReadRegion],
) -> dict[str, list[tuple[int, int]]]:
    """将 ReadRegion 列表按文件聚合为 文件 -> 合并后的区间列表。"""
    by_file: dict[str, list[tuple[int, int]]] = {}
    for path, start, end in regions:
        a, b = _to_concrete_interval(start, end)
        by_file.setdefault(path, []).append((a, b))
    return {f: _merge_intervals(ivals) for f, ivals in by_file.items()}


def _intersect_file_intervals_all(
    per_trajectory: list[dict[str, list[tuple[int, int]]]],
) -> dict[str, list[tuple[int, int]]]:
    """多条轨迹按文件求区间交集。某文件仅部分轨迹有时，该文件不在结果中。"""
    if not per_trajectory:
        return {}
    result = dict(per_trajectory[0])
    for traj in per_trajectory[1:]:
        for f in list(result):
            if f not in traj:
                del result[f]
            else:
                inter = _intersect_intervals(result[f], traj[f])
                if not inter:
                    del result[f]
                else:
                    result[f] = inter
    return result


def _intervals_to_region_list(
    file_intervals: dict[str, list[tuple[int, int]]],
) -> list[dict[str, Any]]:
    """将 文件 -> [(start,end)] 转为 [{"path": ..., "start": ..., "end": ...}, ...]。end 为 MAX_LINE 时输出 -1，便于评估时替换为真实结尾行数。"""
    out: list[dict[str, Any]] = []
    for path in sorted(file_intervals):
        for start, end in file_intervals[path]:
            out.append(
                {
                    "path": path,
                    "start": start,
                    "end": END_OF_FILE if end == MAX_LINE else end,
                }
            )
    return out


def _resolve_region_end_placeholders(
    regions: list[dict[str, Any]],
    repo_dir: Path | None,
) -> None:
    """In-place: 将 regions 中 end=-1 的项替换为 repo 内对应文件的真实末行行数。"""
    if not repo_dir:
        return
    for r in regions:
        if r.get("end") != END_OF_FILE:
            continue
        fp = repo_dir / r["path"]
        if not fp.is_file():
            continue
        try:
            n = sum(1 for _ in fp.open("r", encoding="utf-8", errors="ignore"))
        except (OSError, UnicodeDecodeError):
            # Skip binary files or files that can't be read
            continue
        r["end"] = n


def build_instance_ground_truth(
    instance_id: str,
    traj_paths: list[Path],
    repo_path_placeholder: str = "/testbed",
    repos_root: Path | None = None,
) -> dict[str, Any] | None:
    """Load trajectories for one instance and compute line-level core/optional ground truth.

    G_core = 各轨迹按文件的行区间求交集（所有成功轨迹都读到的行）
    optional_regions_map[model] = 该模型读到的行区间 减去 G_core
    当 repos_root 给定且本地存在对应仓库时，用磁盘文件校验路径存在性并解析 end=-1 为真实末行。
    """
    if not traj_paths:
        return None

    repo_name = instance_id.split("__")[0]
    # detected_repo_path 用于路径归一化，repo_path_placeholder 用于输出
    detected_repo_path: str | None = None
    repo_dir = _resolve_repo_dir(repos_root, instance_id) if repos_root else None
    if not repo_dir:
        logger.warning(
            f"Repo directory not found for instance {instance_id} in {repos_root}"
        )
        return None
    all_read_regions: list[tuple[str, list[ReadRegion]]] = []
    # 带步骤索引的读取区域: model -> [(norm_path, start, end, step_idx, traj_path)]
    all_read_regions_with_step: list[
        tuple[str, list[tuple[str, int, int, int, str]]]
    ] = []
    all_modified_sets: list[tuple[str, set[str]]] = []
    # 记录使用的轨迹路径
    used_traj_paths: list[Path] = []

    for traj_path in traj_paths:
        try:
            data = load_trajectory_data(traj_path)
        except Exception:
            continue

        used_traj_paths.append(traj_path)
        traj_path_str = str(traj_path)

        # 尝试获取实际 repo_path 用于归一化: 1) config.environment.cwd 2) 从轨迹路径检测
        if detected_repo_path is None:
            env = ((data.get("info") or {}).get("config") or {}).get(
                "environment"
            ) or {}
            if env.get("cwd"):
                detected_repo_path = env["cwd"]
            else:
                detected_repo_path = _detect_repo_path_from_trajectory(data)

        # 用于归一化的路径，优先使用检测到的，否则使用占位符
        norm_base = detected_repo_path or repo_path_placeholder

        raw_regions_with_step = extract_read_regions_with_step(data)
        normalized: list[ReadRegion] = []
        normalized_with_step: list[tuple[str, int, int, int, str]] = []
        for p, start, end, step_idx in raw_regions_with_step:
            norm_p = _normalize_path(p, norm_base)
            # Skip empty paths and temp files
            if not norm_p or _is_temp_file(norm_p):
                continue
            if if_path_exists_original(
                p,
                repo_name=repo_name,
                repo_dir=repo_dir,
                norm_path=norm_p if repo_dir else None,
            ):
                normalized.append((norm_p, start, end))
                # (norm_path, start, end, step_idx, traj_path)
                normalized_with_step.append(
                    (norm_p, start or 1, end, step_idx, traj_path_str)
                )
        model_name = (data.get("info") or {}).get(
            "model"
        ) or traj_path.parent.parent.name
        all_read_regions.append((model_name, normalized))
        all_read_regions_with_step.append((model_name, normalized_with_step))

        # 修改文件解析: 1) 从 submission diff 2) 从 str_replace_editor tool calls
        submission = (data.get("info") or {}).get("submission") or ""
        modified_from_diff = set(parse_modified_files_from_submission(submission))
        modified_from_tools = _extract_modified_files_from_tool_calls(
            data.get("messages", [])
        )
        # 合并、归一化路径、过滤临时文件
        all_modified = set()
        for p in modified_from_diff | modified_from_tools:
            norm_p = _normalize_path(p, norm_base)
            if norm_p and not _is_temp_file(norm_p):
                all_modified.add(norm_p)
        all_modified_sets.append((model_name, all_modified))
    if not all_read_regions:
        return None

    # 每条轨迹转为 文件 -> 合并后的行区间
    per_traj_intervals: list[dict[str, list[tuple[int, int]]]] = []
    for _, regions in all_read_regions:
        per_traj_intervals.append(_regions_to_file_intervals(regions))

    # G_core：按文件求所有轨迹的行区间交集
    read_core_intervals = _intersect_file_intervals_all(per_traj_intervals)
    read_core_regions = _intervals_to_region_list(read_core_intervals)
    read_core_files = sorted(read_core_intervals.keys())

    # 按 model 合并同一 model 的多条轨迹（取并集）
    union_regions_by_model: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for model_name, regions in all_read_regions:
        ivals = _regions_to_file_intervals(regions)
        if model_name not in union_regions_by_model:
            union_regions_by_model[model_name] = dict(ivals)
        else:
            for f, ivs in ivals.items():
                union_regions_by_model[model_name].setdefault(f, []).extend(ivs)
    for model_name in union_regions_by_model:
        union_regions_by_model[model_name] = {
            f: _merge_intervals(ivs)
            for f, ivs in union_regions_by_model[model_name].items()
        }

    # optional_regions_map[model] = 该 model 读到的行区间 减去 core
    read_optional_regions_map: dict[str, list[dict[str, Any]]] = {}
    read_optional_files_map: dict[str, list[str]] = {}
    for model_name, union_ivals in union_regions_by_model.items():
        optional_ivals: dict[str, list[tuple[int, int]]] = {}
        for f, ivs in union_ivals.items():
            core_ivs = read_core_intervals.get(f, [])
            remaining = _interval_set_minus(ivs, core_ivs)
            if remaining:
                optional_ivals[f] = remaining
        read_optional_regions_map[model_name] = _intervals_to_region_list(
            optional_ivals
        )
        read_optional_files_map[model_name] = sorted(optional_ivals.keys())

    union_modifies: dict[str, set[str]] = {}
    for model_name, modified_set in all_modified_sets:
        union_modifies.setdefault(model_name, set()).update(modified_set)
    intersection_modifies = set(all_modified_sets[0][1])
    for _, modified_set in all_modified_sets[1:]:
        intersection_modifies &= modified_set
    modified_core_files = sorted(intersection_modifies)
    main_files = sorted(set(read_core_files) & intersection_modifies)

    if repo_dir is not None:
        _resolve_region_end_placeholders(read_core_regions, repo_dir)
        for _model, regions in read_optional_regions_map.items():
            _resolve_region_end_placeholders(regions, repo_dir)

    # 构建 read_step_info: 每个文件对应的读取步骤信息
    # 格式: {path: [{traj_path, step_idx, start, end}, ...]}
    read_step_info: dict[str, list[dict[str, Any]]] = {}
    for _model, regions_with_step in all_read_regions_with_step:
        for norm_path, start, end, step_idx, traj_path_str in regions_with_step:
            if norm_path not in read_step_info:
                read_step_info[norm_path] = []
            read_step_info[norm_path].append(
                {
                    "traj_path": traj_path_str,
                    "step_idx": step_idx,
                    "start": start,
                    "end": end,
                }
            )

    # 仓库目录相对路径
    repo_dir_relative = str(repo_dir) if repo_dir else None

    return {
        "instance_id": instance_id,
        "repo_path": repo_path_placeholder,
        "repo_dir": repo_dir_relative,
        "ground_truth": {
            "read_core_files": read_core_files,
            "read_core_regions": read_core_regions,
            "read_optional_files_map": read_optional_files_map,
            "read_optional_regions_map": read_optional_regions_map,
            "modified_core_files": modified_core_files,
            "main_files": main_files,
        },
        "read_step_info": read_step_info,
        "meta": {
            "num_trajectories": len(all_read_regions),
            "num_read_core": len(read_core_files),
            "num_read_core_regions": len(read_core_regions),
            "num_modified_core": len(modified_core_files),
            "num_main": len(main_files),
        },
    }


@app.command()
def build(
    trajs_dir: Path = typer.Option(
        Path("unify_trajs"),
        "--trajs-dir",
        "-t",
        help="统一轨迹目录轨迹",
    ),
    output: Path = typer.Option(
        Path("bench.jsonl"),
        "--output",
        "-o",
        help="输出 JSONL 文件路径",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="子目录名正则，只扫描该匹配下的轨迹（如 'gemini' 匹配 gemini3、mini-gemini3）",
    ),
    instance_filter: str | None = typer.Option(
        None,
        "--instance-filter",
        "-F",
        help="instance_id 正则，只保留匹配的实例（如 'django__django-'）",
    ),
    repo_filter: str | None = typer.Option(
        None,
        "--repo",
        "-R",
        help="info.repo 正则，只保留轨迹中 repo 匹配的（如 'django/django'）",
    ),
    min_trajectories: int = typer.Option(
        1,
        "--min-trajectories",
        help="每个 instance 至少需要的成功轨迹数才纳入 bench",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="只统计将要处理的 instance 与轨迹数，不解析轨迹、不写文件",
    ),
    repo_path_placeholder: str = typer.Option(
        "/testbed",
        "--repo-path",
        help="bench 中 repo_path 的默认值（若轨迹内无 cwd）",
    ),
    repos: Path | None = typer.Option(
        Path("./repos"),
        "--repos",
        "-r",
        help="本地 repos 根目录；给定时会用磁盘文件校验路径存在性并将 end=-1 解析为真实末行",
    ),
):
    """从轨迹目录构建 SWE-Explore benchmark（Core/Optional 真值）。"""
    trajs_dir = trajs_dir.resolve()
    repos_root = repos.resolve() if repos else None
    if repos_root is not None and not repos_root.is_dir():
        console.print(f"[red]repos 目录不存在或不是目录: {repos_root}[/red]")
        raise typer.Exit(1)
    if not trajs_dir.is_dir():
        console.print(f"[red]轨迹目录不存在或不是目录: {trajs_dir}[/red]")
        raise typer.Exit(1)
    model_re = re.compile(model) if model else None
    instance_re = re.compile(instance_filter) if instance_filter else None
    repo_re = re.compile(repo_filter) if repo_filter else None
    groups = discover_trajectory_groups(trajs_dir, model_re, instance_re, repo_re)

    groups = {k: v for k, v in groups.items() if len(v) >= min_trajectories}

    if dry_run:
        total_trajs = sum(len(v) for v in groups.values())
        console.print(
            f"[dim]Dry run: [bold]{len(groups)}[/bold] instances, "
            f"[bold]{total_trajs}[/bold] trajectory files would be used."
        )
        if model:
            console.print(f"  Model (subdir regex): [cyan]{model}[/cyan]")
        if instance_filter:
            console.print(f"  Instance filter: [cyan]{instance_filter}[/cyan]")
        if repo_filter:
            console.print(f"  Repo filter: [cyan]{repo_filter}[/cyan]")
        console.print(f"  Output would be: [cyan]{output.resolve()}[/cyan]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building benchmark...", total=len(groups))
        written = 0
        with open(output, "w") as f:
            for instance_id, paths in tqdm(
                groups.items(), desc="Building ground truth..."
            ):
                record = build_instance_ground_truth(
                    instance_id,
                    paths,
                    repo_path_placeholder=repo_path_placeholder,
                    repos_root=repos_root,
                )
                if record:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                progress.advance(task)

    console.print(
        f"[green]Wrote [bold]{written}[/bold] instances to [bold]{output}[/bold][/green]"
    )


@app.command()
def list_models(
    trajs_dir: Path = typer.Argument(
        Path("unify_trajs"),
        help="统一轨迹根目录",
    ),
):
    """列出 trajs_dir 下所有模型子目录（用于 --model 正则）。"""
    trajs_dir = trajs_dir.resolve()
    if not trajs_dir.is_dir():
        console.print("[red]Not a directory.[/red]")
        raise typer.Exit(1)
    subdirs = [d.name for d in trajs_dir.iterdir() if d.is_dir()]
    for name in sorted(subdirs):
        console.print(name)


if __name__ == "__main__":
    app()
