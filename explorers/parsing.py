"""Shared output-parsing utilities for agentic explorers."""
from __future__ import annotations

import contextvars
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, List

from .base import ContextRegion, ExplorerResult

# File extensions we recognise in fallback regex
_SRC_EXTS = r"py|js|ts|java|go|rs|c|cpp|h|rb|php|md|txt|toml|yaml|yml|json|rst|cfg|ini|sh|ets"

# Known absolute prefixes that agents may return (e.g. /opt/swe-explore/data/repos/xxx/...)
_ABS_REPO_PATTERN = re.compile(
    r"^/(?:opt|home|root|tmp|workspace|testbed)/.+?/repos/[^/]+/"
)


def _normalize_path(path: str, repo_path: str = "") -> str:
    """Strip absolute repo prefixes to get a relative path.

    Handles patterns like:
      /opt/swe-explore/data/repos/text2num/text_to_num/parsers.py -> text_to_num/parsers.py
      /testbed/src/foo.py -> src/foo.py
      /workspace/repo/bar.py -> bar.py
      org__repo-N/file.py -> file.py  (repo_dir basename prefix)
    """
    path = path.strip()
    # Strip repo_dir basename prefix for relative paths (e.g. CoSIL output)
    if repo_path and not path.startswith("/"):
        import os
        repo_basename = os.path.basename(repo_path)
        if repo_basename and path.startswith(repo_basename + "/"):
            path = path[len(repo_basename) + 1:]
    if not path.startswith("/"):
        return path

    # Pattern 1: .../repos/<repo_name>/relative_path
    m = _ABS_REPO_PATTERN.match(path)
    if m:
        return path[m.end():]

    # Pattern 2: /testbed/relative_path
    if path.startswith("/testbed/"):
        return path[len("/testbed/"):]

    # Pattern 3: /workspace/<anything>/relative_path or /workspace/relative_path
    if path.startswith("/workspace/"):
        rest = path[len("/workspace/"):]
        # skip one directory level if it looks like a repo name
        parts = rest.split("/", 1)
        if len(parts) == 2 and not parts[0].endswith((".py", ".js", ".ts")):
            return parts[1]
        return rest

    # Fallback: strip everything up to and including the first directory
    # that looks like a repo root (contains common project files)
    # If nothing matches, return as-is (better than losing the path)
    return path


def parse_relevant_files(
    text: str,
    instance_id: str,
    *,
    top_k: int | None = None,
) -> List[ExplorerResult]:
    """Parse a RELEVANT_FILES block with optional ``path:start-end`` ranges.

    Falls back to a regex sweep for ``file:line-line`` patterns when the
    structured block is absent.
    """
    results: list[ExplorerResult] = []

    # 1) Try structured RELEVANT_FILES block
    match = re.search(r"RELEVANT_FILES:\s*\n((?:[-*] .+\n?)+)", text)
    if not match:
        match = re.search(r"RELEVANT_FILES:\s*\n((?:[^\n]+\n?)+)", text)

    if match:
        block = match.group(1)
        for line in block.strip().split("\n"):
            line = line.strip().lstrip("-* ").strip()
            if not line:
                continue
            if ":" in line and "-" in line.split(":")[-1]:
                path, range_str = line.rsplit(":", 1)
                parts = range_str.split("-")
                try:
                    start, end = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
            else:
                path = line.split(":")[0]
                if not path or ("/" not in path and "." not in path):
                    continue
                start, end = 1, -1

            path = _normalize_path(path)
            results.append(ExplorerResult(
                instance_id=instance_id,
                score=1.0,
                regions=[ContextRegion(path=path, start=start, end=end)],
            ))
        if results:
            return results[:top_k] if top_k else results

    # 2) Fallback: regex for file:line-line patterns
    pattern = rf"[\w/.-]+\.(?:{_SRC_EXTS}):\d+-\d+"
    for m in re.finditer(pattern, text):
        parts = m.group().rsplit(":", 1)
        path = _normalize_path(parts[0])
        start_s, end_s = parts[1].split("-")
        results.append(ExplorerResult(
            instance_id=instance_id,
            score=1.0,
            regions=[ContextRegion(path=path, start=int(start_s), end=int(end_s))],
        ))

    return results[:top_k] if top_k else results


def parse_file_paths(
    text: str,
    instance_id: str,
    *,
    top_k: int | None = None,
) -> List[ExplorerResult]:
    """Parse a RELEVANT_FILES block that lists plain file paths (no ranges).

    Useful for explorers that report full files rather than line ranges.
    """
    results: list[ExplorerResult] = []

    match = re.search(r"RELEVANT_FILES:\s*\n((?:[-*] .+\n?)+)", text)
    if not match:
        match = re.search(r"RELEVANT_FILES:\s*\n((?:[^\n]+\n?)+)", text)

    if match:
        block = match.group(1)
        for line in block.strip().split("\n"):
            line = line.strip().lstrip("-* ").strip()
            if not line:
                continue
            path = line.split(":")[0].strip()
            if not path or ("/" not in path and "." not in path):
                continue
            path = _normalize_path(path)
            results.append(ExplorerResult(
                instance_id=instance_id,
                score=1.0,
                regions=[ContextRegion(path=path, start=1, end=-1)],
            ))

    # Fallback: any path-like tokens
    if not results:
        pattern = rf"[\w/.-]+\.(?:{_SRC_EXTS})"
        for m in re.finditer(pattern, text):
            path = _normalize_path(m.group())
            results.append(ExplorerResult(
                instance_id=instance_id,
                score=1.0,
                regions=[ContextRegion(path=path, start=1, end=-1)],
            ))

    return results[:top_k] if top_k else results


# ── AST-based entity-to-line resolution ─────────────────────────────────


def resolve_entity_lines(
    repo_path: str, file_path: str, entity_name: str,
) -> tuple[int, int] | None:
    """Resolve a function/class name to (start_line, end_line) via AST.

    *entity_name* can be:
      - ``"func_name"`` — top-level function
      - ``"ClassName"`` — class
      - ``"ClassName.method_name"`` — method inside a class

    Returns ``None`` when the entity cannot be found.
    """
    import ast
    from pathlib import Path

    full = Path(repo_path) / file_path
    if not full.is_file():
        return None
    try:
        source = full.read_text(errors="ignore")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return None

    parts = entity_name.split(".", 1)

    for node in ast.walk(tree):
        if len(parts) == 1:
            # top-level function or class
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == parts[0]:
                    return (node.lineno, node.end_lineno or node.lineno)
        else:
            # ClassName.method_name
            class_name, method_name = parts
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in ast.walk(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            return (item.lineno, item.end_lineno or item.lineno)

    return None


# ── Model-specific output parsers ──────────────────────────────────────


def parse_locagent_jsonl(
    jsonl_path: str,
    instance_id: str,
    repo_path: str,
) -> list[ExplorerResult]:
    """Parse LocAgent merged JSONL output into ExplorerResult list.

    LocAgent ``found_entities`` uses format ``file.py:ClassName.method``
    or ``file.py:func``.  We resolve each to line ranges via AST.
    """
    import json
    from pathlib import Path

    p = Path(jsonl_path)
    if not p.is_file():
        return []

    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("instance_id") != instance_id:
            continue

        regions: list[ContextRegion] = []
        # Prefer found_entities → found_files fallback
        entities = rec.get("found_entities", [])
        # found_entities is list of lists; flatten
        if entities and isinstance(entities[0], list):
            entities = [e for sub in entities for e in sub]

        for entity_str in entities:
            if ":" not in entity_str:
                continue
            fpath, ename = entity_str.split(":", 1)
            fpath = _normalize_path(fpath)
            rng = resolve_entity_lines(repo_path, fpath, ename)
            if rng:
                regions.append(ContextRegion(path=fpath, start=rng[0], end=rng[1]))
            else:
                regions.append(ContextRegion(path=fpath, start=1, end=-1))

        # Fallback A: parse raw_output_loc lines like "file.py:QualifiedName"
        # directly. LocAgent's own parser only recognises "function:" /
        # "class:" prefixed lines; reasoning models often emit a flatter
        # `path:QualifiedName` list which yields empty found_entities.
        if not regions:
            raw_outputs = rec.get("raw_output_loc", []) or []
            seen: set[tuple[str, str]] = set()
            for raw in raw_outputs:
                if not isinstance(raw, str):
                    continue
                for raw_line in raw.splitlines():
                    s = raw_line.strip().strip("`").strip()
                    if not s or s.startswith("#"):
                        continue
                    if ":" not in s:
                        continue
                    fpath, ename = s.split(":", 1)
                    fpath = fpath.strip()
                    ename = ename.strip()
                    # Multilingual: accept any source-like extension
                    if "." not in fpath:
                        continue
                    _ext = fpath.rsplit(".", 1)[1].lower()
                    if _ext not in {
                        "py","go","java","js","ts","tsx","jsx","rs","rb","php",
                        "c","h","cc","cpp","cxx","hpp","hh","hxx","scala","kt",
                        "swift","cs","lua","dart","ex","exs","erl","clj","m",
                        "mm","proto","sh","bash","yml","yaml","sql"
                    }:
                        continue
                    if not ename or any(c.isspace() for c in ename):
                        continue
                    fpath_n = _normalize_path(fpath)
                    key = (fpath_n, ename)
                    if key in seen:
                        continue
                    seen.add(key)
                    rng = resolve_entity_lines(repo_path, fpath_n, ename)
                    if rng:
                        regions.append(ContextRegion(
                            path=fpath_n, start=rng[0], end=rng[1]))
                    else:
                        regions.append(ContextRegion(
                            path=fpath_n, start=1, end=-1))

        # Fallback B: found_files
        if not regions:
            files = rec.get("found_files", [])
            if files and isinstance(files[0], list):
                files = [f for sub in files for f in sub]
            for fp in files:
                fp = _normalize_path(fp)
                regions.append(ContextRegion(path=fp, start=1, end=-1))

        if regions:
            return [ExplorerResult(instance_id=instance_id, score=1.0, regions=regions)]
    return []


def parse_orcaloca_output(
    output_json_path: str,
    instance_id: str,
    repo_path: str,
) -> list[ExplorerResult]:
    """Parse OrcaLoca process_output.py JSON → ExplorerResult.

    Expected structure::

        { "instance_id": { "bug_locations": [
            {"file_path": "...", "class_name": "...", "method_name": "...",
             "line_range": "[start, end]"}
        ]}}
    """
    import json
    from pathlib import Path

    p = Path(output_json_path)
    if not p.is_file():
        return []

    data = json.loads(p.read_text())
    entry = data.get(instance_id)
    if not entry:
        return []

    regions: list[ContextRegion] = []
    for loc in entry.get("bug_locations", []):
        fpath = _normalize_path(loc.get("file_path", ""))
        if not fpath:
            continue
        lr = loc.get("line_range", "")
        start, end = 1, -1
        if lr:
            try:
                import ast as _ast
                rng = _ast.literal_eval(lr)
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    start, end = int(rng[0]), int(rng[1])
            except Exception:
                # Fallback: resolve via class/method name
                entity = ""
                cn = loc.get("class_name", "")
                mn = loc.get("method_name", "")
                if cn and mn:
                    entity = f"{cn}.{mn}"
                elif mn:
                    entity = mn
                elif cn:
                    entity = cn
                if entity:
                    rng = resolve_entity_lines(repo_path, fpath, entity)
                    if rng:
                        start, end = rng
        regions.append(ContextRegion(path=fpath, start=start, end=end))

    if regions:
        return [ExplorerResult(instance_id=instance_id, score=1.0, regions=regions)]
    return []


def parse_cosil_jsonl(
    func_jsonl_path: str,
    instance_id: str,
    repo_path: str,
) -> list[ExplorerResult]:
    """Parse CoSIL func-level JSONL → ExplorerResult.

    ``found_related_locs`` maps ``file.py`` →
    ``["ClassName.method", "func", ...]``.
    """
    import json
    from pathlib import Path

    p = Path(func_jsonl_path)
    if not p.is_file():
        return []

    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("instance_id") != instance_id:
            continue

        regions: list[ContextRegion] = []
        related = rec.get("found_related_locs", {})
        if isinstance(related, dict):
            for fpath, entities in related.items():
                fpath = _normalize_path(fpath, repo_path)
                for ename in (entities or []):
                    rng = resolve_entity_lines(repo_path, fpath, ename)
                    if rng:
                        regions.append(ContextRegion(path=fpath, start=rng[0], end=rng[1]))
                    else:
                        regions.append(ContextRegion(path=fpath, start=1, end=-1))

        # Fallback: file-level
        if not regions:
            for fp in rec.get("found_files", []):
                fp = _normalize_path(fp, repo_path)
                regions.append(ContextRegion(path=fp, start=1, end=-1))

        if regions:
            return [ExplorerResult(instance_id=instance_id, score=1.0, regions=regions)]
    return []


def parse_acr_bug_locations(
    json_path: str,
    instance_id: str,
) -> list[ExplorerResult]:
    """Parse AutoCodeRover ``bug_locations_after_process.json``.

    Each entry has ``rel_file_path``, ``start``, ``end`` (1-based).
    """
    import json
    from pathlib import Path

    p = Path(json_path)
    if not p.is_file():
        return []

    data = json.loads(p.read_text())
    if not isinstance(data, list):
        return []

    regions: list[ContextRegion] = []
    for loc in data:
        fpath = _normalize_path(loc.get("rel_file_path", ""))
        if not fpath:
            continue
        start = loc.get("start") or 1
        end = loc.get("end") or -1
        regions.append(ContextRegion(path=fpath, start=start, end=end))

    if regions:
        return [ExplorerResult(instance_id=instance_id, score=1.0, regions=regions)]
    return []


# ── Token usage extraction ─────────────────────────────────────────────
#
# Each explorer that touches an LLM parses its provider-specific output for
# usage fields and reports a TokenUsage into the active per-case collector
# (see ``usage_collector`` / ``report_usage``).  The runner sums usage per
# case, writes it into the JSONL rows and prints totals at the end.


@dataclass
class TokenUsage:
    """Token consumption for one exploration case.

    Categories are non-overlapping where the provider allows it:
    ``input`` excludes cached reads (for OpenAI-style ``prompt_tokens``
    the cached part is subtracted), ``reasoning`` is the subset of
    ``output`` reported separately by the provider.

    ``reasoning_separate`` marks providers that report reasoning tokens on
    top of ``output_tokens`` — e.g. Gemini's ``thoughts_token_count``
    alongside ``candidates_token_count``, or agent CLIs such as OpenCode
    whose ``tokens`` object carries ``output`` and ``reasoning`` as sibling
    buckets (empirically the case for the GLM/Zai line, where reasoning
    exceeds output, so it cannot be a subset). For OpenAI/Anthropic-style
    schemas reasoning is already inside ``output`` and must not be added
    a second time (see :attr:`total`).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_separate: bool = False

    _CATEGORIES = (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    )

    def add(self, other: "TokenUsage | None") -> None:
        if not other:
            return
        for cat in self._CATEGORIES:
            setattr(self, cat, getattr(self, cat) + getattr(other, cat))
        self.reasoning_separate = self.reasoning_separate or other.reasoning_separate

    def has_any(self) -> bool:
        return any(getattr(self, cat) for cat in self._CATEGORIES)

    @property
    def total(self) -> int:
        """input + output (+ reasoning only when reported separately)."""
        extra = self.reasoning_tokens if self.reasoning_separate else 0
        return self.input_tokens + self.output_tokens + extra

    def to_dict(self) -> dict[str, int]:
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "cache_read": self.cache_read_tokens,
            "cache_write": self.cache_write_tokens,
            "reasoning": self.reasoning_tokens,
            "total": self.total,
            "reasoning_separate": self.reasoning_separate,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "TokenUsage | None":
        if not isinstance(data, dict):
            return None
        return cls(
            input_tokens=int(data.get("input") or 0),
            output_tokens=int(data.get("output") or 0),
            cache_read_tokens=int(data.get("cache_read") or 0),
            cache_write_tokens=int(data.get("cache_write") or 0),
            reasoning_tokens=int(data.get("reasoning") or 0),
            reasoning_separate=bool(data.get("reasoning_separate")),
        )


# Normalised (lowercase, non-letters stripped) key aliases per category.
_INPUT_KEYS = {"inputtokens", "prompttokens", "input", "prompttokencount"}
# Input keys whose value already contains the cached-token portion.
_CACHE_INCLUSIVE_INPUT_KEYS = {"prompttokens", "prompttokencount"}
_OUTPUT_KEYS = {"outputtokens", "completiontokens", "output"}
_SEPARATE_OUTPUT_KEYS = {
    # Gemini-style: candidates + thoughts, reasoning not inside output
    "candidatestokencount",
    "candidates",
}
_CACHE_READ_KEYS = {
    "cachereadinputtokens",
    "cachereadtokens",
    "cacheread",
    "cachedtokens",
    "cachedinputtokens",
    "cachedcontenttokencount",
}
_CACHE_WRITE_KEYS = {
    "cachecreationinputtokens",
    "cachecreationtokens",
    "cachewritetokens",
    "cachewrite",
}
_REASONING_KEYS = {"reasoningtokens", "reasoning", "thoughtstokencount", "thoughts"}
_MAX_SCAN_DEPTH = 12


def _norm_key(key: Any) -> str:
    return re.sub(r"[^a-z]", "", str(key).lower())


def _scan_usage(obj: Any, usage: TokenUsage, state: dict, depth: int) -> None:
    if depth > _MAX_SCAN_DEPTH:
        return
    if isinstance(obj, dict):
        # A reasoning field directly inside the same usage object as an
        # output field is a sibling bucket (opencode ``tokens``, Gemini
        # ``usageMetadata``), not a subset of ``output``. Nested details
        # (OpenAI ``completion_tokens_details.reasoning_tokens``) keep the
        # inclusive OpenAI/Anthropic semantics.
        normed_keys = {_norm_key(k) for k in obj}
        sibling_reasoning = bool(normed_keys & _REASONING_KEYS)
        for key, value in obj.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                nk = _norm_key(key)
                if nk in _INPUT_KEYS:
                    usage.input_tokens += int(value)
                    if nk in _CACHE_INCLUSIVE_INPUT_KEYS:
                        state["prompt_includes_cache"] = True
                elif nk in _OUTPUT_KEYS or nk in _SEPARATE_OUTPUT_KEYS:
                    usage.output_tokens += int(value)
                    if nk in _SEPARATE_OUTPUT_KEYS or sibling_reasoning:
                        state["output_inclusive"] = False
                    elif not state["output_seen"]:
                        state["output_inclusive"] = True
                    state["output_seen"] = True
                elif nk in _CACHE_READ_KEYS:
                    usage.cache_read_tokens += int(value)
                elif nk in _CACHE_WRITE_KEYS:
                    usage.cache_write_tokens += int(value)
                elif nk in _REASONING_KEYS:
                    usage.reasoning_tokens += int(value)
            elif isinstance(value, dict):
                nk = _norm_key(key)
                if nk == "cache":
                    # opencode-style nested counts: "cache": {"read": N, "write": N}
                    cache_read = value.get("read")
                    cache_write = value.get("write")
                    if isinstance(cache_read, (int, float)) and not isinstance(cache_read, bool):
                        usage.cache_read_tokens += int(cache_read)
                    if isinstance(cache_write, (int, float)) and not isinstance(cache_write, bool):
                        usage.cache_write_tokens += int(cache_write)
                else:
                    _scan_usage(value, usage, state, depth + 1)
            elif isinstance(value, list):
                _scan_usage(value, usage, state, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _scan_usage(item, usage, state, depth + 1)


def extract_usage(obj: Any) -> TokenUsage:
    """Recursively scan parsed JSON for token-usage fields.

    Recognises common container names (``usage``, ``token_usage``,
    ``total_usage``, ``llm_usage``, ``model_usage``, ``usage_metadata``, ...)
    plus the field aliases used by Anthropic, OpenAI and LiteLLM:
    ``input_tokens``/``prompt_tokens``, ``output_tokens``/
    ``completion_tokens``, ``cache_read_*``/``cached_tokens``,
    ``cache_creation_*`` and ``reasoning_tokens``.  Only numeric values are
    collected, so text fields such as a reasoning transcript are ignored.

    Sets ``reasoning_separate`` when the reasoning tokens are reported on
    top of ``output`` — Gemini-style candidates/thoughts, opencode-style
    ``tokens`` objects where ``output`` and ``reasoning`` are siblings, or
    reasoning without any output field — so that :attr:`TokenUsage.total`
    does not double-count them.
    """
    usage = TokenUsage()
    state = {
        "prompt_includes_cache": False,
        "output_inclusive": False,
        "output_seen": False,
    }
    _scan_usage(obj, usage, state, 0)
    if state["prompt_includes_cache"] and usage.cache_read_tokens:
        usage.input_tokens = max(0, usage.input_tokens - usage.cache_read_tokens)
    if usage.reasoning_tokens and not state["output_inclusive"]:
        usage.reasoning_separate = True
    return usage


def extract_usage_from_jsonl(raw: str) -> TokenUsage | None:
    """Scan newline-delimited JSON events (agent CLI stdout) for usage."""
    usage = TokenUsage()
    found = False
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or not line.startswith(("{", "[")):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = extract_usage(event)
        if part.has_any():
            usage.add(part)
            found = True
    return usage if found else None


_usage_collector_var: contextvars.ContextVar[TokenUsage | None] = (
    contextvars.ContextVar("swe_explore_token_usage", default=None)
)


def report_usage(usage: TokenUsage | None) -> None:
    """Accumulate *usage* into the active per-case collector (no-op if none)."""
    if usage is None or not usage.has_any():
        return
    active = _usage_collector_var.get()
    if active is not None:
        active.add(usage)


@contextmanager
def usage_collector() -> Iterator[TokenUsage]:
    """Collect all ``report_usage`` calls within the block (thread-safe:
    each thread gets its own collector via contextvars)."""
    tracker = TokenUsage()
    token = _usage_collector_var.set(tracker)
    try:
        yield tracker
    finally:
        _usage_collector_var.reset(token)


def register_litellm_usage_callback() -> None:
    """Report in-process litellm completions (mini-swe-agent, LocAgent, ...)
    into the active collector.  No-op when litellm is not installed."""
    try:
        import litellm
    except Exception:
        return
    if getattr(litellm, "_swe_explore_usage_hook", False):
        return

    def _hook(kwargs, response, start_time, end_time, user_id=None):
        resp_usage = getattr(response, "usage", None)
        if resp_usage is None:
            return
        if hasattr(resp_usage, "model_dump"):
            try:
                resp_usage = resp_usage.model_dump()
            except Exception:
                return
        if isinstance(resp_usage, dict):
            report_usage(extract_usage(resp_usage))

    litellm.success_callback.append(_hook)
    litellm._swe_explore_usage_hook = True
