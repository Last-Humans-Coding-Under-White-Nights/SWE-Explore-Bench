"""Shared output-parsing utilities for agentic explorers."""
from __future__ import annotations

import re
from typing import List

from .base import ContextRegion, ExplorerResult

# File extensions we recognise in fallback regex
_SRC_EXTS = r"py|js|ts|java|go|rs|c|cpp|h|rb|php|md|txt|toml|yaml|yml|json|rst|cfg|ini|sh"

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
