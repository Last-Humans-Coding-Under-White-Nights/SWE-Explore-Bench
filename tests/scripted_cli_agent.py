"""A deterministic stand-in for ``opencode run`` / ``deveco run``.

Used by ``test_cli_agent_arkts_e2e.py`` so CI can drive the real explorer
subprocess path without a model or network. It honours the same contract as
the CLIs it replaces:

* ``run --format json --dir <repo>`` with the prompt on stdin, tolerating the
  approval flags each explorer adds (``--auto``,
  ``--dangerously-skip-permissions``).
* ``$SCRIPTED_AGENT_CLI`` names the CLI being emulated (``opencode`` or
  ``deveco``) and the profile is loaded from that CLI's config-dir variable
  only, so a stray variable for the other CLI cannot hijack a run. Every
  navigation step checks the profile's ``permission`` block first and aborts
  if the tool is not allowed, and it refuses profiles that allow edits.
* Output is OpenCode's ``--format json`` event stream: ``step_start``,
  ``tool_use`` with ``part.tool`` / ``part.state.input``, ``text`` carrying
  the answer, and ``step_finish`` with token counts.

Exploration is scripted, not learned: start from the ArkUI pages, follow each
``import ... from '<relative>'`` to the module it names, score lines by the
identifiers in the issue text, and expand the best line to its enclosing
brace block. That is enough to show that the pipeline finds ``.ets`` files,
follows imports, lands inside a decorated component, and reports exact lines.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explorers.deveco import DevEcoExplorer  # noqa: E402
from explorers.opencode import OpenCodeExplorer  # noqa: E402

CLI_VAR = "SCRIPTED_AGENT_CLI"
#: cli -> (config-dir variable, config filename), from the explorer itself.
CONFIG_VARS = {
    "opencode": (OpenCodeExplorer.config_env_var, OpenCodeExplorer.config_filename),
    "deveco": (DevEcoExplorer.config_env_var, DevEcoExplorer.config_filename),
}
ACCEPTED_FLAGS = {"--auto", "--dangerously-skip-permissions"}
ISSUE_HEADER = "ISSUE DESCRIPTION FROM USER (very important):"
ISSUE_TRAILER = "Do exactly this, but without modifications."
SKIPPED_DIRS = {"oh_modules", "build", ".serena"}

IMPORT_RE = re.compile(r"""^\s*import\b.*?\bfrom\s+['"](\.{1,2}/[^'"]+)['"]""")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
#: A method header: lower-case callee (ArkUI builders such as ``Column() {``
#: are capitalised), not a control-flow keyword, ending in an opening brace.
DEFINITION_RE = re.compile(
    r"^\s*(?!(?:if|for|while|switch|catch|with)\b)[a-z_][A-Za-z0-9_]*"
    r"\s*\(.*\)\s*(?::\s*\w+\s*)?\{\s*$"
)

Location = tuple[Path, int]  # file and 0-based line index


def emit(event_type: str, part: dict) -> None:
    event = {"type": event_type, "timestamp": int(time.time() * 1000),
             "sessionID": "ses_scripted", "part": part}
    sys.stdout.write(json.dumps(event) + "\n")


_calls = 0


def emit_tool_use(tool: str, arguments: dict, output: str = "") -> None:
    global _calls
    _calls += 1
    state = {"status": "completed", "input": arguments, "output": output}
    emit("tool_use", {"type": "tool", "tool": tool, "callID": f"call_{_calls:03d}",
                      "messageID": "msg_scripted", "state": state})


def fail(message: str, code: int) -> NoReturn:
    emit("error", {"error": {"name": "ScriptedAgentError", "data": {"message": message}}})
    sys.stderr.write(f"scripted agent: {message}\n")
    sys.exit(code)


def parse_args(argv: list[str]) -> Path:
    if not argv or argv[0] != "run":
        fail(f"expected `run`, got {argv!r}", 2)
    options: dict[str, str] = {}
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ACCEPTED_FLAGS:
            i += 1
        elif arg in ("--format", "--dir") and i + 1 < len(argv):
            options[arg] = argv[i + 1]
            i += 2
        else:
            fail(f"unsupported argument {arg!r}", 2)
    if options.get("--format") != "json":
        fail("--format json is required", 2)
    if "--dir" not in options:
        fail("--dir is required", 2)
    return Path(options["--dir"])


def read_issue(prompt: str) -> str:
    if "RELEVANT_FILES:" not in prompt or ISSUE_HEADER not in prompt:
        fail("prompt does not follow the explorer contract", 2)
    return prompt.split(ISSUE_HEADER, 1)[1].split(ISSUE_TRAILER, 1)[0]


def load_profile() -> dict:
    cli = os.environ.get(CLI_VAR)
    if cli not in CONFIG_VARS:
        fail(f"{CLI_VAR} must be one of {sorted(CONFIG_VARS)}, got {cli!r}", 2)
    var, filename = CONFIG_VARS[cli]
    config_dir = os.environ.get(var)
    if not config_dir:
        fail(f"{var} is not exported; refusing to run on an implicit profile", 2)
    path = Path(config_dir) / filename
    if not path.is_file():
        fail(f"{var} points at {config_dir} but {filename} is missing", 2)
    return json.loads(path.read_text(encoding="utf-8"))


class Tools:
    """Read-only navigation gated by the profile's permission block."""

    def __init__(self, repo: Path, profile: dict) -> None:
        self.repo = repo
        self.permission = profile.get("permission", {})
        if self.permission.get("edit") != "deny":
            fail("profile is not read-only: permission.edit must be deny", 3)
        self.semantic = bool(profile.get("mcp", {}).get("serena", {}).get("enabled"))

    def _check(self, tool: str) -> None:
        if self.permission.get(tool) != "allow":
            fail(f"tool {tool!r} is not allowed by the profile", 3)

    def rel(self, path: Path) -> str:
        return path.relative_to(self.repo).as_posix()

    def list_ets(self) -> list[Path]:
        self._check("list")
        files = sorted(p for p in self.repo.rglob("*.ets")
                       if not SKIPPED_DIRS & set(p.relative_to(self.repo).parts))
        emit_tool_use("list", {"path": "."}, "\n".join(map(self.rel, files)))
        return files

    def read(self, path: Path) -> list[str]:
        self._check("read")
        lines = path.read_text(encoding="utf-8").splitlines()
        emit_tool_use("read", {"filePath": self.rel(path)},
                      f"<content>{len(lines)} lines</content>")
        return lines

    def grep(self, pattern: str) -> None:
        self._check("grep")
        emit_tool_use("grep", {"pattern": pattern})

    def find_symbol(self, name: str) -> None:
        if self.semantic:
            emit_tool_use("serena_find_symbol", {"name_path": name})


def resolve_import(source: Path, spec: str) -> Path | None:
    target = (source.parent / spec).resolve()
    for candidate in (target.with_suffix(".ets"), target.with_name(target.name + ".d.ets")):
        if candidate.is_file():
            return candidate
    return None


def explore_from_pages(tools: Tools, pages: list[Path]) -> dict[Path, list[str]]:
    """Read pages first, then whatever they import, transitively."""
    visited: dict[Path, list[str]] = {}
    queue = deque(pages)
    while queue:
        path = queue.popleft()
        if path in visited:
            continue
        visited[path] = tools.read(path)
        for line in visited[path]:
            match = IMPORT_RE.match(line)
            target = resolve_import(path, match.group(1)) if match else None
            if target is not None and target not in visited:
                queue.append(target)
    return visited


def score_line(line: str, identifiers: set[str], words: set[str]) -> int:
    if IMPORT_RE.match(line):
        return 0  # an import names the symbol but never contains the bug
    score = 0
    for token in set(IDENT_RE.findall(line)):
        if token in identifiers:
            score += 3
        elif token.lower() in words:
            score += 1
    if DEFINITION_RE.match(line):
        score += 2  # a method definition beats its call sites
    return score


def query_identifiers(issue: str) -> set[str]:
    """Code-like tokens in the issue text: anything with an inner capital."""
    return {t for t in IDENT_RE.findall(issue) if any(c.isupper() for c in t[1:])}


def best_location(visited: dict[Path, list[str]], issue: str) -> Location | None:
    """The line most like the issue text.

    Ties go to the deepest-nested line: a statement inside a handler is more
    specific than the declaration of the same field.
    """
    identifiers = query_identifiers(issue)
    words = {t.lower() for t in IDENT_RE.findall(issue)}
    best: tuple[tuple[int, int], Location] | None = None
    for path, lines in visited.items():
        for index, line in enumerate(lines):
            score = score_line(line, identifiers, words)
            depth = len(line) - len(line.lstrip())
            if score and (best is None or (score, depth) > best[0]):
                best = ((score, depth), (path, index))
    return best[1] if best else None


def enclosing_block(lines: list[str], index: int) -> tuple[int, int]:
    """1-based inclusive range of the brace block that contains ``index``.

    Scanning backwards, every ``}`` opens a sibling block that must be
    skipped; the first ``{`` left unmatched is the enclosing block's opener.
    A line with no enclosing block is returned on its own.
    """
    start, pending = index, 0
    while start >= 0:
        opened = lines[start].count("{") - lines[start].count("}")
        if opened > pending:
            break
        pending = max(0, pending - opened)
        start -= 1
    if start < 0:
        return index + 1, index + 1
    depth = 0
    for end in range(start, len(lines)):
        depth += lines[end].count("{") - lines[end].count("}")
        if depth <= 0:
            return start + 1, end + 1
    return start + 1, len(lines)


def nominal_usage(visited: dict[Path, list[str]]) -> dict[str, int]:
    """Token counts in OpenCode's step_finish shape, sized by lines read, so
    the explorer's usage collector has something real-looking to sum."""
    tokens = {"input": 12 * sum(map(len, visited.values())), "output": 24, "reasoning": 0}
    tokens["total"] = tokens["input"] + tokens["output"]
    return tokens


def main() -> None:
    repo = parse_args(sys.argv[1:])
    # The explorer sends UTF-8; the locale codec on Windows is not.
    sys.stdin.reconfigure(encoding="utf-8")
    issue = read_issue(sys.stdin.read())
    tools = Tools(repo, load_profile())

    emit("step_start", {"type": "step-start"})
    pages = [p for p in tools.list_ets() if "pages" in p.parts]
    visited = explore_from_pages(tools, pages)
    for name in sorted(query_identifiers(issue)):
        tools.grep(name)
        tools.find_symbol(name)

    location = best_location(visited, issue)
    if location is None:
        answer = "I could not find a relevant region."
    else:
        path, index = location
        start, end = enclosing_block(visited[path], index)
        answer = f"RELEVANT_FILES:\n- {tools.rel(path)}:{start}-{end}"
    emit("text", {"type": "text", "text": answer})
    emit("step_finish", {"type": "step-finish", "reason": "stop",
                         "tokens": nominal_usage(visited), "cost": 0})


if __name__ == "__main__":
    main()
