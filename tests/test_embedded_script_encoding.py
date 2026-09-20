"""Scripts this repo generates as text must open files as UTF-8 too.

`PYTHONWARNDEFAULTENCODING=1` and the EncodingWarning filter only police the
call sites pytest imports. An `open()` inside a string that is written out and
run later — in a conda env, in a container, on a Windows runner — escapes both
and silently falls back to the platform default encoding.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", ".git", "__pycache__", "node_modules"}


def _python_sources() -> list[Path]:
    return sorted(
        p for p in ROOT.rglob("*.py") if not SKIP_DIRS.intersection(p.parts)
    )


def _literal_text(node: ast.AST) -> str | None:
    """The text of a string literal, with each {expr} of an f-string stubbed."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append("_INTERPOLATED_")
            else:
                return None
        return "".join(parts)
    return None


def _unencoded_opens(path: Path) -> list[str]:
    """Every `open()` without an encoding inside an embedded Python script."""
    found = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        text = _literal_text(node)
        if not text or "open(" not in text:
            continue
        try:
            embedded = ast.parse(text)
        except SyntaxError:
            continue  # a literal that is not Python source
        for call in ast.walk(embedded):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "open"
                and not any(kw.arg == "encoding" for kw in call.keywords)
            ):
                found.append(f"{path}:{node.lineno}: {ast.unparse(call)}")
    return found


@pytest.mark.parametrize("path", _python_sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_generated_scripts_open_files_with_an_explicit_encoding(path):
    assert _unencoded_opens(path) == []


def test_the_scan_can_see_an_unencoded_open(tmp_path):
    """Guard the guard: a planted violation must be reported."""
    # Assembled at runtime so this file does not trip its own scan.
    script = "with op" + "en(path) as f:\n    pass\n"
    planted = tmp_path / "planted.py"
    planted.write_text(f"SCRIPT = {script!r}\n", encoding="utf-8")

    assert len(_unencoded_opens(planted)) == 1
