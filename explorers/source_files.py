"""Source discovery policy shared by retrieval explorers."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

DEFAULT_EXTENSIONS = frozenset({
    ".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".rst",
    ".ets", ".ts", ".json5",
})
DEFAULT_EXCLUDED_DIRS = frozenset({
    ".git", "node_modules", "oh_modules", "build", ".hvigor", "dist",
    ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
})


def iter_source_files(
    repo_root: Path | str,
    *,
    extensions: Iterable[str] | str = DEFAULT_EXTENSIONS,
    excluded_dirs: Iterable[str] | str = DEFAULT_EXCLUDED_DIRS,
) -> Iterator[Path]:
    """Yield sources in lexicographic repository-relative POSIX path order.

    Extensions and excluded directory names are case-insensitive; either
    argument accepts a single string or an iterable of strings. Extensions
    accept an optional leading dot. Base suffixes include declarations:
    ``.ets`` matches ``.d.ets``, and ``.ts`` matches ``.d.ts``. Compound
    suffixes can also be selected explicitly. Trailing forward slashes or
    backslashes on excluded directory names are ignored.

    Overrides replace the defaults; empty collections disable matching or
    directory exclusions, respectively. Excluded directory names are pruned
    at every depth, before traversal. Directory symlinks are not followed;
    valid file symlinks are included and broken symlinks are skipped.
    """
    repo_root = Path(repo_root)
    if isinstance(extensions, str):
        extensions = (extensions,)
    if isinstance(excluded_dirs, str):
        excluded_dirs = (excluded_dirs,)
    suffixes = tuple(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
    )
    excluded = {name.rstrip("/\\").lower() for name in excluded_dirs}
    paths: list[Path] = []
    for directory, dirs, files in repo_root.walk():
        dirs[:] = [name for name in dirs if name.lower() not in excluded]
        for name in files:
            if not name.lower().endswith(suffixes):
                continue
            path = directory / name
            if path.is_file():
                paths.append(path)
    yield from sorted(paths, key=lambda p: p.relative_to(repo_root).as_posix())
