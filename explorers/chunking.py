"""Shared chunking utilities for retrieval explorers."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from .source_files import iter_source_files

@dataclass
class Chunk:
    """A contiguous block of lines from a source file."""

    path: str  # relative path within repo
    start: int  # 1-based start line
    end: int  # 1-based end line (inclusive)
    content: str


def _iter_chunks(
    repo_root: Path, *, chunk_size: int, chunk_overlap: int
) -> Iterator[Chunk]:
    """Yield the overlapping line windows of every source file, unbounded."""
    step = max(1, chunk_size - chunk_overlap)

    for p in iter_source_files(repo_root):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        if not lines:
            continue
        rel = p.relative_to(repo_root).as_posix()
        for i in range(0, len(lines), step):
            end_idx = min(i + chunk_size, len(lines))
            content = "\n".join(lines[i:end_idx])
            if content.strip():
                yield Chunk(path=rel, start=i + 1, end=end_idx, content=content)
            if end_idx >= len(lines):
                break


def chunk_repo(
    repo_root: Path,
    *,
    chunk_size: int = 80,
    chunk_overlap: int = 20,
    max_chunks: int = 3000,
) -> list[Chunk]:
    """Chunk all source files in a repo into overlapping line windows.

    Returns at most *max_chunks* chunks. Windows are generated lazily and
    truncated once, over the whole corpus, so the budget holds for any mix of
    file sizes: a file short enough to fit in a single window counts against
    it like any other, and files past the budget are never read. A
    non-positive budget returns no chunks at all.
    """
    windows = _iter_chunks(
        repo_root, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return list(islice(windows, max(max_chunks, 0)))
