"""Tests for the chunk budget in :func:`explorers.chunking.chunk_repo`."""
from __future__ import annotations

from pathlib import Path

import pytest

from explorers.chunking import chunk_repo


def _write(root: Path, name: str, lines: int) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"line {i}\n" for i in range(1, lines + 1))
    path.write_text(body, encoding="utf-8")


def _mixed_repo(root: Path) -> Path:
    """A corpus mixing files that fit in one window with files that do not."""
    _write(root, "one_line.py", 1)
    _write(root, "pkg/medium.py", 25)
    _write(root, "pkg/short.py", 2)
    _write(root, "zlong.py", 100)
    return root


def test_short_files_respect_the_chunk_budget(tmp_path):
    for name in ("a.py", "b.py", "c.py"):
        _write(tmp_path, name, 1)

    assert len(chunk_repo(tmp_path, max_chunks=1)) == 1


def test_long_file_respects_the_chunk_budget(tmp_path):
    _write(tmp_path, "long.py", 300)

    chunks = chunk_repo(tmp_path, chunk_size=10, chunk_overlap=0, max_chunks=4)

    assert len(chunks) == 4


@pytest.mark.parametrize("max_chunks", [-1, 0, 1, 2, 3, 5, 8, 13, 1000])
def test_budget_truncates_any_mix_of_file_sizes_to_a_prefix(tmp_path, max_chunks):
    _mixed_repo(tmp_path)
    kwargs = {"chunk_size": 10, "chunk_overlap": 2}
    unlimited = chunk_repo(tmp_path, max_chunks=10_000, **kwargs)

    chunks = chunk_repo(tmp_path, max_chunks=max_chunks, **kwargs)

    # A positive budget returns that prefix of an unbounded run — the same
    # chunks, truncated at the same place every time. Anything else returns
    # nothing.
    assert chunks == (unlimited[:max_chunks] if max_chunks > 0 else [])
