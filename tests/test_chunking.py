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


def _mixed_repo(root: Path) -> None:
    """A corpus mixing files that fit in one window with files that do not."""
    _write(root, "one_line.py", 1)
    _write(root, "pkg/medium.py", 25)
    _write(root, "pkg/short.py", 2)
    _write(root, "zlong.py", 100)


def test_short_files_respect_the_chunk_budget(tmp_path):
    for name in ("a.py", "b.py", "c.py"):
        _write(tmp_path, name, 1)

    chunks = chunk_repo(tmp_path, max_chunks=1)

    # The one chunk is the first of the unbounded run, not an arbitrary file's.
    assert len(chunks) == 1
    assert (chunks[0].path, chunks[0].start) == ("a.py", 1)


def test_long_file_respects_the_chunk_budget(tmp_path):
    _write(tmp_path, "long.py", 300)

    chunks = chunk_repo(tmp_path, chunk_size=10, chunk_overlap=0, max_chunks=4)

    assert len(chunks) == 4


def test_no_budget_returns_every_window(tmp_path):
    """max_chunks=None is the unbounded run, without a stand-in number."""
    _write(tmp_path, "long.py", 300)

    chunks = chunk_repo(tmp_path, chunk_size=10, chunk_overlap=0, max_chunks=None)

    assert len(chunks) == 30


def test_files_past_the_budget_are_never_read(tmp_path, monkeypatch):
    """The budget bounds the reading, not just the returned list."""
    _mixed_repo(tmp_path)
    read_text = Path.read_text
    read: list[Path] = []

    def record(self, *args, **kwargs):
        read.append(self)
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record)
    chunk_repo(tmp_path, chunk_size=10, chunk_overlap=2, max_chunks=1)

    assert read == [tmp_path / "one_line.py"]


def test_a_string_repo_root_is_accepted(tmp_path):
    """Same coercion as iter_source_files, which this walks with."""
    _write(tmp_path, "a.py", 1)

    assert chunk_repo(str(tmp_path)) == chunk_repo(tmp_path)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_a_window_must_cover_a_line(tmp_path, chunk_size):
    """chunk_size=0 indexed nothing; a negative one built inverted ranges."""
    _write(tmp_path, "a.py", 5)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_repo(tmp_path, chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_overlap", [-1, 10, 11])
def test_a_window_must_advance_by_a_line(tmp_path, chunk_overlap):
    """A negative overlap skipped lines; one past the window stalled it."""
    _write(tmp_path, "a.py", 5)

    with pytest.raises(ValueError, match=r"chunk_overlap must be in \[0, 10\)"):
        chunk_repo(tmp_path, chunk_size=10, chunk_overlap=chunk_overlap)


@pytest.mark.parametrize("max_chunks", [-1, 0, 1, 2, 3, 5, 8, 13, 1000, None])
def test_budget_truncates_any_mix_of_file_sizes_to_a_prefix(tmp_path, max_chunks):
    _mixed_repo(tmp_path)
    kwargs = {"chunk_size": 10, "chunk_overlap": 2}
    unlimited = chunk_repo(tmp_path, max_chunks=None, **kwargs)

    chunks = chunk_repo(tmp_path, max_chunks=max_chunks, **kwargs)

    # A positive budget returns that prefix of an unbounded run — the same
    # chunks, truncated at the same place every time. No budget returns all of
    # them; anything else returns nothing.
    expected = unlimited if max_chunks is None else unlimited[: max(max_chunks, 0)]
    assert chunks == expected
