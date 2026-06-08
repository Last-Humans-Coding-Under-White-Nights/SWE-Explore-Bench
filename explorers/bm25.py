"""BM25-based code explorer using shared chunking."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import typer
from rank_bm25 import BM25Okapi

from .base import ContextRegion, Explorer, ExplorerResult
from .chunking import Chunk, chunk_repo

app = typer.Typer(rich_markup_mode="rich")


class BM25Explorer(Explorer):
    """BM25 explorer using chunk-level indexing.

    Uses :func:`chunking.chunk_repo` for overlapping line-window chunking,
    consistent with the TF-IDF and Potion explorers.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        chunk_size: int = 80,
        chunk_overlap: int = 20,
        max_chunks: int = 3000,
    ) -> None:
        self.repo_root = repo_root
        self._chunks: list[Chunk] = chunk_repo(
            repo_root,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
        )
        tokenized = [c.content.split() for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5
    ) -> List[ExplorerResult]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = query.split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[ExplorerResult] = []
        for idx, score in ranked:
            chunk = self._chunks[idx]
            regions = [
                ContextRegion(
                    path=chunk.path, start=chunk.start, end=chunk.end, snippet=None
                )
            ]
            results.append(
                ExplorerResult(
                    instance_id=instance_id,
                    score=float(score),
                    regions=regions,
                )
            )
        return results


@app.command()
def search(
    repo_root: Path = typer.Argument(
        ..., exists=True, file_okay=False,
        help="Source repository root, e.g. repos/django",
    ),
    query: str = typer.Argument(..., help="Issue description or query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    chunk_size: int = typer.Option(80, "--chunk-size", help="Chunk size (lines)"),
    chunk_overlap: int = typer.Option(20, "--chunk-overlap", help="Chunk overlap (lines)"),
) -> None:
    """BM25-based code search CLI."""
    explorer = BM25Explorer(
        repo_root, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    results = explorer.explore(instance_id="adhoc", query=query, top_k=top_k)
    for i, res in enumerate(results, 1):
        typer.echo(f"[{i}] score={res.score:.4f}")
        for region in res.regions:
            typer.echo(f"  - {region.path}:{region.start}-{region.end}")


if __name__ == "__main__":
    app()
