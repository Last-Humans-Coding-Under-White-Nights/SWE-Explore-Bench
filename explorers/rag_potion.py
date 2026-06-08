"""Potion-8M static-embedding RAG explorer.

Uses model2vec StaticModel for 256-dim embeddings.
The model (~32 MB on disk) can be shared across repos via the ``_model``
parameter to avoid reloading.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np

from .base import ContextRegion, Explorer, ExplorerResult
from .chunking import chunk_repo


def _load_potion_model(model_path: str) -> Any:
    """Load a model2vec StaticModel (lazy import)."""
    from model2vec import StaticModel

    return StaticModel.from_pretrained(model_path)


class PotionExplorer(Explorer):
    """Static-embedding RAG retriever using Potion-8M (model2vec).

    Encodes all chunks at init with L2-normalised vectors so that cosine
    similarity reduces to a simple dot product at query time.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        model_path: str = "/tmp/potion-base-8M",
        chunk_size: int = 80,
        chunk_overlap: int = 20,
        max_chunks: int = 3000,
        _model: Any | None = None,
    ) -> None:
        self.repo_root = repo_root
        self._model = _model or _load_potion_model(model_path)
        self._chunks = chunk_repo(
            repo_root,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
        )

        if not self._chunks:
            self._normed: np.ndarray | None = None
            return

        vecs = self._model.encode([c.content for c in self._chunks])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._normed = vecs / norms  # (N, dim)

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5
    ) -> List[ExplorerResult]:
        if self._normed is None or not query.strip():
            return []

        q_vec = self._model.encode([query])  # (1, dim)
        q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
        q_norm = np.where(q_norm == 0, 1.0, q_norm)
        q_vec = q_vec / q_norm

        scores = (self._normed @ q_vec.T).ravel()  # cosine sim
        top_indices = scores.argsort()[-top_k:][::-1]

        results: list[ExplorerResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                ExplorerResult(
                    instance_id=instance_id,
                    score=float(scores[idx]),
                    regions=[
                        ContextRegion(
                            path=chunk.path,
                            start=chunk.start,
                            end=chunk.end,
                        )
                    ],
                )
            )
        return results
