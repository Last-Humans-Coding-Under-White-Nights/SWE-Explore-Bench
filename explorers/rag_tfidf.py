"""TF-IDF based RAG explorer.

Lightweight alternative to neural embeddings — uses scikit-learn
TfidfVectorizer with cosine similarity over line-based chunks.
Peak memory ~5-10 MB per repo (sparse matrix).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .base import ContextRegion, Explorer, ExplorerResult
from .chunking import Chunk, chunk_repo


class TFIDFExplorer(Explorer):
    """TF-IDF + cosine similarity retriever over repo chunks."""

    def __init__(
        self,
        repo_root: Path,
        *,
        chunk_size: int = 80,
        chunk_overlap: int = 20,
        max_features: int = 10_000,
        max_chunks: int = 3000,
    ) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.repo_root = repo_root
        self._chunks = chunk_repo(
            repo_root,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
        )

        if not self._chunks:
            self._chunk_matrix = None
            self._vectorizer = None
            return

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=True,
            stop_words="english",
            token_pattern=r"(?u)\b[A-Za-z_][A-Za-z0-9_]{1,}\b",
        )
        try:
            self._chunk_matrix = self._vectorizer.fit_transform(
                [c.content for c in self._chunks]
            )
        except ValueError:
            # Empty vocabulary
            self._chunk_matrix = None
            self._vectorizer = None

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5
    ) -> List[ExplorerResult]:
        if self._chunk_matrix is None or self._vectorizer is None or not query.strip():
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._chunk_matrix)[0]
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
