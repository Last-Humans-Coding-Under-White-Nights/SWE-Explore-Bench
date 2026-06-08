"""Generic embedding RAG explorer supporting multiple embedding backends.

Supported backends:
  - sentence_transformers: Any HuggingFace SentenceTransformer model
    e.g. BAAI/bge-code-v1, jinaai/jina-embeddings-v4, jinaai/jina-embeddings-v5-text-small
  - openai: OpenAI-compatible embedding API
    e.g. text-embedding-3-large, voyage-code-3
  - google: Google GenAI embedding API
    e.g. gemini-embedding-2-preview
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import ContextRegion, Explorer, ExplorerResult
from .chunking import Chunk, chunk_repo


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

def _embed_sentence_transformers(
    texts: list[str],
    model_name: str,
    *,
    _cache: dict = {},
    query_prefix: str = "",
    is_query: bool = False,
    batch_size: int = 16,
    task: Optional[str] = None,
    **_kw,
) -> np.ndarray:
    """Embed via sentence-transformers (local)."""
    if model_name not in _cache:
        from sentence_transformers import SentenceTransformer
        model_kwargs = {}
        if task:
            model_kwargs["default_task"] = task
        _cache[model_name] = SentenceTransformer(
            model_name, trust_remote_code=True, model_kwargs=model_kwargs,
        )
    model = _cache[model_name]
    if is_query and query_prefix:
        texts = [f"{query_prefix}: {t}" for t in texts]
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 32,
        batch_size=batch_size,
    )


def _embed_openai(
    texts: list[str],
    model_name: str,
    *,
    api_key: str,
    api_base: Optional[str] = None,
    batch_size: int = 512,
    **_kw,
) -> np.ndarray:
    """Embed via OpenAI-compatible API (voyage, jina hosted, etc.)."""
    import openai
    kwargs = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base.rstrip("/")
        if not kwargs["base_url"].endswith("/v1"):
            kwargs["base_url"] += "/v1"
    client = openai.OpenAI(**kwargs)

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model_name, input=batch)
        batch_embs = [d.embedding for d in resp.data]
        all_embeddings.extend(batch_embs)

    arr = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def _embed_google(
    texts: list[str],
    model_name: str,
    *,
    api_key: str,
    batch_size: int = 64,
    **_kw,
) -> np.ndarray:
    """Embed via Google GenAI API (gemini-embedding)."""
    from google import genai
    client = genai.Client(api_key=api_key)

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.models.embed_content(model=model_name, contents=batch)
        all_embeddings.extend([e.values for e in resp.embeddings])

    arr = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


_BACKENDS = {
    "sentence_transformers": _embed_sentence_transformers,
    "openai": _embed_openai,
    "google": _embed_google,
}

# ---------------------------------------------------------------------------
# Preset configurations for popular models
# ---------------------------------------------------------------------------
PRESETS = {
    # Local sentence-transformers models
    "bge-code-v1": {
        "backend": "sentence_transformers",
        "model": "BAAI/bge-code-v1",
        "query_prefix": "",
    },
    "jina-v3": {
        "backend": "sentence_transformers",
        "model": "jinaai/jina-embeddings-v3",
        "query_prefix": "",
    },
    "jina-v4": {
        "backend": "sentence_transformers",
        "model": "jinaai/jina-embeddings-v4",
        "query_prefix": "",
        "task": "retrieval",
    },
    "jina-v5-text-small": {
        "backend": "sentence_transformers",
        "model": "jinaai/jina-embeddings-v5-text-small",
        "query_prefix": "",
        "task": "retrieval",
    },
    # OpenAI-compatible API models
    "text-embedding-3-large": {
        "backend": "openai",
        "model": "text-embedding-3-large",
    },
    "text-embedding-3-small": {
        "backend": "openai",
        "model": "text-embedding-3-small",
    },
    "voyage-code-3": {
        "backend": "openai",
        "model": "voyage-code-3",
        "api_base": "https://api.voyageai.com",
    },
    "qwen3-embedding": {
        "backend": "openai",
        "model": "qwen3-embedding",
    },
    # Google GenAI
    "gemini-embedding": {
        "backend": "google",
        "model": "gemini-embedding-2-preview",
    },
}


# ---------------------------------------------------------------------------
# Explorer
# ---------------------------------------------------------------------------
class EmbedExplorer(Explorer):
    """Generic embedding-based code retriever.

    Args:
        repo_root: Repository directory.
        backend: One of "sentence_transformers", "openai", "google".
        model_name: Model identifier for the chosen backend.
        preset: Shorthand name from PRESETS (overrides backend/model_name).
        api_key: API key (for openai/google backends).
        api_base: API base URL (for openai backend).
        query_prefix: Prefix prepended to query for asymmetric models.
        chunk_size: Lines per chunk.
        chunk_overlap: Overlap between chunks.
        max_chunks: Maximum chunks to index.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        backend: str = "sentence_transformers",
        model_name: str = "BAAI/bge-code-v1",
        preset: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        query_prefix: str = "",
        chunk_size: int = 80,
        chunk_overlap: int = 20,
        max_chunks: int = 3000,
    ) -> None:
        # Apply preset if given
        task = None
        if preset and preset in PRESETS:
            cfg = PRESETS[preset]
            backend = cfg.get("backend", backend)
            model_name = cfg.get("model", model_name)
            query_prefix = cfg.get("query_prefix", query_prefix)
            api_base = cfg.get("api_base", api_base) or api_base
            task = cfg.get("task")

        self.repo_root = repo_root
        self.backend = backend
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.query_prefix = query_prefix
        self._task = task

        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Choose from {list(_BACKENDS)}")
        self._embed_fn = _BACKENDS[backend]

        self._chunks: list[Chunk] = chunk_repo(
            repo_root,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
        )

        if self._chunks:
            embed_kwargs = {
                "model_name": model_name,
                "api_key": api_key or "",
                "api_base": api_base,
                "query_prefix": query_prefix,
                "is_query": False,
                "task": self._task,
            }
            self._embeddings = self._embed_fn(
                [c.content for c in self._chunks], **embed_kwargs,
            )
        else:
            self._embeddings = np.empty((0, 1))

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5,
    ) -> List[ExplorerResult]:
        if self._embeddings.shape[0] == 0 or not query.strip():
            return []

        embed_kwargs = {
            "model_name": self.model_name,
            "api_key": self.api_key or "",
            "api_base": self.api_base,
            "query_prefix": self.query_prefix,
            "is_query": True,
            "task": self._task,
        }
        q_emb = self._embed_fn([query], **embed_kwargs)[0]

        scores = self._embeddings @ q_emb
        top_idx = np.argsort(scores)[::-1][:top_k]

        results: list[ExplorerResult] = []
        for idx in top_idx:
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
