from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from sentence_transformers import SentenceTransformer

from .base import ContextRegion, ExplorerResult
from .chunking import Chunk, chunk_repo


def _post_chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    timeout: int = 120,
) -> str:
    import openai
    # Detect Azure endpoint
    if "openai.azure.com" in api_base:
        client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=api_base,
            api_version="2024-12-01-preview",
        )
    else:
        base = api_base.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        client = openai.OpenAI(api_key=api_key, base_url=base)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
    )
    return resp.choices[0].message.content


def _extract_json_obj(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


@dataclass
class _ScoredChunk:
    chunk: Chunk
    score: float


class SweRankExplorer:
    """Lightweight two-stage explorer compatible with the formal runner.

    Stage 1:
    - Use a local embedding model to retrieve candidate chunks.
    Stage 2:
    - Ask an OpenAI-compatible model to rerank the retrieved chunks and select
      the most useful ones for solving the issue.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        embed_model: str,
        embedder: SentenceTransformer | None = None,
        rerank_model: str,
        chunk_size: int = 80,
        chunk_overlap: int = 20,
        retrieve_top_n: int = 50,
        rerank_top_n: int | None = None,
        max_chunks: int = 1000,
        backend: str = "openai",
        api_key: str,
        api_base: str,
    ) -> None:
        if backend != "openai":
            raise ValueError(f"Unsupported backend: {backend}")

        self.repo_root = Path(repo_root)
        self.rerank_model = rerank_model
        self.retrieve_top_n = retrieve_top_n
        self.rerank_top_n = rerank_top_n or retrieve_top_n
        self.api_key = api_key
        self.api_base = api_base
        self._embedder = embedder or SentenceTransformer(embed_model, trust_remote_code=True)
        self._chunks = chunk_repo(
            self.repo_root,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
        )
        texts = [c.content for c in self._chunks]
        if texts:
            self._chunk_embs = self._embedder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=16,
                show_progress_bar=len(texts) > 50,
            )
        else:
            self._chunk_embs = None

    def _retrieve(self, query: str) -> list[_ScoredChunk]:
        if not self._chunks or self._chunk_embs is None:
            return []
        query_emb = self._embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        scores = self._chunk_embs @ query_emb
        top_idx = scores.argsort()[::-1][: self.retrieve_top_n]
        return [
            _ScoredChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_idx
            if float(scores[i]) > 0
        ]

    def _rerank(self, query: str, retrieved: list[_ScoredChunk], top_k: int) -> list[int]:
        if not retrieved:
            return []

        candidates = retrieved[: min(len(retrieved), self.rerank_top_n)]
        candidate_blocks = []
        for idx, item in enumerate(candidates, start=1):
            snippet = item.chunk.content[:600]
            candidate_blocks.append(
                "\n".join(
                    [
                        f"[{idx}] path={item.chunk.path}",
                        f"lines={item.chunk.start}-{item.chunk.end}",
                        "content:",
                        snippet,
                    ]
                )
            )

        system = (
            "You are reranking retrieved code chunks for a software issue. "
            "Select the chunks most useful for fixing the issue. "
            "Return strict JSON: {\"selected_ids\": [int, ...]} with no extra text."
        )
        user = "\n\n".join(
            [
                f"Issue:\n{query}",
                f"Select up to {top_k} chunk ids that are most relevant.",
                "Candidates:",
                "\n\n".join(candidate_blocks),
            ]
        )
        try:
            text = _post_chat_completion(
                api_base=self.api_base,
                api_key=self.api_key,
                model=self.rerank_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            data = _extract_json_obj(text)
        except Exception as e:
            import sys
            print(f"[swerank] rerank failed, falling back to retrieval order: {e}", file=sys.stderr)
            return []
        selected_ids = data.get("selected_ids") or []
        out: list[int] = []
        for raw in selected_ids:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(candidates) and idx not in out:
                out.append(idx)
            if len(out) >= top_k:
                break
        return out

    def explore(self, *, instance_id: str, query: str, top_k: int = 5) -> list[ExplorerResult]:
        retrieved = self._retrieve(query)
        if not retrieved:
            return []

        chosen = self._rerank(query, retrieved, top_k=top_k)
        selected = [retrieved[idx - 1] for idx in chosen] if chosen else retrieved[:top_k]
        return [
            ExplorerResult(
                instance_id=instance_id,
                score=item.score,
                regions=[
                    ContextRegion(
                        path=item.chunk.path,
                        start=item.chunk.start,
                        end=item.chunk.end,
                        snippet=None,
                    )
                ],
            )
            for item in selected
        ]
