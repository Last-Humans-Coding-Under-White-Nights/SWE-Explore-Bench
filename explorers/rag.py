from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import requests
import typer

from .base import ContextRegion, Explorer, ExplorerResult


app = typer.Typer(rich_markup_mode="rich")


@dataclass
class _EmbeddingDoc:
    path: str
    content: str
    embedding: list[float]


def _default_embed_endpoint() -> str:
    return os.environ.get("RAG_EMBEDDING_ENDPOINT", "")


def _default_embed_api_key() -> str:
    return os.environ.get("RAG_EMBEDDING_API_KEY", "")


def _iter_source_files(repo_root: Path) -> Iterable[Path]:
    exts = {".py", ".md", ".txt"}
    for p in repo_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _load_texts(repo_root: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for p in _iter_source_files(repo_root):
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        rel = p.relative_to(repo_root).as_posix()
        docs.append((rel, text))
    return docs


def _call_embedding_api(
    endpoint: str,
    api_key: str | None,
    texts: list[str],
) -> list[list[float]]:
    """调用外部向量服务。

    这里仅假定一个非常通用的 JSON 协议：
    POST {endpoint}
    {
      "texts": [...],
      "api_key": "...",  # 可选
    }
    响应：
    {
      "embeddings": [[...], [...], ...]
    }

    具体对接时可以按自己的服务协议在此修改。
    """
    headers = {"Content-Type": "application/json"}
    payload: dict = {"texts": texts}
    if api_key:
        payload["api_key"] = api_key
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings") or []
    if not isinstance(embeddings, list):
        raise ValueError("Invalid embedding response: 'embeddings' must be a list")
    return embeddings


def _l2_normalize(vec: list[float]) -> list[float]:
    import math

    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


class RAGExplorer(Explorer):
    """使用外部 embedding 服务的简易 RAG 检索器。

    实际对接哪个服务由环境变量或调用方决定，我们只假设一个非常简单的 JSON 协议。
    """

    def __init__(
        self,
        repo_root: Path,
        endpoint: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.endpoint = endpoint or _default_embed_endpoint()
        self.api_key = api_key or _default_embed_api_key()
        if not self.endpoint:
            raise RuntimeError(
                "RAG_EMBEDDING_ENDPOINT 未设置，无法初始化 RAGExplorer。"
            )

        texts = _load_texts(repo_root)
        paths = [p for p, _ in texts]
        contents = [c for _, c in texts]
        embeddings_raw = _call_embedding_api(self.endpoint, self.api_key, contents)
        if len(embeddings_raw) != len(contents):
            raise ValueError("Embedding 数量与文档数量不一致")

        self._docs: list[_EmbeddingDoc] = []
        for path, content, emb in zip(paths, contents, embeddings_raw):
            self._docs.append(
                _EmbeddingDoc(
                    path=path,
                    content=content,
                    embedding=_l2_normalize([float(x) for x in emb]),
                )
            )

    def explore(self, *, instance_id: str, query: str, top_k: int = 5) -> List[ExplorerResult]:
        query_emb_raw = _call_embedding_api(self.endpoint, self.api_key, [query])[0]
        query_emb = _l2_normalize([float(x) for x in query_emb_raw])

        scored: list[tuple[float, _EmbeddingDoc]] = []
        for doc in self._docs:
            score = _cosine_sim(query_emb, doc.embedding)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:top_k]

        results: list[ExplorerResult] = []
        for score, doc in scored:
            # 同样先返回文件级别的上下文，行级由下游行精修管道承担
            regions = [ContextRegion(path=doc.path, start=1, end=-1, snippet=None)]
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
        ...,
        exists=True,
        file_okay=False,
        help="单个源码仓库根目录，例如 repos/django",
    ),
    query: str = typer.Argument(..., help="Issue 描述或自然语言查询"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="返回的文件数量"),
    endpoint: str | None = typer.Option(
        None,
        "--endpoint",
        "-e",
        help="向量服务地址，默认从 RAG_EMBEDDING_ENDPOINT 读取",
    ),
) -> None:
    """使用外部向量服务的简单 RAG 检索 CLI。"""
    explorer = RAGExplorer(repo_root, endpoint=endpoint)
    results = explorer.explore(instance_id="adhoc", query=query, top_k=top_k)
    for i, res in enumerate(results, 1):
        typer.echo(f"[{i}] score={res.score:.4f}")
        for region in res.regions:
            typer.echo(f"  - {region.path}:{region.start}-{region.end}")


if __name__ == "__main__":
    app()

