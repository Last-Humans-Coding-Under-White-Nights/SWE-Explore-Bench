"""CodeNib's manifest-backed BM25 explorer for SWE-Explore."""

from __future__ import annotations

from pathlib import Path

from .base import ContextRegion, Explorer, ExplorerResult


class CodeNibExplorer(Explorer):
    """Adapt CodeNib's native explorer to SWE-Explore's local protocol.

    Index construction is explicit at this boundary.  ``auto_index=True``
    materializes or updates only CodeNib's BM25 view before the first query;
    ``False`` requires a current manifest for the checkout.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        auto_index: bool = True,
        rebuild: bool = False,
    ) -> None:
        from codenib.integrations.swe_explore import CodeNibSWEExploreExplorer

        self.repo_root = repo_root.expanduser().resolve()
        if auto_index:
            from codenib.cli import detect_languages, index_repository

            languages = detect_languages(self.repo_root)
            if not languages:
                raise RuntimeError(
                    f"CodeNib found no supported source language in {self.repo_root}"
                )
            _manifest, failed = index_repository(
                self.repo_root,
                languages=languages,
                views=("bm25",),
                rebuild=rebuild,
            )
            if failed:
                raise RuntimeError(
                    "CodeNib failed to materialize required views: " + ", ".join(failed)
                )
        self._delegate = CodeNibSWEExploreExplorer.from_repository(self.repo_root)

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5
    ) -> list[ExplorerResult]:
        results = self._delegate.explore(
            instance_id=instance_id,
            query=query,
            top_k=top_k,
        )
        return [
            ExplorerResult(
                instance_id=result.instance_id,
                score=result.score,
                regions=[
                    ContextRegion(
                        path=region.path,
                        start=region.start,
                        end=region.end,
                        snippet=region.snippet,
                    )
                    for region in result.regions
                ],
            )
            for result in results
        ]


__all__ = ["CodeNibExplorer"]
