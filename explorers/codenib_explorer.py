"""Run CodeNib's native repository explorer under SWE-Explore's protocol."""

from __future__ import annotations

from pathlib import Path

from .base import ContextRegion, Explorer, ExplorerResult


class CodeNibExplorer(Explorer):
    """Adapt CodeNib's native explorer to SWE-Explore's local protocol.

    ``bm25`` remains the low-dependency compatibility control, while callers
    can select any native CodeNib repository-explorer policy. Index construction
    is explicit at this boundary: ``auto_index=True`` materializes or updates
    the views declared by that policy before the first query; ``False`` requires
    a current manifest for the checkout.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        auto_index: bool = True,
        rebuild: bool = False,
        policy: str = "bm25",
        planning_budget: str = "balanced",
        retrieval_level: str = "l2",
    ) -> None:
        from codenib.agent import (
            normalize_repository_explorer_policy,
            repository_explorer_build_views,
        )
        from codenib.integrations.swe_explore import CodeNibSWEExploreExplorer

        self.repo_root = repo_root.expanduser().resolve()
        self.policy = normalize_repository_explorer_policy(policy)
        self.required_views = repository_explorer_build_views(self.policy)
        self.planning_budget = planning_budget
        self.retrieval_level = retrieval_level
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
                views=self.required_views,
                rebuild=rebuild,
            )
            if failed:
                raise RuntimeError(
                    "CodeNib failed to materialize required views: " + ", ".join(failed)
                )
        self._delegate = CodeNibSWEExploreExplorer.from_repository(
            self.repo_root,
            policy=self.policy,
            budget=self.planning_budget,
            level=self.retrieval_level,
        )

    def close(self) -> None:
        """Release CodeNib runtime resources loaded for this explorer."""

        self._delegate.close()

    def __enter__(self) -> "CodeNibExplorer":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

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
