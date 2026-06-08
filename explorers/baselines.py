"""Baseline explorers for SWE-Explore evaluation."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

from .base import ContextRegion, Explorer, ExplorerResult


class OracleExplorer(Explorer):
    """Oracle explorer: returns ground truth directly."""

    def __init__(self, bench_path: Path) -> None:
        self.bench_data: dict[str, list[dict]] = {}
        with open(bench_path) as f:
            for line in f:
                data = json.loads(line)
                instance_id = data["instance_id"]
                gt = data.get("ground_truth") or {}
                self.bench_data[instance_id] = gt.get("read_core_regions") or []

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5
    ) -> list[ExplorerResult]:
        regions_raw = self.bench_data.get(instance_id, [])
        regions = [
            ContextRegion(path=r["path"], start=r["start"], end=r["end"])
            for r in regions_raw
        ]
        return [
            ExplorerResult(instance_id=instance_id, score=1.0, regions=regions)
        ]


class RandomExplorer(Explorer):
    """Random explorer: randomly samples regions from ground truth pool."""

    def __init__(self, bench_path: Path, seed: int = 42) -> None:
        self.all_regions: list[dict] = []
        self.rng = random.Random(seed)
        with open(bench_path) as f:
            for line in f:
                data = json.loads(line)
                gt = data.get("ground_truth") or {}
                self.all_regions.extend(gt.get("read_core_regions") or [])

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5
    ) -> list[ExplorerResult]:
        if not self.all_regions:
            return []
        sampled = self.rng.sample(
            self.all_regions, min(top_k * 2, len(self.all_regions))
        )
        regions = [
            ContextRegion(path=r["path"], start=r["start"], end=r["end"])
            for r in sampled
        ]
        return [
            ExplorerResult(instance_id=instance_id, score=0.0, regions=regions)
        ]


class SimpleRuleExplorer(Explorer):
    """Simple heuristic explorer: prioritises common file patterns."""

    PRIORITY_PATTERNS = [
        "__init__.py", "setup.py", "main.py", "__main__.py",
        "config.py", "constants.py", "settings.py",
    ]

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5
    ) -> list[ExplorerResult]:
        if not self.repo_root.is_dir():
            return []

        py_files = sorted(self.repo_root.rglob("*.py"))

        def priority_score(f: Path) -> tuple[int, str]:
            name = f.name
            for i, pat in enumerate(self.PRIORITY_PATTERNS):
                if pat in name:
                    return (i, name)
            return (len(self.PRIORITY_PATTERNS), name)

        sorted_files = sorted(py_files, key=priority_score)[:top_k]
        regions = [
            ContextRegion(
                path=str(f.relative_to(self.repo_root)),
                start=1,
                end=100,
            )
            for f in sorted_files
        ]
        return [
            ExplorerResult(instance_id=instance_id, score=0.5, regions=regions)
        ]
