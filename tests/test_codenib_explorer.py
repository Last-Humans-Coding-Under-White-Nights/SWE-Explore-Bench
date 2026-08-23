from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from explorers.codenib_explorer import CodeNibExplorer


class _Delegate:
    repo_root = None
    policy = None
    budget = None
    level = None
    closed = False

    @classmethod
    def from_repository(cls, repo_root, *, policy, budget, level):
        cls.repo_root = repo_root
        cls.policy = policy
        cls.budget = budget
        cls.level = level
        cls.closed = False
        return cls()

    def close(self):
        type(self).closed = True

    def explore(self, *, instance_id, query, top_k):
        assert query == "find parser"
        assert top_k == 3
        return [
            SimpleNamespace(
                instance_id=instance_id,
                score=0.75,
                regions=(
                    SimpleNamespace(
                        path="src/parser.py",
                        start=4,
                        end=12,
                        snippet=None,
                    ),
                ),
            )
        ]


class _IndexCall:
    kwargs = None


def _install_fake_codenib(monkeypatch, *, failed=()):
    package = ModuleType("codenib")
    package.__path__ = []
    agent = ModuleType("codenib.agent")
    policies = {"auto", "bm25", "dense", "hybrid", "hybrid_rerank", "graph"}
    policy_views = {
        "auto": ("bm25", "vector", "symbol_graph"),
        "bm25": ("bm25",),
        "dense": ("vector",),
        "hybrid": ("bm25", "vector"),
        "hybrid_rerank": ("bm25", "vector"),
        "graph": ("bm25", "symbol_graph"),
    }

    def normalize_policy(value):
        normalized = value.strip().lower().replace("-", "_")
        if normalized not in policies:
            raise ValueError("unsupported policy")
        return normalized

    agent.normalize_repository_explorer_policy = normalize_policy
    agent.repository_explorer_build_views = lambda policy: policy_views[policy]
    integrations = ModuleType("codenib.integrations")
    integrations.__path__ = []
    integration = ModuleType("codenib.integrations.swe_explore")
    integration.CodeNibSWEExploreExplorer = _Delegate
    cli = ModuleType("codenib.cli")
    cli.detect_languages = lambda _repo: ["python"]

    def index_repository(*_args, **kwargs):
        _IndexCall.kwargs = kwargs
        return object(), list(failed)

    cli.index_repository = index_repository
    _IndexCall.kwargs = None
    monkeypatch.setitem(sys.modules, "codenib", package)
    monkeypatch.setitem(sys.modules, "codenib.agent", agent)
    monkeypatch.setitem(sys.modules, "codenib.integrations", integrations)
    monkeypatch.setitem(sys.modules, "codenib.integrations.swe_explore", integration)
    monkeypatch.setitem(sys.modules, "codenib.cli", cli)


def test_wrapper_preserves_official_region_contract(monkeypatch, tmp_path):
    _install_fake_codenib(monkeypatch)
    explorer = CodeNibExplorer(tmp_path, auto_index=False)

    results = explorer.explore(instance_id="org__repo-1", query="find parser", top_k=3)

    assert _Delegate.repo_root == tmp_path.resolve()
    assert _Delegate.policy == "bm25"
    assert _Delegate.budget == "balanced"
    assert _Delegate.level == "l2"
    assert len(results) == 1
    assert results[0].score == 0.75
    assert results[0].regions[0].path == "src/parser.py"
    assert (results[0].regions[0].start, results[0].regions[0].end) == (4, 12)


def test_auto_index_surfaces_failed_required_view(monkeypatch, tmp_path):
    _install_fake_codenib(monkeypatch, failed=("bm25",))

    with pytest.raises(RuntimeError, match="bm25"):
        CodeNibExplorer(tmp_path, auto_index=True)


def test_policy_controls_materialized_views_and_runtime(monkeypatch, tmp_path):
    _install_fake_codenib(monkeypatch)

    with CodeNibExplorer(
        tmp_path,
        policy="hybrid-rerank",
        planning_budget="thorough",
        retrieval_level="l0",
    ) as explorer:
        assert explorer.policy == "hybrid_rerank"
        assert explorer.required_views == ("bm25", "vector")
        assert _IndexCall.kwargs["views"] == ("bm25", "vector")
        assert _Delegate.policy == "hybrid_rerank"
        assert _Delegate.budget == "thorough"
        assert _Delegate.level == "l0"

    assert _Delegate.closed is True
