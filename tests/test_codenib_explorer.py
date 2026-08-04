from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from explorers.codenib_explorer import CodeNibExplorer


class _Delegate:
    repo_root = None

    @classmethod
    def from_repository(cls, repo_root):
        cls.repo_root = repo_root
        return cls()

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


def _install_fake_codenib(monkeypatch, *, failed=()):
    package = ModuleType("codenib")
    package.__path__ = []
    integrations = ModuleType("codenib.integrations")
    integrations.__path__ = []
    integration = ModuleType("codenib.integrations.swe_explore")
    integration.CodeNibSWEExploreExplorer = _Delegate
    cli = ModuleType("codenib.cli")
    cli.detect_languages = lambda _repo: ["python"]
    cli.index_repository = lambda *_args, **_kwargs: (object(), list(failed))
    monkeypatch.setitem(sys.modules, "codenib", package)
    monkeypatch.setitem(sys.modules, "codenib.integrations", integrations)
    monkeypatch.setitem(sys.modules, "codenib.integrations.swe_explore", integration)
    monkeypatch.setitem(sys.modules, "codenib.cli", cli)


def test_wrapper_preserves_official_region_contract(monkeypatch, tmp_path):
    _install_fake_codenib(monkeypatch)
    explorer = CodeNibExplorer(tmp_path, auto_index=False)

    results = explorer.explore(instance_id="org__repo-1", query="find parser", top_k=3)

    assert _Delegate.repo_root == tmp_path.resolve()
    assert len(results) == 1
    assert results[0].score == 0.75
    assert results[0].regions[0].path == "src/parser.py"
    assert (results[0].regions[0].start, results[0].regions[0].end) == (4, 12)


def test_auto_index_surfaces_failed_required_view(monkeypatch, tmp_path):
    _install_fake_codenib(monkeypatch, failed=("bm25",))

    with pytest.raises(RuntimeError, match="bm25"):
        CodeNibExplorer(tmp_path, auto_index=True)
