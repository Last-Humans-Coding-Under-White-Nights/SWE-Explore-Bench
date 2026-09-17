"""Retrieval baselines must see the same, repeatable source corpus."""
import shutil
from pathlib import Path

import pytest

from explorers.baselines import SimpleRuleExplorer
from explorers.bm25 import BM25Explorer
from explorers.chunking import chunk_repo
from explorers.rag import _load_texts
from explorers.rag_tfidf import TFIDFExplorer
from explorers.source_files import iter_source_files


ARKTS_FILES = {
    "build-profile.json5",
    "oh-package.json5",
    "entry/src/main/ets/pages/Index.ets",
    "entry/src/main/ets/model/DataSource.ets",
    "entry/src/main/ets/common/types.d.ets",
    "util.ts",
}


@pytest.fixture
def arkts_repo(tmp_path: Path) -> Path:
    root = tmp_path / "app"
    shutil.copytree(Path(__file__).parent / "fixtures" / "arkts_app", root)
    (root / "util.ts").write_text("export const title = 'Catalog';\n")
    return root


def test_all_arkts_files_are_indexed(arkts_repo):
    paths = {p.relative_to(arkts_repo).as_posix() for p in iter_source_files(arkts_repo)}
    assert ARKTS_FILES <= paths
    assert ARKTS_FILES <= {chunk.path for chunk in chunk_repo(arkts_repo)}
    assert ARKTS_FILES <= {path for path, text in _load_texts(arkts_repo) if text}


def test_simple_rule_returns_arkts_sources(arkts_repo):
    results = SimpleRuleExplorer(arkts_repo).explore(
        instance_id="arkts", query="Next item", top_k=20,
    )
    paths = {region.path for result in results for region in result.regions}
    assert ARKTS_FILES <= paths


@pytest.mark.parametrize("directory", [
    ".git", "node_modules", "oh_modules", "build", ".hvigor", "dist",
    "Build", "DIST", "Node_modules",
    ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
])
def test_generated_and_dependency_directories_are_excluded(tmp_path, directory):
    for prefix in (Path(), Path("entry")):
        generated = tmp_path / prefix / directory / "cache" / "generated.py"
        generated.parent.mkdir(parents=True)
        generated.write_text("generated = True\n")
    source = tmp_path / "src" / "main.py"
    source.parent.mkdir()
    source.write_text("source = True\n")
    assert list(iter_source_files(tmp_path)) == [source]
    assert [path for path, _ in _load_texts(tmp_path)] == ["src/main.py"]
    results = SimpleRuleExplorer(tmp_path).explore(instance_id="test", query="source")
    assert [r.path for result in results for r in result.regions] == ["src/main.py"]


def test_order_is_sorted_and_independent_of_creation_order(tmp_path):
    names = ["z.ts", "b/main.py", "a.ets", "a/types.d.ets", "README.md"]
    roots = [tmp_path / "first", tmp_path / "second"]
    for root, order in zip(roots, (names, names[::-1])):
        for name in order:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("first\nsecond\nthird\n")
    for root in roots:
        for _ in range(2):
            paths = [p.relative_to(root).as_posix() for p in iter_source_files(root)]
            assert paths == sorted(names)
            chunks = chunk_repo(root, chunk_size=1, chunk_overlap=0, max_chunks=2)
            assert [(c.path, c.start) for c in chunks] == [
                ("README.md", 1), ("README.md", 2),
            ]


def test_discovery_configuration_replaces_defaults(tmp_path):
    names = [
        "main.py", "types.d.ets", "page.ets", "data.custom",
        "build/kept.custom", "vendor/skip.custom",
    ]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content\n")
    paths = iter_source_files(
        tmp_path, extensions={".custom", ".d.ets"}, excluded_dirs={"vendor"},
    )
    assert [p.relative_to(tmp_path).as_posix() for p in paths] == [
        "build/kept.custom", "data.custom", "types.d.ets",
    ]
    assert list(iter_source_files(tmp_path, extensions=set())) == []


def test_existing_extensions_and_case_insensitive_matching(tmp_path):
    names = [
        "a.py", "b.md", "c.txt", "d.toml", "e.cfg", "f.ini", "g.yaml",
        "h.yml", "i.json", "j.rst", "k.ETS", "l.TS", "m.JSON5",
    ]
    for name in names + ["ignored.png", "ignored.ets.bak"]:
        (tmp_path / name).write_text("content\n")
    assert [p.name for p in iter_source_files(tmp_path)] == names


def test_string_repository_root(tmp_path):
    source = tmp_path / "main.ets"
    source.write_text("source\n")
    assert list(iter_source_files(str(tmp_path))) == [source]


@pytest.mark.parametrize("extensions", ["ts", ".TS", ["ts"], [".ts"]])
def test_extensions_match_whole_suffixes(tmp_path, extensions):
    for name in ("main.ts", "types.d.ts", "assets", "hosts", "notes", "file.t"):
        (tmp_path / name).write_text("source\n")
    assert [p.name for p in iter_source_files(tmp_path, extensions=extensions)] == [
        "main.ts", "types.d.ts",
    ]


@pytest.mark.parametrize("excluded_dirs", [
    "DIST", ["DIST"], "DIST/", ["DIST/"], "DIST\\", ["DIST\\"],
])
def test_excluded_dirs_accept_names_case_insensitively(tmp_path, excluded_dirs):
    for directory in ("dist", "d", "s", "t"):
        path = tmp_path / directory / "main.py"
        path.parent.mkdir()
        path.write_text("source\n")
    paths = iter_source_files(tmp_path, excluded_dirs=excluded_dirs)
    assert [p.relative_to(tmp_path).as_posix() for p in paths] == [
        "d/main.py", "s/main.py", "t/main.py",
    ]


def test_symlinks_skip_directories_and_broken_targets(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    target = external / "source.ets"
    target.write_text("source\n")
    (root / "directory.ets").symlink_to(external, target_is_directory=True)
    (root / "broken.ets").symlink_to(external / "missing.ets")
    linked_file = root / "linked.ets"
    linked_file.symlink_to(target)
    assert list(iter_source_files(root)) == [linked_file]


@pytest.mark.parametrize("filename", [None, "ignored.png"])
def test_empty_source_corpus(tmp_path, filename):
    if filename:
        (tmp_path / filename).write_text("not source\n")
    assert list(iter_source_files(tmp_path)) == []
    assert chunk_repo(tmp_path) == []
    assert _load_texts(tmp_path) == []


@pytest.mark.parametrize("explorer_class", [BM25Explorer, TFIDFExplorer])
def test_retrievers_find_arkts_fixture_sources(arkts_repo, explorer_class):
    explorer = explorer_class(arkts_repo)
    results = explorer.explore(instance_id="arkts", query="CatalogItem", top_k=20)
    paths = {r.path for result in results if result.score > 0 for r in result.regions}
    assert "entry/src/main/ets/common/types.d.ets" in paths


def test_source_readers_use_utf8(tmp_path, monkeypatch):
    content = "const title = 'Каталог 商品';\n"
    (tmp_path / "main.ets").write_text(content, encoding="utf-8")
    original_open = Path.open

    def open_with_legacy_default(
        path, mode="r", buffering=-1, encoding=None, errors=None, newline=None,
    ):
        if encoding in (None, "locale"):
            encoding = "cp1252"
        return original_open(path, mode, buffering, encoding, errors, newline)

    # Simulate a legacy locale while retaining real file reads.
    monkeypatch.setattr(Path, "open", open_with_legacy_default)
    assert [c.content for c in chunk_repo(tmp_path)] == [content.rstrip("\n")]
    assert _load_texts(tmp_path) == [("main.ets", content)]
