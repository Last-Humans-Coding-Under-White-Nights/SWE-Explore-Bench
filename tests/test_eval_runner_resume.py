"""Resuming an interrupted run must score every case exactly once.

Results for one case are written to one file per top_k budget, one after the
other. An interrupt in that gap leaves a case in some budget files but not
others; resuming must not score the surviving rows *and* re-run the case.
"""
import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import eval_runner

TOP_K = "1,2"


@pytest.fixture(autouse=True)
def wide_console(monkeypatch):
    """Keep the summary table from being truncated to terminal width."""
    monkeypatch.setattr(eval_runner, "console", Console(width=400))


def _write_bench(path: Path, n: int = 4, regions: int | None = None) -> Path:
    """Bench where case i has i core regions, so per-case scores all differ.

    `regions` overrides that count for every case, giving a bench with the same
    instance_ids but different ground truth.
    """
    lines = []
    for i in range(1, n + 1):
        count = i if regions is None else regions
        regions_ = [{"path": f"f{j}.py", "start": 1, "end": 10} for j in range(count)]
        lines.append(
            json.dumps(
                {
                    "instance_id": f"case-{i}",
                    "problem_statement": "issue text",
                    "ground_truth": {
                        "read_core_regions": regions_,
                        "read_optional_regions": [],
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run(
    bench: Path,
    out_dir: Path,
    *extra: str,
    explorers: tuple[str, ...] = ("oracle",),
    top_k: str = TOP_K,
    output: Path | None = None,
):
    """Invoke the CLI over `bench`, writing one file per explorer and budget."""
    explorer_args = [arg for name in explorers for arg in ("--explorers", name)]
    return CliRunner().invoke(
        eval_runner.app,
        [
            "--bench", str(bench),
            *explorer_args,
            "--top-k", top_k,
            "--no-line-counts",
            "--repos", str(out_dir),
            "--output", str(output or out_dir / "{explorer}" / "top{k}.jsonl"),
            *extra,
        ],
    )


def _summary_rows(output: str) -> list[str]:
    """The rendered rows of the per-explorer results table."""
    return [ln.rstrip() for ln in output.splitlines() if ln.lstrip().startswith("│")]


def _flat(text: str) -> str:
    """Console output with rich's line wrapping undone."""
    return " ".join(text.split())


def _rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def test_resume_after_interrupt_matches_uninterrupted_run(tmp_path, monkeypatch):
    bench = _write_bench(tmp_path / "bench.jsonl")
    clean_dir, torn_dir = tmp_path / "clean", tmp_path / "torn"

    clean = _run(bench, clean_dir)
    assert clean.exit_code == 0

    # Interrupt case-3 after its top1.jsonl row is written, before top2.jsonl.
    append_row = eval_runner._append_row
    written = 0

    def interrupt_case_3(fh, row):
        nonlocal written
        if row["instance_id"] == "case-3":
            written += 1
            if written == 2:
                raise KeyboardInterrupt
        append_row(fh, row)

    monkeypatch.setattr(eval_runner, "_append_row", interrupt_case_3)
    interrupted = _run(bench, torn_dir)
    assert interrupted.exit_code == 130
    # The tear under test: case-3 reached top1.jsonl but not top2.jsonl.
    assert len(_rows(torn_dir / "oracle" / "top1.jsonl")) == 3
    assert len(_rows(torn_dir / "oracle" / "top2.jsonl")) == 2

    monkeypatch.setattr(eval_runner, "_append_row", append_row)
    resumed = _run(bench, torn_dir, "--resume")
    assert resumed.exit_code == 0

    assert _summary_rows(resumed.stdout) == _summary_rows(clean.stdout)
    for k in (1, 2):
        rel = Path("oracle") / f"top{k}.jsonl"
        assert _rows(torn_dir / rel) == _rows(clean_dir / rel)
        ids = [r["instance_id"] for r in _rows(torn_dir / rel)]
        assert sorted(ids) == [f"case-{i}" for i in range(1, 5)]


def test_rerun_without_resume_starts_the_output_files_over(tmp_path):
    bench = _write_bench(tmp_path / "bench.jsonl")
    out = tmp_path / "out"

    assert _run(bench, out).exit_code == 0
    assert _run(bench, out).exit_code == 0

    for k in (1, 2):
        ids = [r["instance_id"] for r in _rows(out / "oracle" / f"top{k}.jsonl")]
        assert sorted(ids) == [f"case-{i}" for i in range(1, 5)]


def _seed_results(tmp_path: Path, rows_per_k: dict[int, list[dict]]) -> str:
    """Write result files for the `oracle` explorer; returns the path template."""
    (tmp_path / "oracle").mkdir(parents=True, exist_ok=True)
    for k, rows in rows_per_k.items():
        path = tmp_path / "oracle" / f"top{k}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(tmp_path / "{explorer}" / "top{k}.jsonl")


def test_reconcile_drops_a_case_torn_between_budgets(tmp_path):
    done = {"instance_id": "case-1", "metrics": {"recall": 1.0}}
    torn = {"instance_id": "case-2", "metrics": {"recall": 0.5}}
    template = _seed_results(tmp_path, {1: [done, torn], 2: [done]})

    resumed_ids, kept = eval_runner._reconcile_resume_state(template, "oracle", [1, 2])

    assert resumed_ids == {"case-1"}
    assert kept == {1: [done], 2: [done]}
    assert _rows(tmp_path / "oracle" / "top1.jsonl") == [done]


def test_reconcile_collapses_rows_a_past_run_wrote_twice(tmp_path):
    first = {"instance_id": "case-1", "metrics": {"recall": 0.1}}
    second = {"instance_id": "case-1", "metrics": {"recall": 0.9}}
    template = _seed_results(tmp_path, {1: [first, second], 2: [first]})

    resumed_ids, kept = eval_runner._reconcile_resume_state(template, "oracle", [1, 2])

    assert resumed_ids == {"case-1"}
    assert kept[1] == [second]
    assert _rows(tmp_path / "oracle" / "top1.jsonl") == [second]


def test_resume_refuses_to_prune_against_a_missing_budget_file(tmp_path):
    """Adding a top_k must not wipe the budgets that already finished."""
    bench = _write_bench(tmp_path / "bench.jsonl")
    out = tmp_path / "out"
    assert _run(bench, out).exit_code == 0
    before = {k: _rows(out / "oracle" / f"top{k}.jsonl") for k in (1, 2)}

    added = _run(bench, out, "--resume", top_k="1,2,5")

    assert added.exit_code == 1
    assert "do not match the run being resumed" in _flat(added.stdout)
    assert {k: _rows(out / "oracle" / f"top{k}.jsonl") for k in (1, 2)} == before


def test_resume_repairs_a_line_left_half_written(tmp_path):
    """A hard kill can leave a fragment with no newline; the next append must not glue onto it."""
    done = {"instance_id": "case-1", "metrics": {"recall": 1.0}}
    template = _seed_results(tmp_path, {1: [done], 2: [done]})
    torn = tmp_path / "oracle" / "top1.jsonl"
    with torn.open("a", encoding="utf-8") as f:
        f.write('{"instance_id": "case-2", "met')

    resumed_ids, kept = eval_runner._reconcile_resume_state(template, "oracle", [1, 2])

    assert resumed_ids == {"case-1"}
    assert kept[1] == [done]
    assert _rows(torn) == [done]
    assert torn.read_text(encoding="utf-8").endswith("\n")


def test_budgets_sharing_one_output_file_stay_parseable(tmp_path):
    """An --output template without {k} points every budget at one file."""
    bench = _write_bench(tmp_path / "bench.jsonl")
    shared = tmp_path / "results.jsonl"

    result = _run(bench, tmp_path, output=shared)

    assert result.exit_code == 0
    assert len(_rows(shared)) == 8  # 4 cases x 2 budgets, none overwritten


def test_results_round_trip_non_ascii_paths_as_utf8(tmp_path):
    """Result files are UTF-8 whatever the platform default encoding is."""
    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        json.dumps(
            {
                "instance_id": "case-1",
                "problem_statement": "issue text",
                "ground_truth": {
                    "read_core_regions": [{"path": "src/测试.ets", "start": 1, "end": 10}],
                    "read_optional_regions": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"

    assert _run(bench, out).exit_code == 0

    written = out / "oracle" / "top1.jsonl"
    assert _rows(written)[0]["regions"][0]["path"] == "src/测试.ets"
    # Re-reading through the resume loader must see the same row back.
    assert eval_runner._load_existing_results(written) == _rows(written)


def test_resume_refuses_when_budgets_share_one_output_file(tmp_path):
    """A row records no top_k, so budgets sharing a file cannot be told apart."""
    bench = _write_bench(tmp_path / "bench.jsonl")
    shared = tmp_path / "results.jsonl"
    assert _run(bench, tmp_path, output=shared).exit_code == 0
    before = shared.read_text(encoding="utf-8")

    result = _run(bench, tmp_path, "--resume", output=shared)

    assert result.exit_code == 1
    assert "one file per explorer and budget" in _flat(result.stdout)
    assert shared.read_text(encoding="utf-8") == before


def _seed_two_explorers_into_one_file(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    """Run oracle and random into a single {explorer}-less output file."""
    bench = _write_bench(tmp_path / "bench.jsonl")
    out = tmp_path / "out"
    shared = out / "top{k}.jsonl"
    assert _run(bench, out, explorers=("oracle", "random"), top_k="1", output=shared).exit_code == 0
    written = out / "top1.jsonl"
    return bench, out, shared, written.read_text(encoding="utf-8")


def test_resume_refuses_when_explorers_share_one_output_file(tmp_path):
    """Without {explorer}, one explorer would score and then delete another's rows."""
    bench, out, shared, before = _seed_two_explorers_into_one_file(tmp_path)

    result = _run(bench, out, "--resume", explorers=("oracle", "random"), top_k="1", output=shared)

    assert result.exit_code == 1
    assert {r["explorer"] for r in _rows(out / "top1.jsonl")} == {"oracle", "random"}
    assert (out / "top1.jsonl").read_text(encoding="utf-8") == before


def test_loader_ignores_lines_that_are_not_result_rows(tmp_path):
    path = tmp_path / "top1.jsonl"
    path.write_text('123\n"a string"\nnull\n{"instance_id": "case-1"}\n[]\n', encoding="utf-8")

    assert eval_runner._load_existing_results(path) == [{"instance_id": "case-1"}]


def test_resume_counts_token_usage_once_per_case(tmp_path):
    """Usage is per case, not per budget file, however many budgets hold the row."""
    bench = _write_bench(tmp_path / "bench.jsonl", n=2)
    out = tmp_path / "out"
    rows = [
        {"instance_id": "case-1", "metrics": {}, "token_usage": {"input": 10000, "output": 2000}},
        {"instance_id": "case-2", "metrics": {}, "token_usage": {"input": 5000, "output": 1000}},
    ]
    _seed_results(out, {1: rows, 2: rows})

    result = _run(bench, out, "--resume")

    assert result.exit_code == 0
    assert "15,000" in result.stdout  # summed once across both budget files
    assert "30,000" not in result.stdout


def test_resume_refuses_a_file_holding_another_explorers_rows(tmp_path):
    """A file written by an earlier, wider --explorers set is not ours to prune."""
    bench, out, shared, before = _seed_two_explorers_into_one_file(tmp_path)

    result = _run(bench, out, "--resume", top_k="1", output=shared)

    assert result.exit_code == 1
    assert "also holds rows from random" in _flat(result.stdout)
    assert (out / "top1.jsonl").read_text(encoding="utf-8") == before


def test_fresh_run_with_no_cases_left_clears_stale_output(tmp_path):
    """A fresh run owns its output files even when it has nothing to run."""
    bench = _write_bench(tmp_path / "bench.jsonl")
    out = tmp_path / "out"
    assert _run(bench, out).exit_code == 0
    assert len(_rows(out / "oracle" / "top1.jsonl")) == 4

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    assert _run(empty, out).exit_code == 0
    for k in (1, 2):
        assert _rows(out / "oracle" / f"top{k}.jsonl") == []


def test_interrupted_fresh_run_does_not_leave_a_later_explorer_resumable(tmp_path, monkeypatch):
    """A fresh run owns every file it will write, not just the ones it reached."""
    out = tmp_path / "out"
    both = ("random", "oracle")
    old_bench = _write_bench(tmp_path / "old.jsonl")
    assert _run(old_bench, out, explorers=both, top_k="1").exit_code == 0
    stale = _rows(out / "oracle" / "top1.jsonl")

    # Same cases, different ground truth: oracle must score differently now.
    new_bench = _write_bench(tmp_path / "new.jsonl", regions=4)
    append_row = eval_runner._append_row

    def interrupt_during_first_explorer(fh, row):
        if row["explorer"] == "random":
            raise KeyboardInterrupt
        append_row(fh, row)

    monkeypatch.setattr(eval_runner, "_append_row", interrupt_during_first_explorer)
    assert _run(new_bench, out, explorers=both, top_k="1").exit_code == 130
    monkeypatch.setattr(eval_runner, "_append_row", append_row)
    assert _rows(out / "oracle" / "top1.jsonl") != stale  # not the old run's rows

    assert _run(new_bench, out, "--resume", explorers=both, top_k="1").exit_code == 0
    clean = tmp_path / "clean"
    assert _run(new_bench, clean, explorers=both, top_k="1").exit_code == 0
    assert _rows(out / "oracle" / "top1.jsonl") == _rows(clean / "oracle" / "top1.jsonl")
