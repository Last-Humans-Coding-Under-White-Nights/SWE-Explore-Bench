"""Every attempted case ends in a recorded outcome, and every result file says
which configuration produced it.

The OpenCode tests drive the real explorer subprocess path against a fake
``opencode`` (written below) whose behaviour is chosen per test through
environment variables, which the explorer passes through to the CLI.
"""
import json
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import eval_runner
from explorers.base import BINARY_NOT_FOUND, ERROR, PROVIDER_ERROR, SUCCESS, TIMEOUT

SECRET = "sk-live-do-not-record-0123456789"

FAKE_OPENCODE = r'''
import json, os, sys, time

args = sys.argv[1:]
if args == ["--version"]:
    print(os.environ.get("FAKE_OC_VERSION", "1.18.29"))
    sys.exit(0)
if args[:2] == ["debug", "config"]:
    if os.environ.get("FAKE_OC_CONFIG_FAILS"):
        sys.stderr.write("config unavailable\n")
        sys.exit(1)
    # Resolved configuration, credentials substituted in, as the real CLI prints it.
    config = {
        "model": os.environ.get("FAKE_OC_CONFIG_MODEL", "fake/default"),
        "provider": {"fake": {"options": {"apiKey": os.environ.get("FAKE_OC_KEY", "")}}},
        "mcp": {"serena": {"type": "local", "enabled": True,
                           "environment": {"SERENA_HOME": "/srv/serena"}}},
    }
    if os.environ.get("FAKE_OC_AGENT_MODEL"):
        config["agent"] = {"build": {"model": os.environ["FAKE_OC_AGENT_MODEL"]}}
    print(json.dumps(config, indent=2))
    sys.exit(0)
if args[:1] == ["db"]:
    print("[]")
    sys.exit(0)

with open(os.environ["FAKE_OC_ARGV_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(args) + "\n")
sys.stdin.read()
usage = {"type": "step_finish", "part": {"type": "step-finish", "tokens": {
    "input": 700, "output": 50, "reasoning": 0, "cache": {"read": 0, "write": 0}}}}
print(json.dumps(usage), flush=True)
mode = os.environ.get("FAKE_OC_MODE", "answer")
if mode == "crash":
    # A bad flag or a broken config: non-zero, and nothing on the stream.
    sys.stderr.write("unknown agent 'nope'\n")
    sys.exit(1)
if mode == "error":
    print(json.dumps({"type": "error", "error": {"name": "APIError", "data": {
        "message": "Cannot connect to API"}}}), flush=True)
elif mode == "timeout":
    time.sleep(10)
else:
    print(json.dumps({"type": "text", "part": {"text": "RELEVANT_FILES:\n- f0.py:1-10"}}))
'''


@pytest.fixture(autouse=True)
def wide_console(monkeypatch):
    """Keep the summary table from being truncated to terminal width."""
    monkeypatch.setattr(eval_runner, "console", Console(width=400))


@pytest.fixture
def fake_opencode(tmp_path: Path, monkeypatch) -> str:
    """A launcher for the fake CLI that runs on every CI platform."""
    script = tmp_path / "fake_opencode.py"
    script.write_text(FAKE_OPENCODE, encoding="utf-8")
    if sys.platform == "win32":
        launcher = tmp_path / "fake-opencode.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        launcher = tmp_path / "fake-opencode"
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
        )
        launcher.chmod(0o755)
    monkeypatch.setenv("FAKE_OC_ARGV_LOG", str(tmp_path / "argv.log"))
    monkeypatch.setenv("FAKE_OC_KEY", SECRET)
    return str(launcher)


def _write_bench(tmp_path: Path, n: int = 2, with_repo: int | None = None) -> Path:
    """`n` cases, each with its own checkout under tmp_path/repos (the first
    `with_repo` only, when given)."""
    lines = []
    for i in range(1, n + 1):
        if with_repo is None or i <= with_repo:
            repo = tmp_path / "repos" / f"case-{i}"
            repo.mkdir(parents=True, exist_ok=True)
            (repo / "f0.py").write_text("x = 1\n" * 20, encoding="utf-8")
        lines.append(json.dumps({
            "instance_id": f"case-{i}",
            "repo_dir": f"case-{i}",
            "problem_statement": "issue text",
            "ground_truth": {
                "read_core_regions": [{"path": "f0.py", "start": 1, "end": 10}],
                "read_optional_regions": [],
            },
        }))
    bench = tmp_path / "bench.jsonl"
    bench.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return bench


def _run(tmp_path: Path, bench: Path, *extra: str, explorer: str = "oracle"):
    return CliRunner().invoke(
        eval_runner.app,
        [
            "--bench", str(bench),
            "--explorers", explorer,
            "--top-k", "1",
            "--no-line-counts",
            "--repos", str(tmp_path / "repos"),
            "--output", str(tmp_path / "out" / "{explorer}" / "top{k}.jsonl"),
            *extra,
        ],
    )


def _run_opencode(tmp_path: Path, bench: Path, binary: str, *extra: str):
    return _run(tmp_path, bench, "--opencode-bin", binary, *extra, explorer="opencode")


def _rows(tmp_path: Path, explorer: str) -> list[dict]:
    text = (tmp_path / "out" / explorer / "top1.jsonl").read_text(encoding="utf-8")
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def _sidecar(tmp_path: Path, explorer: str) -> dict:
    path = tmp_path / "out" / explorer / "top1.manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))["explorers"][explorer]


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_error_only_stream_is_a_scored_provider_error(tmp_path, fake_opencode, monkeypatch):
    """Exit 0 with only error events: a zero with its reason, spend kept."""
    monkeypatch.setenv("FAKE_OC_MODE", "error")
    bench = _write_bench(tmp_path)

    result = _run_opencode(tmp_path, bench, fake_opencode)

    assert result.exit_code == 0, result.output
    rows = _rows(tmp_path, "opencode")
    assert [r["outcome"] for r in rows] == [PROVIDER_ERROR, PROVIDER_ERROR]
    assert all("Cannot connect to API" in r["error"] for r in rows)
    assert all(r["metrics"]["recall"] == 0.0 and r["regions"] == [] for r in rows)
    assert all(r["token_usage"]["input"] == 700 for r in rows)
    summary = _sidecar(tmp_path, "opencode")["summary"]["1"]
    assert summary["attempted"] == 2
    assert summary["outcomes"] == {PROVIDER_ERROR: 2}
    assert summary["completion_rate"] == 0.0
    assert summary["token_usage"]["input"] == 1400


def test_timeout_is_a_scored_failure_that_keeps_its_usage(tmp_path, fake_opencode, monkeypatch):
    """The output produced before the kill is read for its spend, not copied."""
    monkeypatch.setenv("FAKE_OC_MODE", "timeout")
    bench = _write_bench(tmp_path, n=1)

    result = _run_opencode(tmp_path, bench, fake_opencode, "--opencode-timeout", "3")

    assert result.exit_code == 0, result.output
    [row] = _rows(tmp_path, "opencode")
    assert row["outcome"] == TIMEOUT
    assert row["error"] == "OpenCode CLI timed out after 3s"
    assert "step_finish" not in row["error"]  # the agent's stream stays in the log
    assert row["token_usage"]["input"] == 700
    assert _sidecar(tmp_path, "opencode")["summary"]["1"]["completion_rate"] == 0.0


def test_a_missing_binary_stops_the_run_at_startup(tmp_path):
    """A setup error, not 300 zero rows: `describe()` learns it before case 1."""
    bench = _write_bench(tmp_path, n=1)

    result = _run_opencode(tmp_path, bench, str(tmp_path / "no-such-opencode"))

    assert result.exit_code == 1
    assert "OpenCode CLI not found" in _flat(result.output)
    assert "--opencode-bin" in _flat(result.output)
    assert not (tmp_path / "out" / "opencode" / "top1.jsonl").exists()


def test_success_rows_name_their_outcome_and_the_pinned_model(tmp_path, fake_opencode, monkeypatch):
    """With no --opencode-model, the configured model is still named on argv."""
    monkeypatch.setenv("FAKE_OC_CONFIG_MODEL", "fake/configured")
    bench = _write_bench(tmp_path)

    result = _run_opencode(tmp_path, bench, fake_opencode)

    assert result.exit_code == 0, result.output
    rows = _rows(tmp_path, "opencode")
    assert [r["outcome"] for r in rows] == [SUCCESS, SUCCESS]
    assert all(r["error"] is None for r in rows)
    assert _sidecar(tmp_path, "opencode")["manifest"]["explorer_config"]["model"] == "fake/configured"
    argv = [json.loads(ln) for ln in (tmp_path / "argv.log").read_text(encoding="utf-8").splitlines()]
    assert all(a[a.index("--model") + 1] == "fake/configured" for a in argv)
    assert _sidecar(tmp_path, "opencode")["summary"]["1"]["completion_rate"] == 1.0


def test_manifest_records_the_configuration_and_no_secret(tmp_path, fake_opencode):
    bench = _write_bench(tmp_path)

    result = _run_opencode(
        tmp_path, bench, fake_opencode, "--opencode-model", "fake/pinned",
        "--opencode-prompt-additions", "Prefer the call graph.",
    )

    assert result.exit_code == 0, result.output
    manifest = _sidecar(tmp_path, "opencode")["manifest"]
    config = manifest["explorer_config"]
    assert config["model"] == "fake/pinned"
    assert config["model_source"] == "flag"
    assert config["cli_version"] == "1.18.29"
    assert len(config["prompt_sha256"]) == 64
    assert len(config["resolved_config_sha256"]) == 64
    assert len(manifest["bench_sha256"]) == 64
    written = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "out").rglob("*") if p.is_file())
    assert SECRET not in written


def test_resume_with_a_different_model_is_refused(tmp_path, fake_opencode, monkeypatch):
    bench = _write_bench(tmp_path, n=3)

    # Interrupt after the first case so there is something left to resume.
    append_row = eval_runner._append_row

    def interrupt_second_case(fh, row):
        if row["instance_id"] == "case-2":
            raise KeyboardInterrupt
        append_row(fh, row)

    monkeypatch.setattr(eval_runner, "_append_row", interrupt_second_case)
    first = _run_opencode(tmp_path, bench, fake_opencode, "--opencode-model", "fake/one")
    assert first.exit_code == 130
    monkeypatch.setattr(eval_runner, "_append_row", append_row)
    before = _rows(tmp_path, "opencode")

    changed = _run_opencode(
        tmp_path, bench, fake_opencode, "--opencode-model", "fake/two", "--resume"
    )

    assert changed.exit_code == 1
    message = _flat(changed.output)
    assert "cannot resume opencode" in message
    assert "model: 'fake/one' -> 'fake/two'" in message
    assert _rows(tmp_path, "opencode") == before  # nothing mixed in

    same = _run_opencode(
        tmp_path, bench, fake_opencode, "--opencode-model", "fake/one", "--resume"
    )
    assert same.exit_code == 0, same.output
    assert [r["instance_id"] for r in _rows(tmp_path, "opencode")] == [
        "case-1", "case-2", "case-3"
    ]


def test_resume_with_a_different_cli_version_is_refused(tmp_path, fake_opencode, monkeypatch):
    bench = _write_bench(tmp_path, n=1)
    assert _run_opencode(tmp_path, bench, fake_opencode).exit_code == 0

    monkeypatch.setenv("FAKE_OC_VERSION", "1.19.0")
    result = _run_opencode(tmp_path, bench, fake_opencode, "--resume")

    assert result.exit_code == 1
    assert "cli_version: '1.18.29' -> '1.19.0'" in _flat(result.output)


def test_rotating_the_api_key_does_not_block_resume(tmp_path, fake_opencode, monkeypatch):
    bench = _write_bench(tmp_path, n=1)
    assert _run_opencode(tmp_path, bench, fake_opencode).exit_code == 0

    monkeypatch.setenv("FAKE_OC_KEY", "sk-live-rotated")
    result = _run_opencode(tmp_path, bench, fake_opencode, "--resume")

    assert result.exit_code == 0, result.output


def test_resume_against_a_changed_bench_is_refused(tmp_path):
    bench = _write_bench(tmp_path)
    assert _run(tmp_path, bench).exit_code == 0

    bench.write_text(bench.read_text(encoding="utf-8").replace('"end": 10', '"end": 12'),
                     encoding="utf-8")
    result = _run(tmp_path, bench, "--resume")

    assert result.exit_code == 1
    assert "bench_sha256" in _flat(result.output)


def test_resume_with_a_different_chunk_size_is_refused(tmp_path):
    """Chunking decides what a retrieval explorer can return at all."""
    bench = _write_bench(tmp_path)
    assert _run(tmp_path, bench, "--chunk-size", "5", explorer="bm25").exit_code == 0

    result = _run(tmp_path, bench, "--chunk-size", "20", "--resume", explorer="bm25")

    assert result.exit_code == 1
    assert "chunk_size: 5 -> 20" in _flat(result.output)


def test_resume_with_a_changed_issue_is_refused(tmp_path):
    """The issue text is the query every explorer is given."""
    bench = _write_bench(tmp_path)
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps({"case-1": "first", "case-2": "first"}), encoding="utf-8")
    assert _run(tmp_path, bench, "--issue-map", str(issues)).exit_code == 0

    issues.write_text(json.dumps({"case-1": "rewritten", "case-2": "first"}),
                      encoding="utf-8")
    result = _run(tmp_path, bench, "--issue-map", str(issues), "--resume")

    assert result.exit_code == 1
    assert "issues_sha256" in _flat(result.output)


def test_resuming_a_narrowed_selection_reports_on_that_selection(tmp_path):
    """--limit on a resume must not count the rows it excluded."""
    bench = _write_bench(tmp_path, n=3)
    assert _run(tmp_path, bench).exit_code == 0

    result = _run(tmp_path, bench, "--resume", "--limit", "1")

    assert result.exit_code == 0, result.output
    summary = _sidecar(tmp_path, "oracle")["summary"]["1"]
    assert summary["cases"] == 1
    assert summary["attempted"] == 1
    assert summary["completion_rate"] == 1.0
    # The excluded rows are still in the file; a narrower run reads less,
    # it does not delete.
    assert len(_rows(tmp_path, "oracle")) == 3


def test_results_without_a_manifest_are_adopted_on_resume(tmp_path):
    """Files written before manifests existed can still be continued."""
    bench = _write_bench(tmp_path)
    assert _run(tmp_path, bench).exit_code == 0
    (tmp_path / "out" / "oracle" / "top1.manifest.json").unlink()

    result = _run(tmp_path, bench, "--resume")

    assert result.exit_code == 0, result.output
    assert "record no run manifest" in _flat(result.output)
    assert _sidecar(tmp_path, "oracle")["manifest"]["explorer"] == "oracle"


def test_an_unclassified_exception_is_a_scored_error(tmp_path, monkeypatch):
    """A crash costs a case exactly what an empty answer does, never less."""
    from explorers.baselines import OracleExplorer

    explore = OracleExplorer.explore

    def crash_on_case_2(self, *, instance_id, query, top_k=5):
        if instance_id == "case-2":
            raise ValueError("explorer bug")
        return explore(self, instance_id=instance_id, query=query, top_k=top_k)

    monkeypatch.setattr(OracleExplorer, "explore", crash_on_case_2)
    bench = _write_bench(tmp_path)

    result = _run(tmp_path, bench)

    assert result.exit_code == 0, result.output
    rows = {r["instance_id"]: r for r in _rows(tmp_path, "oracle")}
    assert rows["case-1"]["outcome"] == SUCCESS
    assert rows["case-2"]["outcome"] == ERROR
    assert rows["case-2"]["error"] == "ValueError: explorer bug"
    assert rows["case-2"]["metrics"]["recall"] == 0.0
    summary = _sidecar(tmp_path, "oracle")["summary"]["1"]
    assert summary["completion_rate"] == 0.5
    assert summary["metrics"]["recall"] == pytest.approx(rows["case-1"]["metrics"]["recall"] / 2)
    assert "completion rate=50.0%" in result.output


def test_a_case_without_a_checkout_is_counted_as_not_attempted(tmp_path, fake_opencode):
    bench = _write_bench(tmp_path, n=2, with_repo=1)

    result = _run_opencode(tmp_path, bench, fake_opencode)

    assert result.exit_code == 0, result.output
    assert [r["instance_id"] for r in _rows(tmp_path, "opencode")] == ["case-1"]
    summary = _sidecar(tmp_path, "opencode")["summary"]["1"]
    assert (summary["cases"], summary["attempted"], summary["not_attempted"]) == (2, 1, 1)
    assert summary["completion_rate"] == 0.5


def test_a_broken_usage_collector_surfaces_its_own_error(tmp_path, monkeypatch):
    """The handler must never mask the real failure with a NameError."""
    def broken_collector():
        raise RuntimeError("collector could not start")

    monkeypatch.setattr(eval_runner, "usage_collector", broken_collector)
    bench = _write_bench(tmp_path, n=1)

    result = _run(tmp_path, bench)

    assert isinstance(result.exception, RuntimeError)
    assert "collector could not start" in str(result.exception)


def test_a_missing_checkout_is_a_scored_error_when_it_may_not_be_skipped(tmp_path):
    """--no-skip-missing-repo asks for the case to be run, so it is scored."""
    bench = _write_bench(tmp_path, n=2, with_repo=1)

    result = _run(tmp_path, bench, "--no-skip-missing-repo", explorer="bm25")

    assert result.exit_code == 0, result.output
    rows = {r["instance_id"]: r for r in _rows(tmp_path, "bm25")}
    assert rows["case-1"]["outcome"] == SUCCESS
    assert rows["case-2"]["outcome"] == ERROR
    assert rows["case-2"]["error"] == "no checkout for case-2 under repos"
    # A row is read on another machine: no local absolute path in it.
    assert str(tmp_path) not in rows["case-2"]["error"]
    assert rows["case-2"]["metrics"]["recall"] == 0.0
    summary = _sidecar(tmp_path, "bm25")["summary"]["1"]
    assert (summary["cases"], summary["attempted"], summary["not_attempted"]) == (2, 2, 0)


def test_an_unreadable_manifest_does_not_crash_a_resume(tmp_path):
    """A sidecar is a file on disk: `explorers` may be anything at all."""
    bench = _write_bench(tmp_path)
    assert _run(tmp_path, bench).exit_code == 0
    sidecar = tmp_path / "out" / "oracle" / "top1.manifest.json"
    sidecar.write_text(json.dumps({"schema": 1, "explorers": [{"oracle": {}}]}),
                       encoding="utf-8")

    result = _run(tmp_path, bench, "--resume")

    assert result.exit_code == 0, result.output
    assert "record no run manifest" in _flat(result.output)


def test_an_interrupted_write_leaves_no_temporary_file(tmp_path):
    """The half-written file must not be left beside the results it is not."""
    target = tmp_path / "out.jsonl"
    target.write_text("kept\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        with eval_runner._atomic_target(target) as tmp:
            tmp.write_text("half", encoding="utf-8")
            raise RuntimeError("interrupted")

    assert target.read_text(encoding="utf-8") == "kept\n"
    assert [p.name for p in tmp_path.iterdir()] == ["out.jsonl"]


def test_the_issue_map_does_not_depend_on_directory_order(tmp_path, monkeypatch):
    """Two trajectories for one instance: the same query on every filesystem.

    `rglob` hands back whatever order the directory is stored in, and the
    first file read wins, so the order is forced here to the one a sorted
    read must not follow.
    """
    trajs = tmp_path / "trajs"
    for name, issue in (("a-run", "first"), ("b-run", "second")):
        path = trajs / name
        path.mkdir(parents=True)
        (path / "case-1.json").write_text(
            json.dumps({"info": {"instance_id": "case-1", "issue": issue}}),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        Path, "rglob", lambda self, pattern: iter(sorted(
            (p for p in self.glob("**/" + pattern)), reverse=True
        )),
    )

    assert eval_runner._load_issue_map(trajs) == {"case-1": "first"}


def test_a_changed_retrieval_model_is_refused_on_resume(tmp_path):
    """`rag` embeds the issue and the files: its model identifies the run."""
    bench = _write_bench(tmp_path, n=1, with_repo=0)
    assert _run(tmp_path, bench, "--embed-model", "org/a", explorer="rag").exit_code == 0
    assert _sidecar(tmp_path, "rag")["manifest"]["explorer_config"] == {"model": "org/a"}

    result = _run(tmp_path, bench, "--embed-model", "org/b", "--resume", explorer="rag")

    assert result.exit_code == 1
    assert "explorer_config.model: 'org/a' -> 'org/b'" in _flat(result.output)


def test_a_changed_embedding_backend_is_refused_on_resume(tmp_path):
    """Chunk settings alone did not say which model scored the chunks."""
    bench = _write_bench(tmp_path, n=1, with_repo=0)
    assert _run(tmp_path, bench, "--embed-backend", "openai", explorer="embed").exit_code == 0
    config = _sidecar(tmp_path, "embed")["manifest"]["explorer_config"]
    assert config["backend"] == "openai"
    assert config["model"] == eval_runner.DEFAULT_EMBED_MODEL

    result = _run(tmp_path, bench, "--embed-backend", "sentence_transformers",
                  "--resume", explorer="embed")

    assert result.exit_code == 1
    assert ("explorer_config.backend: 'openai' -> 'sentence_transformers'"
            in _flat(result.output))


def _argv(tmp_path: Path) -> list[list[str]]:
    text = (tmp_path / "argv.log").read_text(encoding="utf-8")
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def test_resume_runs_the_cases_that_failed_again(tmp_path, fake_opencode, monkeypatch):
    """A rate limit is a reason to retry, not a permanent zero."""
    monkeypatch.setenv("FAKE_OC_MODE", "error")
    bench = _write_bench(tmp_path, n=2)
    assert _run_opencode(tmp_path, bench, fake_opencode).exit_code == 0
    assert [r["outcome"] for r in _rows(tmp_path, "opencode")] == [PROVIDER_ERROR] * 2

    monkeypatch.setenv("FAKE_OC_MODE", "answer")
    result = _run_opencode(tmp_path, bench, fake_opencode, "--resume")

    assert result.exit_code == 0, result.output
    rows = _rows(tmp_path, "opencode")
    # Replaced, not appended: one row per case, and both now succeeded.
    assert [r["instance_id"] for r in rows] == ["case-1", "case-2"]
    assert [r["outcome"] for r in rows] == [SUCCESS, SUCCESS]
    assert "2 case(s) that failed before will be run again" in _flat(result.output)
    assert _sidecar(tmp_path, "opencode")["summary"]["1"]["completion_rate"] == 1.0


def test_no_retry_failed_keeps_a_failed_row_as_done(tmp_path, fake_opencode, monkeypatch):
    monkeypatch.setenv("FAKE_OC_MODE", "error")
    bench = _write_bench(tmp_path, n=2)
    assert _run_opencode(tmp_path, bench, fake_opencode).exit_code == 0

    monkeypatch.setenv("FAKE_OC_MODE", "answer")
    result = _run_opencode(tmp_path, bench, fake_opencode, "--resume", "--no-retry-failed")

    assert result.exit_code == 0, result.output
    assert [r["outcome"] for r in _rows(tmp_path, "opencode")] == [PROVIDER_ERROR] * 2
    assert "All instances already completed" in _flat(result.output)


def test_a_successful_row_is_not_run_again(tmp_path, fake_opencode, monkeypatch):
    """Retrying failures must not turn a resume into a rerun of everything."""
    bench = _write_bench(tmp_path, n=2)
    assert _run_opencode(tmp_path, bench, fake_opencode).exit_code == 0
    runs_before = len(_argv(tmp_path))

    result = _run_opencode(tmp_path, bench, fake_opencode, "--resume")

    assert result.exit_code == 0, result.output
    assert len(_argv(tmp_path)) == runs_before
    assert "All instances already completed" in _flat(result.output)


def test_a_non_zero_exit_without_error_events_is_not_a_provider_error(
    tmp_path, fake_opencode, monkeypatch
):
    """A bad flag is the harness's fault, not the provider's."""
    monkeypatch.setenv("FAKE_OC_MODE", "crash")
    bench = _write_bench(tmp_path, n=1)

    result = _run_opencode(tmp_path, bench, fake_opencode)

    assert result.exit_code == 0, result.output
    [row] = _rows(tmp_path, "opencode")
    assert row["outcome"] == ERROR
    assert row["error"] == "OpenCode CLI failed (rc=1)"
    # The stderr tail is in the log, not in a file that may be published.
    assert "unknown agent" not in row["error"]


def test_a_row_leaves_the_configuration_to_the_manifest(tmp_path, fake_opencode):
    bench = _write_bench(tmp_path, n=1)

    assert _run_opencode(tmp_path, bench, fake_opencode).exit_code == 0

    [row] = _rows(tmp_path, "opencode")
    assert list(row) == [
        "instance_id", "explorer", "outcome", "error", "regions",
        "metrics", "num_regions", "token_usage",
    ]


def test_the_manifest_records_no_absolute_local_path(tmp_path, fake_opencode):
    bench = _write_bench(tmp_path, n=1)

    assert _run_opencode(tmp_path, bench, fake_opencode).exit_code == 0

    manifest = _sidecar(tmp_path, "opencode")["manifest"]
    assert manifest["recorded"]["bench_path"] == "bench.jsonl"
    assert str(tmp_path) not in json.dumps(manifest)


def test_pinning_the_model_the_config_had_already_chosen_still_resumes(
    tmp_path, fake_opencode, monkeypatch
):
    """`model_source` says how the model was chosen, not what ran."""
    monkeypatch.setenv("FAKE_OC_CONFIG_MODEL", "fake/configured")
    bench = _write_bench(tmp_path, n=1)
    assert _run_opencode(tmp_path, bench, fake_opencode).exit_code == 0
    assert _sidecar(tmp_path, "opencode")["manifest"]["explorer_config"][
        "model_source"] == "config"

    result = _run_opencode(tmp_path, bench, fake_opencode, "--resume",
                           "--opencode-model", "fake/configured")

    assert result.exit_code == 0, result.output
    assert "model_source" not in _flat(result.output)


def test_a_probe_that_could_not_run_does_not_refuse_a_resume(
    tmp_path, fake_opencode, monkeypatch
):
    """One slow startup must not cost every later resume."""
    bench = _write_bench(tmp_path, n=1)
    assert _run_opencode(tmp_path, bench, fake_opencode).exit_code == 0

    monkeypatch.setenv("FAKE_OC_CONFIG_FAILS", "1")
    result = _run_opencode(tmp_path, bench, fake_opencode, "--resume")

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "resolved_config_sha256 is unknown on one side" in flat
    assert "was produced by a different configuration" not in flat


def test_the_agent_whose_model_was_taken_is_named_on_the_command_line(
    tmp_path, fake_opencode, monkeypatch
):
    """The fallback agent is a default, not a promise: pin what we assumed."""
    monkeypatch.setenv("FAKE_OC_AGENT_MODEL", "fake/agent-model")
    bench = _write_bench(tmp_path, n=1)

    result = _run_opencode(tmp_path, bench, fake_opencode)

    assert result.exit_code == 0, result.output
    [argv] = [a for a in _argv(tmp_path) if a[0] == "run"]
    assert argv[argv.index("--model") + 1] == "fake/agent-model"
    assert argv[argv.index("--agent") + 1] == "build"


def test_the_model_the_config_chose_pins_no_agent(tmp_path, fake_opencode):
    """Nothing was assumed, so the CLI keeps its own choice of agent."""
    bench = _write_bench(tmp_path, n=1)

    assert _run_opencode(tmp_path, bench, fake_opencode).exit_code == 0

    [argv] = [a for a in _argv(tmp_path) if a[0] == "run"]
    assert "--agent" not in argv


def test_a_zero_token_row_is_not_counted_as_a_case_with_usage(tmp_path):
    """A fresh run counts a case whose tool reported nothing as no usage at
    all; a resume counted the row and depressed the per-case mean."""
    bench = _write_bench(tmp_path, n=2)
    assert _run(tmp_path, bench).exit_code == 0
    assert _sidecar(tmp_path, "oracle")["summary"]["1"]["token_usage_cases"] == 0
    # A tool that reported usage of zero writes the object, not `null`.
    zeroed = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
              "reasoning": 0, "total": 0}
    path = tmp_path / "out" / "oracle" / "top1.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln]
    for row in rows:
        row["token_usage"] = dict(zeroed)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    result = _run(tmp_path, bench, "--resume")

    assert result.exit_code == 0, result.output
    assert _sidecar(tmp_path, "oracle")["summary"]["1"]["token_usage_cases"] == 0


def test_a_sidecar_whose_summary_is_null_is_rewritten(tmp_path):
    """The crash landed after every case had run, which is the worst time."""
    bench = _write_bench(tmp_path, n=1)
    assert _run(tmp_path, bench).exit_code == 0
    sidecar = tmp_path / "out" / "oracle" / "top1.manifest.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["explorers"]["oracle"]["summary"] = None
    sidecar.write_text(json.dumps(data), encoding="utf-8")

    result = _run(tmp_path, bench, "--resume")

    assert result.exit_code == 0, result.output
    assert _sidecar(tmp_path, "oracle")["summary"]["1"]["attempted"] == 1


def test_a_narrowed_resume_keeps_the_failed_rows_it_will_not_run(
    tmp_path, fake_opencode, monkeypatch
):
    """Retry is decided over the file; only the selection is actually run."""
    monkeypatch.setenv("FAKE_OC_MODE", "error")
    bench = _write_bench(tmp_path, n=3)
    assert _run_opencode(tmp_path, bench, fake_opencode).exit_code == 0
    before = {r["instance_id"]: r for r in _rows(tmp_path, "opencode")}

    monkeypatch.setenv("FAKE_OC_MODE", "answer")
    result = _run_opencode(tmp_path, bench, fake_opencode, "--resume", "--limit", "1")

    assert result.exit_code == 0, result.output
    rows = {r["instance_id"]: r for r in _rows(tmp_path, "opencode")}
    assert rows["case-1"]["outcome"] == SUCCESS
    # The two the run never selected keep their rows, their reason and the
    # tokens they already spent.
    assert [rows["case-2"], rows["case-3"]] == [before["case-2"], before["case-3"]]
    assert "1 case(s) that failed before will be run again" in _flat(result.output)
    # And the next unnarrowed resume still retries them.
    assert _run_opencode(tmp_path, bench, fake_opencode, "--resume").exit_code == 0
    assert [r["outcome"] for r in _rows(tmp_path, "opencode")] == [SUCCESS] * 3
