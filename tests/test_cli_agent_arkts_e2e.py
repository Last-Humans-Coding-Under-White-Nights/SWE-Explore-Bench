"""End-to-end ArkTS localization through the OpenCode / DevEco Code explorers.

What each driver does, how the checkout is prepared and how to enable the
real-CLI run are documented in configs/cli_agents/README.md, "Tests". This
module only adds the mechanics: a launcher for the scripted stand-in and a
recorder that keeps the CLI's event stream for inspection.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from explorers._cli_agent_base import BaseCliAgentExplorer
from explorers.deveco import DevEcoExplorer
from explorers.opencode import OpenCodeExplorer
from explorers.parsing import iter_events, usage_collector

TESTS_DIR = Path(__file__).resolve().parent
FIXTURE = TESTS_DIR / "fixtures" / "arkts_app"
PROFILES_ROOT = TESTS_DIR.parent / "configs" / "cli_agents"
SCRIPTED_AGENT = TESTS_DIR / "scripted_cli_agent.py"
CASES = json.loads((FIXTURE / "expected_locations.json").read_text(encoding="utf-8"))

#: Fixture files that would hand the agent the answers; never copied.
ANSWER_KEY = ("expected_locations.json", "README.md")
#: Serena keeps its language-server cache here; it is not part of the checkout.
SERENA_DIR = ".serena"

SMOKE_VAR = "SWE_EXPLORE_CLI_SMOKE"
PROXY_VARS = ("ACADEMIC_API_BASE", "ACADEMIC_API_KEY")
SCRIPTED_CLI_VAR = "SCRIPTED_AGENT_CLI"
SERENA_HOME_VAR = "SWE_EXPLORE_SERENA_HOME"
MCP_TOOL_PREFIX = "serena_"
#: Serena tools that navigate code. Only these count as semantic navigation;
#: its configuration, memory and onboarding tools are administrative.
MCP_NAVIGATION_TOOLS = frozenset({
    "serena_find_symbol", "serena_find_declaration", "serena_find_implementations",
    "serena_find_referencing_symbols", "serena_get_symbols_overview",
    "serena_search_for_pattern",
})
#: parse_relevant_files encodes "the whole file" as end == -1.
WHOLE_FILE = -1
VARIANTS = ("arkts", "arkts-no-mcp")
PAGE = "entry/src/main/ets/pages/Index.ets"
IMPORTED_MODULE = "entry/src/main/ets/model/DataSource.ets"

# The explorer module shares this module object, so keep the original before
# the recorder replaces the attribute for the duration of a run.
_REAL_RUN = subprocess.run

Region = tuple[str, int, int]


@dataclass(frozen=True)
class CliSpec:
    name: str
    explorer_cls: type[BaseCliAgentExplorer]
    bin_env_var: str
    timeout_env_var: str


CLI_SPECS = (
    CliSpec("opencode", OpenCodeExplorer,
            "SWE_EXPLORE_OPENCODE_BIN", "SWE_EXPLORE_OPENCODE_TIMEOUT"),
    CliSpec("deveco", DevEcoExplorer,
            "SWE_EXPLORE_DEVECO_BIN", "SWE_EXPLORE_DEVECO_TIMEOUT"),
)


def tree_digest(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and SERENA_DIR not in p.relative_to(root).parts
    }


def overlaps(region: Region, want: Region) -> bool:
    path, start, end = region
    if end == WHOLE_FILE:
        end = float("inf")
    want_end = float("inf") if want[2] == WHOLE_FILE else want[2]
    return path == want[0] and max(start, want[1]) <= min(end, want_end)


def expected_regions(case: dict) -> list[Region]:
    return [(r["path"], r["start"], r["end"]) for r in case["regions"]]


def as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def call_key(event: dict, part: dict, position: int) -> str:
    """Identity of one tool call across its streamed events.

    Prefer the part id, scoped to its message: parallel OpenCode calls can
    share both ``messageID`` and ``callID``. Without a part id, fall back to
    the message/call pair, then the event id. An event with no identifiers
    counts as its own call.
    """
    message_id = part.get("messageID") or event.get("messageID") or ""
    if part.get("id"):
        return f"{message_id}/{part['id']}"
    call_id = part.get("callID") or event.get("callID")
    if call_id:
        return f"{message_id}/{call_id}"
    return str(event.get("id") or f"event-{position}")


@pytest.fixture
def arkts_repo(tmp_path: Path) -> Path:
    """The fixture as an agent may see it: sources only. The explorer adds
    the Serena project file from the profile for the run itself."""
    root = tmp_path / "arkts_app"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns(*ANSWER_KEY))
    return root


@pytest.fixture(scope="session")
def serena_home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One Serena home for the whole session, so the language server is
    installed once rather than once per smoke run."""
    value = os.environ.get(SERENA_HOME_VAR)
    return Path(value).expanduser() if value else tmp_path_factory.mktemp("serena-home")


@pytest.fixture
def scripted_bin(tmp_path: Path) -> str:
    """A launcher the explorer can exec as argv[0] on every CI platform."""
    if sys.platform == "win32":
        launcher = tmp_path / "scripted-agent.cmd"
        launcher.write_text(
            f'@"{sys.executable}" "{SCRIPTED_AGENT}" %*\r\n', encoding="utf-8"
        )
    else:
        launcher = tmp_path / "scripted-agent"
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{SCRIPTED_AGENT}" "$@"\n',
            encoding="utf-8",
        )
        launcher.chmod(0o755)
    return str(launcher)


class RecordingRun:
    """Run the real subprocess and keep its tool events for inspection."""

    def __init__(self) -> None:
        self.tool_events: list[dict] = []

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        completed = _REAL_RUN(*args, **kwargs)
        self.tool_events = [
            event for event in iter_events(completed.stdout)
            if event.get("type") == "tool_use"
        ]
        return completed

    def parts(self) -> list[dict]:
        return [as_dict(event.get("part")) for event in self.tool_events]

    def tool_names(self) -> list[str]:
        return [part.get("tool") or "" for part in self.parts()]

    def read_paths(self) -> list[str]:
        return [
            as_dict(as_dict(part.get("state")).get("input")).get("filePath", "")
            for part in self.parts() if part.get("tool") == "read"
        ]

    def mcp_outcomes(self) -> McpOutcomes:
        """One entry per Serena call, split by kind and outcome.

        A streamed call may appear more than once (running, then completed),
        so the last status per call wins. Only a completed navigation call is
        evidence that semantic navigation worked.
        """
        calls: dict[str, tuple[str, object]] = {}
        for position, event in enumerate(self.tool_events):
            part = as_dict(event.get("part"))
            tool = part.get("tool") or ""
            if tool.startswith(MCP_TOOL_PREFIX):
                status = as_dict(part.get("state")).get("status")
                calls[call_key(event, part, position)] = (tool, status)
        navigation = [
            status for tool, status in calls.values() if tool in MCP_NAVIGATION_TOOLS
        ]
        completed = navigation.count("completed")
        return McpOutcomes(
            navigation_ok=completed,
            navigation_failed=len(navigation) - completed,
            administrative=len(calls) - len(navigation),
        )


@dataclass(frozen=True)
class McpOutcomes:
    navigation_ok: int
    navigation_failed: int
    administrative: int


def test_mcp_outcomes_keeps_parallel_calls_and_deduplicates_updates():
    recorder = RecordingRun()
    recorder.tool_events = [
        {
            "type": "tool_use",
            "part": {
                "id": part_id,
                "messageID": "msg_001",
                "callID": "call_001",
                "tool": tool,
                "state": {"status": status},
            },
        }
        for part_id, tool, status in [
            ("prt_006", "serena_find_symbol", "running"),
            ("prt_003", "serena_search_for_pattern", "running"),
            ("prt_006", "serena_find_symbol", "completed"),
            ("prt_003", "serena_search_for_pattern", "error"),
            ("prt_007", "serena_activate_project", "completed"),
        ]
    ]

    assert recorder.mcp_outcomes() == McpOutcomes(
        navigation_ok=1, navigation_failed=1, administrative=1,
    )


@dataclass
class Run:
    regions: list[Region]
    recorder: RecordingRun
    tokens: int


def profile_dir(spec: CliSpec, variant: str) -> Path:
    return PROFILES_ROOT / spec.name / variant


def scripted_explorer(
    spec: CliSpec, repo: Path, launcher: str, config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> BaseCliAgentExplorer:
    monkeypatch.setenv(SCRIPTED_CLI_VAR, spec.name)
    return spec.explorer_cls(
        repo_root=repo, bin_path=launcher, timeout=60, config_dir=config_dir,
    )


def run_case(
    explorer: BaseCliAgentExplorer, case: dict, monkeypatch: pytest.MonkeyPatch,
) -> Run:
    recorder = RecordingRun()
    monkeypatch.setattr("explorers._cli_agent_base.subprocess.run", recorder)
    with usage_collector() as usage:
        results = explorer.explore(
            instance_id=case["instance_id"], query=case["query"], top_k=5,
        )
    regions = [(r.path, r.start, r.end) for result in results for r in result.regions]
    return Run(regions, recorder, usage.total)


# ── scripted driver: always on ──

@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("spec", CLI_SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["instance_id"])
def test_scripted_agent_localizes_arkts_bug_exactly(
    spec: CliSpec, variant: str, case: dict, arkts_repo: Path,
    scripted_bin: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ANSWER_KEY:
        assert not (arkts_repo / name).exists(), f"{name} must not reach the agent"
    before = tree_digest(arkts_repo)
    explorer = scripted_explorer(
        spec, arkts_repo, scripted_bin, profile_dir(spec, variant), monkeypatch,
    )

    run = run_case(explorer, case, monkeypatch)

    assert run.regions == expected_regions(case)
    assert tree_digest(arkts_repo) == before, "the agent modified the repository"
    assert run.tokens > 0, "token usage from step_finish events was not collected"

    tools = run.recorder.tool_names()
    assert tools[0] == "list", "exploration must start by finding the .ets files"
    read_paths = run.recorder.read_paths()
    assert PAGE in read_paths and IMPORTED_MODULE in read_paths, (
        f"expected {PAGE} and {IMPORTED_MODULE} to be read, got {read_paths}"
    )
    assert read_paths.index(PAGE) < read_paths.index(IMPORTED_MODULE), (
        "the page must be read before the module it imports"
    )
    mcp = run.recorder.mcp_outcomes()
    assert mcp.navigation_failed == 0 and mcp.administrative == 0
    assert (mcp.navigation_ok > 0) == (variant == "arkts"), (
        "the MCP flag in the profile must reach the CLI"
    )


@pytest.mark.parametrize("spec", CLI_SPECS, ids=lambda s: s.name)
def test_scripted_agent_reads_non_ascii_issue_text(
    spec: CliSpec, arkts_repo: Path, scripted_bin: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prompt travels as UTF-8; the stand-in must not decode it with the
    platform's locale codec (cp1252 on the Windows runner)."""
    case = dict(CASES[1], query=CASES[1]["query"] + " （标题错误 — Überschrift）")
    explorer = scripted_explorer(
        spec, arkts_repo, scripted_bin, profile_dir(spec, "arkts-no-mcp"), monkeypatch,
    )
    assert run_case(explorer, case, monkeypatch).regions == expected_regions(case)


@pytest.mark.parametrize("spec", CLI_SPECS, ids=lambda s: s.name)
def test_scripted_agent_refuses_a_profile_that_is_not_read_only(
    spec: CliSpec, arkts_repo: Path, scripted_bin: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stand-in only navigates with tools the profile allows, so a
    profile regression that grants edits fails here, not in a real run."""
    filename = f"{spec.name}.json"
    loose_dir = tmp_path / "loose-profile"
    loose_dir.mkdir()
    loose = json.loads((profile_dir(spec, "arkts") / filename).read_text(encoding="utf-8"))
    loose["permission"]["edit"] = "allow"
    (loose_dir / filename).write_text(json.dumps(loose), encoding="utf-8")
    explorer = scripted_explorer(spec, arkts_repo, scripted_bin, loose_dir, monkeypatch)

    with pytest.raises(RuntimeError, match="read-only"):
        run_case(explorer, CASES[0], monkeypatch)


# ── real driver: opt-in smoke test ──

def require_real_binary(spec: CliSpec, variant: str) -> str:
    binary = os.environ.get(spec.bin_env_var) or shutil.which(spec.name)
    if not binary:
        pytest.skip(f"{spec.name} binary not found; set {spec.bin_env_var}")
    for var in PROXY_VARS:
        if not os.environ.get(var):
            pytest.skip(f"{var} is required by the committed profile")
    if variant == "arkts" and not shutil.which("serena"):
        pytest.skip("serena not on PATH (uv tool install serena-agent); "
                    "without it the MCP-on arm would silently equal the control")
    return binary


@pytest.mark.skipif(
    os.environ.get(SMOKE_VAR) != "1", reason=f"set {SMOKE_VAR}=1 to run the real CLIs",
)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("spec", CLI_SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["instance_id"])
def test_real_cli_localizes_arkts_bug(
    spec: CliSpec, variant: str, case: dict, arkts_repo: Path, serena_home: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    binary = require_real_binary(spec, variant)
    monkeypatch.setenv(SERENA_HOME_VAR, str(serena_home))
    before = tree_digest(arkts_repo)
    explorer = spec.explorer_cls(
        repo_root=arkts_repo, bin_path=binary,
        timeout=int(os.environ.get(spec.timeout_env_var) or "600"),
        config_dir=profile_dir(spec, variant),
    )

    run = run_case(explorer, case, monkeypatch)

    assert tree_digest(arkts_repo) == before, "the agent modified the repository"
    expected = expected_regions(case)
    hits = [region for region in run.regions if any(overlaps(region, want) for want in expected)]
    assert hits, f"no returned region in {run.regions} overlaps {expected}"

    tools = run.recorder.tool_names()
    mcp = run.recorder.mcp_outcomes()
    if variant == "arkts":
        assert mcp.navigation_ok, (
            f"MCP is enabled but no Serena navigation call completed ({mcp}); "
            "is the language server running?"
        )
    else:
        assert mcp == McpOutcomes(0, 0, 0), f"MCP is disabled but Serena was called: {mcp}"

    best = hits[0]
    width = "all" if best[2] == WHOLE_FILE else best[2] - best[1] + 1
    with capsys.disabled():
        print(
            f"\n[cli-smoke] cli={spec.name} variant={variant} case={case['instance_id']} "
            f"exact={best in expected} width={width} "
            f"rank={run.regions.index(best) + 1} regions={len(run.regions)} "
            f"tool_calls={len(tools)} mcp_nav_ok={mcp.navigation_ok} "
            f"mcp_nav_failed={mcp.navigation_failed} mcp_admin={mcp.administrative} "
            f"tokens={run.tokens}"
        )


@pytest.mark.parametrize("prompt", [
    "RELEVANT_FILES: ISSUE DESCRIPTION FROM USER (very important): issue",
    "Do exactly this, but without modifications. RELEVANT_FILES: "
    "ISSUE DESCRIPTION FROM USER (very important): issue",
])
def test_scripted_agent_rejects_missing_or_misplaced_trailer(prompt, tmp_path):
    completed = _REAL_RUN(
        [sys.executable, str(SCRIPTED_AGENT), "run", "--format", "json", "--dir", str(tmp_path)],
        input=prompt, capture_output=True, text=True, encoding="utf-8",
    )
    assert completed.returncode == 2
    assert "prompt does not follow the explorer contract" in completed.stderr


def test_scripted_agent_reports_unicode_errors_under_ascii_locale():
    completed = _REAL_RUN(
        [sys.executable, str(SCRIPTED_AGENT), "标题"],
        env={**os.environ, "PYTHONIOENCODING": "ascii"},
        capture_output=True, text=True, encoding="utf-8",
    )
    assert completed.returncode == 2
    assert "标题" in completed.stderr
    assert "UnicodeEncodeError" not in completed.stderr


@pytest.mark.parametrize("region,want,expected", [
    ((PAGE, 10, 12), (PAGE, 1, WHOLE_FILE), True),
    ((PAGE, 1, WHOLE_FILE), (PAGE, 10, 12), True),
    ((PAGE, 1, 9), (PAGE, 10, 12), False),
    ((PAGE, 10, 12), (IMPORTED_MODULE, 1, WHOLE_FILE), False),
])
def test_overlap_handles_whole_files(region, want, expected):
    assert overlaps(region, want) is expected
