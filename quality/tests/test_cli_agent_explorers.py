"""Table-driven tests for the agentic CLI explorers.

The CLIs are external binaries absent from CI, so every case mocks
``subprocess.run``. A new explorer is one row in ``CLI_EXPLORER_CASES``.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Type
from unittest.mock import patch

from explorers._cli_agent_base import BaseCliAgentExplorer, _extract_output_text
from explorers.deveco import DevEcoExplorer
from explorers.opencode import OpenCodeExplorer

FIXTURES = Path(__file__).parent / "fixtures"

ANSWER_EVENT = json.dumps(
    {"type": "text", "part": {"text": "RELEVANT_FILES:\n- src/main.py:10-20"}}
)


@dataclass(frozen=True)
class CliExplorerCase:
    """One CLI explorer subclass and the contract it is expected to honour."""

    explorer_cls: Type[BaseCliAgentExplorer]
    bin_path: str
    #: Called with (bin_path, resolved repo root) -> expected argv.
    build_expected_cmd: Callable[[str, str], list[str]]
    expected_config_env_var: str
    expected_config_filename: str
    expected_local_config_dirname: str
    missing_binary_pattern: str


CLI_EXPLORER_CASES = (
    CliExplorerCase(
        explorer_cls=OpenCodeExplorer,
        bin_path="opencode-test",
        build_expected_cmd=lambda binary, repo: [
            binary, "run", "--auto", "--format", "json", "--dir", repo,
        ],
        expected_config_env_var="OPENCODE_CONFIG_DIR",
        expected_config_filename="opencode.json",
        expected_local_config_dirname=".opencode",
        missing_binary_pattern="OpenCode CLI not found",
    ),
    CliExplorerCase(
        explorer_cls=DevEcoExplorer,
        bin_path="deveco-test",
        build_expected_cmd=lambda binary, repo: [
            binary, "run", "--format", "json", "--dir", repo,
            "--dangerously-skip-permissions",
        ],
        expected_config_env_var="DEVECO_CONFIG_DIR",
        expected_config_filename="deveco.json",
        expected_local_config_dirname=".deveco",
        missing_binary_pattern="deveco CLI not found",
    ),
)


class CliAgentExplorerContractTest(unittest.TestCase):
    """Every subclass must satisfy the same invocation and parsing contract."""

    def test_explore_builds_cmd_and_parses_relevant_files(self) -> None:
        for case in CLI_EXPLORER_CASES:
            with self.subTest(explorer=case.explorer_cls.__name__):
                with tempfile.TemporaryDirectory() as repo:
                    seen: dict[str, object] = {}

                    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                        seen.update(
                            cmd=cmd, cwd=kwargs["cwd"], env=kwargs["env"],
                            input=kwargs["input"], timeout=kwargs["timeout"],
                        )
                        return subprocess.CompletedProcess(
                            cmd, 0, stdout=ANSWER_EVENT, stderr=""
                        )

                    explorer = case.explorer_cls(
                        repo_root=Path(repo), bin_path=case.bin_path, timeout=12
                    )
                    with patch(
                        "explorers._cli_agent_base.subprocess.run", side_effect=fake_run
                    ):
                        results = explorer.explore(
                            instance_id="inst-1", query="broken behavior", top_k=5
                        )

                    self.assertEqual(seen["cwd"], repo)
                    self.assertEqual(seen["timeout"], 12)
                    self.assertEqual(
                        seen["cmd"],
                        case.build_expected_cmd(case.bin_path, str(Path(repo).resolve())),
                    )
                    self.assertIn("broken behavior", seen["input"])  # type: ignore[operator]
                    self.assertEqual(results[0].regions[0].path, "src/main.py")
                    self.assertEqual(results[0].regions[0].start, 10)
                    self.assertEqual(results[0].regions[0].end, 20)

    def test_explore_isolates_home_from_the_operator_environment(self) -> None:
        for case in CLI_EXPLORER_CASES:
            with self.subTest(explorer=case.explorer_cls.__name__):
                with tempfile.TemporaryDirectory() as repo:
                    seen: dict[str, object] = {}

                    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                        seen["env"] = kwargs["env"]
                        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

                    explorer = case.explorer_cls(
                        repo_root=Path(repo), bin_path=case.bin_path
                    )
                    with patch(
                        "explorers._cli_agent_base.subprocess.run", side_effect=fake_run
                    ):
                        explorer.explore(instance_id="inst-1", query="issue")

                    env = seen["env"]
                    self.assertNotEqual(env["HOME"], os.environ.get("HOME"))  # type: ignore[index]
                    self.assertEqual(env["HOME"], env["USERPROFILE"])  # type: ignore[index]

    def test_explore_exports_config_dir_and_copies_config_into_repo(self) -> None:
        for case in CLI_EXPLORER_CASES:
            with self.subTest(explorer=case.explorer_cls.__name__):
                # Created under cwd so a relative path can be passed in,
                # proving the explorer resolves it before exporting it.
                with tempfile.TemporaryDirectory() as repo, \
                        tempfile.TemporaryDirectory(dir=".") as cfg:
                    (Path(cfg) / case.expected_config_filename).write_text(
                        "{}", encoding="utf-8"
                    )
                    seen: dict[str, object] = {}

                    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                        seen["env"] = kwargs["env"]
                        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

                    explorer = case.explorer_cls(
                        repo_root=Path(repo),
                        bin_path=case.bin_path,
                        config_dir=Path(os.path.relpath(cfg)),
                    )
                    with patch(
                        "explorers._cli_agent_base.subprocess.run", side_effect=fake_run
                    ):
                        explorer.explore(instance_id="inst-1", query="issue")

                    self.assertEqual(
                        seen["env"][case.expected_config_env_var],  # type: ignore[index]
                        str(Path(cfg).resolve()),
                    )
                    copied = (Path(repo) / case.expected_local_config_dirname
                              / case.expected_config_filename)
                    self.assertTrue(copied.exists())

    def test_missing_config_file_raises_file_not_found(self) -> None:
        for case in CLI_EXPLORER_CASES:
            with self.subTest(explorer=case.explorer_cls.__name__):
                with tempfile.TemporaryDirectory() as repo, \
                        tempfile.TemporaryDirectory() as cfg:
                    explorer = case.explorer_cls(
                        repo_root=Path(repo), bin_path=case.bin_path,
                        config_dir=Path(cfg),
                    )
                    with patch("explorers._cli_agent_base.subprocess.run") as run:
                        with self.assertRaises(FileNotFoundError):
                            explorer.explore(instance_id="inst-1", query="issue")
                    run.assert_not_called()

    def test_missing_binary_raises_clear_runtime_error(self) -> None:
        for case in CLI_EXPLORER_CASES:
            with self.subTest(explorer=case.explorer_cls.__name__):
                with tempfile.TemporaryDirectory() as repo:
                    explorer = case.explorer_cls(
                        repo_root=Path(repo), bin_path="missing-binary"
                    )
                    with patch(
                        "explorers._cli_agent_base.subprocess.run",
                        side_effect=FileNotFoundError,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, case.missing_binary_pattern
                        ):
                            explorer.explore(instance_id="inst-1", query="issue")

    def test_nonzero_return_code_surfaces_stdout_and_stderr(self) -> None:
        for case in CLI_EXPLORER_CASES:
            with self.subTest(explorer=case.explorer_cls.__name__):
                with tempfile.TemporaryDirectory() as repo:
                    explorer = case.explorer_cls(
                        repo_root=Path(repo), bin_path=case.bin_path
                    )
                    completed = subprocess.CompletedProcess(
                        [], 3, stdout="partial out", stderr="boom"
                    )
                    with patch(
                        "explorers._cli_agent_base.subprocess.run", return_value=completed
                    ):
                        with self.assertRaisesRegex(RuntimeError, "rc=3"):
                            explorer.explore(instance_id="inst-1", query="issue")

    def test_empty_stdout_returns_no_results(self) -> None:
        for case in CLI_EXPLORER_CASES:
            with self.subTest(explorer=case.explorer_cls.__name__):
                with tempfile.TemporaryDirectory() as repo:
                    explorer = case.explorer_cls(
                        repo_root=Path(repo), bin_path=case.bin_path
                    )
                    completed = subprocess.CompletedProcess([], 0, stdout="  \n", stderr="")
                    with patch(
                        "explorers._cli_agent_base.subprocess.run", return_value=completed
                    ):
                        self.assertEqual(
                            explorer.explore(instance_id="inst-1", query="issue"), []
                        )

    def test_prompt_additions_reach_the_prompt(self) -> None:
        for case in CLI_EXPLORER_CASES:
            with self.subTest(explorer=case.explorer_cls.__name__):
                with tempfile.TemporaryDirectory() as repo:
                    seen: dict[str, object] = {}

                    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                        seen["input"] = kwargs["input"]
                        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

                    explorer = case.explorer_cls(
                        repo_root=Path(repo), bin_path=case.bin_path,
                        prompt_additions="Prefer the LSP hub.",
                    )
                    with patch(
                        "explorers._cli_agent_base.subprocess.run", side_effect=fake_run
                    ):
                        explorer.explore(instance_id="inst-1", query="issue")

                    self.assertIn("Prefer the LSP hub.", seen["input"])  # type: ignore[operator]


class DevEcoInvocationTest(unittest.TestCase):
    """Guards the parts of the deveco 0.1.9 contract that are easy to get wrong."""

    def _cmd_for(self, **kwargs) -> list[str]:  # type: ignore[no-untyped-def]
        with tempfile.TemporaryDirectory() as repo:
            return DevEcoExplorer(repo_root=Path(repo), **kwargs).build_cmd()

    def test_prompt_is_never_passed_as_dash_p(self) -> None:
        """``-p`` is deveco's --password flag; the prompt goes on stdin."""
        cmd = self._cmd_for()
        self.assertNotIn("-p", cmd)
        self.assertNotIn("--password", cmd)

    def test_no_auto_flag_is_sent(self) -> None:
        """``--auto`` is OpenCode-only; deveco 0.1.9 does not define it."""
        self.assertNotIn("--auto", self._cmd_for())

    def test_skip_permissions_can_be_disabled(self) -> None:
        self.assertIn("--dangerously-skip-permissions", self._cmd_for())
        self.assertNotIn(
            "--dangerously-skip-permissions", self._cmd_for(skip_permissions=False)
        )


class ExtractOutputTextTest(unittest.TestCase):
    """Driven by real captures from both CLIs.

    Both emit JSONL events and carry the answer as ``part.text`` on events of
    type ``text``.
    """

    def _fixture(self, name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    def test_real_opencode_capture_yields_only_the_answer(self) -> None:
        out = _extract_output_text(self._fixture("opencode_events.jsonl"))
        self.assertEqual(out, "RELEVANT_FILES:\n- src/button.py:1-2")

    def test_real_deveco_capture_yields_only_the_answer(self) -> None:
        out = _extract_output_text(self._fixture("deveco_events.jsonl"))
        self.assertTrue(out.startswith("RELEVANT_FILES:"))
        self.assertIn("src/render.cpp:10-50", out)

    def test_tool_output_never_reaches_the_parser(self) -> None:
        """Tool events embed file contents and paths; none may leak through."""
        for name in ("opencode_events.jsonl", "deveco_events.jsonl"):
            with self.subTest(fixture=name):
                raw = self._fixture(name)
                out = _extract_output_text(raw)
                self.assertNotIn("<content>", out)
                self.assertNotIn("/repo", out)
                self.assertLess(len(out), len(raw) / 4)

    def test_answer_is_the_last_text_event(self) -> None:
        raw = "\n".join([
            json.dumps({"type": "text", "part": {"text": "thinking out loud"}}),
            json.dumps({"type": "text",
                        "part": {"text": "RELEVANT_FILES:\n- a.py:1-2"}}),
        ])
        self.assertEqual(_extract_output_text(raw), "RELEVANT_FILES:\n- a.py:1-2")

    def test_decoy_in_tool_event_does_not_win(self) -> None:
        raw = "\n".join([
            json.dumps({"type": "tool_use",
                        "part": {"state": {"output": "RELEVANT_FILES:\n- decoy.py:1-2"}}}),
            json.dumps({"type": "text",
                        "part": {"text": "RELEVANT_FILES:\n- real.py:5-6"}}),
        ])
        self.assertEqual(_extract_output_text(raw), "RELEVANT_FILES:\n- real.py:5-6")

    def test_malformed_line_does_not_discard_the_stream(self) -> None:
        # An unescaped Windows path: "\u" is not a valid JSON escape.
        bad = '{"type":"tool_use","part":{"input":{"path":"D:' + chr(92) + 'umd"}}}'
        raw = "\n".join([
            bad,
            json.dumps({"type": "text",
                        "part": {"text": "RELEVANT_FILES:\n- a.py:1-2"}}),
        ])
        with self.assertRaises(json.JSONDecodeError):
            json.loads(bad)
        self.assertEqual(_extract_output_text(raw), "RELEVANT_FILES:\n- a.py:1-2")

    def test_non_json_text_passes_through_unchanged(self) -> None:
        raw = "RELEVANT_FILES:\n- src/main.py:10-20"
        self.assertEqual(_extract_output_text(raw), raw)

    def test_empty_input_returns_empty_string(self) -> None:
        self.assertEqual(_extract_output_text("  \n "), "")

    def test_events_without_any_text_yield_nothing(self) -> None:
        raw = json.dumps({"type": "step_finish", "part": {"reason": "stop"}})
        self.assertEqual(_extract_output_text(raw), "")


if __name__ == "__main__":
    unittest.main()
