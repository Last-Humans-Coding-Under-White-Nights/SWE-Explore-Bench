"""Table-driven tests for the agentic CLI explorers.

The CLIs are external binaries absent from CI, so every case mocks
``subprocess.run``. A new explorer is one row in ``CLI_EXPLORER_CASES``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Type
from unittest.mock import patch

from explorers._cli_agent_base import (
    XDG_VARS,
    BaseCliAgentExplorer,
    _extract_output_text,
)
from explorers.deveco import DevEcoExplorer
from explorers.opencode import OpenCodeExplorer
from explorers.parsing import parse_relevant_files

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
    expected_override_vars: tuple[str, ...]
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
        expected_override_vars=("OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT",
                                "OPENCODE_CONFIG_DIR"),
        missing_binary_pattern="OpenCode CLI not found",
    ),
    CliExplorerCase(
        explorer_cls=DevEcoExplorer,
        bin_path="deveco-test",
        build_expected_cmd=lambda binary, repo: [
            binary, "run", "--auto", "--format", "json", "--dir", repo,
            "--dangerously-skip-permissions",
        ],
        expected_config_env_var="DEVECO_CONFIG_DIR",
        expected_config_filename="deveco.json",
        expected_override_vars=("DEVECO_CONFIG", "DEVECO_CONFIG_CONTENT",
                                "DEVECO_CONFIG_DIR"),
        missing_binary_pattern="deveco CLI not found",
    ),
)


class CliAgentExplorerContractTest(unittest.TestCase):
    """Every subclass must satisfy the same invocation and parsing contract."""

    def test_explore_builds_cmd_and_parses_relevant_files(self) -> None:
        for case in CLI_EXPLORER_CASES:
            with self.subTest(explorer=case.explorer_cls.__name__):
                with tempfile.TemporaryDirectory() as repo:
                    (Path(repo) / "src").mkdir()
                    (Path(repo) / "src/main.py").write_text("# source\n" * 20, encoding="utf-8")
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

    def test_explore_isolates_config_discovery_from_inherited_env(self) -> None:
        """A temporary HOME alone is not isolation.
        XDG paths and the CLI's own override variables each redirect config
        discovery past HOME, so all of them must be cleared too.
        """
        for case in CLI_EXPLORER_CASES:
            with self.subTest(explorer=case.explorer_cls.__name__):
                with tempfile.TemporaryDirectory() as repo:
                    seen: dict[str, object] = {}

                    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                        seen["env"] = kwargs["env"]
                        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

                    redirects = XDG_VARS + case.expected_override_vars
                    inherited = dict(os.environ, HOME="/user")
                    inherited.update({var: "/user/leaked" for var in redirects})

                    explorer = case.explorer_cls(
                        repo_root=Path(repo), bin_path=case.bin_path
                    )
                    with patch.dict(os.environ, inherited, clear=True), patch(
                        "explorers._cli_agent_base.subprocess.run", side_effect=fake_run
                    ):
                        explorer.explore(instance_id="inst-1", query="issue")

                    env = seen["env"]
                    self.assertNotEqual(env["HOME"], "/user")  # type: ignore[index]
                    self.assertEqual(env["HOME"], env["USERPROFILE"])  # type: ignore[index]
                    for var in redirects:
                        self.assertNotIn(var, env)  # type: ignore[operator]

    def test_explore_exports_config_dir_without_copying_it(self) -> None:
        for case in CLI_EXPLORER_CASES:
            with self.subTest(explorer=case.explorer_cls.__name__):
                with tempfile.TemporaryDirectory() as repo, \
                        tempfile.TemporaryDirectory() as cfg:
                    (Path(cfg) / case.expected_config_filename).write_text(
                        "{}", encoding="utf-8"
                    )
                    (Path(cfg) / "sub").mkdir()
                    unnormalised = Path(cfg) / "sub" / ".."
                    seen: dict[str, object] = {}

                    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                        seen["env"] = kwargs["env"]
                        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

                    explorer = case.explorer_cls(
                        repo_root=Path(repo),
                        bin_path=case.bin_path,
                        config_dir=unnormalised,
                    )
                    with patch(
                        "explorers._cli_agent_base.subprocess.run", side_effect=fake_run
                    ):
                        explorer.explore(instance_id="inst-1", query="issue")

                    self.assertEqual(
                        seen["env"][case.expected_config_env_var],  # type: ignore[index]
                        str(Path(cfg).resolve()),
                    )
                    # A copy elsewhere breaks {file:./...} references
                    self.assertEqual(list(Path(repo).iterdir()), [])

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
    """Guards the parts of the deveco argv contract that are easy to get wrong."""

    def _cmd_for(self, **kwargs) -> list[str]:  # type: ignore[no-untyped-def]
        with tempfile.TemporaryDirectory() as repo:
            return DevEcoExplorer(repo_root=Path(repo), **kwargs).build_cmd()

    def test_prompt_is_never_passed_as_dash_p(self) -> None:
        """``-p`` is deveco's --password flag; the prompt goes on stdin."""
        cmd = self._cmd_for()
        self.assertNotIn("-p", cmd)
        self.assertNotIn("--password", cmd)

    def test_auto_flag_respects_skip_permissions(self) -> None:
        """deveco 0.1.12 resolves ``ask`` rules with ``--auto`` and ignores the
        skip flag below; 0.1.9 ignores unknown flags, so both are sent."""
        self.assertIn("--auto", self._cmd_for())
        self.assertNotIn("--auto", self._cmd_for(skip_permissions=False))

    def test_skip_permissions_can_be_disabled(self) -> None:
        self.assertIn("--dangerously-skip-permissions", self._cmd_for())
        self.assertNotIn(
            "--dangerously-skip-permissions", self._cmd_for(skip_permissions=False)
        )


def _snapshot(root: Path) -> dict[str, str]:
    """Relative path -> text for every file under ``root``."""
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class CheckoutSeedingTest(unittest.TestCase):
    """Files under ``<config_dir>/checkout`` are placed into the repository for
    the run (a Serena project file, for instance) and removed again afterwards
    unless the run changed them. Files the checkout already has are never
    overwritten."""

    def setUp(self) -> None:
        self.repo = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Path(self.enterContext(tempfile.TemporaryDirectory()))
        _write(self.repo, "README.md", "original")
        _write(self.cfg, "opencode.json", "{}")

    def _run(self, during_run=None) -> dict[str, str]:  # type: ignore[no-untyped-def]
        """Explore once; return the checkout as the CLI saw it."""
        seen: dict[str, str] = {}

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            seen.update(_snapshot(self.repo))
            if during_run:
                during_run()
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        explorer = OpenCodeExplorer(repo_root=self.repo, bin_path="x", config_dir=self.cfg)
        with patch("explorers._cli_agent_base.subprocess.run", side_effect=fake_run):
            explorer.explore(instance_id="inst-1", query="issue")
        return seen

    def test_seed_files_exist_during_the_run_and_are_removed_after(self) -> None:
        _write(self.cfg, "checkout/.serena/project.yml", "read_only: true\n")
        _write(self.cfg, "checkout/README.md", "from profile")

        during = self._run()

        self.assertEqual(during, {"README.md": "original",
                                  ".serena/project.yml": "read_only: true\n"})
        self.assertEqual(_snapshot(self.repo), {"README.md": "original"})
        self.assertFalse((self.repo / ".serena").exists())

    def test_seed_file_changed_by_the_run_is_left_in_place(self) -> None:
        _write(self.cfg, "checkout/.serena/project.yml", "a\n")

        def mutate() -> None:
            _write(self.repo, ".serena/project.yml", "b\n")
            _write(self.repo, ".serena/cache/index", "")

        self._run(during_run=mutate)

        self.assertEqual(_snapshot(self.repo), {
            "README.md": "original",
            ".serena/project.yml": "b\n",
            ".serena/cache/index": "",
        })

    def test_deleted_seed_directory_does_not_mask_success_or_timeout(self) -> None:
        _write(self.cfg, "checkout/.serena/project.yml", "a")
        self._run(lambda: shutil.rmtree(self.repo / ".serena"))

        def timeout() -> None:
            shutil.rmtree(self.repo / ".serena")
            raise subprocess.TimeoutExpired("x", 1)

        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self._run(timeout)

    def test_preexisting_empty_directory_survives_cleanup(self) -> None:
        (self.repo / ".serena").mkdir()
        _write(self.cfg, "checkout/.serena/nested/project.yml", "a")
        self._run()
        self.assertTrue((self.repo / ".serena").is_dir())
        self.assertEqual(list((self.repo / ".serena").iterdir()), [])

    def test_broken_symlink_is_left_untouched(self) -> None:
        target = self.repo / "link"
        try:
            target.symlink_to(self.repo / "missing")
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        _write(self.cfg, "checkout/link", "a")
        self._run()
        self.assertTrue(target.is_symlink())
        self.assertFalse((self.repo / "missing").exists())

    def test_failed_seeding_rolls_back_files_and_directories(self) -> None:
        _write(self.cfg, "checkout/a/first", "a")
        _write(self.cfg, "checkout/b/second", "b")
        original = Path.read_bytes

        def read(path):
            if path == self.cfg / "checkout/b/second":
                raise FileNotFoundError("seed source disappeared")
            return original(path)

        with patch.object(Path, "read_bytes", read):
            with self.assertRaisesRegex(FileNotFoundError, "seed source disappeared"):
                self._run()
        self.assertEqual(sorted(p.name for p in self.repo.iterdir()), ["README.md"])

    def test_write_failure_rolls_back_created_directories(self) -> None:
        _write(self.cfg, "checkout/a/first", "a")
        _write(self.cfg, "checkout/b/second", "b")
        original = Path.open

        def open_file(path, *args, **kwargs):
            if path == self.repo / "b/second":
                raise PermissionError("seed write denied")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", open_file):
            with self.assertRaisesRegex(PermissionError, "seed write denied"):
                self._run()
        self.assertEqual(sorted(p.name for p in self.repo.iterdir()), ["README.md"])

    def test_no_checkout_dir_means_no_seeding(self) -> None:
        self.assertEqual(self._run(), {"README.md": "original"})
        self.assertEqual(_snapshot(self.repo), {"README.md": "original"})


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

    def test_answer_split_across_events_is_joined(self) -> None:
        """The list can arrive after the header; returning only the header
        that matched the marker would yield no regions at all."""
        raw = "\n".join([
            json.dumps({"type": "text", "part": {"text": "RELEVANT_FILES:"}}),
            json.dumps({"type": "text", "part": {"text": "- a.py:1-2\n- b.py:3-4"}}),
        ])
        out = _extract_output_text(raw)
        self.assertEqual(out, "RELEVANT_FILES:\n- a.py:1-2\n- b.py:3-4")
        self.assertEqual(len(parse_relevant_files(out, "i", top_k=5)), 2)

    def test_string_part_is_accepted(self) -> None:
        """A string ``part`` used to raise AttributeError out of explore()."""
        raw = json.dumps({"type": "text", "part": "RELEVANT_FILES:\n- a.py:1-2"})
        self.assertEqual(_extract_output_text(raw), "RELEVANT_FILES:\n- a.py:1-2")

    def test_stray_json_line_does_not_discard_plain_text(self) -> None:
        """A log line is not a protocol event, so it must not suppress the
        plain-text answer that follows it."""
        raw = '{"level":"info","msg":"telemetry ping"}\nRELEVANT_FILES:\n- a.py:1-2'
        self.assertEqual(len(parse_relevant_files(_extract_output_text(raw), "i")), 1)

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
