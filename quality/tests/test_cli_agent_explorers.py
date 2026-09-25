"""Table-driven tests for the agentic CLI explorers.

The CLIs are external binaries absent from CI, so every case mocks
the shared CLI process runner. A new explorer is one row in ``CLI_EXPLORER_CASES``.
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
    REDACTED,
    XDG_VARS,
    BaseCliAgentExplorer,
    _extract_output_text,
    redact_config,
)
from explorers.base import (
    BINARY_NOT_FOUND,
    INVALID_OUTPUT,
    PROVIDER_ERROR,
    TIMEOUT,
    ExplorerFailure,
)
from explorers.deveco import DevEcoExplorer
from explorers.opencode import OpenCodeExplorer
from explorers.parsing import parse_relevant_files, usage_collector

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
                        if cmd[1] == "db":
                            return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")
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
                        "explorers._cli_agent_base.run_cli", side_effect=fake_run
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
                        return subprocess.CompletedProcess(cmd, 0, stdout=ANSWER_EVENT, stderr="")

                    redirects = XDG_VARS + case.expected_override_vars
                    inherited = dict(os.environ, HOME="/user")
                    inherited.update({var: "/user/leaked" for var in redirects})

                    explorer = case.explorer_cls(
                        repo_root=Path(repo), bin_path=case.bin_path
                    )
                    with patch.dict(os.environ, inherited, clear=True), patch(
                        "explorers._cli_agent_base.run_cli", side_effect=fake_run
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
                        return subprocess.CompletedProcess(cmd, 0, stdout=ANSWER_EVENT, stderr="")

                    explorer = case.explorer_cls(
                        repo_root=Path(repo),
                        bin_path=case.bin_path,
                        config_dir=unnormalised,
                    )
                    with patch(
                        "explorers._cli_agent_base.run_cli", side_effect=fake_run
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
                    with patch("explorers._cli_agent_base.run_cli") as run:
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
                        "explorers._cli_agent_base.run_cli",
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
                        "explorers._cli_agent_base.run_cli", return_value=completed
                    ):
                        with self.assertRaisesRegex(RuntimeError, "rc=3"):
                            explorer.explore(instance_id="inst-1", query="issue")

    def test_empty_stdout_is_invalid_output(self) -> None:
        """A clean exit that printed nothing never answered."""
        for case in CLI_EXPLORER_CASES:
            with self.subTest(explorer=case.explorer_cls.__name__):
                with tempfile.TemporaryDirectory() as repo:
                    explorer = case.explorer_cls(
                        repo_root=Path(repo), bin_path=case.bin_path
                    )
                    completed = subprocess.CompletedProcess([], 0, stdout="  \n", stderr="")
                    with patch(
                        "explorers._cli_agent_base.run_cli", return_value=completed
                    ):
                        with self.assertRaises(ExplorerFailure) as caught:
                            explorer.explore(instance_id="inst-1", query="issue")
                    self.assertEqual(caught.exception.outcome, INVALID_OUTPUT)

    def test_prompt_additions_reach_the_prompt(self) -> None:
        for case in CLI_EXPLORER_CASES:
            with self.subTest(explorer=case.explorer_cls.__name__):
                with tempfile.TemporaryDirectory() as repo:
                    seen: dict[str, object] = {}

                    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                        if cmd[1] == "db":
                            return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")
                        seen["input"] = kwargs["input"]
                        return subprocess.CompletedProcess(cmd, 0, stdout=ANSWER_EVENT, stderr="")

                    explorer = case.explorer_cls(
                        repo_root=Path(repo), bin_path=case.bin_path,
                        prompt_additions="Prefer the LSP hub.",
                    )
                    with patch(
                        "explorers._cli_agent_base.run_cli", side_effect=fake_run
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


# One error event as OpenCode 1.18.29 prints it when the provider is unreachable.
ERROR_EVENT = json.dumps({
    "type": "error", "timestamp": 1790164155111, "sessionID": "ses_1",
    "error": {"name": "APIError", "data": {
        "message": "Cannot connect to API: Unable to connect.", "isRetryable": True,
    }},
})
USAGE_EVENT = json.dumps({
    "type": "step_finish",
    "part": {"type": "step-finish", "tokens": {"input": 700, "output": 50,
                                               "reasoning": 0, "cache": {"read": 0, "write": 0}}},
})


class CliAgentOutcomeTest(unittest.TestCase):
    """Each way a run can go wrong surfaces as its own outcome, spend included."""

    def _explore(self, explorer_cls, run):  # type: ignore[no-untyped-def]
        """Run one case against ``run`` (the fake agent); the session store is empty."""
        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            if cmd[1] == "db":
                return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")
            return run(cmd, **kwargs)

        with tempfile.TemporaryDirectory() as repo:
            explorer = explorer_cls(repo_root=Path(repo), bin_path="agent-test", timeout=5)
            with patch("explorers._cli_agent_base.run_cli", side_effect=fake_run):
                with usage_collector() as tracker:
                    try:
                        return explorer.explore(instance_id="inst-1", query="issue"), tracker
                    except ExplorerFailure as exc:
                        return exc, tracker

    def _each_cli(self):  # type: ignore[no-untyped-def]
        for cls in (OpenCodeExplorer, DevEcoExplorer):
            with self.subTest(explorer=cls.__name__):
                yield cls

    def test_error_only_stream_with_clean_exit_is_a_provider_error(self) -> None:
        stdout = "\n".join([USAGE_EVENT, ERROR_EVENT])
        for cls in self._each_cli():
            failure, tracker = self._explore(
                cls, lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout, "")
            )
            self.assertIsInstance(failure, ExplorerFailure)
            self.assertEqual(failure.outcome, PROVIDER_ERROR)
            self.assertIn("Cannot connect to API", str(failure))
            self.assertEqual(tracker.input_tokens, 700)  # spend is kept

    def test_error_stream_with_failing_exit_is_a_provider_error(self) -> None:
        for cls in self._each_cli():
            failure, _ = self._explore(
                cls, lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, ERROR_EVENT, "")
            )
            self.assertEqual(failure.outcome, PROVIDER_ERROR)
            self.assertIn("rc=1", str(failure))

    def test_text_without_an_answer_is_invalid_output(self) -> None:
        text = json.dumps({"type": "text", "part": {"text": "I could not decide."}})
        for cls in self._each_cli():
            failure, _ = self._explore(
                cls, lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, text, "")
            )
            self.assertEqual(failure.outcome, INVALID_OUTPUT)

    def test_an_answer_naming_no_region_is_an_empty_success(self) -> None:
        text = json.dumps({"type": "text", "part": {"text": "RELEVANT_FILES:\n(none)"}})
        for cls in self._each_cli():
            results, _ = self._explore(
                cls, lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, text, "")
            )
            self.assertEqual(results, [])

    def test_an_answer_despite_an_error_event_is_a_success(self) -> None:
        stdout = "\n".join([ERROR_EVENT, ANSWER_EVENT])
        with tempfile.TemporaryDirectory() as repo:
            (Path(repo) / "src").mkdir()
            (Path(repo) / "src/main.py").write_text("x\n" * 30, encoding="utf-8")
            explorer = OpenCodeExplorer(repo_root=Path(repo), bin_path="agent-test")

            def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                out = "[]" if cmd[1] == "db" else stdout
                return subprocess.CompletedProcess(cmd, 0, out, "")

            with patch("explorers._cli_agent_base.run_cli", side_effect=fake_run):
                results = explorer.explore(instance_id="inst-1", query="issue")
        self.assertEqual(results[0].regions[0].path, "src/main.py")

    def test_timeout_keeps_the_usage_but_not_the_raw_output(self) -> None:
        """The spend is in what the CLI printed before the kill; the row is not."""
        partial = (USAGE_EVENT + "\n" + json.dumps(
            {"type": "text", "part": {"text": "still reading render.py"}}
        )).encode("utf-8")

        def times_out(cmd, **kwargs):  # type: ignore[no-untyped-def]
            # subprocess.run hands back raw bytes on a timeout, even with text=True.
            raise subprocess.TimeoutExpired(cmd, 5, output=partial, stderr=b"slow provider")

        for cls in self._each_cli():
            failure, tracker = self._explore(cls, times_out)
            self.assertEqual(failure.outcome, TIMEOUT)
            self.assertTrue(str(failure).endswith("timed out after 5s"), str(failure))
            # The streams are still read — that is where the usage is — but
            # the agent's own text stays out of a row that may be published.
            self.assertNotIn("still reading render.py", str(failure))
            self.assertNotIn("slow provider", str(failure))
            self.assertEqual(tracker.input_tokens, 700)

    def test_repeated_error_events_are_reported_once(self) -> None:
        """A retried call prints its error per attempt; the row says it once."""
        repeated = "\n".join([ERROR_EVENT] * 4)
        for cls in self._each_cli():
            failure, _ = self._explore(
                cls, lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, repeated, "")
            )
            self.assertIsInstance(failure, ExplorerFailure)
            self.assertEqual(failure.outcome, PROVIDER_ERROR)
            self.assertEqual(str(failure).count("Cannot connect to API"), 1)

    def test_missing_binary_is_binary_not_found(self) -> None:
        def missing(cmd, **kwargs):  # type: ignore[no-untyped-def]
            raise FileNotFoundError(cmd[0])

        for cls in self._each_cli():
            failure, _ = self._explore(cls, missing)
            self.assertEqual(failure.outcome, BINARY_NOT_FOUND)


class CliAgentSelectionTest(unittest.TestCase):
    """The model and agent are named on the command line, not left implicit."""

    def test_model_and_agent_reach_the_argv(self) -> None:
        for cls in (OpenCodeExplorer, DevEcoExplorer):
            with self.subTest(explorer=cls.__name__):
                cmd = cls(repo_root=Path("."), model="p/m", agent="explore").build_cmd()
                self.assertEqual(cmd[cmd.index("--model") + 1], "p/m")
                self.assertEqual(cmd[cmd.index("--agent") + 1], "explore")

    def test_no_selection_adds_no_flags(self) -> None:
        cmd = OpenCodeExplorer(repo_root=Path(".")).build_cmd()
        self.assertNotIn("--model", cmd)
        self.assertNotIn("--agent", cmd)


class CliAgentDescribeTest(unittest.TestCase):
    """The manifest entry pins the configuration without recording a secret."""

    SECRET = "sk-live-0123456789abcdef"

    def _resolved(self, api_key: str, model: str = "swe-explore/gpt-5.4") -> dict:
        return {
            "model": model,
            "provider": {"swe-explore": {"options": {
                "baseURL": "http://127.0.0.1:4000/v1", "apiKey": api_key,
            }}},
            "mcp": {
                "serena": {"type": "local", "enabled": True,
                           "environment": {"SERENA_HOME": "/home/me/.serena"}},
                "off": {"type": "remote", "enabled": False,
                        "headers": {"Authorization": f"Bearer {api_key}"}},
            },
        }

    def _describe(self, resolved: dict, **kwargs) -> tuple[dict, list[list[str]]]:  # type: ignore[no-untyped-def]
        calls: list[list[str]] = []

        def fake_run(cmd, **kw):  # type: ignore[no-untyped-def]
            calls.append(cmd)
            if cmd[1:] == ["--version"]:
                return subprocess.CompletedProcess(cmd, 0, "1.18.29\n", "")
            return subprocess.CompletedProcess(cmd, 0, json.dumps(resolved, indent=2), "")

        explorer = OpenCodeExplorer(repo_root=Path("."), bin_path="oc-test", **kwargs)
        with patch("explorers._cli_agent_base.run_cli", side_effect=fake_run):
            return explorer.describe(), calls

    def test_describe_records_the_resolved_configuration(self) -> None:
        described, calls = self._describe(self._resolved(self.SECRET))
        self.assertEqual(described["cli_version"], "1.18.29")
        self.assertEqual(described["model"], "swe-explore/gpt-5.4")
        self.assertEqual(described["model_source"], "config")
        self.assertEqual(described["mcp_servers"], {
            "off": {"type": "remote", "enabled": False},
            "serena": {"type": "local", "enabled": True},
        })
        self.assertIsNotNone(described["resolved_config_sha256"])
        self.assertIsNotNone(described["prompt_sha256"])
        self.assertIn(["oc-test", "debug", "config", "--pure"], calls)

    def test_describe_holds_no_secret_and_ignores_key_rotation(self) -> None:
        first, _ = self._describe(self._resolved(self.SECRET))
        rotated, _ = self._describe(self._resolved("sk-live-rotated"))
        self.assertNotIn(self.SECRET, json.dumps(first))
        self.assertEqual(first, rotated)

    def test_describe_tells_models_and_prompts_apart(self) -> None:
        base, _ = self._describe(self._resolved(self.SECRET))
        other, _ = self._describe(self._resolved(self.SECRET, model="swe-explore/glm-5"))
        flagged, _ = self._describe(self._resolved(self.SECRET), model="p/pinned")
        prompted, _ = self._describe(self._resolved(self.SECRET), prompt_additions="Be brief.")
        self.assertNotEqual(base["resolved_config_sha256"], other["resolved_config_sha256"])
        self.assertEqual((flagged["model"], flagged["model_source"]), ("p/pinned", "flag"))
        self.assertNotEqual(base["prompt_sha256"], prompted["prompt_sha256"])

    def test_the_selected_agents_model_outranks_the_global_one(self) -> None:
        """Pinning the global model would override what --agent selected."""
        resolved = self._resolved(self.SECRET)
        resolved["agent"] = {"probe": {"model": "swe-explore/special"}}
        described, _ = self._describe(resolved, agent="probe")
        self.assertEqual(described["agent"], "probe")
        self.assertEqual(described["model"], "swe-explore/special")
        self.assertEqual(described["model_source"], "agent")
        # --model still wins over both.
        flagged, _ = self._describe(resolved, agent="probe", model="p/pinned")
        self.assertEqual((flagged["model"], flagged["model_source"]), ("p/pinned", "flag"))
        # An agent that names no model falls back to the global one.
        plain, _ = self._describe(resolved, agent="other")
        self.assertEqual((plain["model"], plain["model_source"]),
                         ("swe-explore/gpt-5.4", "config"))

    def test_the_implicit_default_agents_model_outranks_the_global_one(self) -> None:
        """With no agent named, the CLI still runs one, and its model wins."""
        resolved = self._resolved(self.SECRET)
        resolved["agent"] = {"build": {"model": "swe-explore/agent-default"}}
        described, _ = self._describe(resolved)
        self.assertEqual(described["agent"], "build")
        self.assertEqual(described["model"], "swe-explore/agent-default")
        self.assertEqual(described["model_source"], "agent")
        # An explicitly named agent still takes precedence over the implicit one.
        resolved["agent"]["probe"] = {"model": "swe-explore/probe"}
        named, _ = self._describe(resolved, agent="probe")
        self.assertEqual(named["model"], "swe-explore/probe")
        # And a configuration whose agent names no model is unaffected.
        plain, _ = self._describe(self._resolved(self.SECRET))
        self.assertEqual((plain["model"], plain["model_source"]),
                         ("swe-explore/gpt-5.4", "config"))

    def test_profile_hash_ignores_a_rotated_key_but_not_a_real_change(self) -> None:
        """A profile may hold an inline credential; rotating it must not
        look like a different configuration (README "Run manifest")."""
        def profile_hash(api_key: str, model: str = "m/one") -> str | None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "opencode.json").write_text(
                    json.dumps({"model": model, "provider": {"p": {"options": {
                        "apiKey": api_key}}}}),
                    encoding="utf-8",
                )
                (root / "notes.txt").write_text("not json", encoding="utf-8")
                described, _ = self._describe(
                    self._resolved(self.SECRET), config_dir=root
                )
                return described["profile_sha256"]

        self.assertIsNotNone(profile_hash("sk-one"))
        self.assertEqual(profile_hash("sk-one"), profile_hash("sk-rotated"))
        self.assertNotEqual(profile_hash("sk-one"), profile_hash("sk-one", model="m/two"))

    def test_trailing_output_after_the_config_is_tolerated(self) -> None:
        """A plugin's "Done in 12ms" line must not cost us the configuration."""
        resolved = self._resolved(self.SECRET)

        def fake_run(cmd, **kw):  # type: ignore[no-untyped-def]
            if cmd[1:] == ["--version"]:
                return subprocess.CompletedProcess(cmd, 0, "1.18.29\n", "")
            noisy = f"reading config\n{json.dumps(resolved)}\nDone in 12ms\n"
            return subprocess.CompletedProcess(cmd, 0, noisy, "")

        explorer = OpenCodeExplorer(repo_root=Path("."), bin_path="oc-test")
        with patch("explorers._cli_agent_base.run_cli", side_effect=fake_run):
            described = explorer.describe()
        clean, _ = self._describe(resolved)
        self.assertEqual(described["model"], "swe-explore/gpt-5.4")
        self.assertEqual(described["mcp_servers"], clean["mcp_servers"])
        self.assertEqual(
            described["resolved_config_sha256"], clean["resolved_config_sha256"]
        )

    def test_profile_hash_ignores_desktop_metadata_files(self) -> None:
        """Opening the profile in Finder must not look like a config change."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "opencode.json").write_text('{"model": "m/one"}', encoding="utf-8")
            before, _ = self._describe(self._resolved(self.SECRET), config_dir=root)
            (root / ".DS_Store").write_bytes(b"\x00\x01finder")
            after, _ = self._describe(self._resolved(self.SECRET), config_dir=root)
            # A dotfile that is real configuration still counts.
            (root / ".env").write_text("MODE=fast\n", encoding="utf-8")
            with_env, _ = self._describe(self._resolved(self.SECRET), config_dir=root)
        self.assertEqual(before["profile_sha256"], after["profile_sha256"])
        self.assertNotEqual(before["profile_sha256"], with_env["profile_sha256"])

    def test_profile_hash_skips_installed_dependency_trees(self) -> None:
        """node_modules is pinned by the lockfile and is not read."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "opencode.json").write_text('{"model": "m/one"}', encoding="utf-8")
            deps = root / "node_modules" / "pkg"
            deps.mkdir(parents=True)
            (deps / "index.js") .write_text("module.exports = 1", encoding="utf-8")
            before, _ = self._describe(self._resolved(self.SECRET), config_dir=root)
            (deps / "index.js").write_text("module.exports = 2", encoding="utf-8")
            after, _ = self._describe(self._resolved(self.SECRET), config_dir=root)
        self.assertEqual(before["profile_sha256"], after["profile_sha256"])

    def test_profile_hash_skips_what_the_cli_generated(self) -> None:
        """OpenCode installs plugins into the profile and lists what it wrote."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "opencode.json").write_text('{"model": "m/one"}', encoding="utf-8")
            before, _ = self._describe(self._resolved(self.SECRET), config_dir=root)
            # What `opencode` writes on first use, `.gitignore` included.
            (root / ".gitignore").write_text(
                "node_modules\npackage.json\npackage-lock.json\nbun.lock\n",
                encoding="utf-8",
            )
            (root / "package.json").write_text('{"dependencies": {}}', encoding="utf-8")
            (root / "bun.lock").write_text("lockfile v1", encoding="utf-8")
            after, _ = self._describe(self._resolved(self.SECRET), config_dir=root)
            # A file it did not generate still counts.
            (root / "plugin.json").write_text('{"on": "chat"}', encoding="utf-8")
            with_plugin, _ = self._describe(self._resolved(self.SECRET), config_dir=root)
        self.assertEqual(before["profile_sha256"], after["profile_sha256"])
        self.assertNotEqual(before["profile_sha256"], with_plugin["profile_sha256"])

    def test_profile_hash_ignores_the_git_file_of_a_work_tree(self) -> None:
        """In a work tree `.git` is a file holding this machine's path."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "opencode.json").write_text('{"model": "m/one"}', encoding="utf-8")
            before, _ = self._describe(self._resolved(self.SECRET), config_dir=root)
            (root / ".git").write_text(
                "gdir: /Users/someone/checkouts/bench/.git/worktrees/w1\n",
                encoding="utf-8",
            )
            after, _ = self._describe(self._resolved(self.SECRET), config_dir=root)
        self.assertEqual(before["profile_sha256"], after["profile_sha256"])

    def test_a_dangling_symlink_in_the_profile_does_not_crash_the_run(self) -> None:
        """os.walk lists a broken link among the files; opening it raises."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "opencode.json").write_text('{"model": "m/one"}', encoding="utf-8")
            os.symlink(str(root / "never-created.json"), str(root / "dangling"))
            described, _ = self._describe(self._resolved(self.SECRET), config_dir=root)
        self.assertIsNotNone(described["profile_sha256"])

    def test_config_is_found_among_the_cli_s_own_chatter(self) -> None:
        """A brace in a log line before the object must not be mistaken for it."""
        resolved = self._resolved(self.SECRET)
        noisy = f"[plugin] loaded {{foo}}\n{json.dumps(resolved)}\nDone in 12ms\n"

        def fake_run(cmd, **kw):  # type: ignore[no-untyped-def]
            if cmd[1:] == ["--version"]:
                return subprocess.CompletedProcess(cmd, 0, "1.18.29\n", "")
            return subprocess.CompletedProcess(cmd, 0, noisy, "")

        explorer = OpenCodeExplorer(repo_root=Path("."), bin_path="oc-test")
        with patch("explorers._cli_agent_base.run_cli", side_effect=fake_run):
            described = explorer.describe()
        clean, _ = self._describe(resolved)
        self.assertEqual(described["model"], "swe-explore/gpt-5.4")
        self.assertEqual(
            described["resolved_config_sha256"], clean["resolved_config_sha256"]
        )

    def test_redaction_reaches_plural_keys_and_list_values(self) -> None:
        redacted = redact_config({
            "api_keys": ["sk-1", "sk-2"], "access_tokens": "t", "secrets": ["s"],
            "cookies": "c", "maxTokens": 4096, "numKeys": 2, "keybinds": {"a": "b"},
        })
        self.assertEqual(redacted, {
            "api_keys": [REDACTED, REDACTED], "access_tokens": REDACTED,
            "secrets": [REDACTED], "cookies": REDACTED,
            # A key that counts something is a limit, and still identifies a run.
            "maxTokens": 4096, "numKeys": 2, "keybinds": {"a": "b"},
        })

    def test_redaction_keeps_limits_and_drops_credentials(self) -> None:
        redacted = redact_config({
            "apiKey": "k", "api_key": "k", "accessToken": "t", "password": "p",
            "maxTokens": 4096, "keybinds": {"leader": "ctrl+x"},
            "headers": {"Authorization": "Bearer t"},
        })
        self.assertEqual(redacted, {
            "apiKey": REDACTED, "api_key": REDACTED, "accessToken": REDACTED,
            "password": REDACTED, "maxTokens": 4096, "keybinds": {"leader": "ctrl+x"},
            "headers": {"Authorization": REDACTED},
        })

    def test_redaction_reaches_a_credential_object(self) -> None:
        """A secret named only by its parent key was left in the clear."""
        redacted = redact_config({
            "credentials": {"data": "sk-live-1", "client_id": "acme", "n": 2},
            "provider": {"fake": {"options": {"apiKey": "sk-live-2"}}},
        })
        self.assertEqual(redacted, {
            "credentials": {"data": REDACTED, "client_id": REDACTED, "n": 2},
            "provider": {"fake": {"options": {"apiKey": REDACTED}}},
        })

    def test_redaction_keeps_numbers_and_flags_under_credential_names(self) -> None:
        """Only a string is a credential; a count is configuration."""
        redacted = redact_config({"tokens": 4096, "keys": 12, "cache_key": True})

        self.assertEqual(redacted, {"tokens": 4096, "keys": 12, "cache_key": True})

    def test_redaction_reaches_an_opaque_map_inside_the_configuration(self) -> None:
        """`environment` and `headers` are handed over verbatim, wherever they sit."""
        redacted = redact_config({
            "mcp": {"serena": {"environment": {"SERENA_HOME": "/srv", "TOKEN_A": "t"}}},
            "provider": {"fake": {"headers": {"X-Tenant": "acme"}}},
        })
        self.assertEqual(redacted, {
            "mcp": {"serena": {"environment": {"SERENA_HOME": REDACTED,
                                               "TOKEN_A": REDACTED}}},
            "provider": {"fake": {"headers": {"X-Tenant": REDACTED}}},
        })

    def test_redaction_covers_key_ids_passphrases_and_auth(self) -> None:
        redacted = redact_config({
            "access_key_id": "AKIA0", "passphrase": "p", "auth": "Bearer t",
            # `oauth` names a provider block, not a secret of its own.
            "oauth": {"issuer": "https://issuer.example", "token": "t"},
        })
        self.assertEqual(redacted, {
            "access_key_id": REDACTED, "passphrase": REDACTED, "auth": REDACTED,
            "oauth": {"issuer": "https://issuer.example", "token": REDACTED},
        })


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
            return subprocess.CompletedProcess(cmd, 0, stdout=ANSWER_EVENT, stderr="")

        explorer = OpenCodeExplorer(repo_root=self.repo, bin_path="x", config_dir=self.cfg)
        with patch("explorers._cli_agent_base.run_cli", side_effect=fake_run):
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
