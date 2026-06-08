from __future__ import annotations

import unittest
from pathlib import Path

from quality.run_mini_swe_agent import _build_trajectory_info, _read_regions_from_docker, resolve_runtime_config


CONFIG_PATH = Path("/root/jialiang/swe-explore/quality/configs/mini_swe_agent.yaml")


class _FakeDockerEnv:
    def execute(self, action: dict, timeout: int | None = None) -> dict[str, str]:
        command = action.get("command", "")
        if command == "sed -n '1,$p' /testbed/foo.py":
            return {"output": "line1\nline2\nline3\n"}
        raise AssertionError(f"unexpected command: {command}")



class RunMiniSweAgentConfigTest(unittest.TestCase):
    def test_resolve_runtime_config_uses_yaml_step_limit_when_cli_absent(self) -> None:
        resolved, resolution = resolve_runtime_config(
            config_path=CONFIG_PATH,
            model_name="gpt-5-mini-2025-08-07",
            api_key="sk-test-key",
            base_url="https://api.example.test/v1",
            image_name="swebench/test-image:latest",
            max_steps=None,
        )

        self.assertEqual(resolution["agent"]["yaml_step_limit"], 250)
        self.assertIsNone(resolution["agent"]["cli_max_steps"])
        self.assertEqual(resolution["agent"]["effective_step_limit"], 250)
        self.assertEqual(resolution["agent"]["step_limit_source"], "config:agent.step_limit")
        self.assertEqual((resolved.get("agent") or {}).get("step_limit"), 250)

    def test_resolve_runtime_config_cli_max_steps_overrides_yaml(self) -> None:
        resolved, resolution = resolve_runtime_config(
            config_path=CONFIG_PATH,
            model_name="gpt-5-mini-2025-08-07",
            api_key="sk-test-key",
            base_url="https://api.example.test/v1",
            image_name="swebench/test-image:latest",
            max_steps=24,
        )

        self.assertEqual(resolution["agent"]["yaml_step_limit"], 250)
        self.assertEqual(resolution["agent"]["cli_max_steps"], 24)
        self.assertEqual(resolution["agent"]["effective_step_limit"], 24)
        self.assertEqual(resolution["agent"]["step_limit_source"], "cli:max_steps")
        self.assertEqual((resolved.get("agent") or {}).get("step_limit"), 24)

    def test_resolve_runtime_config_has_no_explicit_token_limit_by_default(self) -> None:
        _, resolution = resolve_runtime_config(
            config_path=CONFIG_PATH,
            model_name="gpt-5-mini-2025-08-07",
            api_key="sk-test-key",
            base_url="https://api.example.test/v1",
            image_name="swebench/test-image:latest",
            max_steps=24,
        )

        self.assertFalse(resolution["model"]["has_explicit_token_limit"])
        token_limits = resolution["model"]["token_limits"]
        self.assertEqual(
            set(token_limits),
            {"max_tokens", "max_completion_tokens", "max_output_tokens", "max_reasoning_tokens"},
        )
        for values in token_limits.values():
            self.assertIsNone(values["yaml"])
            self.assertIsNone(values["effective"])

    def test_read_regions_from_docker_preserves_whole_file_end_minus_one(self) -> None:
        regions = _read_regions_from_docker(
            _FakeDockerEnv(),
            [{"path": "foo.py", "start": 1, "end": -1}],
        )

        self.assertEqual(regions, [{"path": "foo.py", "start": 1, "end": -1, "content": "line1\nline2\nline3\n"}])

    def test_build_trajectory_info_carries_config_resolution(self) -> None:
        info = _build_trajectory_info(
            exit_status="Submitted",
            submission="diff --git a/foo.py b/foo.py",
            config_resolution={"agent": {"effective_step_limit": 24}},
        )

        self.assertEqual(info["exit_status"], "Submitted")
        self.assertIn("config_resolution", info)
        self.assertEqual(info["config_resolution"]["agent"]["effective_step_limit"], 24)


if __name__ == "__main__":
    unittest.main()
