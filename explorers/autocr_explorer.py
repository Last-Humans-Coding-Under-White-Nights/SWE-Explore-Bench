"""AutoCodeRover explorer — AST-aware search API + statistical fault localization.

Paper: AutoCodeRover (ISSTA'24)
Repo:  https://github.com/AutoCodeRoverSG/auto-code-rover

AutoCodeRover uses a stratified context retrieval approach with AST-aware
search APIs and spectrum-based fault localization to locate buggy code.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .base import Explorer, ExplorerResult
from ._paths import autocr_path as _default_autocr_path
from .parsing import parse_acr_bug_locations
from .subprocess_utils import run_in_conda


@dataclass
class AutoCodeRoverExplorer(Explorer):
    """Wraps AutoCodeRover via subprocess (conda run) in local-issue mode.

    Args:
        repo_root: Repository directory to explore.
        acr_path: Path to auto-code-rover installation.
        model: LLM model name (e.g. "gpt-4o-2024-05-13").
        conda_env: Conda environment name.
        api_key: API key (set as OPENAI_KEY env var).
        api_base: API base URL (set as OPENAI_API_BASE env var).
        conv_round_limit: Max conversation rounds.
        timeout: Per-instance wall-clock timeout in seconds.
    """

    repo_root: Path
    acr_path: Path = field(default_factory=_default_autocr_path)
    model: str = "gpt-4o-2024-05-13"
    conda_env: str = "auto-code-rover"
    api_key: str | None = None
    api_base: str | None = None
    conv_round_limit: int = 10
    timeout: int = 1200

    # Known ACR built-in model names (no prefix needed)
    _ACR_BUILTIN_MODELS = {
        "gpt-4o-2024-08-06", "gpt-4o-2024-05-13", "gpt-4-turbo-2024-04-09",
        "gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613",
        "gpt-4-0613", "gpt-4o-mini-2024-07-18", "o1-mini", "o1",
    }

    def _resolve_model(self) -> str:
        """Prefix model name with litellm-generic-openai/ if not built-in."""
        m = self.model
        if m in self._ACR_BUILTIN_MODELS or m.startswith("litellm-"):
            return m
        # Route through LiteLLM with OpenAI-compatible provider
        if not m.startswith("openai/"):
            m = f"openai/{m}"
        return f"litellm-generic-{m}"

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5,
    ) -> List[ExplorerResult]:
        """Run AutoCodeRover local-issue mode and parse bug locations."""
        with tempfile.TemporaryDirectory(prefix="acr_") as tmpdir:
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Write issue text to a file
            issue_file = os.path.join(tmpdir, "issue.txt")
            with open(issue_file, "w") as f:
                f.write(query)

            # Write a shim script that patches ACR's runtime environment
            shim_path = os.path.join(tmpdir, "acr_shim.py")
            with open(shim_path, "w") as sf:
                sf.write(f"""import sys, os, argparse

# 1. Inject ACR root into sys.path
sys.path.insert(0, "{self.acr_path}")
os.chdir("{self.acr_path}")

# 2. Configure litellm for custom API endpoints
import litellm
litellm.drop_params = True
_api_base = os.getenv("OPENAI_API_BASE_URL", os.getenv("OPENAI_API_BASE", ""))
if _api_base:
    litellm.api_base = _api_base

# 3. Patch argparse to allow custom model names (bypass choices whitelist)
_orig_add_argument = argparse.ArgumentParser.add_argument
def _patched_add_argument(self, *args, **kwargs):
    if args and args[0] == "--model" and "choices" in kwargs:
        kwargs.pop("choices")
    return _orig_add_argument(self, *args, **kwargs)
argparse.ArgumentParser.add_argument = _patched_add_argument

# 4. local-issue mode uses PlainTask whose execute_reproducer is unimplemented
#    and would abort the workflow before the search/localization step.
#    Patch it (and Task base) to always return a non-reproduced result so the
#    pipeline proceeds to bug-location search.
from app.task import Task as _AcrTask, PlainTask as _AcrPlainTask
from app.data_structures import ReproResult as _AcrReproResult
def _stub_execute_reproducer(self, *_a, **_kw):
    return _AcrReproResult("", "", 0)
_AcrTask.execute_reproducer = _stub_execute_reproducer
_AcrPlainTask.execute_reproducer = _stub_execute_reproducer

# 5. Run ACR main
import runpy
runpy.run_path(os.path.join("{self.acr_path}", "app", "main.py"), run_name="__main__")
""")

            cmd = [
                "python", "-u", shim_path, "local-issue",
                "--output-dir", output_dir,
                "--model", self._resolve_model(),
                "--model-temperature", "0.2",
                "--task-id", instance_id,
                "--local-repo", str(self.repo_root),
                "--issue-file", issue_file,
                "--conv-round-limit", str(self.conv_round_limit),
                "--enable-layered",
            ]

            env: dict[str, str] = {
                "PYTHONPATH": str(self.acr_path),
            }
            if self.api_key:
                env["OPENAI_KEY"] = self.api_key
                env["OPENAI_API_KEY"] = self.api_key
                env["ANTHROPIC_API_KEY"] = self.api_key
            if self.api_base:
                env["OPENAI_API_BASE"] = self.api_base
                env["OPENAI_API_BASE_URL"] = self.api_base
                env["ANTHROPIC_API_BASE"] = self.api_base

            try:
                proc = run_in_conda(
                    self.conda_env, cmd,
                    env=env,
                    cwd=str(self.acr_path),
                    timeout=self.timeout,
                )
            except Exception as e:
                raise RuntimeError(
                    f"AutoCodeRover failed for {instance_id}: {e}"
                ) from e

            # Find the timestamped output directory
            bug_loc_path = self._find_bug_locations(output_dir, instance_id)
            if not bug_loc_path:
                # Persist subprocess output AND ACR's output tree for offline debugging
                import sys as _sys, shutil as _shutil
                _dbg = Path("/tmp") / f"acr_debug_{instance_id}"
                if _dbg.exists():
                    _shutil.rmtree(str(_dbg))
                _dbg.mkdir(parents=True, exist_ok=True)
                (_dbg / "stdout.txt").write_text(getattr(proc, "stdout", "") or "")
                (_dbg / "stderr.txt").write_text(getattr(proc, "stderr", "") or "")
                # Snapshot ACR's full output tree so we can inspect what was actually written
                try:
                    _shutil.copytree(output_dir, str(_dbg / "output"), dirs_exist_ok=True)
                except Exception:
                    pass
                _sys.stderr.write(
                    f"  [autocr] no bug_locations for {instance_id}; "
                    f"subprocess output saved to {_dbg}/\n"
                )
                return []

            return parse_acr_bug_locations(bug_loc_path, instance_id)

    @staticmethod
    def _find_bug_locations(output_dir: str, instance_id: str) -> str | None:
        """Locate ``bug_locations_after_process.json`` in the output tree.

        AutoCodeRover creates ``{output_dir}/{task_id}_{timestamp}/output_N/``
        directories.  We search for the latest one.
        """
        output_root = Path(output_dir)
        candidates: list[Path] = []

        for d in sorted(output_root.iterdir(), reverse=True):
            if not d.is_dir() or not d.name.startswith(instance_id):
                continue
            # Search output_0, output_1, ...  (prefer higher index = later retry)
            for sub in sorted(d.iterdir(), reverse=True):
                if not sub.is_dir() or not sub.name.startswith("output_"):
                    continue
                # ACR stores in output_N/search/ or output_N/
                for loc in [sub / "search" / "bug_locations_after_process.json",
                            sub / "bug_locations_after_process.json"]:
                    if loc.is_file():
                        candidates.append(loc)
                        break

        if candidates:
            return str(candidates[0])
        return None
