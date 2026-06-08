"""OrcaLoca explorer — priority-queue dynamic scheduling + relevance scoring.

Paper: OrcaLoca (ICML'25)
Repo:  https://github.com/fishmingyu/OrcaLoca

OrcaLoca uses Docker containers (based on SWE-Agent image) to clone and
explore repositories.  It produces bug locations with file/class/method
and optional line ranges.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .base import Explorer, ExplorerResult
from ._paths import orcaloca_path as _default_orcaloca_path
from .parsing import parse_orcaloca_output
from .subprocess_utils import run_in_conda


@dataclass
class OrcaLocaExplorer(Explorer):
    """Wraps OrcaLoca via subprocess (conda run + Docker).

    Args:
        repo_root: Repository directory to explore.
        orcaloca_path: Path to OrcaLoca installation.
        model: LLM model name (e.g. "claude-3-5-sonnet-20241022").
        conda_env: Conda environment with OrcaLoca deps.
        docker_image: Docker image for the OrcaLoca runtime.
        api_key: API key (written to key.cfg).
        api_base: API base URL (written to key.cfg).
        timeout: Per-instance wall-clock timeout in seconds.
    """

    repo_root: Path
    orcaloca_path: Path = field(default_factory=_default_orcaloca_path)
    model: str = "gpt-4o"
    conda_env: str = "orcaloca"
    docker_image: str = "hejiaz/swe-agent:latest"
    api_key: str | None = None
    api_base: str | None = None
    timeout: int = 900

    def __post_init__(self):
        self._check_docker()

    def _check_docker(self):
        """Verify Docker daemon is reachable."""
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True, timeout=15, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Docker is required for OrcaLoca but is not available."
            ) from e

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5,
    ) -> List[ExplorerResult]:
        """Run OrcaLoca search stage on a single instance."""
        # Generate unique container name per run so multiple workers can run
        # concurrently without colliding on the default "orcar_swe_bench_run_ctr".
        container_name = f"orcar_run_{uuid.uuid4().hex[:10]}"
        # Best-effort cleanup of any leftover container with this exact name
        # (shouldn't exist since name is unique, but harmless).
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True, timeout=30,
            )
        except Exception:
            pass
        with tempfile.TemporaryDirectory(prefix="orcaloca_") as tmpdir:
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Isolate per-worker HOME so OrcaLoca's host-side cache
            # (~/.orcar/<repo>) does not collide between concurrent workers.
            # We symlink common shared dirs (.cache, .conda, .condarc,
            # .gitconfig) to the real user's HOME so conda/pip/HF/git keep
            # working, but ~/.orcar lives under tmpdir for this worker only.
            fake_home = os.path.join(tmpdir, "home")
            os.makedirs(fake_home, exist_ok=True)
            real_home = os.path.expanduser("~")
            for name in (".cache", ".conda", ".condarc", ".gitconfig", ".config"):
                src = os.path.join(real_home, name)
                if os.path.exists(src):
                    try:
                        os.symlink(src, os.path.join(fake_home, name))
                    except FileExistsError:
                        pass

            # Write key.cfg for API access
            cfg_path = os.path.join(tmpdir, "key.cfg")
            self._write_key_cfg(cfg_path)

            # Step 1: Run evaluation/run.py for the search stage
            cmd = [
                "python", "evaluation/run.py",
                "--model", self.model,
                "--image", self.docker_image,
                "--dataset", "princeton-nlp/SWE-bench_Verified",
                "--instance_ids", instance_id,
                "--final_stage", "search",
                "--cfg_path", cfg_path,
                "--persistent", "False",
                "--redirect_log", "False",
                "--container_name", container_name,
            ]

            env: dict[str, str] = {}
            # Override HOME so OrcaLoca's `~/.orcar` host cache is per-worker.
            env["HOME"] = fake_home
            if self.api_key:
                env["OPENAI_API_KEY"] = self.api_key
            if self.api_base:
                # OrcaLoca's gen_config.py does NOT pass api_base to llama_index's
                # OpenAI(); we instead set the env vars that the underlying openai
                # python SDK respects so requests are routed to our LiteLLM proxy.
                env["OPENAI_API_BASE_URL"] = self.api_base
                env["OPENAI_BASE_URL"] = self.api_base
                env["OPENAI_API_BASE"] = self.api_base

            try:
                run_in_conda(
                    self.conda_env, cmd,
                    env=env,
                    cwd=str(self.orcaloca_path),
                    timeout=self.timeout,
                )
            except Exception as e:
                raise RuntimeError(
                    f"OrcaLoca search failed for {instance_id}: {e}"
                ) from e

            # Step 2: Run process_output.py to resolve line ranges
            process_cmd = [
                "python", "evaluation/process_output.py",
                "--output_dir", output_dir,
            ]

            # Copy searcher output to output_dir for process_output
            searcher_file = os.path.join(
                str(self.orcaloca_path), "output", instance_id,
                f"searcher_{instance_id}.json",
            )
            if os.path.isfile(searcher_file):
                inst_out = os.path.join(output_dir, instance_id)
                os.makedirs(inst_out, exist_ok=True)
                shutil.copy2(searcher_file, inst_out)

            try:
                run_in_conda(
                    self.conda_env, process_cmd,
                    env=env,
                    cwd=str(self.orcaloca_path),
                    timeout=120,
                )
            except Exception:
                pass  # process_output may fail; fall back to raw parsing

            # Parse output.json (from process_output) or raw searcher JSON
            output_json = os.path.join(output_dir, "output.json")
            if os.path.isfile(output_json):
                return parse_orcaloca_output(
                    output_json, instance_id, str(self.repo_root),
                )

            # Fallback: parse raw searcher output directly
            if os.path.isfile(searcher_file):
                return self._parse_raw_searcher(searcher_file, instance_id)
            return []

    def _write_key_cfg(self, path: str):
        """Write a key.cfg file for OrcaLoca API access.

        OrcaLoca uses the red-dove CFG format which requires string values to
        be quoted (otherwise raw URLs containing ':' confuse the parser).
        """
        def _esc(v: str) -> str:
            return v.replace("\\", "\\\\").replace('"', '\\"')

        lines = []
        if self.api_key:
            lines.append(f'OPENAI_API_KEY: "{_esc(self.api_key)}"')
        if self.api_base:
            lines.append(f'OPENAI_API_BASE_URL: "{_esc(self.api_base)}"')
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _parse_raw_searcher(
        self, searcher_path: str, instance_id: str,
    ) -> list[ExplorerResult]:
        """Fallback parser for raw searcher JSON without line ranges."""
        from .base import ContextRegion
        from .parsing import _normalize_path, resolve_entity_lines

        with open(searcher_path) as f:
            data = json.load(f)

        regions: list[ContextRegion] = []
        for loc in data.get("bug_locations", []):
            fpath = _normalize_path(loc.get("file_path", ""))
            if not fpath:
                continue
            cn = loc.get("class_name", "")
            mn = loc.get("method_name", "")
            entity = f"{cn}.{mn}" if cn and mn else (mn or cn or "")
            start, end = 1, -1
            if entity:
                rng = resolve_entity_lines(str(self.repo_root), fpath, entity)
                if rng:
                    start, end = rng
            regions.append(ContextRegion(path=fpath, start=start, end=end))

        if regions:
            return [ExplorerResult(instance_id=instance_id, score=1.0, regions=regions)]
        return []
