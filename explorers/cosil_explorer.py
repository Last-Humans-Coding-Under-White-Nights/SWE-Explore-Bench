"""CoSIL explorer — module-call-graph → function-call-graph two-stage search.

Paper: CoSIL (ASE'25)
Repo:  https://github.com/ZhonghaoJiang/CoSIL

CoSIL uses a two-phase approach: first file-level localization via module
call graph context, then function-level localization with a pruning agent.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .base import Explorer, ExplorerResult
from ._paths import cosil_path as _default_cosil_path
from .parsing import parse_cosil_jsonl
from .subprocess_utils import run_in_conda


@dataclass
class CoSILExplorer(Explorer):
    """Wraps CoSIL via subprocess (conda run).

    Because CoSIL hardcodes API keys in ``afl/util/api_requests.py``,
    we create a wrapper script that patches the OpenAI client at runtime.

    Args:
        repo_root: Repository directory to explore.
        cosil_path: Path to CoSIL installation.
        model: LLM model name (e.g. "gpt-4o-2024-08-06").
        conda_env: Conda environment with CoSIL deps.
        api_key: OpenAI-compatible API key.
        api_base: OpenAI-compatible API base URL.
        structure_cache_dir: Directory for pre-generated repo structures.
        timeout: Per-instance wall-clock timeout in seconds.
    """

    repo_root: Path
    cosil_path: Path = field(default_factory=_default_cosil_path)
    model: str = "gpt-4o-2024-08-06"
    conda_env: str = "cosil"
    api_key: str | None = None
    api_base: str | None = None
    structure_cache_dir: str | None = None
    timeout: int = 900

    def _derive_github_repo(self, instance_id: str) -> str:
        """Derive GitHub org/repo from repo_root name and instance_id.

        Handles both SWE-bench format (org__repo-N) and ScaleSWE format
        (org_repo_prN with repo_dir org__repo-N).
        """
        basename = self.repo_root.name

        if "__" in basename:
            org, repo_part = basename.split("__", 1)
            m = re.search(r'[-_](\d+)$', instance_id)
            if m:
                suffix = f"-{m.group(1)}"
                if repo_part.endswith(suffix):
                    repo_part = repo_part[:-len(suffix)]
            return f"{org}/{repo_part}"

        prefix = instance_id.rsplit("-", 1)[0]
        parts = prefix.split("__", 1)
        return "/".join(parts) if len(parts) == 2 else prefix

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5,
    ) -> List[ExplorerResult]:
        """Run CoSIL two-phase localization on a single instance."""
        self._ensure_repo_structure(instance_id)
        with tempfile.TemporaryDirectory(prefix="cosil_") as tmpdir:
            file_out = os.path.join(tmpdir, "file_level")
            func_out = os.path.join(tmpdir, "func_level")
            os.makedirs(file_out, exist_ok=True)
            os.makedirs(func_out, exist_ok=True)

            env = self._build_env(tmpdir)

            # Write single-instance dataset JSONL for CoSIL
            dataset_file = os.path.join(tmpdir, "instance.jsonl")
            github_repo = self._derive_github_repo(instance_id)
            instance_data = {
                "instance_id": instance_id,
                "repo": github_repo,
                "base_commit": "HEAD",
                "problem_statement": query,
                "patch": "",
            }
            with open(dataset_file, "w") as f:
                f.write(json.dumps(instance_data) + "\n")

            # Create the API-patching wrapper script
            wrapper = self._write_wrapper_script(tmpdir, dataset_file)

            # Phase 1: file-level localization
            file_cmd = [
                "python", wrapper, "file",
                "--file_level",
                "--output_folder", file_out,
                "--output_file", "loc_outputs.jsonl",
                "--model", self.model,
                "--backend", "openai",
                "--dataset", dataset_file,
                "--target_id", instance_id,
                "--top_n", str(top_k),
                "--num_threads", "1",
            ]

            try:
                run_in_conda(
                    self.conda_env, file_cmd,
                    env=env,
                    cwd=str(self.cosil_path),
                    timeout=self.timeout // 2,
                )
            except Exception as e:
                raise RuntimeError(
                    f"CoSIL file-level failed for {instance_id}: {e}"
                ) from e

            file_loc = os.path.join(file_out, "loc_outputs.jsonl")
            if not os.path.isfile(file_loc):
                return []

            # Phase 2: function-level localization
            func_cmd = [
                "python", wrapper, "func",
                "--output_folder", func_out,
                "--output_file", "loc_func.jsonl",
                "--loc_file", file_loc,
                "--model", self.model,
                "--backend", "openai",
                "--dataset", dataset_file,
                "--target_id", instance_id,
                "--top_n", str(top_k),
                "--num_threads", "1",
            ]

            try:
                run_in_conda(
                    self.conda_env, func_cmd,
                    env=env,
                    cwd=str(self.cosil_path),
                    timeout=self.timeout // 2,
                )
            except Exception as e:
                # func-level may fail; fall back to file-level results
                pass

            func_loc = os.path.join(func_out, "loc_func.jsonl")
            results = parse_cosil_jsonl(
                func_loc, instance_id, str(self.repo_root),
            )
            if results:
                return results

            # Fallback: file-level only
            return parse_cosil_jsonl(
                file_loc, instance_id, str(self.repo_root),
            )


    def _ensure_repo_structure(self, instance_id: str) -> None:
        """Generate repo_structures/{instance_id}.json if it does not exist.

        CoSIL requires a pre-computed AST structure file for each instance.
        We generate it on-the-fly using CoSIL's own create_structure.
        """
        struct_dir = self.cosil_path / "repo_structures"
        struct_file = struct_dir / f"{instance_id}.json"
        if struct_file.is_file():
            return

        struct_dir.mkdir(exist_ok=True)

        github_repo = self._derive_github_repo(instance_id)

        import tempfile as _tf
        gen_script_path = os.path.join(_tf.gettempdir(), f"gen_struct_{instance_id}.py")
        with open(gen_script_path, "w") as f:
            f.write(f"""import sys, json, os
sys.path.insert(0, "{self.cosil_path}")
from get_repo_structure.get_repo_structure import create_structure

structure = create_structure("{self.repo_root}")
d = {{"repo": "{github_repo}", "base_commit": "HEAD", "structure": structure, "instance_id": "{instance_id}"}}
os.makedirs("{struct_dir}", exist_ok=True)
with open("{struct_file}", "w") as fout:
    json.dump(d, fout)
print(f"Generated {struct_file}")
""")
        run_in_conda(
            self.conda_env,
            ["python", gen_script_path],
            cwd=str(self.cosil_path),
            timeout=120,
        )

    def _build_env(self, tmpdir: str) -> dict[str, str]:
        """Build environment variables for CoSIL subprocess."""
        env: dict[str, str] = {}
        if self.api_key:
            env["OPENAI_API_KEY"] = self.api_key
        if self.api_base:
            base = self.api_base.rstrip("/")
            if not base.endswith("/v1"):
                base += "/v1"
            env["OPENAI_BASE_URL"] = base
        # Always point to the repo_structures directory (generated by _ensure_repo_structure)
        env["PROJECT_FILE_LOC"] = self.structure_cache_dir or str(self.cosil_path / "repo_structures")
        env["PYTHONPATH"] = str(self.cosil_path)
        return env

    def _write_wrapper_script(self, tmpdir: str, dataset_file: str) -> str:
        """Create a wrapper script that monkey-patches CoSIL's API client
        and dataset loading.

        CoSIL hardcodes API keys in ``afl/util/api_requests.py`` and
        uses ``load_dataset`` which expects HuggingFace datasets. We patch
        both to work with our custom JSONL dataset and API endpoint.
        """
        wrapper_path = os.path.join(tmpdir, "cosil_wrapper.py")
        api_key = self.api_key or ""
        api_base = self.api_base or ""

        script = f'''\
import os, sys
sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
sys.path.insert(0, os.path.join(os.environ.get("PYTHONPATH", "."), "afl", "fl"))

# Monkey-patch openai.OpenAI to inject our API key/base
import openai
_OrigOpenAI = openai.OpenAI
def _patched_openai(**kwargs):
    api_key = os.environ.get("OPENAI_API_KEY", "") or "{api_key}"
    base_url = os.environ.get("OPENAI_BASE_URL", "") or "{api_base}"
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return _OrigOpenAI(**kwargs)
openai.OpenAI = _patched_openai

# Monkey-patch datasets.load_dataset to handle local JSONL files
import datasets as _ds_mod
_orig_load = _ds_mod.load_dataset
def _patched_load(path, *a, **kw):
    if isinstance(path, str) and os.path.isfile(path) and path.endswith(('.jsonl', '.json')):
        kw.pop('split', None)
        return _orig_load('json', *a, data_files=path, split='train', **kw)
    return _orig_load(path, *a, **kw)
_ds_mod.load_dataset = _patched_load

# Dispatch to the correct CoSIL entry point
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["file", "func"])
phase, remaining = parser.parse_known_args()

if phase.phase == "file":
    sys.argv = ["AFL_localize_file.py"] + remaining
    exec(open("afl/fl/AFL_localize_file.py").read())
else:
    sys.argv = ["AFL_localize_func.py"] + remaining
    exec(open("afl/fl/AFL_localize_func.py").read())
'''
        with open(wrapper_path, "w") as f:
            f.write(script)
        return wrapper_path
