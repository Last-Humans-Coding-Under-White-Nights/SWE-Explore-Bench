"""LocAgent explorer — graph-guided LLM agent for code localization.

Paper: Graph-Guided LLM Agents for Code Localization (ACL'25)
Repo:  https://github.com/gersteinlab/LocAgent

LocAgent parses codebases into directed heterogeneous graphs and uses
LLM agents to search and locate relevant entities via multi-hop reasoning.
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
from ._paths import locagent_path as _default_locagent_path
from .parsing import parse_locagent_jsonl
from .subprocess_utils import run_in_conda


@dataclass
class LocAgentExplorer(Explorer):
    """Wraps LocAgent via subprocess (conda run).

    Args:
        repo_root: Repository directory to explore.
        locagent_path: Path to LocAgent installation.
        model: LiteLLM model name (e.g. "openai/gpt-4o").
        conda_env: Conda environment with LocAgent deps.
        api_key: OpenAI-compatible API key.
        api_base: OpenAI-compatible API base URL.
        graph_index_dir: Directory for pre-built graph indices.
        num_samples: Number of localization samples.
        timeout: Per-instance wall-clock timeout in seconds.
    """

    repo_root: Path
    locagent_path: Path = field(default_factory=_default_locagent_path)
    model: str = "openai/gpt-4o"
    conda_env: str = "locagent"
    api_key: str | None = None
    api_base: str | None = None
    graph_index_dir: str | None = None
    bm25_index_dir: str | None = None
    num_samples: int = 1
    timeout: int = 1800

    def _derive_github_repo(self, instance_id: str) -> str:
        """Derive GitHub org/repo from repo_root name and instance_id.

        Handles both SWE-bench format (org__repo-N) and ScaleSWE format
        (org_repo_prN with repo_dir org__repo-N).
        """
        basename = self.repo_root.name  # e.g. "facelessuser__pymdown-extensions-2634"

        if "__" in basename:
            org, repo_part = basename.split("__", 1)
            # Strip trailing issue number (e.g. "-2634") from repo part.
            # Extract issue number from instance_id for precise stripping.
            m = re.search(r'[-_](\d+)$', instance_id)
            if m:
                suffix = f"-{m.group(1)}"
                if repo_part.endswith(suffix):
                    repo_part = repo_part[:-len(suffix)]
            return f"{org}/{repo_part}"

        # Fallback: old SWE-bench style from instance_id
        prefix = instance_id.rsplit("-", 1)[0]
        parts = prefix.split("__", 1)
        return "/".join(parts) if len(parts) == 2 else prefix

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5,
    ) -> List[ExplorerResult]:
        """Run LocAgent on a single instance and parse results."""
        with tempfile.TemporaryDirectory(prefix="locagent_") as tmpdir:
            output_folder = os.path.join(tmpdir, "output")
            os.makedirs(output_folder, exist_ok=True)

            # Write a single-instance dataset JSONL for LocAgent
            dataset_file = os.path.join(tmpdir, "instance.jsonl")
            github_repo = self._derive_github_repo(instance_id)
            _head = getattr(self, "base_commit", None) or "HEAD"
            # Inject explicit top-K + early-finish directive at the end of the
            # issue body. LocAgent's auto_search PR_TEMPLATE uses the first
            # line as title and the rest as description, so appending here
            # preserves the title and adds a clear stop signal for the agent.
            top_k_hint = (
                f"\n\n---\n"
                f"IMPORTANT TASK INSTRUCTIONS:\n"
                f"- Your goal is to identify the **top {top_k} most relevant code regions** "
                f"(file + class/function or file:line range) that need to be inspected to resolve this issue.\n"
                f"- Use AT MOST 8 search/explore tool calls. Be focused; avoid redundant or repeated searches.\n"
                f"- As soon as you have {top_k} candidate regions, output them in the format "
                f"`file_path:QualifiedName` (one per line, ordered by relevance) and "
                f"IMMEDIATELY call <finish></finish>. Do not continue searching.\n"
            )
            problem_with_hint = (query or "").rstrip() + top_k_hint
            instance_data = {
                "instance_id": instance_id,
                "repo": github_repo,
                "base_commit": _head,
                "problem_statement": problem_with_hint,
                "patch": "",
            }
            with open(dataset_file, "w") as f:
                f.write(json.dumps(instance_data) + "\n")

            cmd = [
                "python", str(Path(__file__).parent / "_locagent_shim.py"),
                "--localize",
                "--merge",
                "--model", self.model,
                "--dataset", dataset_file,
                "--split", "train",
                "--output_folder", output_folder,
                "--num_samples", str(self.num_samples),
                "--num_processes", "1",
                "--eval_n_limit", "1",
                "--use_function_calling",
                "--simple_desc",
                "--timeout", str(self.timeout - 60),
            ]

            env: dict[str, str] = {"LOCAGENT_ROOT": str(self.locagent_path)}
            # Child processes spawned by torch.multiprocessing need LocAgent on PYTHONPATH
            env["PYTHONPATH"] = str(self.locagent_path) + ":" + env.get("PYTHONPATH", "")
            # LocAgent requires graph/BM25 index dirs (creates them on first run).
            # Use a persistent per-instance cache so index build (slow on big
            # repos like django) is reused across runs.
            persistent_idx_root = os.environ.get(
                "LOCAGENT_INDEX_CACHE",
                str(Path.home() / ".cache" / "locagent_index"),
            )
            idx_dir = os.path.join(persistent_idx_root, instance_id)
            os.makedirs(idx_dir, exist_ok=True)
            env.setdefault("GRAPH_INDEX_DIR", os.path.join(idx_dir, "graph_index"))
            env.setdefault("BM25_INDEX_DIR", os.path.join(idx_dir, "bm25_index"))
            if self.api_key:
                env["OPENAI_API_KEY"] = self.api_key
            if self.api_base:
                _base = self.api_base.rstrip("/")
                if not _base.endswith("/v1"):
                    _base += "/v1"
                env["OPENAI_API_BASE"] = _base
            if self.graph_index_dir:
                env["GRAPH_INDEX_DIR"] = self.graph_index_dir
            if self.bm25_index_dir:
                env["BM25_INDEX_DIR"] = self.bm25_index_dir
            # Pass local repo path so the shim can symlink instead of cloning
            env["LOCAL_REPO_PATH"] = str(self.repo_root)
            # Forward iteration caps to LocAgent (forces termination of
            # non-cooperative reasoning models that ignore <finish>).
            for _k in ("LOCAGENT_MAX_ITER", "LOCAGENT_HARD_MAX_ITER"):
                if _k in os.environ:
                    env[_k] = os.environ[_k]
            env.setdefault("LOCAGENT_MAX_ITER", "10")
            env.setdefault("LOCAGENT_HARD_MAX_ITER", "13")

            try:
                run_in_conda(
                    self.conda_env, cmd,
                    env=env,
                    cwd=str(self.locagent_path),
                    timeout=self.timeout,
                )
            except Exception as e:
                raise RuntimeError(f"LocAgent failed for {instance_id}: {e}") from e

            # Parse merged output (preferred) or raw output
            merged = os.path.join(output_folder, "merged_loc_outputs.jsonl")
            if not os.path.isfile(merged):
                merged = os.path.join(output_folder, "merged_loc_outputs_mrr.jsonl")
            if not os.path.isfile(merged):
                merged = os.path.join(output_folder, "loc_outputs.jsonl")

            if os.environ.get("LOCAGENT_KEEP_TMP"):
                import shutil as _sh
                _keep_root = os.environ.get(
                    "LOCAGENT_KEEP_TMP_ROOT",
                    str(Path.home() / "locagent_keep"),
                )
                _dst = os.path.join(_keep_root, instance_id)
                _sh.rmtree(_dst, ignore_errors=True)
                _sh.copytree(tmpdir, _dst)
                print(f"[LOCAGENT_KEEP_TMP] copied tmpdir -> {_dst}", flush=True)

            return parse_locagent_jsonl(
                merged, instance_id, str(self.repo_root),
            )
