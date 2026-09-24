"""Shared utilities for running models in isolated conda environments."""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Dict, List

from ._cli_process import run_cli
from ._paths import resolve_conda_exe

logger = logging.getLogger(__name__)


def run_in_conda(
    conda_env: str,
    cmd: List[str],
    *,
    env: Dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: int = 900,
) -> subprocess.CompletedProcess:
    """Run *cmd* inside a conda environment via ``conda run``.

    Args:
        conda_env: Name of the target conda environment.
        cmd: Command tokens, e.g. ``["python", "main.py", "--flag"]``.
        env: Extra environment variables (merged with ``os.environ``).
        cwd: Working directory for the subprocess.
        timeout: Wall-clock timeout in seconds (default 15 min).

    Returns:
        ``subprocess.CompletedProcess`` with captured stdout/stderr.

    Raises:
        RuntimeError: If the process returns a non-zero exit code.
        subprocess.TimeoutExpired: If *timeout* is exceeded.
    """
    # Use full conda path to avoid PATH issues in non-interactive shells
    conda_exe = resolve_conda_exe()
    full_cmd = [conda_exe, "run", "--no-capture-output", "-n", conda_env] + cmd

    merged_env = {**os.environ, **(env or {})}
    # Ensure conda is on PATH
    if "CONDA_EXE" in merged_env:
        conda_bin = os.path.dirname(merged_env["CONDA_EXE"])
        merged_env["PATH"] = conda_bin + ":" + merged_env.get("PATH", "")

    logger.info("conda run -n %s: %s  (cwd=%s)", conda_env, " ".join(cmd), cwd)
    result = run_cli(
        full_cmd, capture_output=True, text=True, cwd=cwd, env=merged_env,
        timeout=timeout, encoding="utf-8", errors="replace",
    )

    if result.returncode != 0:
        stderr_preview = (result.stderr or "")[:2000]
        stdout_preview = (result.stdout or "")[:2000]
        detail = stderr_preview
        if stdout_preview:
            detail = f"STDOUT:\n{stdout_preview}\nSTDERR:\n{stderr_preview}"
        raise RuntimeError(
            f"conda run -n {conda_env} failed (rc={result.returncode}):\n{detail}"
        )
    return result
