"""Centralized resolution of external academic-agent source paths and conda.

All paths can be overridden via environment variables, with defaults pointing
at the shared Explorer-Codes workspace.  Centralizing this avoids hard-coded
user-specific paths across individual explorer wrappers.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# Root directory containing all external agent source checkouts.
# Override via the EXPLORER_CODES_ROOT environment variable.
EXPLORER_CODES_ROOT = Path(
    os.environ.get("EXPLORER_CODES_ROOT", "./third_party/explorer-codes")
)


def _resolve(env_var: str, subdir: str) -> Path:
    """Return ``$env_var`` if set, else ``$EXPLORER_CODES_ROOT/<subdir>``."""
    val = os.environ.get(env_var)
    if val:
        return Path(val)
    return EXPLORER_CODES_ROOT / subdir


def autocr_path() -> Path:
    return _resolve("AUTOCR_PATH", "auto-code-rover")


def cosil_path() -> Path:
    return _resolve("COSIL_PATH", "CoSIL")


def locagent_path() -> Path:
    return _resolve("LOCAGENT_PATH", "LocAgent")


def orcaloca_path() -> Path:
    return _resolve("ORCALOCA_PATH", "OrcaLoca")


def awe_agent_path() -> Path:
    return _resolve("AWE_AGENT_PATH", "AweAgent")


def resolve_conda_exe() -> str:
    """Locate the conda executable.

    Precedence: ``$CONDA_EXE`` → ``which conda`` → common install locations.
    """
    env_val = os.environ.get("CONDA_EXE")
    if env_val and os.path.isfile(env_val):
        return env_val

    which = shutil.which("conda")
    if which:
        return which

    for candidate in (
        "/opt/miniconda3/bin/conda",
        "/opt/conda/bin/conda",
        os.path.expanduser("~/miniconda3/bin/conda"),
        os.path.expanduser("~/anaconda3/bin/conda"),
        "/root/miniconda3/bin/conda",
    ):
        if os.path.isfile(candidate):
            return candidate

    # Last-resort: return "conda" and let the shell resolve / fail loudly.
    return "conda"
