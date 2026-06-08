"""Shared LLM defaults.

Used by:
1. ``quality/*`` scripts that talk to an OpenAI-compatible HTTP client
2. ``line_refine`` / ``run_line_refine_call`` via the LangChain ``ChatOpenAI`` wrapper

All values can be overridden via environment variables (``OPENAI_API_KEY``,
``OPENAI_BASE_URL``, ``OPENAI_MODEL_NAME``); the constants below are only used as
fallback defaults when nothing is set.
"""

from __future__ import annotations

import os

# Empty defaults — set OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL_NAME
# in your environment (e.g. via .env). See README and .env.example.
DEFAULT_API_KEY = ""
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL_NAME = "gpt-4o-mini"


def ensure_openai_env_defaults() -> None:
    """Populate OpenAI-compatible env vars with fallback defaults if unset."""
    os.environ.setdefault("DEFAULT_LLM_PROVIDER", "openai")
    if DEFAULT_API_KEY:
        os.environ.setdefault("OPENAI_API_KEY", DEFAULT_API_KEY)
    os.environ.setdefault("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    os.environ.setdefault("OPENAI_API_BASE", DEFAULT_BASE_URL)
    os.environ.setdefault("OPENAI_MODEL_NAME", DEFAULT_MODEL_NAME)

