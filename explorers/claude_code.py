"""Claude Code CLI explorer.

Uses the ``claude`` CLI (>= 2.x) to explore a local codebase given an issue
description.  Parses structured RELEVANT_FILES output via shared
:func:`parsing.parse_relevant_files`.

Supports routing through any Anthropic-compatible endpoint (e.g. a LiteLLM
proxy that wraps OpenAI / Azure models) via the ``api_base`` / ``api_key``
parameters: when set we inject ``ANTHROPIC_BASE_URL``, ``ANTHROPIC_AUTH_TOKEN``,
``ANTHROPIC_MODEL`` and ``ANTHROPIC_SMALL_FAST_MODEL`` into the CLI's
environment so Claude Code calls the proxy instead of api.anthropic.com.

Requirements:
    - Claude Code CLI installed (``claude`` binary in PATH, version >= 2.x)
    - Either ANTHROPIC_API_KEY in env (default Anthropic), or pass
      ``api_base`` + ``api_key`` to route through a proxy.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .base import Explorer, ExplorerResult
from .parsing import parse_relevant_files

EXPLORE_PROMPT = """You are a code exploration specialist. Explore this repository to find the
source files and line ranges most relevant to understanding and fixing the
following issue. Do NOT make any code changes.

Use Glob, Grep, and Read tools to explore the codebase. Focus on finding
the ROOT CAUSE, not just symptom locations.

After exploration, output your findings in EXACTLY this format:

RELEVANT_FILES:
- path/to/file1.py:10-50
- path/to/file2.py:1-100

Focus on the root cause. Limit to top {top_k} most relevant regions.

ISSUE:
{issue}
"""


@dataclass
class ClaudeCodeExplorer(Explorer):
    """Claude Code CLI explorer for local codebases.

    Uses ``claude -p --output-format json --permission-mode bypassPermissions``
    on Claude Code CLI 2.x. The prompt is delivered via stdin to avoid
    argv length / shell-escaping issues.
    """

    repo_root: Path
    model: str = "sonnet"
    timeout: int = 600
    api_base: str | None = None  # e.g. "http://127.0.0.1:4000" → routes via LiteLLM
    api_key: str | None = None   # used as ANTHROPIC_AUTH_TOKEN when api_base set

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5
    ) -> List[ExplorerResult]:
        prompt = EXPLORE_PROMPT.format(issue=query, top_k=top_k)

        cmd = [
            "claude", "-p",
            "--output-format", "json",
            "--permission-mode", "bypassPermissions",
            "--model", self.model,
            "--allowed-tools", "Read,Glob,Grep",
            "--no-session-persistence",
        ]

        # Strip CLAUDECODE so a parent claude session can't bleed into the child.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        # When api_base is set, route through the (Anthropic-compatible) proxy.
        if self.api_base:
            # Strip trailing slash and any /v1 suffix — Claude Code appends paths itself.
            base = self.api_base.rstrip("/")
            if base.endswith("/v1"):
                base = base[: -len("/v1")]
            env["ANTHROPIC_BASE_URL"] = base
            if self.api_key:
                env["ANTHROPIC_AUTH_TOKEN"] = self.api_key
                # Some installs read API_KEY too — set both to be safe.
                env.setdefault("ANTHROPIC_API_KEY", self.api_key)
            env["ANTHROPIC_MODEL"] = self.model
            env["ANTHROPIC_SMALL_FAST_MODEL"] = self.model

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            raw = completed.stdout.strip()
        except FileNotFoundError:
            raise RuntimeError(
                "Claude Code CLI not found. Install with: "
                "npm install -g @anthropic-ai/claude-code"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Claude Code CLI timed out after {self.timeout}s"
            )

        if not raw:
            return []

        try:
            data = json.loads(raw)
            output = data.get("result", "")
        except json.JSONDecodeError:
            output = raw

        if not output:
            return []

        return parse_relevant_files(output, instance_id, top_k=top_k)
