"""Cursor Agent CLI explorer.

Uses the Cursor Agent CLI (``agent``) to explore a local codebase given an
issue description.

Requirements:
    - Cursor Agent CLI installed (``agent`` binary in PATH)
    - CURSOR_API_KEY env var or --api-key parameter for authentication
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .base import ContextRegion, Explorer, ExplorerResult
from .parsing import parse_relevant_files

EXPLORE_PROMPT = """You are a code exploration specialist. Explore this repository to find the
source files and line ranges most relevant to understanding and fixing the
following issue. Do NOT make any code changes.

After exploration, output your findings in EXACTLY this format:

RELEVANT_FILES:
- path/to/file1.py:10-50
- path/to/file2.py:1-100

Focus on the root cause. Limit to top {top_k} most relevant regions.

ISSUE:
{issue}
"""


@dataclass
class CursorAgentExplorer(Explorer):
    """Cursor Agent CLI explorer for local codebases.

    Uses ``agent --trust -p --output-format json`` to explore a local repo.
    """

    repo_root: Path
    api_key: str = ""
    model: str = ""
    timeout: int = 600

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5
    ) -> List[ExplorerResult]:
        prompt = EXPLORE_PROMPT.format(issue=query, top_k=top_k)

        api_key = self.api_key or os.environ.get("CURSOR_API_KEY", "")
        model = self.model or os.environ.get("CURSOR_MODEL", "")

        agent_bin = os.environ.get(
            "CURSOR_AGENT_BIN", os.path.expanduser("~/.local/bin/agent")
        )
        cmd = [agent_bin, "--trust", "-p", "--output-format", "json"]
        if api_key:
            cmd.extend(["--api-key", api_key])
        if model:
            cmd.extend(["--model", model])
        cmd.append(prompt)

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            raw = completed.stdout.strip()
        except FileNotFoundError:
            raise RuntimeError(
                "Cursor Agent CLI not found. Install with: "
                "curl https://cursor.com/install -fsS | bash"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Cursor Agent CLI timed out after {self.timeout}s"
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
