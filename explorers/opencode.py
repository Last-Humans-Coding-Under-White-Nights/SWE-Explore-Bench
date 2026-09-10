"""OpenCode CLI explorer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ._cli_agent_base import BaseCliAgentExplorer

EXPLORE_PROMPT = """You are a code exploration specialist. Explore this repository to find the
source files and line ranges most relevant to understanding and fixing the
following issue. Do NOT make any code changes.

Use available read-only repository navigation tools. Focus on finding the ROOT
CAUSE, not just symptom locations. If semantic navigation or MCP tools are
available in this OpenCode configuration, prefer them for code navigation.

VERY IMPORTANT: After exploration, output your findings in EXACTLY this format:

RELEVANT_FILES:
- path/to/file1.py:10-50
- path/to/file2.py:1-100

Focus on the root cause. Limit to top {top_k} most relevant regions.
{prompt_additions}

ISSUE:
{issue}
"""


@dataclass
class OpenCodeExplorer(BaseCliAgentExplorer):
    """OpenCode CLI explorer for local codebases.

    Uses ``opencode run --auto --format json --dir ...`` with the prompt on
    stdin, and parses the final response for the shared ``RELEVANT_FILES``
    output contract.
    """

    bin_path: str = "opencode"

    cli_display_name: ClassVar[str] = "OpenCode CLI"
    config_env_var: ClassVar[str] = "OPENCODE_CONFIG_DIR"
    config_filename: ClassVar[str] = "opencode.json"
    config_override_vars: ClassVar[tuple[str, ...]] = (
        "OPENCODE_CONFIG",
        "OPENCODE_CONFIG_CONTENT",
        "OPENCODE_CONFIG_DIR",
    )
    install_hint: ClassVar[str] = (
        "Install and configure the `opencode` binary, or pass --opencode-bin."
    )
    prompt_template: ClassVar[str] = EXPLORE_PROMPT

    def build_cmd(self) -> list[str]:
        return [
            self.bin_path,
            "run",
            "--auto",
            "--format",
            "json",
            "--dir",
            str(self.repo_root.resolve()),
        ]
