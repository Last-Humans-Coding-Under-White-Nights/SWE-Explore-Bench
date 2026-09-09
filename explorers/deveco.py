"""DevEco Code CLI explorer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ._cli_agent_base import BaseCliAgentExplorer

EXPLORE_PROMPT = """You are a code exploration specialist. Explore this repository to find the
source files and line ranges most relevant to understanding and fixing the
following issue. Do NOT make any code changes.

Use available read-only repository navigation tools. Focus on finding the ROOT
CAUSE, not just symptom locations. If semantic navigation or MCP tools are
available in this DevEco Code configuration, prefer them for code navigation.

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
class DevEcoExplorer(BaseCliAgentExplorer):
    """DevEco Code CLI explorer for local codebases.

    Uses ``deveco run --format json --dir ... <prompt>`` and parses the final
    response for the shared ``RELEVANT_FILES`` output contract.
    """

    bin_path: str = "deveco"
    #: OpenCode's ``--auto`` has no DevEco equivalent; this is the analogue that
    #: keeps an unattended run from stalling on an approval prompt.
    skip_permissions: bool = True

    cli_display_name: ClassVar[str] = "deveco CLI"
    config_env_var: ClassVar[str] = "DEVECO_CONFIG_DIR"
    config_filename: ClassVar[str] = "deveco.json"
    config_override_vars: ClassVar[tuple[str, ...]] = (
        "DEVECO_CONFIG",
        "DEVECO_CONFIG_CONTENT",
        "DEVECO_CONFIG_DIR",
    )
    install_hint: ClassVar[str] = (
        "Install and configure the `deveco` binary, or pass --deveco-bin."
    )
    prompt_template: ClassVar[str] = EXPLORE_PROMPT

    def build_cmd(self) -> list[str]:
        cmd = [
            self.bin_path,
            "run",
            "--format",
            "json",
            "--dir",
            str(self.repo_root.resolve()),
        ]
        if self.skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        return cmd
