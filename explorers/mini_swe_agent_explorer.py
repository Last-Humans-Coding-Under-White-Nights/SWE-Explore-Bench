"""mini-swe-agent based explorer.

Wraps SWE-agent/mini-swe-agent (vendored under
``third_party/mini-swe-agent``) to do code exploration only.  We reuse:

- ``minisweagent.environments.local.LocalEnvironment`` to run bash commands in
  the locally checked-out repository.
- ``minisweagent.models.litellm_model.LitellmModel`` to talk to the unified
  LiteLLM proxy (Azure GPT-5.4).
- ``minisweagent.agents.default.DefaultAgent`` for the basic THOUGHT/ACTION
  loop.

The explorer customises the system + instance templates so that the final
agent action prints a ``RELEVANT_FILES:`` block followed by the standard
mini-swe-agent submit sentinel.  Region parsing reuses
``parse_relevant_files`` from :mod:`explorers.parsing`.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .base import Explorer, ExplorerResult
from .parsing import parse_relevant_files

# Vendored mini-swe-agent
_MSWEA_SRC = Path(__file__).resolve().parent.parent / "third_party" / "mini-swe-agent" / "src"
if _MSWEA_SRC.is_dir() and str(_MSWEA_SRC) not in sys.path:
    sys.path.insert(0, str(_MSWEA_SRC))

# Silence the mini-swe-agent rich startup banner
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
# Default to "ignore_errors" so missing cost metadata for proxy models doesn't abort
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

logger = logging.getLogger(__name__)


_SYSTEM_TEMPLATE = """You are a code-exploration specialist. Your job is to read the repository
and identify the source files (and line ranges) most relevant to a given issue.
Do NOT modify any files.

Your response MUST contain exactly ONE bash code block with ONE command.
Include a brief THOUGHT before your command. Format:

<format_example>
THOUGHT: explanation here.

```bash
your_command_here
```
</format_example>

You can use any read-only bash command (find, grep, ls, cat, head, tail, sed -n,
nl, etc.) to navigate the codebase.

When you have identified the most relevant files, finish by emitting EXACTLY
ONE bash command that prints the answer in the required format. The first line
of stdout MUST be the literal sentinel ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``
followed on the next line by ``RELEVANT_FILES:`` and one bullet per region:

```bash
cat <<'EOF'
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
RELEVANT_FILES:
- path/to/file1.py:10-50
- path/to/file2.py:120-160
EOF
```

Use repository-relative paths and inclusive line ranges (start-end).  List at
most {{top_k}} regions, ordered by relevance."""


_INSTANCE_TEMPLATE = """ISSUE:
{{task}}

Explore the repository (cwd is the repo root) to find the {{top_k}} most
relevant code regions. Limit yourself to {{step_limit}} steps. Once you are
confident, submit using the COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT sentinel
described in the system message.

<system_information>
{{system}} {{release}} {{machine}}
</system_information>
"""


@dataclass
class MiniSWEAgentExplorer(Explorer):
    """Code-exploration explorer powered by mini-swe-agent."""

    repo_root: Path
    model: str = "openai/gpt-5.4"
    api_key: str = ""
    api_base: str = ""
    step_limit: int = 25
    cost_limit: float = 3.0
    bash_timeout: int = 60
    timeout: int = 600  # informational; mini-swe-agent has its own step_limit/cost_limit

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5
    ) -> List[ExplorerResult]:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_model import LitellmModel
        from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

        # Wire LiteLLM to use the unified proxy if api_base/api_key were provided.
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
        if self.api_base:
            os.environ["OPENAI_API_BASE"] = self.api_base
            os.environ["OPENAI_BASE_URL"] = self.api_base

        env = LocalEnvironment(cwd=str(self.repo_root), timeout=self.bash_timeout)

        # Some upstream models (Fireworks accounts/msft/deployments/* incl. Kimi)
        # (a) reject 'provider_specific_fields'/'reasoning'/etc. when an assistant
        #     message is fed back, and (b) don't reliably emit OpenAI tool_calls
        #     for our explorer prompt. Use the textbased model variant so the
        #     agent looks for ```bash``` blocks in plain content (matching our
        #     existing system template), and strip non-OpenAI-spec keys before
        #     re-sending messages.
        _ALLOWED_KEYS = {
            "role", "content", "name", "tool_calls", "tool_call_id",
            "function_call", "refusal",
        }
        _BASH_REGEX = r"```bash\s*\n(.*?)\n```"

        class _StrictTextModel(LitellmTextbasedModel):
            def __init__(self, **kw):
                super().__init__(**kw)
                # Override the default 'mswea_bash_command' regex to plain ```bash``` blocks.
                self.config.action_regex = _BASH_REGEX
                self.config.format_error_template = (
                    "Please provide EXACTLY ONE action in a ```bash``` triple-backtick block; "
                    "found {{actions|length}} actions."
                )

            def _prepare_messages_for_api(self, messages):
                prepared = super()._prepare_messages_for_api(messages)
                return [
                    {k: v for k, v in m.items() if k in _ALLOWED_KEYS}
                    for m in prepared
                ]

        model = _StrictTextModel(model_name=self.model)
        agent = DefaultAgent(
            model,
            env,
            system_template=_SYSTEM_TEMPLATE,
            instance_template=_INSTANCE_TEMPLATE,
            step_limit=self.step_limit,
            cost_limit=self.cost_limit,
        )

        try:
            result = agent.run(task=query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mini-swe-agent run failed for %s: %s", instance_id, exc)
            return []

        submission = (result or {}).get("submission") or ""
        if not submission:
            # Best-effort fallback: scan the last assistant message for a RELEVANT_FILES block.
            for msg in reversed(getattr(agent, "messages", [])):
                content = msg.get("content") or ""
                if "RELEVANT_FILES:" in content:
                    submission = content
                    break
        if not submission:
            return []
        return parse_relevant_files(submission, instance_id, top_k=top_k)
