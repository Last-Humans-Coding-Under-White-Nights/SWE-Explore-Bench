"""AweAgent explorer — uses SearchSWEAgent with a local bash session.

Integrates AweAgent's SearchSWEAgent into the swe-explore benchmark,
running it on a local repository directory without Docker.

Requirements:
    - AweAgent installed or available at /root/jialiang/AweAgent
    - LLM backend configured via constructor args or env vars:
        OPENAI_API_KEY, OPENAI_BASE_URL (for openai backend)
        ARK_API_KEY (for ark backend)
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .base import Explorer, ExplorerResult
from ._paths import awe_agent_path as _default_awe_agent_path
from .parsing import parse_relevant_files

_AWE_AGENT_ROOT = _default_awe_agent_path()

EXPLORE_PROMPT = """You are a code exploration specialist. Explore this repository to find
the source files and line ranges most relevant to understanding and fixing the
following issue. Do NOT make any code changes.

Use bash commands (find, grep, cat, head) and the editor view command to explore.
Focus on finding the ROOT CAUSE, not just symptom locations.

When done, call the `finish` tool with a `lines` argument listing your top {top_k}
most relevant regions in the format `path/to/file.py:start-end` (one per line).
Example:
  src/foo/bar.py:10-50
  src/foo/baz.py:100-130

ISSUE:
{issue}
"""


class LocalBashSession:
    """Minimal RuntimeSession that executes bash commands on the local filesystem.

    Implements the RuntimeSession protocol from AweAgent without Docker.
    """

    def __init__(self, workdir: str) -> None:
        self._workdir = workdir

    async def execute(self, command: str, cwd: str | None = None,
                      timeout: int | None = None, env: dict | None = None):
        _cwd = cwd or self._workdir

        def _run():
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=_cwd,
                capture_output=True,
                text=True,
                timeout=timeout or 60,
                env={**os.environ, **(env or {})},
            )
            return result.returncode, result.stdout, result.stderr

        loop = asyncio.get_event_loop()
        try:
            exit_code, stdout, stderr = await loop.run_in_executor(None, _run)
        except subprocess.TimeoutExpired:
            # Import here to avoid circular import at module load time
            awe_root = str(_AWE_AGENT_ROOT)
            if awe_root not in sys.path:
                sys.path.insert(0, awe_root)
            from awe_agent.core.runtime.types import ExecutionResult
            return ExecutionResult(stdout="", stderr="Command timed out", exit_code=1)

        awe_root = str(_AWE_AGENT_ROOT)
        if awe_root not in sys.path:
            sys.path.insert(0, awe_root)
        from awe_agent.core.runtime.types import ExecutionResult
        return ExecutionResult(stdout=stdout, stderr=stderr, exit_code=exit_code)

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        Path(remote_path).parent.mkdir(parents=True, exist_ok=True)
        Path(remote_path).write_bytes(content)

    async def download_file(self, remote_path: str) -> bytes:
        return Path(remote_path).read_bytes()

    async def list_files(self, path: str, recursive: bool = False) -> list[str]:
        p = Path(path)
        if recursive:
            return [str(f) for f in p.rglob("*") if f.is_file()]
        return [str(f) for f in p.iterdir()]

    async def get_patch(self, *args, **kwargs) -> str:
        # Exploration only — no patch needed
        return ""

    async def close(self) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@dataclass
class AweAgentExplorer(Explorer):
    """AweAgent SearchSWEAgent explorer running on local filesystem.

    Wraps AweAgent's SearchSWEAgent with a lightweight local bash session,
    enabling evaluation without Docker. Supports any OpenAI-compatible backend.

    Args:
        repo_root: Path to the repository to explore.
        model: LLM model name (e.g. "gpt-4o", "glm-4.7-flash").
        backend: LLM backend ("openai", "ark", "sglang").
        base_url: OpenAI-compatible API base URL. Falls back to OPENAI_BASE_URL.
        api_key: API key. Falls back to OPENAI_API_KEY / ARK_API_KEY env vars.
        max_steps: Max agent steps (default 30).
        bash_timeout: Per-bash-command timeout in seconds (default 60).
    """

    repo_root: Path
    model: str = "gpt-4o"
    backend: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    max_steps: int = 30
    bash_timeout: int = 60

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5
    ) -> List[ExplorerResult]:
        prompt = EXPLORE_PROMPT.format(issue=query, top_k=top_k)
        try:
            output, finish_lines = asyncio.run(self._run_agent(prompt))
        except Exception as e:
            raise RuntimeError(f"AweAgent exploration failed: {e}") from e

        # Prefer structured finish payload (file -> [line numbers])
        if finish_lines:
            regions_text = self._lines_to_regions_text(finish_lines)
            results = parse_relevant_files(regions_text, instance_id, top_k=top_k)
            if results:
                return results

        if not output:
            return []
        return parse_relevant_files(output, instance_id, top_k=top_k)

    @staticmethod
    def _lines_to_regions_text(lines_map: dict) -> str:
        """Convert {file: [line_numbers]} into RELEVANT_FILES text blocks."""
        out = ["RELEVANT_FILES:"]
        for path, nums in lines_map.items():
            if not nums:
                continue
            ints = sorted(set(int(n) for n in nums))
            # Group consecutive line numbers into ranges
            start = prev = ints[0]
            for n in ints[1:] + [None]:
                if n is None or n != prev + 1:
                    out.append(f"- {path}:{start}-{prev}")
                    if n is not None:
                        start = n
                if n is not None:
                    prev = n
        return "\n".join(out)

    async def _run_agent(self, prompt: str) -> tuple[str, dict]:
        awe_root = str(_AWE_AGENT_ROOT)
        if awe_root not in sys.path:
            sys.path.insert(0, awe_root)

        from awe_agent.core.agent.context import AgentContext
        from awe_agent.core.agent.loop import AgentLoop
        from awe_agent.core.llm.client import LLMClient, llm_registry
        from awe_agent.core.llm.config import LLMConfig
        from awe_agent.core.llm.backends.openai import OpenAIBackend
        if "openai" not in llm_registry._items:
            llm_registry.register("openai", OpenAIBackend)
        from awe_agent.scaffold.search_swe import SearchSWEAgent
        from awe_agent.core.tool.code.finish import LineFLFinishTool

        # Resolve API key: constructor > env var (backend-specific fallback)
        api_key = (
            self.api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ARK_API_KEY")
            or "dummy"
        )
        base_url = self.base_url or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            base_url = base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"

        llm_config = LLMConfig(
            backend=self.backend,
            model=self.model,
            base_url=base_url,
            api_key=api_key,
        )
        llm = LLMClient(llm_config)
        agent = SearchSWEAgent(
            enable_search=False,
            bash_timeout=self.bash_timeout,
            max_output_length=16000,
        )
        # Replace the default FinishTool with LineFLFinishTool so the model
        # is forced to submit structured `lines` output (file:line ranges).
        line_finish = LineFLFinishTool()
        agent._tools = [
            t for t in agent._tools if t.name != "finish"
        ] + [line_finish]

        session = LocalBashSession(workdir=str(self.repo_root))
        ctx = AgentContext(
            llm=llm,
            session=session,
            tools=agent.get_tools(),
            task_info={"dataset_id": "swe_explore", "workdir": str(self.repo_root)},
            max_steps=self.max_steps,
        )
        loop = AgentLoop(agent, ctx)
        result = await loop.run(prompt)

        # Extract finish payload (lines={file: [line_numbers]})
        finish_lines: dict = {}
        try:
            import json as _json
            for step in reversed(result.trajectory.steps):
                action = step.action
                if not action or action.type != "finish" or not action.tool_calls:
                    continue
                for tc in action.tool_calls:
                    name = tc.get("name") or tc.get("function", {}).get("name", "")
                    if name != "finish":
                        continue
                    args = tc.get("arguments") or tc.get("function", {}).get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = _json.loads(args)
                        except Exception:
                            args = {}
                    parsed = line_finish.submit(args or {})
                    if parsed:
                        finish_lines = parsed
                        break
                if finish_lines:
                    break
        except Exception:
            pass

        # Concatenate all assistant message contents (fallback for free-form output)
        all_text_parts: list[str] = []
        for msg in result.messages:
            if hasattr(msg, "role") and msg.role == "assistant" and msg.content:
                all_text_parts.append(msg.content)
        return ("\n".join(all_text_parts), finish_lines)
