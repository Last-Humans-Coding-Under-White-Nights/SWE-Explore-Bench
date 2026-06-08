"""think tool: 让 agent 显式推理，不产生副作用。"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class ThinkInput(BaseModel):
    thought: str = Field(description="Your step-by-step reasoning about the problem.")


class ThinkTool(BaseTool):
    """Agent 调用此工具进行显式推理，内容会记录在 trajectory 中。"""

    name: str = "think"
    description: str = (
        "Use this tool to think step-by-step about the problem before submitting a patch. "
        "Write your reasoning as the 'thought' argument. This does not produce any side effects."
    )
    args_schema: type[BaseModel] = ThinkInput

    def _run(self, thought: str = "", **kwargs) -> str:
        return "Thought recorded. Continue reasoning or submit your patch."

    async def _arun(self, thought: str = "", **kwargs) -> str:
        return self._run(thought)
