"""submit_patch tool: agent 通过此工具提交最终的 unified diff patch。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class SubmitPatchInput(BaseModel):
    patch: str = Field(description="The final patch in standard unified diff format.")


class SubmitPatchTool(BaseTool):
    """Agent 调用此工具提交最终 patch。patch 内容从 tool_input 中提取。"""

    name: str = "submit_patch"
    description: str = (
        "Submit your final patch in standard unified diff format. "
        "CRITICAL: Each hunk MUST have a complete header with line numbers: "
        "@@ -<old_start>,<old_count> +<new_start>,<new_count> @@. "
        "NEVER write a bare '@@' without line numbers - this will cause the patch to fail. "
        "The patch must start with 'diff --git a/<path> b/<path>' and include "
        "proper --- / +++ headers. Use the line numbers from the provided code context. "
        "Call this tool exactly once when you are ready to submit."
    )
    args_schema: type[BaseModel] = SubmitPatchInput

    # 运行时由外部注入，用于存储 patch
    _result_store: dict[str, Any] | None = None

    class Config:
        arbitrary_types_allowed = True

    def bind_store(self, store: dict[str, Any]) -> None:
        """绑定外部 dict，agent 提交 patch 后写入 store["patch"]。"""
        self._result_store = store

    def _run(self, patch: str = "", **kwargs) -> str:
        if self._result_store is not None:
            self._result_store["patch"] = patch
        return "Patch submitted successfully."

    async def _arun(self, patch: str = "", **kwargs) -> str:
        return self._run(patch)
