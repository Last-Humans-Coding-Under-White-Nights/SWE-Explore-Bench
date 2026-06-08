"""Default agent：封装好的 LangChain agent，支持 tool calling，参数由 env 提供。"""

from __future__ import annotations

import os
from typing import Any

from langchain.agents import create_agent

from .azure_openai_client import get_default_azure_llm
from .openai_client import get_default_openai_llm


def get_default_llm(provider: str | None = None, **overrides: Any):
    """返回默认 LLM（Azure 或 OpenAI），由环境变量决定，不传参。

    - provider=azure 或 DEFAULT_LLM_PROVIDER=azure 或已设置 MSWEA_API_KEY 时用 Azure
    - 否则用 OpenAI（OPENAI_API_KEY）
    """
    p = (provider or os.getenv("DEFAULT_LLM_PROVIDER") or "").lower()
    if p == "openai":
        return get_default_openai_llm(**overrides)
    if p == "azure":
        return get_default_azure_llm(**overrides)
    if os.getenv("MSWEA_API_KEY"):
        return get_default_azure_llm(**overrides)
    return get_default_openai_llm(**overrides)


def get_default_agent(
    tools: list | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
):
    """返回封装好的 default agent（LangChain ReAct agent），支持 tool calling。

    配置由环境变量提供（ref 的 MSWEA_* 已适配到 LangChain），调用方直接 .invoke() 等。
    """
    model = get_default_llm(**kwargs)
    return create_agent(
        model=model,
        tools=tools or [],
        system_prompt=system_prompt or "",
    )
