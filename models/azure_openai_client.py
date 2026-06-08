"""Azure OpenAI 默认 LLM：直接使用 LangChain AzureChatOpenAI，参数来自 ref 的 MSWEA_* 适配。"""

from __future__ import annotations

from typing import Any

from langchain_openai import AzureChatOpenAI

from .ref_to_langchain import ref_to_azure_langchain_kwargs

_default_azure_llm: AzureChatOpenAI | None = None


def get_default_azure_llm(**overrides: Any) -> AzureChatOpenAI:
    """返回封装好的 Azure Chat 模型（default agent），无需调用方传参。

    配置由环境变量 MSWEA_* 经 ref_to_langchain 适配后传入 LangChain，
    支持 tool calling、流式等 LangChain 原生能力。
    """
    global _default_azure_llm
    if overrides:
        return AzureChatOpenAI(**ref_to_azure_langchain_kwargs(overrides))
    if _default_azure_llm is None:
        _default_azure_llm = AzureChatOpenAI(**ref_to_azure_langchain_kwargs())
    return _default_azure_llm
