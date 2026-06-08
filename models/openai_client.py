"""OpenAI 默认 LLM：直接使用 LangChain ChatOpenAI，参数来自环境变量。"""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI

from quality.models import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    ensure_openai_env_defaults,
)

_default_openai_llm: ChatOpenAI | None = None


def get_default_openai_llm(**overrides: Any) -> ChatOpenAI:
    """返回封装好的 OpenAI Chat 模型（default agent），无需调用方传参。

    配置由 OPENAI_API_KEY、OPENAI_API_BASE 等环境变量提供，
    支持 tool calling、流式等 LangChain 原生能力。
    """
    global _default_openai_llm
    ensure_openai_env_defaults()
    if overrides:
        kwargs = _openai_env_kwargs() | overrides
        return ChatOpenAI(**kwargs)
    if _default_openai_llm is None:
        _default_openai_llm = ChatOpenAI(**_openai_env_kwargs())
    return _default_openai_llm


def _openai_env_kwargs() -> dict[str, Any]:
    model = os.getenv("OPENAI_MODEL_NAME", DEFAULT_MODEL_NAME)
    api_key = os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY)
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or DEFAULT_BASE_URL
    )
    temperature = 0.0
    t = os.getenv("OPENAI_TEMPERATURE")
    if t is not None:
        try:
            temperature = float(t)
        except ValueError:
            pass
    return {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
        "base_url": base_url,
    }
