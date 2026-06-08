"""直接使用 OpenAI-compatible chat completions 调用 OpenRouter。"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from quality.models import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    ensure_openai_env_defaults,
)

from .base import ChatMessage, messages_to_openai


class OpenRouterChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 32000,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def invoke(self, messages: list[ChatMessage]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages_to_openai(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        message = response.choices[0].message
        content = message.content or ""
        if isinstance(content, str):
            return content
        return str(content)


_default_openrouter_client: OpenRouterChatClient | None = None


def _openrouter_env_kwargs() -> dict[str, Any]:
    ensure_openai_env_defaults()
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

    max_tokens = 32000
    mt = os.getenv("OPENAI_MAX_TOKENS")
    if mt is not None:
        try:
            max_tokens = int(mt)
        except ValueError:
            pass

    return {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def get_default_openrouter_client(**overrides: Any) -> OpenRouterChatClient:
    global _default_openrouter_client
    if overrides:
        return OpenRouterChatClient(**(_openrouter_env_kwargs() | overrides))
    if _default_openrouter_client is None:
        _default_openrouter_client = OpenRouterChatClient(**_openrouter_env_kwargs())
    return _default_openrouter_client
