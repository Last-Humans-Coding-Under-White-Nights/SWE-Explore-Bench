"""将 refs/azure_proxy_model 的 MSWEA_* 参数适配为 LangChain Azure/OpenAI 的构造参数。

不二次封装 LangChain，仅做 env → kwargs 的映射，供 default agent 使用。
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any


def ref_to_azure_langchain_kwargs(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """从 MSWEA_* 环境变量生成 LangChain AzureChatOpenAI 的 kwargs。

    与 refs/azure_proxy_model.AzureProxyModelConfig 对齐，便于复用同一套 env 配置。
    """
    overrides = overrides or {}
    model_name = overrides.get("model_name") or os.getenv("MSWEA_MODEL_NAME", "glm-4.6")
    api_key = overrides.get("api_key") or os.getenv("MSWEA_API_KEY", "")
    azure_endpoint = overrides.get("azure_endpoint") or os.getenv(
        "MSWEA_AZURE_ENDPOINT",
        "https://search.bytedance.net/gpt/openapi/online/v2/crawl",
    )
    api_version = overrides.get("api_version") or os.getenv("MSWEA_API_VERSION", "2024-02-01")

    max_tokens_val = os.getenv("MSWEA_MAX_TOKENS")
    max_tokens = int(max_tokens_val) if max_tokens_val else overrides.get("max_tokens")

    temp_val = os.getenv("MSWEA_TEMPERATURE")
    temperature = float(temp_val) if temp_val else overrides.get("temperature", 0.0)

    use_responses = os.getenv("MSWEA_USE_RESPONSES", "").lower() in ("true", "1", "yes")
    if "use_responses_api" in overrides:
        use_responses = overrides["use_responses_api"]

    enable_thinking = os.getenv("MSWEA_ENABLE_THINKING", "").lower() == "true"
    thinking_budget_val = os.getenv("MSWEA_THINKING_BUDGET")
    thinking_budget = int(thinking_budget_val) if thinking_budget_val else None

    enable_prompt_cache = os.getenv("MSWEA_ENABLE_PROMPT_CACHE", "").lower() == "true"
    session_id = os.getenv("MSWEA_SESSION_ID") or (str(uuid.uuid4().int)[:20] if enable_prompt_cache else None)

    default_headers: dict[str, str] = {"X-TT-LOGID": os.getenv("TT_LOGID", "swe-explore")}
    if session_id:
        default_headers["extra"] = json.dumps({"session_id": session_id})

    extra_body: dict[str, Any] = {}
    if enable_thinking:
        thinking_config: dict[str, Any] = {"include_thoughts": True}
        if thinking_budget is not None:
            thinking_config["budget_tokens"] = thinking_budget
        extra_body["thinking"] = thinking_config

    kwargs: dict[str, Any] = {
        "azure_endpoint": azure_endpoint,
        "api_key": api_key,
        "api_version": api_version,
        "azure_deployment": model_name,
        "model": model_name,
        "temperature": temperature,
        "default_headers": default_headers,
        "use_responses_api": use_responses,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if extra_body:
        kwargs["extra_body"] = extra_body

    model_kwargs = overrides.get("model_kwargs") or {}
    kwargs["model_kwargs"] = model_kwargs
    return kwargs
