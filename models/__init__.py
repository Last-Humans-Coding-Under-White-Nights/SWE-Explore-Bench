"""
模型层：提供 default agent（LangChain ReAct agent），支持 tool calling。

- get_default_agent()：封装好的 default agent，直接 .invoke() 等，不传参
- get_default_llm()：底层 LLM（由 env 选 Azure/OpenAI），ref 的 MSWEA_* 已适配到 LangChain
- ChatMessage / messages_to_agent_input：拼成 agent 输入用
- get_default_openrouter_client()：直接走 OpenAI-compatible chat completions

不做二次封装，调用方直接使用 LangChain 能力。
"""

from .azure_openai_client import get_default_azure_llm
from .base import (
    ChatMessage,
    agent_input_to_messages,
    messages_to_agent_input,
    messages_to_langchain,
    messages_to_openai,
)
from .default_agent import get_default_agent, get_default_llm
from .openai_client import get_default_openai_llm
from .openrouter_client import get_default_openrouter_client

__all__ = [
    "ChatMessage",
    "agent_input_to_messages",
    "get_default_agent",
    "get_default_azure_llm",
    "get_default_llm",
    "get_default_openai_llm",
    "get_default_openrouter_client",
    "messages_to_agent_input",
    "messages_to_langchain",
    "messages_to_openai",
]
