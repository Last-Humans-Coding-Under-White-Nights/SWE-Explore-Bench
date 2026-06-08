"""模型层：提供 default agent（LangChain Chat 模型），支持 tool calling 等高级能力。

不对 LangChain 做二次封装，调用方直接使用 LangChain 的 .invoke()、.bind_tools() 等。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """统一的对话消息表示，便于与 LangChain BaseMessage 互转。

    role: "system" | "user" | "assistant"
    content: 纯文本内容（如需多模态可后续扩展）
    """

    role: str
    content: str


def messages_to_langchain(messages: list[ChatMessage]):
    """将 ChatMessage 列表转为 LangChain BaseMessage 列表，供 llm.invoke() 使用。"""
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

    out: list[BaseMessage] = []
    for m in messages:
        r, c = m.role.lower(), m.content
        if r == "system":
            out.append(SystemMessage(content=c))
        elif r == "user":
            out.append(HumanMessage(content=c))
        elif r == "assistant":
            out.append(AIMessage(content=c))
        else:
            out.append(HumanMessage(content=c))
    return out


def messages_to_openai(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """将 ChatMessage 列表转为 OpenAI-compatible messages。"""
    return [{"role": m.role.lower(), "content": m.content} for m in messages]


def messages_to_agent_input(messages: list[ChatMessage]) -> str:
    """将 ChatMessage 列表拼成单段输入，供 default agent.invoke({\"input\": ...}) 使用。"""
    parts: list[str] = []
    for m in messages:
        r, c = m.role.lower(), m.content
        if r == "system":
            parts.append(f"[System]\n{c}")
        elif r == "user":
            parts.append(f"[User]\n{c}")
        elif r == "assistant":
            parts.append(f"[Assistant]\n{c}")
        else:
            parts.append(c)
    return "\n\n".join(parts)


def agent_input_to_messages(agent_input: str) -> list[ChatMessage]:
    """将历史的 agent_input 文本还原回 ChatMessage 列表。

    兼容 messages_to_agent_input 生成的格式：
    [System]\n...

    [User]\n...
    """
    text = agent_input.strip()
    if not text:
        return []

    role_map = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
    }
    pattern = re.compile(r"^\[(System|User|Assistant)\]\n", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [ChatMessage(role="user", content=text)]

    out: list[ChatMessage] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        role = role_map[match.group(1).lower()]
        content = text[start:end].strip()
        out.append(ChatMessage(role=role, content=content))
    return out
