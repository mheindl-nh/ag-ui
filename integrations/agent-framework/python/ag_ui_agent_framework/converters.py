from __future__ import annotations

import json
import uuid
from typing import Iterable, List, Sequence

from ag_ui.core.types import (
    AssistantMessage,
    DeveloperMessage,
    FunctionCall,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from agent_framework import (
    ChatMessage,
    FunctionCallContent,
    FunctionResultContent,
    Role,
    TextContent,
    TextReasoningContent,
)

__all__ = [
    "agui_messages_to_agent_framework",
    "agent_framework_messages_to_agui",
]


def agui_messages_to_agent_framework(messages: Sequence[Message]) -> List[ChatMessage]:
    """Convert AG-UI protocol messages into Agent Framework chat messages."""
    converted: List[ChatMessage] = []

    for message in messages:
        role = message.role
        message_id = getattr(message, "id", None)
        name = getattr(message, "name", None)

        if role == "assistant":
            contents: List[object] = []
            content_text = getattr(message, "content", None)
            if content_text:
                contents.append(TextContent(text=content_text))

            for tool_call in getattr(message, "tool_calls", []) or []:
                contents.append(
                    FunctionCallContent(
                        call_id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                )

            converted.append(
                ChatMessage(
                    role=Role.ASSISTANT,
                    contents=contents,
                    message_id=message_id,
                    author_name=name,
                )
            )
            continue

        if role == "user":
            content_text = getattr(message, "content", "") or ""
            converted.append(
                ChatMessage(
                    role=Role.USER,
                    contents=[TextContent(text=content_text)],
                    message_id=message_id,
                    author_name=name,
                )
            )
            continue

        if role == "system":
            content_text = getattr(message, "content", "") or ""
            converted.append(
                ChatMessage(
                    role=Role.SYSTEM,
                    contents=[TextContent(text=content_text)],
                    message_id=message_id,
                    author_name=name,
                )
            )
            continue

        if role == "developer":
            content_text = getattr(message, "content", "") or ""
            converted.append(
                ChatMessage(
                    role=Role.SYSTEM,
                    contents=[TextContent(text=content_text)],
                    message_id=message_id,
                    author_name=name,
                )
            )
            continue

        if role == "tool":
            content_text = getattr(message, "content", "") or ""
            tool_call_id = getattr(message, "tool_call_id", None)
            converted.append(
                ChatMessage(
                    role=Role.TOOL,
                    contents=[
                        FunctionResultContent(
                            call_id=tool_call_id,
                            result=content_text,
                            exception=getattr(message, "error", None),
                        )
                    ],
                    message_id=message_id,
                    author_name=name,
                )
            )
            continue

    return converted


def agent_framework_messages_to_agui(messages: Iterable[ChatMessage]) -> List[Message]:
    """Convert Agent Framework chat messages back into AG-UI protocol messages."""
    converted: List[Message] = []

    for msg in messages:
        role = _resolve_role(msg.role)
        message_id = msg.message_id or str(uuid.uuid4())
        author_name = getattr(msg, "author_name", None)

        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        tool_results: List[FunctionResultContent] = []

        for content in msg.contents or []:
            if isinstance(content, TextContent):
                text_parts.append(content.text)
            elif isinstance(content, TextReasoningContent):
                reasoning_parts.append(content.text)
            elif isinstance(content, FunctionCallContent):
                call_id = content.call_id or str(uuid.uuid4())
                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        type="function",
                        function=FunctionCall(
                            name=content.name or "tool",
                            arguments=content.arguments or "{}",
                        ),
                    )
                )
            elif isinstance(content, FunctionResultContent):
                tool_results.append(content)

        aggregated_text = "\n".join(filter(None, text_parts)) or None

        if role == "assistant":
            converted.append(
                AssistantMessage(
                    id=message_id,
                    role="assistant",
                    content=aggregated_text,
                    tool_calls=tool_calls or None,
                    name=author_name,
                )
            )
            continue

        if role == "user":
            converted.append(
                UserMessage(
                    id=message_id,
                    role="user",
                    content=aggregated_text or "",
                    name=author_name,
                )
            )
            continue

        if role == "system":
            converted.append(
                SystemMessage(
                    id=message_id,
                    role="system",
                    content=aggregated_text or "",
                    name=author_name,
                )
            )
            continue

        if role == "developer":
            converted.append(
                DeveloperMessage(
                    id=message_id,
                    role="developer",
                    content=aggregated_text or "",
                    name=author_name,
                )
            )
            continue

        if role == "tool":
            for result in tool_results or []:
                converted.append(
                    ToolMessage(
                        id=message_id,
                        role="tool",
                        content=_stringify(result.result),
                        tool_call_id=result.call_id or _pick_tool_call_id(tool_calls),
                        error=_stringify(result.exception) if getattr(result, "exception", None) else None,
                        name=author_name,
                    )
                )
            if not tool_results:
                converted.append(
                    ToolMessage(
                        id=message_id,
                        role="tool",
                        content=aggregated_text or "",
                        tool_call_id=_pick_tool_call_id(tool_calls),
                        name=author_name,
                    )
                )
            continue

    return converted


def _resolve_role(role: Role | str | None) -> str:
    if isinstance(role, Role):
        value = getattr(role, "value", None)
        if value:
            return value
        return str(role)
    if role is None:
        return "assistant"
    return str(role)


def _pick_tool_call_id(tool_calls: List[ToolCall]) -> str:
    return tool_calls[0].id if tool_calls else str(uuid.uuid4())


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="ignore")
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
