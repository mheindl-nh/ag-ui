from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from ag_ui.core import EventType
from ag_ui.core.events import (
    BaseEvent,
    CustomEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ThinkingTextMessageContentEvent,
    ThinkingTextMessageEndEvent,
    ThinkingTextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from agent_framework import (
    AgentRunResponseUpdate,
    FunctionCallContent,
    FunctionResultContent,
    Role,
    TextContent,
    TextReasoningContent,
    UsageContent,
)

__all__ = ["AgentFrameworkEventTranslator"]


@dataclass
class _MessageState:
    role: str
    started: bool = False
    closed: bool = False


@dataclass
class _ToolState:
    name: str
    parent_message_id: Optional[str]
    arguments: str = ""
    ended: bool = False


class AgentFrameworkEventTranslator:
    """Translate Agent Framework streaming updates into AG-UI protocol events."""

    def __init__(self) -> None:
        self._messages: Dict[str, _MessageState] = {}
        self._tools: Dict[str, _ToolState] = {}
        self._current_message_id: Optional[str] = None
        self._thinking_active: bool = False
        self._thinking_phase: bool = False

    def translate(self, update: AgentRunResponseUpdate) -> List[BaseEvent]:
        events: List[BaseEvent] = []
        message_id = self._resolve_message_id(update)
        role = self._resolve_role(update.role)
        state = self._messages.setdefault(message_id, _MessageState(role=role))

        text_chunk = getattr(update, "text", None)
        if text_chunk:
            events.extend(self._handle_text_chunk(message_id, state, text_chunk))

        for content in update.contents or []:
            if isinstance(content, TextReasoningContent):
                events.extend(self._handle_reasoning(content))
                continue

            if isinstance(content, TextContent):
                events.extend(self._handle_text_chunk(message_id, state, content.text or ""))
                continue

            if isinstance(content, FunctionCallContent):
                events.extend(self._handle_tool_call(message_id, content))
                continue

            if isinstance(content, FunctionResultContent):
                events.extend(self._handle_tool_result(message_id, content))
                continue

            if isinstance(content, UsageContent):
                events.append(
                    CustomEvent(
                        type=EventType.CUSTOM,
                        name="usage",
                        value=content.details.to_dict(exclude_none=True),
                    )
                )
                continue

            events.append(
                CustomEvent(
                    type=EventType.CUSTOM,
                    name="agent_framework:raw_content",
                    value=self._safe_json(content),
                )
            )

        return events

    def finalize(self) -> List[BaseEvent]:
        events: List[BaseEvent] = []

        events.extend(self._end_thinking())

        for message_id, state in self._messages.items():
            if state.started and not state.closed:
                events.append(TextMessageEndEvent(message_id=message_id))
                state.closed = True

        for tool_id, tool_state in self._tools.items():
            if not tool_state.ended:
                events.append(ToolCallEndEvent(tool_call_id=tool_id))
                tool_state.ended = True

        return events

    def _handle_text_chunk(
        self,
        message_id: str,
        state: _MessageState,
        chunk: str,
    ) -> List[BaseEvent]:
        if not chunk:
            return []

        events: List[BaseEvent] = []
        if self._thinking_active or self._thinking_phase:
            events.extend(self._end_thinking())
        if not state.started:
            events.append(TextMessageStartEvent(message_id=message_id, role=state.role))
            state.started = True
        events.append(TextMessageContentEvent(message_id=message_id, delta=chunk))
        return events

    def _handle_reasoning(self, content: TextReasoningContent) -> List[BaseEvent]:
        text = content.text or ""
        if not text:
            return []
        events: List[BaseEvent] = []
        if not self._thinking_phase:
            events.append(ThinkingStartEvent())
            self._thinking_phase = True
        if not self._thinking_active:
            events.append(ThinkingTextMessageStartEvent())
            self._thinking_active = True
        events.append(ThinkingTextMessageContentEvent(delta=text))
        return events

    def _handle_tool_call(self, message_id: str, content: FunctionCallContent) -> List[BaseEvent]:
        tool_id = content.call_id or str(uuid.uuid4())
        tool_state = self._tools.get(tool_id)
        events: List[CustomEvent] = []

        if tool_state is None:
            tool_state = _ToolState(name=content.name or "tool", parent_message_id=message_id)
            self._tools[tool_id] = tool_state
            events.append(
                ToolCallStartEvent(
                    tool_call_id=tool_id,
                    tool_call_name=tool_state.name,
                    parent_message_id=message_id,
                )
            )

        arguments = content.arguments or ""
        if arguments:
            delta = arguments[len(tool_state.arguments) :] if arguments.startswith(tool_state.arguments) else arguments
            if delta:
                events.append(ToolCallArgsEvent(tool_call_id=tool_id, delta=delta))
                tool_state.arguments = arguments

        return events

    def _handle_tool_result(self, message_id: str, content: FunctionResultContent) -> List[BaseEvent]:
        tool_id = content.call_id or str(uuid.uuid4())
        tool_state = self._tools.setdefault(
            tool_id,
            _ToolState(name="tool", parent_message_id=message_id),
        )

        events: List[CustomEvent] = [
            ToolCallResultEvent(
                message_id=message_id,
                tool_call_id=tool_id,
                content=self._stringify(content.result),
                role="tool",
            )
        ]

        if not tool_state.ended:
            events.append(ToolCallEndEvent(tool_call_id=tool_id))
            tool_state.ended = True

        return events

    def _end_thinking(self) -> List[BaseEvent]:
        events: List[BaseEvent] = []
        if self._thinking_active:
            events.append(ThinkingTextMessageEndEvent())
            self._thinking_active = False
        if self._thinking_phase:
            events.append(ThinkingEndEvent())
            self._thinking_phase = False
        return events

    def _resolve_message_id(self, update: AgentRunResponseUpdate) -> str:
        message_id = update.message_id
        if message_id:
            self._current_message_id = message_id
            return message_id

        if self._current_message_id is None:
            self._current_message_id = str(uuid.uuid4())
        return self._current_message_id

    @staticmethod
    def _resolve_role(role: Role | str | None) -> str:
        if isinstance(role, Role):
            value = getattr(role, "value", None)
            if value:
                return value
            return str(role)
        if role is None:
            return "assistant"
        return str(role)

    @staticmethod
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

    def _safe_json(self, content: object) -> object:
        if hasattr(content, "to_dict"):
            try:
                return content.to_dict()  # type: ignore[call-arg]
            except Exception:  # pragma: no cover - best effort
                return str(content)
        if hasattr(content, "model_dump"):
            return content.model_dump()  # type: ignore[attr-defined]
        return self._stringify(content)
