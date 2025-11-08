from __future__ import annotations

import logging
from dataclasses import dataclass
import inspect
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from ag_ui.core import EventType
from ag_ui.core.events import (
    MessagesSnapshotEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    CustomEvent,
)
from ag_ui.core.types import Message, RunAgentInput
from agent_framework import (
    AgentProtocol,
    AgentRunResponse,
    AgentRunResponseUpdate,
    ChatMessage,
    Role,
    TextContent,
    TextReasoningContent,
)

from .converters import agui_messages_to_agent_framework, agent_framework_messages_to_agui
from .translator import AgentFrameworkEventTranslator

logger = logging.getLogger(__name__)


@dataclass
class AgentFrameworkRunnerConfig:
    """Configuration for the Agent Framework -> AG-UI bridge."""

    emit_initial_state_snapshot: bool = True
    emit_initial_messages_snapshot: bool = True
    forward_usage_events: bool = True
    run_kwargs_key: str = "run_kwargs"
    chat_options_key: str = "chat_options"


class AgentFrameworkRunner:
    """Stream an Agent Framework agent into AG-UI events."""

    def __init__(
        self,
        agent: AgentProtocol,
        *,
        config: Optional[AgentFrameworkRunnerConfig] = None,
    ) -> None:
        self._agent = agent
        self._config = config or AgentFrameworkRunnerConfig()

    async def run(self, input_data: RunAgentInput) -> AsyncGenerator[Any, None]:
        translator = AgentFrameworkEventTranslator()
        forwarded_props = input_data.forwarded_props or {}
        run_kwargs, additional_chat_options = self._resolve_forwarded_props(forwarded_props)

        agui_messages: list[Message] = input_data.messages or []
        agent_messages = agui_messages_to_agent_framework(agui_messages)

        yield RunStartedEvent(thread_id=input_data.thread_id, run_id=input_data.run_id)

        if self._config.emit_initial_state_snapshot and input_data.state is not None:
            yield StateSnapshotEvent(snapshot=input_data.state)

        if self._config.emit_initial_messages_snapshot and agui_messages:
            yield MessagesSnapshotEvent(messages=agui_messages)

        updates: list[AgentRunResponseUpdate] = []
        response: AgentRunResponse | None = None

        try:
            if self._supports_run_stream():
                async for update in self._agent.run_stream(  # type: ignore[attr-defined]
                    messages=agent_messages,
                    additional_chat_options=additional_chat_options,
                    **run_kwargs,
                ):
                    updates.append(update)
                    for event in translator.translate(update):
                        yield event
                response = AgentRunResponse.from_agent_run_response_updates(updates)
            else:
                response, fallback_updates = await self._run_without_stream(
                    agent_messages,
                    run_kwargs,
                    additional_chat_options,
                )
                for update in fallback_updates:
                    updates.append(update)
                    for event in translator.translate(update):
                        yield event
                if response is None:
                    synthesized_updates = [u for u in updates if not self._is_reasoning_update(u)]
                    response = AgentRunResponse.from_agent_run_response_updates(synthesized_updates)

        except Exception as exc:  # pragma: no cover - transport errors surfaced to client
            logger.exception("Agent Framework run failed")
            for event in translator.finalize():
                yield event
            yield RunErrorEvent(message=str(exc))
            raise

        if response is None:
            response = AgentRunResponse.from_agent_run_response_updates(updates)

        for event in translator.finalize():
            yield event

        if response.messages:
            response_messages = agent_framework_messages_to_agui(response.messages)
            if response_messages:
                final_snapshot = [*agui_messages, *response_messages]
                yield MessagesSnapshotEvent(messages=final_snapshot)

        if self._config.forward_usage_events and response.usage_details:
            payload = response.usage_details.to_dict(exclude_none=True)
            yield self._usage_event(payload)

        yield RunFinishedEvent(
            thread_id=input_data.thread_id,
            run_id=input_data.run_id,
            result=self._build_result_payload(response),
        )

    def _supports_run_stream(self) -> bool:
        run_stream = getattr(self._agent, "run_stream", None)
        return callable(run_stream)

    async def _run_without_stream(
        self,
        messages: list[ChatMessage],
        run_kwargs: Dict[str, Any],
        additional_chat_options: Dict[str, Any] | None,
    ) -> tuple[AgentRunResponse | None, list[AgentRunResponseUpdate]]:
        response_id = str(uuid.uuid4())
        updates: list[AgentRunResponseUpdate] = [
            AgentRunResponseUpdate(
                response_id=response_id,
                role=Role.ASSISTANT,
                contents=[TextReasoningContent(text="Analyzing input...")],
            )
        ]

        result = await self._invoke_agent_run(messages, run_kwargs, additional_chat_options)
        agent_response = self._extract_agent_run_response(result)

        updates.extend(self._updates_from_result(response_id, result if agent_response is None else agent_response))

        return agent_response, updates

    async def _invoke_agent_run(
        self,
        messages: list[ChatMessage],
        run_kwargs: Dict[str, Any],
        additional_chat_options: Dict[str, Any] | None,
    ) -> Any:
        run_callable = getattr(self._agent, "run", None)
        if run_callable is None:
            raise AttributeError("Agent does not expose run_stream or run")

        result = run_callable(
            messages=messages,
            additional_chat_options=additional_chat_options,
            **run_kwargs,
        )

        if inspect.isawaitable(result):
            result = await result
        return result

    def _extract_agent_run_response(self, result: Any) -> AgentRunResponse | None:
        if isinstance(result, AgentRunResponse):
            return result

        candidate = getattr(result, "agent_run_response", None)
        if isinstance(candidate, AgentRunResponse):
            return candidate

        return None

    def _updates_from_result(
        self,
        response_id: str,
        result: Any,
    ) -> list[AgentRunResponseUpdate]:
        if isinstance(result, AgentRunResponse):
            return self._updates_from_agent_run_response(response_id, result)

        get_outputs = getattr(result, "get_outputs", None)
        if callable(get_outputs):
            try:
                outputs = get_outputs()
            except TypeError:
                outputs = None
            if outputs:
                final_output = outputs[-1]
                return self._updates_from_result(response_id, final_output)

        return [
            AgentRunResponseUpdate(
                response_id=response_id,
                role=Role.ASSISTANT,
                text=str(result),
                contents=[TextContent(text=str(result))],
            )
        ]

    def _updates_from_agent_run_response(
        self,
        response_id: str,
        response: AgentRunResponse,
    ) -> list[AgentRunResponseUpdate]:
        updates: list[AgentRunResponseUpdate] = []

        if response.messages:
            for message in response.messages:
                updates.append(
                    AgentRunResponseUpdate(
                        response_id=response_id,
                        role=message.role,
                        contents=message.contents,
                        author_name=getattr(message, "author_name", None),
                        message_id=getattr(message, "message_id", None),
                    )
                )
            return updates

        text_value = getattr(response, "text", None)
        if text_value:
            updates.append(
                AgentRunResponseUpdate(
                    response_id=response_id,
                    role=Role.ASSISTANT,
                    text=text_value,
                    contents=[TextContent(text=text_value)],
                )
            )

        if not updates:
            value = getattr(response, "value", None)
            if value is not None:
                text_value = str(value)
                updates.append(
                    AgentRunResponseUpdate(
                        response_id=response_id,
                        role=Role.ASSISTANT,
                        text=text_value,
                        contents=[TextContent(text=text_value)],
                    )
                )

        return updates

    @staticmethod
    def _is_reasoning_update(update: AgentRunResponseUpdate) -> bool:
        if not update.contents:
            return False
        return any(isinstance(content, TextReasoningContent) for content in update.contents)

    def _resolve_forwarded_props(self, forwarded: Any) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
        if not isinstance(forwarded, dict):
            return {}, None

        run_kwargs = forwarded.get(self._config.run_kwargs_key, {})
        chat_options = forwarded.get(self._config.chat_options_key)

        if not isinstance(run_kwargs, dict):
            run_kwargs = {}
        if chat_options is not None and not isinstance(chat_options, dict):
            chat_options = None

        return run_kwargs, chat_options

    @staticmethod
    def _usage_event(payload: Dict[str, Any]) -> CustomEvent:
        return CustomEvent(type=EventType.CUSTOM, name="usage", value=payload)

    @staticmethod
    def _build_result_payload(response: AgentRunResponse) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": response.text}
        if response.value is not None:
            payload["value"] = response.value
        if response.usage_details is not None:
            payload["usage"] = response.usage_details.to_dict(exclude_none=True)
        return payload
