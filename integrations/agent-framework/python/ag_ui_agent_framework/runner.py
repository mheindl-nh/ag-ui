from __future__ import annotations

import logging
from dataclasses import dataclass
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
from agent_framework import AgentProtocol, AgentRunResponse, AgentRunResponseUpdate

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

        try:
            async for update in self._agent.run_stream(  # type: ignore[attr-defined]
                messages=agent_messages,
                additional_chat_options=additional_chat_options,
                **run_kwargs,
            ):
                updates.append(update)
                for event in translator.translate(update):
                    yield event

        except Exception as exc:  # pragma: no cover - transport errors surfaced to client
            logger.exception("Agent Framework run failed")
            for event in translator.finalize():
                yield event
            yield RunErrorEvent(message=str(exc))
            raise

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
