from __future__ import annotations

import uuid
from typing import Any, Iterable, Sequence

from agent_framework import (
    AgentExecutorRequest,
    AgentExecutorResponse,
    AgentProtocol,
    AgentRunEvent,
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentRunUpdateEvent,
    ChatMessage,
    Role,
    TextContent,
    Workflow,
    WorkflowOutputEvent,
)

__all__ = ["WorkflowAgentAdapter", "workflow_agent"]


class WorkflowAgentAdapter(AgentProtocol):
    """Expose a Workflow instance via the Agent Protocol streaming surface."""

    def __init__(
        self,
        workflow: Workflow,
        *,
        agent_id: str | None = None,
        name: str | None = None,
        display_name: str | None = None,
        description: str | None = None,
    ) -> None:
        self._workflow = workflow
        fallback_id = f"workflow-{uuid.uuid4().hex}"[:32]
        self._id = agent_id or getattr(workflow, "id", None) or fallback_id
        derived_name = name or getattr(workflow, "name", None) or "WorkflowAgent"
        self._name = derived_name
        self._display_name = display_name or getattr(workflow, "display_name", None) or derived_name
        workflow_description = getattr(workflow, "description", None)
        self._description = description or workflow_description or "Workflow-backed agent"

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def description(self) -> str:
        return self._description

    async def run(
        self,
        messages: Any = None,
        *,
        thread: Any | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        updates = [update async for update in self.run_stream(messages=messages, thread=thread, **kwargs)]
        if not updates:
            return AgentRunResponse(messages=[], response_id=str(uuid.uuid4()))
        return AgentRunResponse.from_agent_run_response_updates(updates)

    async def run_stream(
        self,
        messages: Any = None,
        *,
        thread: Any | None = None,
        **kwargs: Any,
    ):
        chat_messages = self._normalize_messages(messages)
        if chat_messages is None:
            chat_messages = []

        run_kwargs = dict(kwargs)
        if thread is not None and "thread" not in run_kwargs:
            run_kwargs["thread"] = thread

        response_id = str(uuid.uuid4())
        request = AgentExecutorRequest(messages=chat_messages, should_respond=True)

        streamed_updates: list[AgentRunResponseUpdate] = []
        emitted_signatures: set[tuple[Any, ...]] = set()

        def _emit(update: AgentRunResponseUpdate) -> bool:
            signature = self._update_signature(update)
            if signature in emitted_signatures:
                return False
            emitted_signatures.add(signature)
            streamed_updates.append(update)
            return True

        agent_final_response: AgentRunResponse | None = None
        workflow_outputs: list[Any] = []

        async for event in self._workflow.run_stream(request, **run_kwargs):
            if isinstance(event, AgentRunUpdateEvent):
                update = event.data
                if update is None:
                    continue
                if update.response_id is None:
                    update.response_id = response_id
                if _emit(update):
                    yield update
                continue

            if isinstance(event, AgentRunEvent):
                agent_final_response = event.data
                continue

            if isinstance(event, WorkflowOutputEvent):
                workflow_outputs.append(event.data)
                continue

        if agent_final_response is not None:
            emitted_any = False
            for update in self._build_updates(response_id, agent_final_response):
                if _emit(update):
                    emitted_any = True
                    yield update
            if emitted_any:
                return

        if workflow_outputs:
            emitted_any = False
            final_output = workflow_outputs[-1]
            for update in self._build_updates(response_id, final_output):
                if _emit(update):
                    emitted_any = True
                    yield update
            if emitted_any:
                return

        # Ensure consumers receive a terminal chunk even if nothing was emitted.
        if all(not (upd.text or upd.contents) for upd in streamed_updates):
            yield AgentRunResponseUpdate(response_id=response_id, role=Role.ASSISTANT, text="")

    def _normalize_messages(self, messages: Any) -> list[ChatMessage] | None:
        if messages is None:
            return None
        if isinstance(messages, ChatMessage):
            return [messages]
        if isinstance(messages, str):
            return [ChatMessage(role=Role.USER, text=messages)]
        if isinstance(messages, Sequence):
            normalized: list[ChatMessage] = []
            for message in messages:
                if isinstance(message, ChatMessage):
                    normalized.append(message)
                elif isinstance(message, str):
                    normalized.append(ChatMessage(role=Role.USER, text=message))
            return normalized
        return None

    def _collect_outputs(self, result: Any) -> list[Any]:
        get_outputs = getattr(result, "get_outputs", None)
        if callable(get_outputs):
            try:
                outputs = get_outputs()
                if isinstance(outputs, list):
                    return outputs
            except TypeError:
                return []
        if isinstance(result, list):
            return result
        return [result]

    def _update_signature(self, update: AgentRunResponseUpdate) -> tuple[Any, ...]:
        contents_repr = tuple(
            (
                content.__class__.__name__,
                getattr(content, "text", None),
                repr(getattr(content, "arguments", None)),
                repr(getattr(content, "result", None)),
            )
            for content in update.contents or []
        )
        return (
            getattr(update, "role", None),
            update.text,
            contents_repr,
            getattr(update, "author_name", None),
            getattr(update, "message_id", None),
        )

    def _build_updates(self, response_id: str, output: Any) -> Iterable[AgentRunResponseUpdate]:
        if isinstance(output, AgentExecutorResponse):
            agent_response = output.agent_run_response
            if agent_response.messages:
                for message in agent_response.messages:
                    contents = list(message.contents or [])
                    if not contents and getattr(message, "text", None):
                        contents = [TextContent(text=message.text)]
                    yield AgentRunResponseUpdate(
                        response_id=response_id,
                        role=message.role,
                        contents=contents,
                        author_name=message.author_name,
                        message_id=message.message_id,
                    )
                return
            if agent_response.text:
                yield AgentRunResponseUpdate(
                    response_id=response_id,
                    role=Role.ASSISTANT,
                    text=agent_response.text,
                    contents=[TextContent(text=agent_response.text)],
                )
                return

        if isinstance(output, AgentRunResponse):
            if output.messages:
                for message in output.messages:
                    contents = list(message.contents or [])
                    if not contents and getattr(message, "text", None):
                        contents = [TextContent(text=message.text)]
                    yield AgentRunResponseUpdate(
                        response_id=response_id,
                        role=message.role,
                        contents=contents,
                        author_name=message.author_name,
                        message_id=message.message_id,
                    )
                return
            if output.text:
                yield AgentRunResponseUpdate(
                    response_id=response_id,
                    role=Role.ASSISTANT,
                    text=output.text,
                    contents=[TextContent(text=output.text)],
                )
                return

        yield AgentRunResponseUpdate(
            response_id=response_id,
            role=Role.ASSISTANT,
            text=str(output),
            contents=[TextContent(text=str(output))],
        )


def workflow_agent(
    workflow: Workflow,
    *,
    agent_id: str | None = None,
    name: str | None = None,
    display_name: str | None = None,
    description: str | None = None,
) -> AgentProtocol:
    """Create a Workflow-backed AgentProtocol for use with ag-ui integrations."""

    return WorkflowAgentAdapter(
        workflow,
        agent_id=agent_id,
        name=name,
        display_name=display_name,
        description=description,
    )
