import uuid

from ag_ui_agent_framework.translator import AgentFrameworkEventTranslator
from ag_ui.core import (
    CustomEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ThinkingTextMessageContentEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from agent_framework import (
    AgentRunResponseUpdate,
    FunctionApprovalRequestContent,
    FunctionCallContent,
    FunctionResultContent,
    Role,
    TextContent,
    TextReasoningContent,
)


def _random_id() -> str:
    return uuid.uuid4().hex


def test_translate_text_emits_single_delta():
    translator = AgentFrameworkEventTranslator()

    update = AgentRunResponseUpdate(
        contents=[TextContent(text="Hello")],
        role=Role.ASSISTANT,
        response_id=_random_id(),
    )

    events = translator.translate(update)

    # Expect one start and one delta event, no duplicates.
    assert len(events) == 2
    assert isinstance(events[0], TextMessageStartEvent)
    assert isinstance(events[1], TextMessageContentEvent)
    assert events[1].delta == "Hello"


def test_function_approval_emits_custom_event_and_closes_tool():
    translator = AgentFrameworkEventTranslator()

    call_id = _random_id()
    message_id = _random_id()

    # Seed the translator with an in-flight tool call so approval handling has context.
    call_update = AgentRunResponseUpdate(
        contents=[
            FunctionCallContent(
                call_id=call_id,
                name="fetch_data",
                arguments="{}",
            )
        ],
        role=Role.ASSISTANT,
        message_id=message_id,
        response_id=_random_id(),
    )
    translator.translate(call_update)

    approval_update = AgentRunResponseUpdate(
        contents=[
            FunctionApprovalRequestContent(
                id=_random_id(),
                function_call=FunctionCallContent(
                    call_id=call_id,
                    name="fetch_data",
                    arguments='{"result": 42}',
                ),
            )
        ],
        role=Role.ASSISTANT,
        message_id=message_id,
        response_id=_random_id(),
    )

    events = translator.translate(approval_update)

    assert any(isinstance(event, ToolCallEndEvent) and event.tool_call_id == call_id for event in events)

    approval_events = [event for event in events if isinstance(event, CustomEvent)]
    assert len(approval_events) == 1
    assert approval_events[0].name == "function_approval_request"
    assert approval_events[0].value["function_call"]["call_id"] == call_id
    assert approval_events[0].value["function_call"]["arguments"] == {"result": 42}


def test_reasoning_content_produces_thinking_events():
    translator = AgentFrameworkEventTranslator()

    update = AgentRunResponseUpdate(
        contents=[TextReasoningContent(text="Working on it"), TextContent(text="")],
        role=Role.ASSISTANT,
        response_id=_random_id(),
    )

    events = translator.translate(update)

    thinking_events = [event for event in events if isinstance(event, (ThinkingStartEvent, ThinkingTextMessageContentEvent))]
    assert any(isinstance(event, ThinkingStartEvent) for event in thinking_events)
    assert any(isinstance(event, ThinkingTextMessageContentEvent) and event.delta == "Working on it" for event in thinking_events)

    finalize_events = translator.finalize()
    assert any(isinstance(event, ThinkingEndEvent) for event in finalize_events)


def test_tool_result_emits_snapshot_and_result():
    translator = AgentFrameworkEventTranslator()

    call_id = _random_id()
    message_id = _random_id()

    call_update = AgentRunResponseUpdate(
        contents=[
            FunctionCallContent(
                call_id=call_id,
                name="process_data",
                arguments='{"input": "abc"}',
            )
        ],
        role=Role.ASSISTANT,
        message_id=message_id,
        response_id=_random_id(),
    )

    events: list = []

    events.extend(translator.translate(call_update))

    events.extend(
        translator.translate(
            AgentRunResponseUpdate(
                contents=[TextContent(text="Done")],
                role=Role.ASSISTANT,
                message_id=message_id,
                response_id=_random_id(),
            )
        )
    )

    events.extend(
        translator.translate(
            AgentRunResponseUpdate(
                contents=[
                    FunctionResultContent(
                        call_id=call_id,
                        result={"status": "ok"},
                    )
                ],
                role=Role.ASSISTANT,
                message_id=message_id,
                response_id=_random_id(),
            )
        )
    )

    assert any(isinstance(event, ToolCallStartEvent) and event.tool_call_id == call_id for event in events)
    assert any(isinstance(event, ToolCallArgsEvent) and event.tool_call_id == call_id for event in events)
    assert any(isinstance(event, ToolCallResultEvent) and event.tool_call_id == call_id for event in events)
    assert any(isinstance(event, ToolCallEndEvent) and event.tool_call_id == call_id for event in events)


def test_finalize_closes_open_message_and_tool():
    translator = AgentFrameworkEventTranslator()

    message_id = _random_id()
    call_id = _random_id()

    translator.translate(
        AgentRunResponseUpdate(
            contents=[TextContent(text="Streaming")],
            role=Role.ASSISTANT,
            message_id=message_id,
            response_id=_random_id(),
        )
    )

    translator.translate(
        AgentRunResponseUpdate(
            contents=[FunctionCallContent(call_id=call_id, name="do_work", arguments="{}")],
            role=Role.ASSISTANT,
            message_id=message_id,
            response_id=_random_id(),
        )
    )

    final_events = translator.finalize()

    assert any(isinstance(event, TextMessageEndEvent) for event in final_events)
    assert any(isinstance(event, ToolCallEndEvent) and event.tool_call_id == call_id for event in final_events)
