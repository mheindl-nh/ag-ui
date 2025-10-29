"""FastAPI bridge showcasing streaming tool calls with the Agent Framework."""

from __future__ import annotations

import json
import os
from typing import Annotated

from fastapi import FastAPI
from pydantic import Field
from agent_framework import ChatAgent, OpenAIChatClient

from ag_ui_agent_framework import AgentFrameworkRunner, add_agent_framework_fastapi_endpoint

app = FastAPI(title="AG-UI × Agent Framework (Tools)")


def get_weather(
    location: Annotated[str, Field(description="Location to retrieve weather for")],
) -> str:
    """Sample tool that returns mock weather data."""
    payload = {
        "location": location,
        "condition": "sunny",
        "high": 22,
        "low": 12,
    }
    return json.dumps(payload)


def _build_agent() -> ChatAgent:
    deployment = os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-4o-mini")
    client = OpenAIChatClient(model_id=deployment)
    return ChatAgent(
        chat_client=client,
        name="tool-agent",
        instructions="Answer user questions using the available tools before responding.",
        tools=[get_weather],
    )


runner = AgentFrameworkRunner(_build_agent())
add_agent_framework_fastapi_endpoint(app, runner, path="/agent")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
