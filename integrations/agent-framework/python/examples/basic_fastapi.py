"""Minimal FastAPI bridge between the Microsoft Agent Framework and AG-UI."""

from __future__ import annotations

import os

from fastapi import FastAPI
from agent_framework import ChatAgent, OpenAIChatClient

from ag_ui_agent_framework import AgentFrameworkRunner, add_agent_framework_fastapi_endpoint

app = FastAPI(title="AG-UI × Agent Framework")


def _build_agent() -> ChatAgent:
    deployment = os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-4o-mini")
    client = OpenAIChatClient(model_id=deployment)
    return ChatAgent(
        chat_client=client,
        name="support-agent",
        instructions="You are a helpful assistant that answers concisely.",
    )


def _build_runner() -> AgentFrameworkRunner:
    agent = _build_agent()
    return AgentFrameworkRunner(agent)


runner = _build_runner()
add_agent_framework_fastapi_endpoint(app, runner, path="/agent")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
