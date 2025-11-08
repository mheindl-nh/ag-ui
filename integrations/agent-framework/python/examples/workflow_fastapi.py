"""FastAPI example exposing an Agent Framework Workflow through AG-UI."""

from __future__ import annotations

import os

from fastapi import FastAPI
from agent_framework import ChatAgent, Workflow, WorkflowBuilder
from agent_framework.azure import AzureOpenAIChatClient

from ag_ui_agent_framework import (
    AgentFrameworkRunner,
    AgentFrameworkRunnerConfig,
    add_agent_framework_fastapi_endpoint,
    workflow_agent,
)

app = FastAPI(title="AG-UI × Agent Framework Workflows")


def _build_chat_agent(name: str, instructions: str) -> ChatAgent:
    deployment = os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-4o-mini")
    client = AzureOpenAIChatClient(
        deployment_name=deployment,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    return ChatAgent(chat_client=client, name=name, instructions=instructions)


def _build_workflow() -> Workflow:
    """Create a simple two-step workflow that triages before executing."""

    planner = _build_chat_agent(
        name="planner-agent",
        instructions="Triage the user's request and outline a short plan before handing off.",
    )

    executor = _build_chat_agent(
        name="executor-agent",
        instructions="Act on the planner's outline and return the final answer succinctly.",
    )

    builder = WorkflowBuilder(name="SupportWorkflow", description="Planner/executor hand-off workflow")
    builder.add_edge(planner, executor)
    builder.set_start_executor(planner)
    return builder.build()


def _build_runner() -> AgentFrameworkRunner:
    workflow = _build_workflow()
    agent = workflow_agent(
        workflow,
        name="support-workflow",
        description="Routes requests through planner/executor agents",
    )
    config = AgentFrameworkRunnerConfig(emit_initial_state_snapshot=False)
    return AgentFrameworkRunner(agent, config=config)


runner = _build_runner()
add_agent_framework_fastapi_endpoint(app, runner, path="/agent")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
