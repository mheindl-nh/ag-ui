# ag-ui-agent-framework

Production-ready AG-UI protocol middleware for the [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) Python SDK.

Wrap any Agent Framework `ChatAgent` (or compatible `AgentProtocol` implementation) with AG-UI's event-based runtime so you can surface rich, streaming agent experiences inside AG-UI clients.

## Installation

```bash
pip install ag-ui-agent-framework
```

The package depends on the Agent Framework preview build `agent-framework==1.0.0b251016`. The `--pre` flag is only required if you install the Agent Framework dependency yourself.

## Quickstart

```python
from fastapi import FastAPI
from agent_framework import ChatAgent, OpenAIChatClient
from ag_ui_agent_framework import AgentFrameworkRunner, add_agent_framework_fastapi_endpoint

# Create or load your Agent Framework agent
chat_client = OpenAIChatClient(model_id="gpt-4o-mini")
agent = ChatAgent(
    chat_client=chat_client,
    name="support-agent",
    instructions="You are a helpful support specialist",
)

# Bridge the agent into AG-UI
runner = AgentFrameworkRunner(agent)
app = FastAPI()
add_agent_framework_fastapi_endpoint(app, runner, path="/agent")
```

Deploy the FastAPI app and point any AG-UI client at `/agent`. The adapter handles:

- Translating AG-UI payloads into Agent Framework conversations
- Streaming Agent Framework updates as AG-UI protocol events (messages, tool calls, thinking, snapshots)
- Emitting final message snapshots so AG-UI clients stay synchronized

## Features

- ✅ Drop-in FastAPI endpoint mirroring the reference AG-UI transport pattern
- ✅ Text, tool-call, and reasoning streaming with automatic delta handling
- ✅ State & message snapshots, run lifecycle events, and usage metrics as AG-UI custom events
- ✅ Message conversion helpers between AG-UI and Agent Framework types
- ✅ Configurable hooks for advanced scenarios (telemetry, custom transports)
- ✅ Built-in adapter for `Workflow` integrations with automatic streaming synthesis

## Examples

Interactive examples live under `examples/`:

- `examples/basic_fastapi.py` – Minimal FastAPI bridge with OpenAI chat client
- `examples/tools_fastapi.py` – Tool calling workflow with streaming tool arguments

Install dev dependencies and run an example with Poetry:

```bash
poetry install
poetry run python examples/basic_fastapi.py
```

Then connect your AG-UI client to `http://localhost:8000/agent`.

## Using Workflows

The Agent Framework `Workflow` API does not expose the streaming-friendly `AgentProtocol` interface directly. Use the bundled adapter to bridge a workflow without writing custom glue code:

```python
from agent_framework import Workflow
from ag_ui_agent_framework import AgentFrameworkRunner, workflow_agent

workflow: Workflow = build_workflow_somehow()
agent = workflow_agent(
    workflow,
    agent_id="medical-workflow",
    name="MedicalWorkflow",
    description="Routes medical questions through triage and responder agents.",
)

runner = AgentFrameworkRunner(agent)
```

The runner now falls back to a synthesized stream whenever `run_stream` is unavailable, so workflows still render real-time thinking indicators and message deltas inside AG-UI.

## Contributing

1. `poetry install`
2. `poetry run pytest` (tests coming soon)
3. Submit a PR 💜

Follow the [AG-UI contributing guide](https://github.com/ag-ui-protocol/ag-ui/blob/main/CONTRIBUTING.md) for repository-wide conventions.
