from __future__ import annotations

from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from ag_ui.core.types import RunAgentInput
from ag_ui.encoder import EventEncoder

from .runner import AgentFrameworkRunner

__all__ = ["add_agent_framework_fastapi_endpoint"]


def add_agent_framework_fastapi_endpoint(
    app: FastAPI,
    runner: AgentFrameworkRunner,
    path: str = "/agent",
    *,
    health_check_name: str | None = "agent-framework",
) -> None:
    """Register a FastAPI endpoint that streams AG-UI events from an Agent Framework runner."""

    if not path.startswith("/"):
        raise ValueError("Endpoint path must start with '/'")

    @app.post(path)
    async def agent_framework_endpoint(input_data: RunAgentInput, request: Request):  # type: ignore[override]
        accept_header = request.headers.get("accept")
        encoder = EventEncoder(accept=accept_header)

        async def event_generator():
            try:
                async for event in runner.run(input_data):
                    yield encoder.encode(event)
            except Exception as exc:  # pragma: no cover - surfaced via HTTP error
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        return StreamingResponse(event_generator(), media_type=encoder.get_content_type())

    if health_check_name is not None:
        health_path = path.rstrip("/") + "/health"

        @app.get(health_path)
        def health() -> dict[str, object]:  # type: ignore[override]
            return {
                "status": "ok",
                "runner": health_check_name,
            }
