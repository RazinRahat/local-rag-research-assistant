"""FastAPI entry point for the Local RAG Research Assistant."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI

from research_assistant.api.routes import build_api_router
from research_assistant.api.runtime import build_local_service
from research_assistant.api.service import (
    ResearchAPI,
    ResearchService,
)

_api_service: ResearchService | None = None
_service_lock = Lock()


def get_api_service() -> ResearchAPI:
    """Create and reuse the local research service."""

    global _api_service

    with _service_lock:
        if _api_service is None:
            _api_service = build_local_service()

        return _api_service


@asynccontextmanager
async def lifespan(
    _app: FastAPI,
) -> AsyncIterator[None]:
    """Close local runtime resources on shutdown."""

    global _api_service

    try:
        yield

    finally:
        with _service_lock:
            if _api_service is not None:
                _api_service.close()
                _api_service = None


app = FastAPI(
    title="Local RAG Research Assistant",
    version="0.9.0",
    lifespan=lifespan,
)

app.include_router(build_api_router(get_api_service))


@app.get(
    "/health",
    tags=["Health"],
)
def health() -> dict[str, str]:
    """Return application health status."""

    return {
        "status": "ok",
        "service": ("local-rag-research-assistant"),
    }
