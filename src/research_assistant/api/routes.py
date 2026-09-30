"""HTTP routes for the Local RAG Research Assistant."""

from collections.abc import Callable
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Path,
    Response,
    UploadFile,
    status,
)

from research_assistant.api.exceptions import APIError
from research_assistant.api.models import (
    QueryRequest,
    SearchRequest,
    SearchResponse,
)
from research_assistant.api.service import ResearchAPI
from research_assistant.citations.exceptions import CitationError
from research_assistant.citations.models import CitedRAGResponse
from research_assistant.generation.exceptions import (
    LLMConnectionError,
    LLMResponseError,
)
from research_assistant.vector_store.models import StoredDocument


def build_api_router(
    get_service: Callable[[], ResearchAPI],
) -> APIRouter:
    """Build API routes around a research-service provider."""

    router = APIRouter()

    @router.post(
        "/documents",
        response_model=StoredDocument,
        status_code=status.HTTP_201_CREATED,
        tags=["Documents"],
    )
    def add_document(
        file: Annotated[
            UploadFile,
            File(...),
        ],
    ) -> StoredDocument:
        try:
            return get_service().add_document(
                file.filename or "",
                file.file,
            )

        except APIError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=str(exc),
            ) from exc

    @router.get(
        "/documents",
        response_model=tuple[StoredDocument, ...],
        tags=["Documents"],
    )
    def list_documents() -> tuple[StoredDocument, ...]:
        return get_service().list_documents()

    @router.delete(
        "/documents/{document_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        response_class=Response,
        tags=["Documents"],
    )
    def delete_document(
        document_id: Annotated[
            str,
            Path(pattern=r"^[a-f0-9]{64}$"),
        ],
    ) -> Response:
        try:
            get_service().delete_document(document_id)

        except KeyError as exc:
            message = str(exc.args[0]) if exc.args else "Document not found"

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=message,
            ) from exc

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.post(
        "/search",
        response_model=SearchResponse,
        tags=["Research"],
    )
    def search(
        request: SearchRequest,
    ) -> SearchResponse:
        results = get_service().search(
            request.query,
            request.to_retrieval_config(),
        )

        return SearchResponse(
            query=request.query,
            results=results,
        )

    @router.post(
        "/query",
        response_model=CitedRAGResponse,
        tags=["Research"],
    )
    def query(
        request: QueryRequest,
    ) -> CitedRAGResponse:
        try:
            return get_service().query(
                request.question,
                request.to_retrieval_config(),
            )

        except LLMConnectionError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Local language model is unavailable",
            ) from exc

        except (
            LLMResponseError,
            CitationError,
        ) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=("Generated response could not be validated"),
            ) from exc

    return router
