"""Tests for the research API routes."""

from typing import BinaryIO

from fastapi import FastAPI
from fastapi.testclient import TestClient

from research_assistant.api.models import SearchResponse
from research_assistant.api.routes import build_api_router
from research_assistant.citations.models import (
    CitationValidationResult,
    CitedRAGResponse,
)
from research_assistant.rag.models import RAGResponse
from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalResult,
)
from research_assistant.vector_store.models import StoredDocument

DOCUMENT_ID = "a" * 64


class FakeResearchAPI:
    def __init__(self) -> None:
        self.uploaded_filename: str | None = None
        self.uploaded_data: bytes | None = None
        self.deleted_document_id: str | None = None
        self.search_config: RetrievalConfig | None = None
        self.query_config: RetrievalConfig | None = None

    def add_document(
        self,
        filename: str,
        source: BinaryIO,
    ) -> StoredDocument:
        self.uploaded_filename = filename
        self.uploaded_data = source.read()

        return StoredDocument(
            document_id=DOCUMENT_ID,
            file_name=filename,
            chunk_count=2,
        )

    def list_documents(
        self,
    ) -> tuple[StoredDocument, ...]:
        return (
            StoredDocument(
                document_id=DOCUMENT_ID,
                file_name="paper.pdf",
                chunk_count=2,
            ),
        )

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        if document_id != DOCUMENT_ID:
            raise KeyError(f"Document not found: {document_id}")

        self.deleted_document_id = document_id

    def search(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> tuple[RetrievalResult, ...]:
        self.search_config = config
        return ()

    def query(
        self,
        question: str,
        config: RetrievalConfig,
    ) -> CitedRAGResponse:
        self.query_config = config

        rag = RAGResponse(
            question=question,
            answer=(
                "I couldn't find sufficient evidence "
                "in the indexed documents to answer "
                "that question."
            ),
            evidence=(),
            retrieved_count=0,
            used_evidence_count=0,
            insufficient_evidence=True,
        )

        return CitedRAGResponse(
            rag=rag,
            citations=CitationValidationResult(
                references=(),
                sources=(),
            ),
        )

    def close(self) -> None:
        pass


def make_client() -> tuple[
    TestClient,
    FakeResearchAPI,
]:
    service = FakeResearchAPI()

    app = FastAPI()

    app.include_router(build_api_router(lambda: service))

    return TestClient(app), service


def test_upload_document() -> None:
    client, service = make_client()

    response = client.post(
        "/documents",
        files={
            "file": (
                "paper.pdf",
                b"%PDF-1.4\nexample",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["document_id"] == DOCUMENT_ID
    assert response.json()["file_name"] == "paper.pdf"

    assert service.uploaded_filename == "paper.pdf"
    assert service.uploaded_data == (b"%PDF-1.4\nexample")


def test_list_documents() -> None:
    client, _ = make_client()

    response = client.get("/documents")

    assert response.status_code == 200

    assert response.json() == [
        {
            "document_id": DOCUMENT_ID,
            "file_name": "paper.pdf",
            "chunk_count": 2,
        }
    ]


def test_delete_document() -> None:
    client, service = make_client()

    response = client.delete(f"/documents/{DOCUMENT_ID}")

    assert response.status_code == 204
    assert response.content == b""

    assert service.deleted_document_id == DOCUMENT_ID


def test_invalid_document_id_is_rejected() -> None:
    client, _ = make_client()

    response = client.delete("/documents/not-a-document-id")

    assert response.status_code == 422


def test_search_request() -> None:
    client, service = make_client()

    response = client.post(
        "/search",
        json={
            "query": "  What is attention?  ",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    SearchResponse.model_validate(response.json())

    assert service.search_config is not None
    assert service.search_config.top_k == 3


def test_query_request() -> None:
    client, service = make_client()

    response = client.post(
        "/query",
        json={
            "question": "What is attention?",
            "top_k": 4,
        },
    )

    assert response.status_code == 200

    assert response.json()["rag"]["question"] == "What is attention?"

    assert service.query_config is not None
    assert service.query_config.top_k == 4


def test_blank_search_is_rejected() -> None:
    client, _ = make_client()

    response = client.post(
        "/search",
        json={
            "query": "   ",
        },
    )

    assert response.status_code == 422


def test_invalid_top_k_is_rejected() -> None:
    client, _ = make_client()

    response = client.post(
        "/search",
        json={
            "query": "attention",
            "top_k": 0,
        },
    )

    assert response.status_code == 422
