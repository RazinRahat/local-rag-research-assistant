"""Tests for the research application service."""

from io import BytesIO
from typing import BinaryIO

import pytest

from research_assistant.api.service import ResearchService
from research_assistant.citations.service import CitationService
from research_assistant.rag.models import RAGResponse
from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalMode,
    RetrievalResult,
)
from research_assistant.vector_store.models import StoredDocument

DOCUMENT_ID = "a" * 64


class FakeIndexer:
    def __init__(self) -> None:
        self.filename: str | None = None

    def index(
        self,
        filename: str,
        source: BinaryIO,
    ) -> StoredDocument:
        self.filename = filename

        return StoredDocument(
            document_id=DOCUMENT_ID,
            file_name=filename,
            chunk_count=2,
        )


class FakeRegistry:
    def __init__(self) -> None:
        self.documents = (
            StoredDocument(
                document_id=DOCUMENT_ID,
                file_name="paper.pdf",
                chunk_count=2,
            ),
        )
        self.deleted_id: str | None = None
        self.closed = False

    def list_documents(
        self,
    ) -> tuple[StoredDocument, ...]:
        return self.documents

    def count_document(
        self,
        document_id: str,
    ) -> int:
        if document_id == DOCUMENT_ID:
            return 2

        return 0

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        self.deleted_id = document_id

    def close(self) -> None:
        self.closed = True


class FakeRetriever:
    def __init__(self) -> None:
        self.query: str | None = None
        self.config: RetrievalConfig | None = None

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> tuple[RetrievalResult, ...]:
        self.query = query
        self.config = config

        return ()


class FakeRAG:
    def ask(
        self,
        question: str,
        retrieval_config: RetrievalConfig | None = None,
    ) -> RAGResponse:
        return RAGResponse(
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


class FakeLLM:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def make_service() -> tuple[
    ResearchService,
    FakeIndexer,
    FakeRegistry,
    FakeRetriever,
    FakeLLM,
]:
    indexer = FakeIndexer()
    registry = FakeRegistry()
    retriever = FakeRetriever()
    llm = FakeLLM()

    service = ResearchService(
        indexer=indexer,
        registry=registry,
        retriever=retriever,
        rag=FakeRAG(),
        citation_service=CitationService(),
        llm=llm,
    )

    return (
        service,
        indexer,
        registry,
        retriever,
        llm,
    )


def test_add_document_delegates_to_indexer() -> None:
    service, indexer, _, _, _ = make_service()

    result = service.add_document(
        "paper.pdf",
        BytesIO(b"example"),
    )

    assert result.document_id == DOCUMENT_ID
    assert result.file_name == "paper.pdf"
    assert indexer.filename == "paper.pdf"


def test_list_documents_uses_registry() -> None:
    service, _, registry, _, _ = make_service()

    assert service.list_documents() == registry.documents


def test_delete_document_uses_registry() -> None:
    service, _, registry, _, _ = make_service()

    service.delete_document(DOCUMENT_ID)

    assert registry.deleted_id == DOCUMENT_ID


def test_missing_document_is_rejected() -> None:
    service, _, _, _, _ = make_service()

    with pytest.raises(KeyError):
        service.delete_document("b" * 64)


def test_search_uses_retriever() -> None:
    service, _, _, retriever, _ = make_service()

    config = RetrievalConfig(
        top_k=3,
        mode=RetrievalMode.HYBRID,
    )

    results = service.search(
        "What is attention?",
        config,
    )

    assert results == ()

    assert retriever.query == "What is attention?"

    assert retriever.config is config


def test_query_validates_rag_response() -> None:
    service, _, _, _, _ = make_service()

    result = service.query(
        "What is attention?",
        RetrievalConfig(top_k=3),
    )

    assert result.rag.insufficient_evidence is True
    assert result.citations.references == ()
    assert result.citations.sources == ()


def test_close_releases_resources() -> None:
    service, _, registry, _, llm = make_service()

    service.close()
    service.close()

    assert registry.closed is True
    assert llm.closed is True
