"""Application service for the local research API."""

from threading import RLock
from typing import BinaryIO, Protocol

from research_assistant.citations.models import CitedRAGResponse
from research_assistant.citations.service import CitationService
from research_assistant.rag.models import RAGResponse
from research_assistant.retrieval.base import Retriever
from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalResult,
)
from research_assistant.vector_store.models import StoredDocument


class DocumentIndexer(Protocol):
    """Indexes uploaded research documents."""

    def index(
        self,
        filename: str,
        source: BinaryIO,
    ) -> StoredDocument: ...


class DocumentRegistry(Protocol):
    """Document operations required from persistent storage."""

    def list_documents(
        self,
    ) -> tuple[StoredDocument, ...]: ...

    def count_document(
        self,
        document_id: str,
    ) -> int: ...

    def delete_document(
        self,
        document_id: str,
    ) -> None: ...

    def close(self) -> None: ...


class RAGAnswerer(Protocol):
    """Interface satisfied by RAGService."""

    def ask(
        self,
        question: str,
        retrieval_config: RetrievalConfig | None = None,
    ) -> RAGResponse: ...


class Closable(Protocol):
    """Resource supporting explicit shutdown."""

    def close(self) -> None: ...


class ResearchAPI(Protocol):
    """Operations exposed to the HTTP layer."""

    def add_document(
        self,
        filename: str,
        source: BinaryIO,
    ) -> StoredDocument: ...

    def list_documents(
        self,
    ) -> tuple[StoredDocument, ...]: ...

    def delete_document(
        self,
        document_id: str,
    ) -> None: ...

    def search(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> tuple[RetrievalResult, ...]: ...

    def query(
        self,
        question: str,
        config: RetrievalConfig,
    ) -> CitedRAGResponse: ...

    def close(self) -> None: ...


class ResearchService:
    """Coordinates the existing research-assistant components."""

    def __init__(
        self,
        *,
        indexer: DocumentIndexer,
        registry: DocumentRegistry,
        retriever: Retriever,
        rag: RAGAnswerer,
        citation_service: CitationService,
        llm: Closable,
    ) -> None:
        self._indexer = indexer
        self._registry = registry
        self._retriever = retriever
        self._rag = rag
        self._citation_service = citation_service
        self._llm = llm

        self._lock = RLock()
        self._closed = False

    def add_document(
        self,
        filename: str,
        source: BinaryIO,
    ) -> StoredDocument:
        with self._lock:
            return self._indexer.index(
                filename,
                source,
            )

    def list_documents(
        self,
    ) -> tuple[StoredDocument, ...]:
        with self._lock:
            return self._registry.list_documents()

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        with self._lock:
            if self._registry.count_document(document_id) == 0:
                raise KeyError(f"Document not found: {document_id}")

            self._registry.delete_document(document_id)

    def search(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> tuple[RetrievalResult, ...]:
        with self._lock:
            return self._retriever.retrieve(
                query,
                config,
            )

    def query(
        self,
        question: str,
        config: RetrievalConfig,
    ) -> CitedRAGResponse:
        with self._lock:
            response = self._rag.ask(
                question,
                retrieval_config=config,
            )

            return self._citation_service.validate(response)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return

            try:
                self._llm.close()
            finally:
                self._registry.close()
                self._closed = True
