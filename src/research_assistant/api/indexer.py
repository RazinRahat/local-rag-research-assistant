"""PDF indexing pipeline used by the local research API."""

import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from research_assistant.api.exceptions import (
    InvalidUploadError,
    UnsupportedMediaError,
    UploadTooLargeError,
)
from research_assistant.chunking.chunker import chunk_document
from research_assistant.chunking.models import ChunkingConfig
from research_assistant.chunking.tokenizer import Tokenizer
from research_assistant.embeddings.embedder import EmbeddingProvider
from research_assistant.embeddings.service import embed_chunks
from research_assistant.ingestion.exceptions import (
    DocumentIngestionError,
)
from research_assistant.ingestion.pdf_loader import load_pdf
from research_assistant.vector_store.base import VectorStore
from research_assistant.vector_store.models import StoredDocument


class LocalPDFIndexer:
    """Index uploaded PDFs through the existing local pipeline."""

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        embedder: EmbeddingProvider,
        store: VectorStore,
        chunking_config: ChunkingConfig | None = None,
        max_upload_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        if max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be positive")

        self._tokenizer = tokenizer
        self._embedder = embedder
        self._store = store

        self._chunking_config = chunking_config or ChunkingConfig()

        self._max_upload_bytes = max_upload_bytes

    @staticmethod
    def _safe_filename(
        filename: str,
    ) -> str:
        name = PurePosixPath(filename.replace("\\", "/")).name

        if not name:
            raise InvalidUploadError("Upload filename cannot be empty")

        if not name.lower().endswith(".pdf"):
            raise UnsupportedMediaError("Only PDF documents are supported")

        return name

    def index(
        self,
        filename: str,
        source: BinaryIO,
    ) -> StoredDocument:
        """Parse, chunk, embed, and store one uploaded PDF."""

        safe_name = self._safe_filename(filename)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / safe_name

            total_bytes = 0

            with path.open("wb") as destination:
                while block := source.read(1024 * 1024):
                    total_bytes += len(block)

                    if total_bytes > self._max_upload_bytes:
                        raise UploadTooLargeError("PDF exceeds the maximum upload size")

                    destination.write(block)

            if total_bytes == 0:
                raise InvalidUploadError("Uploaded PDF is empty")

            try:
                document = load_pdf(path)
            except DocumentIngestionError as exc:
                raise InvalidUploadError("PDF could not be ingested") from exc

            chunks = chunk_document(
                document=document,
                tokenizer=self._tokenizer,
                config=self._chunking_config,
            )

            if not chunks:
                raise InvalidUploadError("PDF contains no indexable text")

            embeddings = embed_chunks(
                chunks,
                self._embedder,
            )

            chunk_count = self._store.replace_document(
                document.metadata.document_id,
                embeddings,
            )

            return StoredDocument(
                document_id=(document.metadata.document_id),
                file_name=(document.metadata.file_name),
                chunk_count=chunk_count,
            )
