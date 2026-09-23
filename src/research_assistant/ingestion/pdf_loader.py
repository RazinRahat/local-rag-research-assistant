from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import pymupdf

from research_assistant.ingestion.exceptions import (
    DocumentIngestionError,
    EmptyDocumentError,
    EncryptedDocumentError,
    UnsupportedDocumentError,
)
from research_assistant.ingestion.models import (
    DocumentMetadata,
    DocumentPage,
    ParsedDocument,
)
from research_assistant.ingestion.normalizer import normalize_text


def _calculate_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 fingerprint without loading the whole file into memory."""

    digest = sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _clean_metadata_value(
    metadata: Mapping[str, Any],
    key: str,
) -> str | None:
    """Return meaningful PDF metadata values or None."""

    value = metadata.get(key)

    if not isinstance(value, str):
        return None

    value = value.strip()

    if not value or value.lower() == "none":
        return None

    return value


def load_pdf(file_path: str | Path) -> ParsedDocument:
    """Load and normalize a text-based PDF document."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document does not exist: {path}")

    if not path.is_file():
        raise DocumentIngestionError(f"Document path is not a file: {path}")

    if path.suffix.lower() != ".pdf":
        raise UnsupportedDocumentError(
            f"Unsupported document type: {path.suffix or 'unknown'}"
        )

    document_hash = _calculate_sha256(path)

    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise DocumentIngestionError(f"Unable to open PDF: {path.name}") from exc

    try:
        if document.needs_pass:
            raise EncryptedDocumentError(f"PDF requires a password: {path.name}")

        if document.page_count == 0:
            raise EmptyDocumentError(f"PDF contains no pages: {path.name}")

        pages: list[DocumentPage] = []

        for page_index in range(document.page_count):
            page = document.load_page(page_index)

            raw_text = page.get_text("text", sort=True)

            if not isinstance(raw_text, str):
                raise DocumentIngestionError(
                    f"Unexpected text extraction result on page {page_index + 1}"
                )

            text = normalize_text(raw_text)

            pages.append(
                DocumentPage(
                    page_number=page_index + 1,
                    text=text,
                    char_count=len(text),
                    word_count=len(text.split()),
                    has_text=bool(text),
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    rotation=int(page.rotation),
                )
            )

        raw_metadata = document.metadata or {}

        text_page_count = sum(1 for page in pages if page.has_text)

        metadata = DocumentMetadata(
            document_id=document_hash,
            file_name=path.name,
            file_size_bytes=path.stat().st_size,
            sha256=document_hash,
            page_count=document.page_count,
            text_page_count=text_page_count,
            title=_clean_metadata_value(raw_metadata, "title"),
            author=_clean_metadata_value(raw_metadata, "author"),
            subject=_clean_metadata_value(raw_metadata, "subject"),
            keywords=_clean_metadata_value(raw_metadata, "keywords"),
            creator=_clean_metadata_value(raw_metadata, "creator"),
            producer=_clean_metadata_value(raw_metadata, "producer"),
            creation_date=_clean_metadata_value(
                raw_metadata,
                "creationDate",
            ),
            modification_date=_clean_metadata_value(
                raw_metadata,
                "modDate",
            ),
        )

        return ParsedDocument(
            metadata=metadata,
            pages=tuple(pages),
        )

    finally:
        document.close()
