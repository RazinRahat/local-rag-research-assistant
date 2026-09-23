from pathlib import Path

import pymupdf
import pytest

from research_assistant.ingestion.exceptions import (
    UnsupportedDocumentError,
)
from research_assistant.ingestion.pdf_loader import load_pdf


def _create_test_pdf(path: Path) -> None:
    document = pymupdf.open()

    first_page = document.new_page()
    first_page.insert_text(
        (72, 72),
        "Retrieval-augmented generation combines retrieval and generation.",
    )

    second_page = document.new_page()
    second_page.insert_text(
        (72, 72),
        "The second page contains evaluation methodology.",
    )

    document.set_metadata(
        {
            "title": "RAG Test Paper",
            "author": "Research Assistant Test",
        }
    )

    document.save(path)
    document.close()


def test_load_pdf_extracts_pages_and_metadata(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "research-paper.pdf"

    _create_test_pdf(pdf_path)

    result = load_pdf(pdf_path)

    assert result.metadata.file_name == "research-paper.pdf"
    assert result.metadata.title == "RAG Test Paper"
    assert result.metadata.author == "Research Assistant Test"

    assert result.metadata.page_count == 2
    assert result.metadata.text_page_count == 2

    assert len(result.metadata.sha256) == 64
    assert result.metadata.document_id == result.metadata.sha256

    assert len(result.pages) == 2

    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2

    assert "Retrieval-augmented generation" in result.pages[0].text
    assert "evaluation methodology" in result.pages[1].text

    assert result.pages[0].has_text is True
    assert result.pages[0].word_count > 0
    assert result.pages[0].char_count > 0


def test_load_pdf_raises_for_missing_file(
    tmp_path: Path,
) -> None:
    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf(missing_file)


def test_load_pdf_rejects_non_pdf(
    tmp_path: Path,
) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text(
        "This is not a PDF.",
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedDocumentError):
        load_pdf(text_file)


def test_identical_files_produce_identical_document_ids(
    tmp_path: Path,
) -> None:
    first_pdf = tmp_path / "first.pdf"

    _create_test_pdf(first_pdf)

    second_pdf = tmp_path / "second.pdf"
    second_pdf.write_bytes(first_pdf.read_bytes())

    first_result = load_pdf(first_pdf)
    second_result = load_pdf(second_pdf)

    assert first_result.metadata.document_id == second_result.metadata.document_id
