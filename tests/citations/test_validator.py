import pytest

from research_assistant.chunking.models import (
    DocumentChunk,
)
from research_assistant.citations.exceptions import (
    MissingCitationError,
    UnknownCitationError,
)
from research_assistant.citations.validator import (
    validate_citations,
)
from research_assistant.rag.models import (
    EvidenceBlock,
)


def _make_evidence(
    source_id: str,
    rank: int,
) -> EvidenceBlock:
    text = f"Evidence text for {source_id}."

    chunk = DocumentChunk(
        chunk_id=f"chunk-{source_id}",
        document_id="document-a",
        file_name="paper.pdf",
        page_number=rank,
        chunk_index=rank - 1,
        page_chunk_index=0,
        text=text,
        token_count=5,
        char_count=len(text),
        token_start=0,
        token_end=5,
    )

    return EvidenceBlock(
        source_id=source_id,
        retrieval_rank=rank,
        score=0.9,
        chunk=chunk,
    )


def test_validate_known_citation() -> None:
    evidence = (_make_evidence("S1", 1),)

    result = validate_citations(
        answer="The result follows from attention. [S1]",
        evidence=evidence,
    )

    assert len(result.references) == 1
    assert len(result.sources) == 1

    assert result.sources[0].source_id == "S1"

    assert result.sources[0].evidence.chunk.file_name == "paper.pdf"


def test_repeated_references_have_one_unique_source() -> None:
    evidence = (_make_evidence("S1", 1),)

    result = validate_citations(
        answer=("First statement. [S1] Second statement. [S1]"),
        evidence=evidence,
    )

    assert len(result.references) == 2
    assert len(result.sources) == 1


def test_unknown_citation_is_rejected() -> None:
    evidence = (_make_evidence("S1", 1),)

    with pytest.raises(UnknownCitationError):
        validate_citations(
            answer=("Unsupported source reference. [S9]"),
            evidence=evidence,
        )


def test_missing_citations_are_rejected() -> None:
    evidence = (_make_evidence("S1", 1),)

    with pytest.raises(MissingCitationError):
        validate_citations(
            answer=("The model uses attention."),
            evidence=evidence,
        )
