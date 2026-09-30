from research_assistant.chunking.models import (
    DocumentChunk,
)
from research_assistant.citations.service import (
    CitationService,
)
from research_assistant.generation.models import (
    GenerationResult,
)
from research_assistant.rag.models import (
    EvidenceBlock,
    RAGResponse,
)


def _make_evidence() -> EvidenceBlock:
    text = "The Transformer relies entirely on attention."

    chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="document-a",
        file_name="paper.pdf",
        page_number=1,
        chunk_index=0,
        page_chunk_index=0,
        text=text,
        token_count=8,
        char_count=len(text),
        token_start=0,
        token_end=8,
    )

    return EvidenceBlock(
        source_id="S1",
        retrieval_rank=1,
        score=0.91,
        chunk=chunk,
    )


def test_service_validates_rag_response() -> None:
    evidence = _make_evidence()

    response = RAGResponse(
        question=("How does the Transformer work?"),
        answer=("The Transformer relies entirely on attention. [S1]"),
        evidence=(evidence,),
        retrieved_count=1,
        used_evidence_count=1,
        estimated_prompt_tokens=100,
        actual_prompt_tokens=105,
        generation=GenerationResult(
            content=("The Transformer relies entirely on attention. [S1]"),
            model_name="test-model",
        ),
    )

    service = CitationService()

    cited = service.validate(response)

    assert len(cited.citations.references) == 1

    assert len(cited.citations.sources) == 1


def test_insufficient_evidence_needs_no_citations() -> None:
    response = RAGResponse(
        question="Unknown question",
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

    service = CitationService()

    cited = service.validate(response)

    assert cited.citations.references == ()

    assert cited.citations.sources == ()
