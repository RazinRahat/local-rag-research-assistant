from research_assistant.citations.models import (
    CitationValidationResult,
    CitedRAGResponse,
)
from research_assistant.citations.validator import (
    validate_citations,
)
from research_assistant.rag.models import (
    RAGResponse,
)


class CitationService:
    """Validate source citations in completed RAG responses."""

    def validate(
        self,
        response: RAGResponse,
    ) -> CitedRAGResponse:
        """Validate citations against evidence used for generation."""

        if response.insufficient_evidence:
            return CitedRAGResponse(
                rag=response,
                citations=CitationValidationResult(
                    references=(),
                    sources=(),
                ),
            )

        citations = validate_citations(
            answer=response.answer,
            evidence=response.evidence,
            require_citations=True,
        )

        return CitedRAGResponse(
            rag=response,
            citations=citations,
        )
