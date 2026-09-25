from research_assistant.generation.base import (
    LLMProvider,
)
from research_assistant.rag.context_builder import (
    ContextBuilder,
)
from research_assistant.rag.models import (
    RAGResponse,
)
from research_assistant.rag.prompts import (
    INSUFFICIENT_EVIDENCE_ANSWER,
)
from research_assistant.retrieval.base import (
    Retriever,
)
from research_assistant.retrieval.models import (
    RetrievalConfig,
)


class RAGService:
    """Coordinate retrieval, context building, and generation."""

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        llm: LLMProvider,
    ) -> None:
        self._retriever = retriever
        self._context_builder = context_builder
        self._llm = llm

    def ask(
        self,
        question: str,
        retrieval_config: RetrievalConfig | None = None,
    ) -> RAGResponse:
        """Answer a question using retrieved document evidence."""

        clean_question = question.strip()

        if not clean_question:
            raise ValueError("RAG question cannot be empty")

        retrieved = self._retriever.retrieve(
            query=clean_question,
            config=retrieval_config,
        )

        if not retrieved:
            return RAGResponse(
                question=clean_question,
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                evidence=(),
                retrieved_count=0,
                used_evidence_count=0,
                insufficient_evidence=True,
            )

        context = self._context_builder.build(
            question=clean_question,
            results=retrieved,
            context_size=self._llm.context_size,
            reserved_output_tokens=(self._llm.max_output_tokens),
        )

        if not context.evidence:
            return RAGResponse(
                question=clean_question,
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                evidence=(),
                retrieved_count=len(retrieved),
                used_evidence_count=0,
                estimated_prompt_tokens=(context.estimated_prompt_tokens),
                context_truncated=context.truncated,
                insufficient_evidence=True,
            )

        generation = self._llm.chat(context.messages)

        return RAGResponse(
            question=clean_question,
            answer=generation.content,
            evidence=context.evidence,
            retrieved_count=len(retrieved),
            used_evidence_count=len(context.evidence),
            estimated_prompt_tokens=(context.estimated_prompt_tokens),
            actual_prompt_tokens=(generation.prompt_tokens),
            context_truncated=context.truncated,
            insufficient_evidence=False,
            generation=generation,
        )
