from collections.abc import Sequence

from research_assistant.chunking.models import (
    DocumentChunk,
)
from research_assistant.generation.models import (
    ChatMessage,
    GenerationResult,
)
from research_assistant.rag.context_builder import (
    ContextBuilder,
)
from research_assistant.rag.service import (
    RAGService,
)
from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalResult,
)


class FakeTokenCounter:
    def count_messages(
        self,
        messages: Sequence[ChatMessage],
    ) -> int:
        return sum(len(message.content.split()) for message in messages)


def _make_result() -> RetrievalResult:
    text = "The Transformer dispenses with recurrence and relies entirely on attention."

    chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="document-a",
        file_name="paper.pdf",
        page_number=1,
        chunk_index=0,
        page_chunk_index=0,
        text=text,
        token_count=12,
        char_count=len(text),
        token_start=0,
        token_end=12,
    )

    return RetrievalResult(
        rank=1,
        score=0.91,
        chunk=chunk,
    )


class FakeRetriever:
    def __init__(
        self,
        results: tuple[RetrievalResult, ...],
    ) -> None:
        self._results = results

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> tuple[RetrievalResult, ...]:
        return self._results


class FakeLLM:
    def __init__(self) -> None:
        self.called = False

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def context_size(self) -> int:
        return 8192

    @property
    def max_output_tokens(self) -> int:
        return 512

    def chat(
        self,
        messages: Sequence[ChatMessage],
    ) -> GenerationResult:
        self.called = True

        return GenerationResult(
            content=("The Transformer avoids recurrence by relying on attention."),
            model_name="fake-model",
            prompt_tokens=120,
            completion_tokens=14,
        )

    def close(self) -> None:
        return None


def test_rag_service_generates_from_evidence() -> None:
    llm = FakeLLM()

    service = RAGService(
        retriever=FakeRetriever((_make_result(),)),
        context_builder=ContextBuilder(FakeTokenCounter()),
        llm=llm,
    )

    response = service.ask("Why does the Transformer avoid recurrence?")

    assert llm.called is True

    assert response.insufficient_evidence is False

    assert response.used_evidence_count == 1

    assert len(response.evidence) == 1

    assert response.generation is not None

    assert response.actual_prompt_tokens == 120


def test_rag_service_does_not_call_llm_without_evidence() -> None:
    llm = FakeLLM()

    service = RAGService(
        retriever=FakeRetriever(()),
        context_builder=ContextBuilder(FakeTokenCounter()),
        llm=llm,
    )

    response = service.ask("What is the answer?")

    assert llm.called is False

    assert response.insufficient_evidence is True

    assert response.evidence == ()

    assert response.generation is None
