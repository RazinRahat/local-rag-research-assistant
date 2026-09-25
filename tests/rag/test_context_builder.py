from collections.abc import Sequence

from research_assistant.chunking.models import (
    DocumentChunk,
)
from research_assistant.generation.models import (
    ChatMessage,
)
from research_assistant.rag.context_builder import (
    ContextBuilder,
)
from research_assistant.rag.models import (
    ContextConfig,
)
from research_assistant.retrieval.models import (
    RetrievalResult,
)


class FakeTokenCounter:
    def count_messages(
        self,
        messages: Sequence[ChatMessage],
    ) -> int:
        return sum(len(message.content.split()) for message in messages) + (
            len(messages) * 4
        )


def _make_result(
    rank: int,
    token_start: int,
    token_end: int,
    text: str,
) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=f"chunk-{rank}",
        document_id="document-a",
        file_name="paper.pdf",
        page_number=1,
        chunk_index=rank - 1,
        page_chunk_index=rank - 1,
        text=text,
        token_count=(token_end - token_start),
        char_count=len(text),
        token_start=token_start,
        token_end=token_end,
    )

    return RetrievalResult(
        rank=rank,
        score=1.0 - (rank * 0.05),
        chunk=chunk,
    )


def test_context_builder_preserves_retrieval_order() -> None:
    builder = ContextBuilder(token_counter=FakeTokenCounter())

    results = (
        _make_result(
            1,
            0,
            100,
            "First relevant evidence.",
        ),
        _make_result(
            2,
            100,
            200,
            "Second relevant evidence.",
        ),
    )

    context = builder.build(
        question="What does the paper say?",
        results=results,
        context_size=2000,
        reserved_output_tokens=200,
    )

    assert len(context.evidence) == 2

    assert context.evidence[0].retrieval_rank == 1

    assert context.evidence[1].retrieval_rank == 2

    assert context.evidence[0].source_id == "S1"


def test_context_builder_removes_duplicate_chunk() -> None:
    builder = ContextBuilder(token_counter=FakeTokenCounter())

    first = _make_result(
        1,
        0,
        100,
        "Same evidence.",
    )

    duplicate = RetrievalResult(
        rank=2,
        score=0.9,
        chunk=first.chunk,
    )

    context = builder.build(
        question="Question?",
        results=(first, duplicate),
        context_size=2000,
        reserved_output_tokens=200,
    )

    assert len(context.evidence) == 1
    assert context.skipped_redundant == 1


def test_context_builder_respects_budget() -> None:
    builder = ContextBuilder(
        token_counter=FakeTokenCounter(),
        config=ContextConfig(
            safety_margin_tokens=20,
        ),
    )

    long_text = "evidence " * 100

    results = tuple(
        _make_result(
            rank=index + 1,
            token_start=index * 100,
            token_end=(index + 1) * 100,
            text=long_text,
        )
        for index in range(5)
    )

    context = builder.build(
        question="Explain the evidence.",
        results=results,
        context_size=300,
        reserved_output_tokens=50,
    )

    assert context.skipped_for_budget > 0
    assert context.truncated is True

    assert context.estimated_prompt_tokens <= context.prompt_budget_tokens
