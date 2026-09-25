from collections.abc import Sequence

from research_assistant.generation.models import (
    ChatMessage,
)
from research_assistant.rag.exceptions import (
    ContextBudgetError,
)
from research_assistant.rag.models import (
    ContextBuildResult,
    ContextConfig,
    EvidenceBlock,
)
from research_assistant.rag.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from research_assistant.rag.token_counter import (
    ChatTokenCounter,
)
from research_assistant.retrieval.models import (
    RetrievalResult,
)


class ContextBuilder:
    """Build bounded grounded prompts from retrieved evidence."""

    def __init__(
        self,
        token_counter: ChatTokenCounter,
        config: ContextConfig | None = None,
    ) -> None:
        self._token_counter = token_counter
        self._config = config or ContextConfig()

    def _is_redundant(
        self,
        candidate: RetrievalResult,
        selected: Sequence[EvidenceBlock],
    ) -> bool:
        candidate_chunk = candidate.chunk

        for existing in selected:
            existing_chunk = existing.chunk

            if candidate_chunk.chunk_id == existing_chunk.chunk_id:
                return True

            if (
                candidate_chunk.document_id != existing_chunk.document_id
                or candidate_chunk.page_number != existing_chunk.page_number
            ):
                continue

            overlap_start = max(
                candidate_chunk.token_start,
                existing_chunk.token_start,
            )

            overlap_end = min(
                candidate_chunk.token_end,
                existing_chunk.token_end,
            )

            overlap = max(
                0,
                overlap_end - overlap_start,
            )

            candidate_length = candidate_chunk.token_end - candidate_chunk.token_start

            existing_length = existing_chunk.token_end - existing_chunk.token_start

            shorter_length = min(
                candidate_length,
                existing_length,
            )

            if shorter_length <= 0:
                continue

            overlap_ratio = overlap / shorter_length

            if overlap_ratio >= self._config.overlap_dedupe_threshold:
                return True

        return False

    def _build_messages(
        self,
        question: str,
        evidence: tuple[EvidenceBlock, ...],
    ) -> tuple[ChatMessage, ...]:
        return (
            ChatMessage(
                role="system",
                content=SYSTEM_PROMPT,
            ),
            ChatMessage(
                role="user",
                content=build_user_prompt(
                    question=question,
                    evidence=evidence,
                ),
            ),
        )

    def build(
        self,
        question: str,
        results: Sequence[RetrievalResult],
        context_size: int,
        reserved_output_tokens: int,
    ) -> ContextBuildResult:
        """Build a prompt that fits the model context budget."""

        clean_question = question.strip()

        if not clean_question:
            raise ValueError("RAG question cannot be empty")

        prompt_budget = (
            context_size - reserved_output_tokens - self._config.safety_margin_tokens
        )

        if prompt_budget <= 0:
            raise ContextBudgetError(
                "No prompt budget remains after reserving generation and safety tokens"
            )

        empty_messages = self._build_messages(
            question=clean_question,
            evidence=(),
        )

        base_tokens = self._token_counter.count_messages(empty_messages)

        if base_tokens > prompt_budget:
            raise ContextBudgetError(
                "System instructions and question exceed the available prompt budget"
            )

        selected: list[EvidenceBlock] = []

        skipped_redundant = 0
        skipped_for_budget = 0

        for result in results:
            if self._is_redundant(
                candidate=result,
                selected=selected,
            ):
                skipped_redundant += 1
                continue

            candidate = EvidenceBlock(
                source_id=f"S{len(selected) + 1}",
                retrieval_rank=result.rank,
                score=result.score,
                chunk=result.chunk,
            )

            proposed = tuple(
                [
                    *selected,
                    candidate,
                ]
            )

            proposed_messages = self._build_messages(
                question=clean_question,
                evidence=proposed,
            )

            proposed_tokens = self._token_counter.count_messages(proposed_messages)

            if proposed_tokens <= prompt_budget:
                selected.append(candidate)
            else:
                skipped_for_budget += 1

        evidence = tuple(selected)

        final_messages = self._build_messages(
            question=clean_question,
            evidence=evidence,
        )

        estimated_tokens = self._token_counter.count_messages(final_messages)

        return ContextBuildResult(
            messages=final_messages,
            evidence=evidence,
            estimated_prompt_tokens=estimated_tokens,
            prompt_budget_tokens=prompt_budget,
            skipped_redundant=skipped_redundant,
            skipped_for_budget=skipped_for_budget,
            truncated=skipped_for_budget > 0,
        )
