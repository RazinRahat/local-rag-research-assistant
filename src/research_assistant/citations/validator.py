from collections.abc import Sequence

from research_assistant.citations.exceptions import (
    DuplicateSourceIdError,
    MissingCitationError,
    UnknownCitationError,
)
from research_assistant.citations.models import (
    CitationSource,
    CitationValidationResult,
)
from research_assistant.citations.parser import (
    parse_citations,
)
from research_assistant.rag.models import (
    EvidenceBlock,
)


def validate_citations(
    answer: str,
    evidence: Sequence[EvidenceBlock],
    *,
    require_citations: bool = True,
) -> CitationValidationResult:
    """Validate generated source references against supplied evidence."""

    references = parse_citations(answer)

    if require_citations and not references:
        raise MissingCitationError(
            "Generated answer does not contain any valid source citations"
        )

    evidence_by_id: dict[str, EvidenceBlock] = {}

    for item in evidence:
        if item.source_id in evidence_by_id:
            raise DuplicateSourceIdError(
                f"Duplicate evidence source identifier: {item.source_id}"
            )

        evidence_by_id[item.source_id] = item

    unknown_ids = tuple(
        dict.fromkeys(
            reference.source_id
            for reference in references
            if reference.source_id not in evidence_by_id
        )
    )

    if unknown_ids:
        joined = ", ".join(unknown_ids)

        raise UnknownCitationError(
            f"Generated answer referenced unknown source identifiers: {joined}"
        )

    unique_sources: list[CitationSource] = []

    seen: set[str] = set()

    for reference in references:
        source_id = reference.source_id

        if source_id in seen:
            continue

        seen.add(source_id)

        unique_sources.append(
            CitationSource(
                source_id=source_id,
                evidence=evidence_by_id[source_id],
            )
        )

    return CitationValidationResult(
        references=references,
        sources=tuple(unique_sources),
    )
