import re

from research_assistant.citations.models import (
    CitationReference,
)

_CITATION_PATTERN = re.compile(r"\[(S[1-9]\d*)\]")


def parse_citations(
    answer: str,
) -> tuple[CitationReference, ...]:
    """Extract strict inline source references from generated text."""

    references: list[CitationReference] = []

    for match in _CITATION_PATTERN.finditer(answer):
        references.append(
            CitationReference(
                source_id=match.group(1),
                start_index=match.start(),
                end_index=match.end(),
            )
        )

    return tuple(references)
