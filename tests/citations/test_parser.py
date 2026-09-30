from research_assistant.citations.parser import (
    parse_citations,
)


def test_parse_single_citation() -> None:
    answer = "The Transformer uses attention. [S1]"

    references = parse_citations(answer)

    assert len(references) == 1

    assert references[0].source_id == "S1"

    assert answer[references[0].start_index : references[0].end_index] == "[S1]"


def test_parse_multiple_citations() -> None:
    answer = "Attention enables parallelisation. [S1][S2]"

    references = parse_citations(answer)

    assert tuple(reference.source_id for reference in references) == (
        "S1",
        "S2",
    )


def test_parse_repeated_citation() -> None:
    answer = "First claim. [S1] Second claim. [S1]"

    references = parse_citations(answer)

    assert len(references) == 2

    assert all(reference.source_id == "S1" for reference in references)


def test_parser_ignores_non_citation_brackets() -> None:
    answer = "The value is represented as [x, y]."

    assert parse_citations(answer) == ()
