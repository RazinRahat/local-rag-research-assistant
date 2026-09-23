from research_assistant.ingestion.normalizer import normalize_text


def test_normalize_text_removes_null_characters() -> None:
    text = "Retrieval\x00 augmented generation"

    result = normalize_text(text)

    assert result == "Retrieval augmented generation"


def test_normalize_text_normalizes_line_endings() -> None:
    text = "First line\r\nSecond line\rThird line"

    result = normalize_text(text)

    assert result == "First line\nSecond line\nThird line"


def test_normalize_text_collapses_excess_blank_lines() -> None:
    text = "Introduction\n\n\n\nMethods"

    result = normalize_text(text)

    assert result == "Introduction\n\nMethods"


def test_normalize_text_removes_trailing_whitespace() -> None:
    text = "First line   \nSecond line\t"

    result = normalize_text(text)

    assert result == "First line\nSecond line"
