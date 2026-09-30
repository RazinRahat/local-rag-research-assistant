"""Tests for API error classification."""

import pytest

from research_assistant.api.exceptions import (
    APIError,
    DocumentNotFoundError,
    IndexingError,
    InvalidUploadError,
    UnsupportedMediaError,
    UploadTooLargeError,
)


@pytest.mark.parametrize(
    ("exception_type", "expected_status"),
    [
        (InvalidUploadError, 422),
        (UnsupportedMediaError, 415),
        (UploadTooLargeError, 413),
        (DocumentNotFoundError, 404),
        (IndexingError, 500),
    ],
)
def test_api_error_status_codes(
    exception_type: type[APIError],
    expected_status: int,
) -> None:
    error = exception_type("Test failure")

    assert isinstance(error, APIError)
    assert error.status_code == expected_status
    assert str(error) == "Test failure"
