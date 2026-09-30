"""Application errors exposed through the HTTP API."""

from typing import ClassVar


class APIError(Exception):
    """Base exception for expected application-level failures."""

    status_code: ClassVar[int] = 500


class InvalidUploadError(APIError):
    """The uploaded document cannot be processed."""

    status_code = 422


class UnsupportedMediaError(APIError):
    """The uploaded file type is unsupported."""

    status_code = 415


class UploadTooLargeError(APIError):
    """The document exceeds the configured upload limit."""

    status_code = 413


class DocumentNotFoundError(APIError):
    """The requested indexed document does not exist."""

    status_code = 404


class IndexingError(APIError):
    """Document indexing could not be completed."""

    status_code = 500
