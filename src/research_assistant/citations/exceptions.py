class CitationError(Exception):
    """Base exception for citation processing failures."""


class MissingCitationError(CitationError):
    """Raised when an answer requires citations but contains none."""


class UnknownCitationError(CitationError):
    """Raised when an answer references evidence that was not supplied."""


class DuplicateSourceIdError(CitationError):
    """Raised when context evidence contains duplicate source identifiers."""
