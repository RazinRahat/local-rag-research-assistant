class LLMError(Exception):
    """Base exception for language-model failures."""


class LLMConnectionError(LLMError):
    """Raised when the local model runtime cannot be reached."""


class LLMResponseError(LLMError):
    """Raised when the model runtime returns an invalid response."""
