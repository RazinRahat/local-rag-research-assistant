class RAGError(Exception):
    """Base exception for RAG pipeline failures."""


class ContextBudgetError(RAGError):
    """Raised when the prompt cannot fit within the model context."""
