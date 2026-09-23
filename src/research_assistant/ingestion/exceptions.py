class DocumentIngestionError(Exception):
    """Base exception for document ingestion failures."""


class UnsupportedDocumentError(DocumentIngestionError):
    """Raised when an unsupported document type is provided."""


class EncryptedDocumentError(DocumentIngestionError):
    """Raised when a PDF requires a password."""


class EmptyDocumentError(DocumentIngestionError):
    """Raised when a document contains no pages."""
