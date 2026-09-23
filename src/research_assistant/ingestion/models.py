from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """Metadata describing an ingested document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    file_name: str
    file_size_bytes: int = Field(ge=0)
    sha256: str

    page_count: int = Field(ge=1)
    text_page_count: int = Field(ge=0)

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None


class DocumentPage(BaseModel):
    """Text and basic layout information extracted from one PDF page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)

    text: str
    char_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    has_text: bool

    width: float = Field(gt=0)
    height: float = Field(gt=0)
    rotation: int


class ParsedDocument(BaseModel):
    """Normalized representation of an ingested document."""

    model_config = ConfigDict(frozen=True)

    metadata: DocumentMetadata
    pages: tuple[DocumentPage, ...]
