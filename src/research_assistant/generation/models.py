from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ChatRole = Literal[
    "system",
    "user",
    "assistant",
]


class ChatMessage(BaseModel):
    """A single message supplied to a language model."""

    model_config = ConfigDict(frozen=True)

    role: ChatRole
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Chat message content cannot be empty")

        return value


class LLMConfig(BaseModel):
    """Configuration for local language generation."""

    model_config = ConfigDict(frozen=True)

    base_url: str = "http://127.0.0.1:11434"

    model_name: str = "qwen3.5:9b"

    temperature: float = Field(
        default=0.2,
        ge=0.0,
    )

    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
    )

    context_size: int = Field(
        default=8192,
        ge=1,
    )

    max_output_tokens: int = Field(
        default=512,
        ge=1,
    )

    seed: int = 42

    think: bool = False

    keep_alive: str = "5m"

    timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
    )


class GenerationResult(BaseModel):
    """Normalized result returned by an LLM provider."""

    model_config = ConfigDict(frozen=True)

    content: str
    model_name: str

    thinking: str | None = None
    done_reason: str | None = None

    prompt_tokens: int = Field(
        default=0,
        ge=0,
    )

    completion_tokens: int = Field(
        default=0,
        ge=0,
    )

    total_duration_ns: int = Field(
        default=0,
        ge=0,
    )

    load_duration_ns: int = Field(
        default=0,
        ge=0,
    )

    prompt_eval_duration_ns: int = Field(
        default=0,
        ge=0,
    )

    generation_duration_ns: int = Field(
        default=0,
        ge=0,
    )
