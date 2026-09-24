from collections.abc import Sequence
from typing import Protocol

from research_assistant.generation.models import (
    ChatMessage,
    GenerationResult,
)


class LLMProvider(Protocol):
    """Interface implemented by language-model providers."""

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        ...

    def chat(
        self,
        messages: Sequence[ChatMessage],
    ) -> GenerationResult:
        """Generate the next assistant message."""
        ...

    def close(self) -> None:
        """Release provider resources."""
        ...
