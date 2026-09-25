from collections.abc import Sequence
from typing import Protocol, cast

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from research_assistant.generation.models import (
    ChatMessage,
)


class ChatTokenCounter(Protocol):
    """Estimate tokens used by complete chat messages."""

    def count_messages(
        self,
        messages: Sequence[ChatMessage],
    ) -> int:
        """Count formatted chat tokens."""
        ...


class HuggingFaceChatTokenCounter:
    """Chat token counter using a Hugging Face chat template."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
    ) -> None:
        self._tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(
                model_name
            ),
        )

    def count_messages(
        self,
        messages: Sequence[ChatMessage],
    ) -> int:
        conversation = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

        formatted = self._tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        if not isinstance(formatted, str):
            raise TypeError(
                "Expected chat template to return formatted text"
            )

        token_ids = self._tokenizer.encode(
            formatted,
            add_special_tokens=False,
            truncation=False,
        )

        return len(token_ids)