from typing import Protocol, cast

import tiktoken
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class Tokenizer(Protocol):
    """Interface required by the document chunker."""

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Convert token IDs back into text."""
        ...

    def count(self, text: str) -> int:
        """Return the number of tokens in text."""
        ...


class TiktokenTokenizer:
    """Tokenizer backed by a tiktoken encoding."""

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(
            text,
            disallowed_special=(),
        )

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


class HuggingFaceTokenizer:
    """Tokenizer backed by a Hugging Face model tokenizer."""

    def __init__(self, model_name: str) -> None:
        self._tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(model_name),
        )

    def encode(self, text: str) -> list[int]:
        token_ids = self._tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )

        return list(token_ids)

    def decode(self, tokens: list[int]) -> str:
        decoded = self._tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if not isinstance(decoded, str):
            raise TypeError(
                "Tokenizer returned batched output for a single token sequence"
            )

        return decoded

    def count(self, text: str) -> int:
        return len(self.encode(text))
