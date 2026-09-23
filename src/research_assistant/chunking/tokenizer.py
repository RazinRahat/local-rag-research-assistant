from typing import Protocol

import tiktoken


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
