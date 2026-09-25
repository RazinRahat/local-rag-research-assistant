from collections.abc import Sequence

import httpx
from pydantic import BaseModel, ValidationError

from research_assistant.generation.exceptions import (
    LLMConnectionError,
    LLMResponseError,
)
from research_assistant.generation.models import (
    ChatMessage,
    GenerationResult,
    LLMConfig,
)


class _OllamaMessage(BaseModel):
    role: str
    content: str
    thinking: str | None = None


class _OllamaChatResponse(BaseModel):
    model: str
    message: _OllamaMessage

    done: bool
    done_reason: str | None = None

    total_duration: int = 0
    load_duration: int = 0

    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0

    eval_count: int = 0
    eval_duration: int = 0


class OllamaProvider:
    """Local LLM provider backed by Ollama."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config or LLMConfig()

        self._client = (
            client
            if client is not None
            else httpx.Client(
                base_url=self._config.base_url,
                timeout=self._config.timeout_seconds,
            )
        )

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def context_size(self) -> int:
        return self._config.context_size

    @property
    def max_output_tokens(self) -> int:
        return self._config.max_output_tokens

    def chat(
        self,
        messages: Sequence[ChatMessage],
    ) -> GenerationResult:
        """Generate a non-streaming chat response."""

        if not messages:
            raise ValueError("At least one chat message is required")

        payload = {
            "model": self._config.model_name,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
            "stream": False,
            "think": self._config.think,
            "keep_alive": self._config.keep_alive,
            "options": {
                "temperature": (self._config.temperature),
                "top_p": self._config.top_p,
                "num_ctx": (self._config.context_size),
                "num_predict": (self._config.max_output_tokens),
                "seed": self._config.seed,
            },
        }

        try:
            response = self._client.post(
                "/api/chat",
                json=payload,
            )

            response.raise_for_status()

        except httpx.RequestError as exc:
            raise LLMConnectionError("Unable to connect to the Ollama runtime") from exc

        except httpx.HTTPStatusError as exc:
            raise LLMResponseError(
                f"Ollama returned an HTTP error: {exc.response.status_code}"
            ) from exc

        try:
            parsed = _OllamaChatResponse.model_validate(response.json())

        except (
            ValueError,
            ValidationError,
        ) as exc:
            raise LLMResponseError("Ollama returned an invalid response") from exc

        if not parsed.done:
            raise LLMResponseError("Expected a completed non-streaming Ollama response")

        content = parsed.message.content.strip()

        if not content:
            raise LLMResponseError("Ollama returned an empty response")

        thinking = parsed.message.thinking

        if thinking is not None:
            thinking = thinking.strip() or None

        return GenerationResult(
            content=content,
            model_name=parsed.model,
            thinking=thinking,
            done_reason=parsed.done_reason,
            prompt_tokens=parsed.prompt_eval_count,
            completion_tokens=parsed.eval_count,
            total_duration_ns=parsed.total_duration,
            load_duration_ns=parsed.load_duration,
            prompt_eval_duration_ns=(parsed.prompt_eval_duration),
            generation_duration_ns=(parsed.eval_duration),
        )

    def close(self) -> None:
        self._client.close()
