import json

import httpx
import pytest

from research_assistant.generation.exceptions import (
    LLMResponseError,
)
from research_assistant.generation.models import (
    ChatMessage,
    LLMConfig,
)
from research_assistant.generation.ollama import (
    OllamaProvider,
)


def test_ollama_provider_parses_response() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.url.path == "/api/chat"

        body = json.loads(request.content.decode("utf-8"))

        assert body["model"] == "test-model"
        assert body["stream"] is False
        assert body["think"] is False

        assert body["options"]["num_ctx"] == 8192

        return httpx.Response(
            status_code=200,
            json={
                "model": "test-model",
                "message": {
                    "role": "assistant",
                    "content": ("Dense retrieval finds relevant evidence."),
                },
                "done": True,
                "done_reason": "stop",
                "total_duration": 1000,
                "load_duration": 100,
                "prompt_eval_count": 20,
                "prompt_eval_duration": 200,
                "eval_count": 8,
                "eval_duration": 700,
            },
        )

    transport = httpx.MockTransport(handler)

    client = httpx.Client(
        transport=transport,
        base_url="http://test",
    )

    provider = OllamaProvider(
        config=LLMConfig(
            base_url="http://test",
            model_name="test-model",
        ),
        client=client,
    )

    try:
        result = provider.chat(
            (
                ChatMessage(
                    role="user",
                    content="Explain dense retrieval.",
                ),
            )
        )

        assert result.model_name == "test-model"

        assert result.content == ("Dense retrieval finds relevant evidence.")

        assert result.prompt_tokens == 20
        assert result.completion_tokens == 8

        assert result.done_reason == "stop"

    finally:
        provider.close()


def test_ollama_provider_rejects_empty_response() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "model": "test-model",
                "message": {
                    "role": "assistant",
                    "content": "",
                },
                "done": True,
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
    )

    provider = OllamaProvider(
        config=LLMConfig(
            base_url="http://test",
            model_name="test-model",
        ),
        client=client,
    )

    try:
        with pytest.raises(LLMResponseError):
            provider.chat(
                (
                    ChatMessage(
                        role="user",
                        content="Hello",
                    ),
                )
            )
    finally:
        provider.close()
