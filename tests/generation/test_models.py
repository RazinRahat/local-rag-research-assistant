import pytest
from pydantic import ValidationError

from research_assistant.generation.models import (
    ChatMessage,
    LLMConfig,
)


def test_chat_message_strips_whitespace() -> None:
    message = ChatMessage(
        role="user",
        content="  Hello model.  ",
    )

    assert message.content == "Hello model."


def test_chat_message_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(
            role="user",
            content="   ",
        )


def test_llm_config_defaults() -> None:
    config = LLMConfig()

    assert config.model_name == "qwen3.5:9b"
    assert config.context_size == 8192
    assert config.max_output_tokens == 512
    assert config.think is False
