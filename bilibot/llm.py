"""OpenAI-compatible LLM client."""

from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from .config import Settings


Message = dict[str, str]


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.settings = settings
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        self.client = OpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key,
            timeout=timeout or settings.llm_timeout,
        )

    def complete(self, messages: Sequence[Message]) -> str:
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            **kwargs,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
