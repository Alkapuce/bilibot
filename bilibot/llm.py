"""OpenAI-compatible LLM client."""

from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from .config import Settings


Message = dict[str, str]


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
        )

    def complete(self, messages: Sequence[Message]) -> str:
        kwargs = {}
        if self.settings.llm_temperature is not None:
            kwargs["temperature"] = self.settings.llm_temperature

        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=list(messages),
            **kwargs,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
