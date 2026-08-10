"""OpenAI-compatible LLM client."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from .config import Settings
from .progress import ProgressCallback, emit


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
        extra_body: dict[str, Any] | None = None,
    ):
        self.settings = settings
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        self.extra_body = extra_body
        self.client = OpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key,
            timeout=timeout or settings.llm_timeout,
        )

    def complete(self, messages: Sequence[Message]) -> str:
        return self._complete_impl(messages)

    def complete_stream(
        self,
        messages: Sequence[Message],
        *,
        task_name: str = "llm_generating",
        progress: ProgressCallback | None = None,
    ) -> str:
        return self._complete_impl(messages, task_name=task_name, progress=progress)

    def _complete_impl(
        self,
        messages: Sequence[Message],
        *,
        task_name: str = "llm_generating",
        progress: ProgressCallback | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.extra_body is not None:
            kwargs["extra_body"] = self.extra_body

        if progress is not None:
            kwargs["stream"] = True
            chars = 0
            pieces: list[str] = []
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                **kwargs,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    pieces.append(delta.content)
                    chars += len(delta.content)
                    if chars % 150 == 0:
                        emit(
                            progress,
                            "task_update",
                            task_name,
                            f"LLM 生成中 ({chars} 字符)",
                            advance=150,
                        )
            if chars % 150 != 0:
                emit(
                    progress,
                    "task_update",
                    task_name,
                    f"LLM 生成完成 ({chars} 字符)",
                    advance=chars % 150,
                )
            content = "".join(pieces).strip()
            return content

        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            **kwargs,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
