"""OpenAI-compatible LLM client."""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

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
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout if timeout is not None else settings.llm_timeout
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        self.extra_body = extra_body
        self.last_model: str | None = None
        self.client: Any | None = None
        self._clients: dict[tuple[str, str], Any] = {}

    def complete(self, messages: Sequence[Message]) -> str:
        return self._complete_with_fallbacks(messages)

    def complete_stream(
        self,
        messages: Sequence[Message],
        *,
        task_name: str = "llm_generating",
        progress: ProgressCallback | None = None,
    ) -> str:
        return self._complete_with_fallbacks(messages, task_name=task_name, progress=progress, stream=True)

    def model_sequence(self) -> list[str]:
        return self._model_sequence()

    def model_sequence_label(self) -> str:
        return " -> ".join(self._model_sequence())

    def _complete_with_fallbacks(
        self,
        messages: Sequence[Message],
        *,
        task_name: str = "llm_generating",
        progress: ProgressCallback | None = None,
        stream: bool = False,
    ) -> str:
        models = self._model_sequence()
        for index, model in enumerate(models):
            try:
                return self._complete_with_retries(
                    messages,
                    model=model,
                    task_name=task_name,
                    progress=progress,
                    stream=stream,
                )
            except Exception as exc:
                if index >= len(models) - 1 or not _is_fallbackable_error(exc):
                    raise
                next_model = models[index + 1]
                emit(
                    progress,
                    "log",
                    task_name,
                    f"LLM 模型 {model} 失败，切换到 {next_model}：{_error_summary(exc)}",
                )
        raise RuntimeError("LLM fallback loop exited unexpectedly")

    def _complete_with_retries(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        task_name: str = "llm_generating",
        progress: ProgressCallback | None = None,
        stream: bool = False,
    ) -> str:
        max_retries = max(0, int(self.settings.llm_max_retries))
        attempts = max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._complete_once(
                    messages,
                    model=model,
                    task_name=task_name,
                    progress=progress,
                    stream=stream,
                )
            except Exception as exc:
                if attempt >= attempts or not _is_retryable_error(exc):
                    raise
                delay = _retry_delay(
                    attempt,
                    base_delay=self.settings.llm_retry_base_delay,
                    max_delay=self.settings.llm_retry_max_delay,
                )
                emit(
                    progress,
                    "log",
                    task_name,
                    f"LLM 模型 {model} 调用失败，{delay:g}s 后重试 ({attempt}/{max_retries})：{_error_summary(exc)}",
                )
                if delay > 0:
                    time.sleep(delay)
        raise RuntimeError("LLM retry loop exited unexpectedly")

    def _complete_once(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        task_name: str = "llm_generating",
        progress: ProgressCallback | None = None,
        stream: bool = False,
    ) -> str:
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.extra_body is not None:
            kwargs["extra_body"] = self.extra_body

        if stream:
            kwargs["stream"] = True
            chars = 0
            reported = 0
            pieces: list[str] = []
            client = self._client_for_model(model)
            stream = client.chat.completions.create(
                model=model,
                messages=list(messages),
                **kwargs,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    pieces.append(delta.content)
                    chars += len(delta.content)
                    while chars - reported >= 150:
                        reported += 150
                        emit(
                            progress,
                            "task_update",
                            task_name,
                            f"LLM 生成中 ({chars} 字符)",
                        )
            remaining = chars - reported
            if remaining:
                emit(
                    progress,
                    "task_update",
                    task_name,
                    f"LLM 生成完成 ({chars} 字符)",
                )
            content = "".join(pieces).strip()
            self.last_model = model
            return content

        client = self._client_for_model(model)
        response = client.chat.completions.create(
            model=model,
            messages=list(messages),
            **kwargs,
        )
        content = response.choices[0].message.content
        self.last_model = model
        return content.strip() if content else ""

    def _model_sequence(self) -> list[str]:
        models = [self.model, *self.settings.llm_fallback_models]
        sequence: list[str] = []
        seen: set[str] = set()
        for model in models:
            model = model.strip()
            if model and model not in seen:
                seen.add(model)
                sequence.append(model)
        return sequence

    def _client_for_model(self, model: str) -> Any:
        if self.client is not None:
            return self.client
        base_url, api_key = self._endpoint_for_model(model)
        cache_key = (base_url, api_key)
        client = self._clients.get(cache_key)
        if client is None:
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=self.timeout)
            self._clients[cache_key] = client
        return client

    def _endpoint_for_model(self, model: str) -> tuple[str, str]:
        if self.base_url is not None or self.api_key is not None:
            return (
                self.base_url or self.settings.llm_base_url,
                self.api_key or self.settings.llm_api_key,
            )

        provider = self.settings.llm_model_providers.get(model.strip())
        if provider:
            provider = provider.lower()
            return (
                self.settings.llm_provider_base_urls.get(provider) or self.settings.llm_base_url,
                self.settings.llm_provider_api_keys.get(provider) or self.settings.llm_api_key,
            )
        return self.settings.llm_base_url, self.settings.llm_api_key


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 425, 429} or exc.status_code >= 500
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


def _is_fallbackable_error(exc: Exception) -> bool:
    if _is_retryable_error(exc):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {400, 404}
    return False


def _retry_delay(attempt: int, *, base_delay: float, max_delay: float) -> float:
    base = max(0.0, float(base_delay))
    cap = max(base, float(max_delay))
    delay = min(cap, base * (2 ** max(0, attempt - 1)))
    if delay <= 0:
        return 0.0
    return round(delay + random.uniform(0, delay * 0.25), 2)


def _error_summary(exc: Exception, max_chars: int = 240) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
