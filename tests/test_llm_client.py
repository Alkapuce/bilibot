from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from bilibot.config import Settings, load_settings
from bilibot.llm import LLMClient


class _Delta:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _BrokenStream:
    def __iter__(self):
        raise httpx.RemoteProtocolError("peer closed connection")


class _GoodStream:
    def __iter__(self):
        yield _Chunk("重试")
        yield _Chunk("成功")


class _Completions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _BrokenStream()
        return _GoodStream()


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class _Client:
    def __init__(self):
        self.chat = _Chat()


class _Message:
    def __init__(self, content: str):
        self.content = content


class _ResponseChoice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str):
        self.choices = [_ResponseChoice(content)]


class _FallbackCompletions:
    def __init__(self):
        self.models: list[str] = []

    def create(self, **kwargs):
        model = kwargs["model"]
        self.models.append(model)
        if model == "grok-4.6":
            raise httpx.RemoteProtocolError("peer closed connection")
        return _Response(f"{model} ok")


class _FallbackChat:
    def __init__(self):
        self.completions = _FallbackCompletions()


class _FallbackClient:
    def __init__(self):
        self.chat = _FallbackChat()


class _EndpointCompletions:
    def __init__(self, client, factory):
        self.client = client
        self.factory = factory

    def create(self, **kwargs):
        model = kwargs["model"]
        self.factory.calls.append((self.client.base_url, self.client.api_key, model))
        if model in self.factory.fail_models:
            raise httpx.RemoteProtocolError(f"{model} failed")
        return _Response(f"{model} ok")


class _EndpointChat:
    def __init__(self, client, factory):
        self.completions = _EndpointCompletions(client, factory)


class _EndpointClient:
    def __init__(self, factory, *, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.chat = _EndpointChat(self, factory)


class _OpenAIFactory:
    def __init__(self, *, fail_models: set[str]):
        self.fail_models = fail_models
        self.calls: list[tuple[str, str, str]] = []
        self.clients: list[_EndpointClient] = []

    def __call__(self, *, base_url: str, api_key: str, timeout: float):
        client = _EndpointClient(self, base_url=base_url, api_key=api_key)
        self.clients.append(client)
        return client


class _Responses:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": "responses ok"})()


class _ResponsesClient:
    def __init__(self):
        self.responses = _Responses()


class LLMClientRetryTests(unittest.TestCase):
    def test_env_parses_fallback_model_chain(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "LLM_MODEL": "grok-4.6",
                "LLM_FALLBACK_MODELS": "grok-4.5 dsv4flash,dsv4pro",
                "LLM_PROVIDER_GROK_BASE_URL": "https://grok.example/v1",
                "LLM_PROVIDER_GROK_API_KEY": "grok-key",
            },
        ):
            settings = load_settings()

        self.assertEqual(settings.llm_model, "grok-4.6")
        self.assertEqual(settings.llm_fallback_models, ("grok-4.5", "dsv4flash", "dsv4pro"))
        self.assertEqual(settings.llm_model_providers["grok-4.6"], "grok")
        self.assertEqual(settings.llm_model_providers["dsv4flash"], "deepseek")
        self.assertEqual(settings.llm_provider_base_urls["grok"], "https://grok.example/v1")
        self.assertEqual(settings.llm_provider_api_keys["grok"], "grok-key")

    def test_fallback_uses_model_specific_provider_endpoint(self) -> None:
        factory = _OpenAIFactory(fail_models={"grok-4.6", "grok-4.5"})
        settings = Settings(
            llm_base_url="https://deepseek.example/v1",
            llm_api_key="deepseek-key",
            llm_model="grok-4.6",
            llm_fallback_models=("grok-4.5", "dsv4flash", "dsv4pro"),
            llm_provider_base_urls={"grok": "https://grok.example/v1"},
            llm_provider_api_keys={"grok": "grok-key"},
            llm_max_retries=0,
        )

        with patch("bilibot.llm.OpenAI", factory):
            llm = LLMClient(settings)
            result = llm.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "dsv4flash ok")
        self.assertEqual(llm.last_model, "dsv4flash")
        self.assertEqual(
            factory.calls,
            [
                ("https://grok.example/v1", "grok-key", "grok-4.6"),
                ("https://grok.example/v1", "grok-key", "grok-4.5"),
                ("https://deepseek.example/v1", "deepseek-key", "dsv4flash"),
            ],
        )
        self.assertEqual(len(factory.clients), 2)

    def test_non_streaming_falls_back_to_next_model(self) -> None:
        llm = LLMClient(
            Settings(
                llm_api_key="test-key",
                llm_model="grok-4.6",
                llm_fallback_models=("grok-4.5", "dsv4flash", "dsv4pro"),
                llm_max_retries=0,
            )
        )
        fake_client = _FallbackClient()
        llm.client = fake_client
        events = []

        result = llm._complete_with_fallbacks([{"role": "user", "content": "hi"}], progress=events.append)

        self.assertEqual(result, "grok-4.5 ok")
        self.assertEqual(llm.model_sequence_label(), "grok-4.6 -> grok-4.5 -> dsv4flash -> dsv4pro")
        self.assertEqual(llm.last_model, "grok-4.5")
        self.assertEqual(fake_client.chat.completions.models, ["grok-4.6", "grok-4.5"])
        self.assertTrue(any("切换到 grok-4.5" in event.message for event in events))

    def test_streaming_retry_replays_transient_transport_error(self) -> None:
        llm = LLMClient(
            Settings(
                llm_api_key="test-key",
                llm_max_retries=1,
                llm_retry_base_delay=0,
                llm_retry_max_delay=0,
            )
        )
        fake_client = _Client()
        llm.client = fake_client
        events = []

        result = llm.complete_stream(
            [{"role": "user", "content": "hi"}],
            progress=events.append,
        )

        self.assertEqual(result, "重试成功")
        self.assertEqual(llm.last_model, "grok-4.6")
        self.assertEqual(fake_client.chat.completions.calls, 2)
        self.assertTrue(any(event.kind == "log" and "重试" in event.message for event in events))
        self.assertTrue(any(event.kind == "task_update" and "LLM 生成完成" in event.message for event in events))
        self.assertTrue(all(event.advance is None for event in events if event.kind == "task_update"))

    def test_responses_api_uses_input_and_max_output_tokens(self) -> None:
        llm = LLMClient(
            Settings(
                llm_api_key="test-key",
                llm_model="gpt-5.5",
                llm_max_tokens=321,
                llm_temperature=0.2,
                llm_model_providers={"gpt-5.5": "codex"},
                llm_provider_wire_apis={"codex": "responses"},
            )
        )
        fake_client = _ResponsesClient()
        llm.client = fake_client

        result = llm.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "responses ok")
        self.assertEqual(llm.last_model, "gpt-5.5")
        self.assertEqual(
            fake_client.responses.calls,
            [{
                "model": "gpt-5.5",
                "input": [{"role": "user", "content": "hi"}],
                "max_output_tokens": 321,
            }],
        )


if __name__ == "__main__":
    unittest.main()
