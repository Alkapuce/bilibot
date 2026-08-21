"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


DEFAULT_LLM_MODEL_PROVIDERS = {
    "grok-4.6": "grok",
    "grok-4.5": "grok",
    "dsv4flash": "deepseek",
    "dsv4pro": "deepseek",
}

@dataclass(frozen=True)
class Settings:
    llm_base_url: str = "http://localhost:5001/v1"
    llm_api_key: str = ""
    llm_model: str = "grok-4.6"
    llm_fallback_models: tuple[str, ...] = ("grok-4.5", "dsv4flash", "dsv4pro")
    llm_model_providers: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LLM_MODEL_PROVIDERS))
    llm_provider_base_urls: dict[str, str] = field(default_factory=dict)
    llm_provider_api_keys: dict[str, str] = field(default_factory=dict)
    llm_wire_api: str = "chat_completions"
    llm_provider_wire_apis: dict[str, str] = field(default_factory=dict)
    llm_timeout: float = 180.0
    llm_temperature: float | None = None
    llm_max_tokens: int | None = None
    llm_max_retries: int = 2
    llm_retry_base_delay: float = 2.0
    llm_retry_max_delay: float = 20.0
    chunk_chars: int = 200000
    summary_max_single_chunk_chars: int = 60000

    output_dir: Path = Path("data")
    language: str = "zh"

    asr_backend: str = "auto"
    asr_preset: str = "auto"
    asr_model: str = ""
    asr_device: str = ""
    asr_compute_type: str = ""
    asr_task: str = "transcribe"
    asr_beam_size: int = 5
    asr_batch_size: int | None = None
    asr_vad_filter: bool | None = None
    asr_vad_min_silence_ms: int | None = None
    asr_condition_on_previous_text: bool | None = None
    asr_cpu_threads: int = 0
    asr_num_workers: int = 1
    asr_download_root: str = ""
    asr_local_files_only: bool = False
    asr_hotwords: str = ""
    asr_initial_prompt: str = ""
    asr_gguf_model_dir: str = ""
    asr_gguf_llama_bin: str = ""
    asr_gguf_cli: str = "llama-mtmd-cli"

    subtitle_postprocess: bool = False
    subtitle_postprocess_base_url: str = ""
    subtitle_postprocess_api_key: str = ""
    subtitle_postprocess_model: str = ""
    subtitle_postprocess_temperature: float | None = None
    subtitle_postprocess_chunk_chars: int = 200000
    subtitle_postprocess_style: str = "clean"
    subtitle_postprocess_disable_thinking: bool = True

    download_timeout: float = 60.0
    download_chunk_size: int = 1024 * 1024
    yt_dlp_format: str = "bestaudio"
    yt_dlp_audio_format: str = "mp3"
    yt_dlp_audio_quality: str = "5"

    bili_sessdata: str = ""
    bili_jct: str = ""
    bili_buvid3: str = ""
    cookie_file: str = ""


def load_settings(**overrides: Any) -> Settings:
    load_dotenv()
    settings = Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", Settings.llm_base_url),
        llm_api_key=os.getenv("LLM_API_KEY", Settings.llm_api_key),
        llm_model=os.getenv("LLM_MODEL", Settings.llm_model),
        llm_fallback_models=_env_list("LLM_FALLBACK_MODELS", Settings.llm_fallback_models),
        llm_model_providers=_env_model_provider_map(),
        llm_provider_base_urls=_env_provider_values("BASE_URL"),
        llm_provider_api_keys=_env_provider_values("API_KEY"),
        llm_wire_api=os.getenv("LLM_WIRE_API", Settings.llm_wire_api),
        llm_provider_wire_apis=_env_provider_values("WIRE_API"),
        llm_timeout=_env_float("LLM_TIMEOUT", Settings.llm_timeout),
        llm_temperature=_env_optional_float("LLM_TEMPERATURE"),
        llm_max_tokens=_env_optional_int("LLM_MAX_TOKENS"),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", Settings.llm_max_retries),
        llm_retry_base_delay=_env_float("LLM_RETRY_BASE_DELAY", Settings.llm_retry_base_delay),
        llm_retry_max_delay=_env_float("LLM_RETRY_MAX_DELAY", Settings.llm_retry_max_delay),
        chunk_chars=_env_int("CHUNK_CHARS", Settings.chunk_chars),
        summary_max_single_chunk_chars=_env_int(
            "SUMMARY_MAX_SINGLE_CHUNK_CHARS",
            Settings.summary_max_single_chunk_chars,
        ),
        output_dir=Path(os.getenv("BILIBOT_OUTPUT_DIR", str(Settings.output_dir))),
        language=os.getenv("TRANSCRIPT_LANGUAGE", Settings.language),
        asr_backend=os.getenv("ASR_BACKEND", Settings.asr_backend),
        asr_preset=os.getenv("ASR_PRESET", Settings.asr_preset),
        asr_model=os.getenv("ASR_MODEL", os.getenv("WHISPER_MODEL", Settings.asr_model)),
        asr_device=os.getenv("ASR_DEVICE", os.getenv("WHISPER_DEVICE", Settings.asr_device)),
        asr_compute_type=os.getenv(
            "ASR_COMPUTE_TYPE",
            os.getenv("WHISPER_COMPUTE_TYPE", Settings.asr_compute_type),
        ),
        asr_task=os.getenv("ASR_TASK", Settings.asr_task),
        asr_beam_size=_env_int("ASR_BEAM_SIZE", Settings.asr_beam_size),
        asr_batch_size=_env_optional_int("ASR_BATCH_SIZE"),
        asr_vad_filter=_env_optional_bool("ASR_VAD_FILTER"),
        asr_vad_min_silence_ms=_env_optional_int("ASR_VAD_MIN_SILENCE_MS"),
        asr_condition_on_previous_text=_env_optional_bool("ASR_CONDITION_ON_PREVIOUS_TEXT"),
        asr_cpu_threads=_env_int("ASR_CPU_THREADS", Settings.asr_cpu_threads),
        asr_num_workers=_env_int("ASR_NUM_WORKERS", Settings.asr_num_workers),
        asr_download_root=os.getenv("ASR_DOWNLOAD_ROOT", Settings.asr_download_root),
        asr_local_files_only=_env_bool("ASR_LOCAL_FILES_ONLY", Settings.asr_local_files_only),
        asr_hotwords=os.getenv("ASR_HOTWORDS", Settings.asr_hotwords),
        asr_initial_prompt=os.getenv("ASR_INITIAL_PROMPT", Settings.asr_initial_prompt),
        asr_gguf_model_dir=os.getenv("ASR_GGUF_MODEL_DIR", Settings.asr_gguf_model_dir),
        asr_gguf_llama_bin=os.getenv("ASR_GGUF_LLAMA_BIN", Settings.asr_gguf_llama_bin),
        asr_gguf_cli=os.getenv("ASR_GGUF_CLI", Settings.asr_gguf_cli),
        subtitle_postprocess=_env_bool("SUBTITLE_POSTPROCESS", Settings.subtitle_postprocess),
        subtitle_postprocess_base_url=os.getenv(
            "SUBTITLE_POSTPROCESS_BASE_URL",
            Settings.subtitle_postprocess_base_url,
        ),
        subtitle_postprocess_api_key=os.getenv(
            "SUBTITLE_POSTPROCESS_API_KEY",
            Settings.subtitle_postprocess_api_key,
        ),
        subtitle_postprocess_model=os.getenv(
            "SUBTITLE_POSTPROCESS_MODEL",
            Settings.subtitle_postprocess_model,
        ),
        subtitle_postprocess_temperature=_env_optional_float("SUBTITLE_POSTPROCESS_TEMPERATURE"),
        subtitle_postprocess_chunk_chars=_env_int(
            "SUBTITLE_POSTPROCESS_CHUNK_CHARS",
            Settings.subtitle_postprocess_chunk_chars,
        ),
        subtitle_postprocess_style=os.getenv(
            "SUBTITLE_POSTPROCESS_STYLE",
            Settings.subtitle_postprocess_style,
        ),
        subtitle_postprocess_disable_thinking=_env_bool(
            "SUBTITLE_POSTPROCESS_DISABLE_THINKING",
            Settings.subtitle_postprocess_disable_thinking,
        ),
        download_timeout=_env_float("DOWNLOAD_TIMEOUT", Settings.download_timeout),
        download_chunk_size=_env_int("DOWNLOAD_CHUNK_SIZE", Settings.download_chunk_size),
        yt_dlp_format=os.getenv("YT_DLP_FORMAT", Settings.yt_dlp_format),
        yt_dlp_audio_format=os.getenv("YT_DLP_AUDIO_FORMAT", Settings.yt_dlp_audio_format),
        yt_dlp_audio_quality=os.getenv("YT_DLP_AUDIO_QUALITY", Settings.yt_dlp_audio_quality),
        bili_sessdata=os.getenv("BILI_SESSDATA", ""),
        bili_jct=os.getenv("BILI_JCT", ""),
        bili_buvid3=os.getenv("BILI_BUVID3", ""),
        cookie_file=os.getenv("BILI_COOKIE_FILE", ""),
    )

    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    if "output_dir" in clean_overrides:
        clean_overrides["output_dir"] = Path(clean_overrides["output_dir"])
    if "llm_fallback_models" in clean_overrides:
        clean_overrides["llm_fallback_models"] = _parse_list(clean_overrides["llm_fallback_models"])
    if "llm_model_providers" in clean_overrides:
        clean_overrides["llm_model_providers"] = _normalize_model_provider_map(
            _parse_mapping(clean_overrides["llm_model_providers"])
        )
    if "llm_provider_base_urls" in clean_overrides:
        clean_overrides["llm_provider_base_urls"] = _normalize_provider_value_map(
            _parse_mapping(clean_overrides["llm_provider_base_urls"])
        )
    if "llm_provider_api_keys" in clean_overrides:
        clean_overrides["llm_provider_api_keys"] = _normalize_provider_value_map(
            _parse_mapping(clean_overrides["llm_provider_api_keys"])
        )
    if "llm_provider_wire_apis" in clean_overrides:
        clean_overrides["llm_provider_wire_apis"] = _normalize_provider_value_map(
            _parse_mapping(clean_overrides["llm_provider_wire_apis"])
        )
    return replace(settings, **clean_overrides)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return float(raw)


def _env_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    return float(raw)


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    return int(raw)


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw == "":
        return ()
    return _parse_list(raw)


def _parse_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    else:
        parts = [str(item) for item in value]
    return tuple(part.strip() for part in parts if part.strip())


def _env_model_provider_map() -> dict[str, str]:
    return _normalize_model_provider_map(
        _env_mapping("LLM_MODEL_PROVIDERS", DEFAULT_LLM_MODEL_PROVIDERS)
    )


def _env_provider_values(kind: str, default: dict[str, str] | None = None) -> dict[str, str]:
    values = _normalize_provider_value_map(_env_mapping(f"LLM_PROVIDER_{kind}S", default or {}))
    prefix = "LLM_PROVIDER_"
    suffix = f"_{kind}"
    for name, raw in os.environ.items():
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        provider = name[len(prefix) : -len(suffix)].lower()
        value = raw.strip()
        if provider and value:
            values[provider] = value
    return values


def _env_mapping(name: str, default: dict[str, str]) -> dict[str, str]:
    raw = os.getenv(name)
    if raw in (None, ""):
        return dict(default)
    return {**default, **_parse_mapping(raw)}


def _parse_mapping(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        items = value.items()
    else:
        tokens = str(value).replace(",", " ").replace(";", " ").split()
        items = (token.split("=", 1) for token in tokens if "=" in token)
    return {str(key).strip(): str(raw).strip() for key, raw in items if str(key).strip() and str(raw).strip()}


def _normalize_model_provider_map(values: dict[str, str]) -> dict[str, str]:
    return {model.strip(): provider.strip().lower() for model, provider in values.items() if model.strip()}


def _normalize_provider_value_map(values: dict[str, str]) -> dict[str, str]:
    return {provider.strip().lower(): value.strip() for provider, value in values.items() if provider.strip()}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}
