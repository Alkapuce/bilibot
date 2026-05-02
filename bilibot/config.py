"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = "http://localhost:5001/v1"
    llm_api_key: str = "replace-with-your-key"
    llm_model: str = "deepseek-v4-pro"
    llm_timeout: float = 180.0
    llm_temperature: float | None = None
    llm_max_tokens: int | None = None
    chunk_chars: int = 8000

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

    subtitle_postprocess: bool = False
    subtitle_postprocess_base_url: str = ""
    subtitle_postprocess_api_key: str = ""
    subtitle_postprocess_model: str = ""
    subtitle_postprocess_temperature: float | None = None
    subtitle_postprocess_chunk_chars: int = 6000
    subtitle_postprocess_style: str = "clean"

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
        llm_timeout=_env_float("LLM_TIMEOUT", Settings.llm_timeout),
        llm_temperature=_env_optional_float("LLM_TEMPERATURE"),
        llm_max_tokens=_env_optional_int("LLM_MAX_TOKENS"),
        chunk_chars=_env_int("CHUNK_CHARS", Settings.chunk_chars),
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
