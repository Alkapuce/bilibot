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
    chunk_chars: int = 8000

    output_dir: Path = Path("data")
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    language: str = "zh"

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
        chunk_chars=_env_int("CHUNK_CHARS", Settings.chunk_chars),
        output_dir=Path(os.getenv("BILIBOT_OUTPUT_DIR", str(Settings.output_dir))),
        whisper_model=os.getenv("WHISPER_MODEL", Settings.whisper_model),
        whisper_device=os.getenv("WHISPER_DEVICE", Settings.whisper_device),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", Settings.whisper_compute_type),
        language=os.getenv("TRANSCRIPT_LANGUAGE", Settings.language),
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
