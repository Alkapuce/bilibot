"""Qwen3-ASR backend — state-of-the-art Chinese speech recognition."""

from __future__ import annotations

import logging
import math
import subprocess
import sys
import tempfile
import types
import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Any

from .config import Settings
from .models import Transcript, TranscriptSegment
from .progress import ProgressCallback, emit

logger = logging.getLogger("bilibot.qwen_asr")

QWEN3_MODEL_IDS: dict[str, str] = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
}

QWEN3_DEFAULT_MODEL = "qwen3-asr-1.7b"
QWEN3_DEFAULT_FORCED_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
QWEN3_LOCAL_FORCED_ALIGNER_DIR = "Qwen3-ForcedAligner-0.6B"
QWEN3_CHUNK_SECONDS = 300
QWEN3_LANGUAGE_ALIASES: dict[str, str] = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "chinese": "Chinese",
    "cn": "Chinese",
    "en": "English",
    "english": "English",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
}


def _check_qwen_asr() -> bool:
    _install_optional_runtime_shims()
    try:
        import qwen_asr  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_model(settings: Settings) -> str:
    model = settings.asr_model
    if not model:
        return QWEN3_MODEL_IDS[QWEN3_DEFAULT_MODEL]
    if model in QWEN3_MODEL_IDS:
        return QWEN3_MODEL_IDS[model]
    if Path(model).is_dir():
        return str(Path(model).resolve())
    if "/" in model or "\\" in model:
        return model
    return model


def _resolve_forced_aligner_model(settings: Settings, asr_model: str) -> str:
    """Resolve the aligner, preferring a local checkpoint next to the ASR model."""
    configured = settings.asr_forced_aligner_model
    if configured:
        path = Path(configured).expanduser()
        return str(path.resolve()) if path.is_dir() else configured

    asr_path = Path(asr_model)
    candidates = [Path.cwd() / ".models" / QWEN3_LOCAL_FORCED_ALIGNER_DIR]
    if asr_path.is_dir():
        candidates.append(asr_path.parent / QWEN3_LOCAL_FORCED_ALIGNER_DIR)
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate.resolve())
    return QWEN3_DEFAULT_FORCED_ALIGNER


def _resolve_device(settings: Settings) -> str:
    requested = (settings.asr_device or "").strip().lower()
    try:
        import torch
        if requested in ("", "auto"):
            if torch.cuda.is_available():
                return "cuda:0"
            return "cpu"
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("ASR_DEVICE=cuda 但 torch.cuda.is_available() 为 False")
            return "cuda:0"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"ASR_DEVICE={settings.asr_device} 但 torch.cuda.is_available() 为 False")
        if requested:
            return settings.asr_device
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def _resolve_dtype(settings: Settings, device: str):
    import torch

    compute_type = (settings.asr_compute_type or "").strip().lower()
    if compute_type in ("bf16", "bfloat16"):
        return torch.bfloat16
    if compute_type in ("fp16", "float16", "half"):
        return torch.float16
    if compute_type in ("fp32", "float32", "full"):
        return torch.float32
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def _resolve_language(language: str) -> str | None:
    if not language or language == "auto":
        return None
    return QWEN3_LANGUAGE_ALIASES.get(language.strip().lower(), language)


def transcribe(
    audio_path: str,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
) -> Transcript:
    if not _check_qwen_asr():
        raise RuntimeError(
            "Qwen3-ASR 后端需要安装额外依赖。请运行：\n"
            "  uv sync --extra qwen3\n"
            "或：\n"
            "  uv add qwen-asr torch"
        )

    import torch
    _install_optional_runtime_shims()
    from qwen_asr import Qwen3ASRModel

    model_id = _resolve_model(settings)
    device = _resolve_device(settings)
    dtype = _resolve_dtype(settings, device)
    aligner_requested = settings.asr_return_time_stamps is not False and (
        settings.asr_return_time_stamps is True or bool(settings.asr_forced_aligner_model)
    )
    aligner_id = _resolve_forced_aligner_model(settings, model_id) if aligner_requested else ""

    emit(progress, "task_start", "asr_model_load", f"加载 Qwen3-ASR 模型：{model_id}")
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device,
        "max_inference_batch_size": 1,
        "max_new_tokens": 1024,
    }
    if aligner_id:
        model_kwargs["forced_aligner"] = aligner_id
        model_kwargs["forced_aligner_kwargs"] = {"dtype": dtype, "device_map": device}
        emit(progress, "log", "asr_model_load", f"加载 Qwen3-ForcedAligner：{aligner_id}")
    model = Qwen3ASRModel.from_pretrained(model_id, **model_kwargs)
    emit(progress, "task_done", "asr_model_load", f"Qwen3-ASR 模型已加载：{model_id}")

    language = _resolve_language(settings.language)

    emit(progress, "task_start", "asr_transcribe", "Qwen3-ASR 语音识别中")
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_paths = _split_audio_for_qwen(audio_path, QWEN3_CHUNK_SECONDS, Path(tmpdir))
        if chunk_paths:
            segments: list[TranscriptSegment] = []
            languages: list[str] = []
            time_stamps: list[dict[str, Any]] = []
            previous_text = ""
            for index, (chunk_path, start, end) in enumerate(chunk_paths, start=1):
                emit(
                    progress,
                    "task_update",
                    "asr_transcribe",
                    f"Qwen3-ASR 语音识别中：第 {index}/{len(chunk_paths)} 段",
                    completed=index - 1,
                    total=len(chunk_paths),
                )
                results = model.transcribe(
                    audio=str(chunk_path),
                    context=previous_text[-500:],
                    language=language,
                    return_time_stamps=aligner_requested,
                )
                if not results:
                    continue
                r = results[0]
                text = r.text.strip()
                if text:
                    segments.append(TranscriptSegment(start=start, end=end, text=text))
                    previous_text += text
                time_stamps.extend(_normalize_time_stamps(getattr(r, "time_stamps", None), offset=start))
                if r.language:
                    languages.append(r.language)
            if not segments:
                raise RuntimeError("Qwen3-ASR returned empty result")
            detected_lang = _merge_detected_languages(languages)
            emit(
                progress,
                "task_done",
                "asr_transcribe",
                f"Qwen3-ASR 识别完成：{detected_lang}，{len(segments)} 段",
            )
            return Transcript(
                source=f"qwen3:{model_id}",
                language=detected_lang,
                segments=segments,
                time_stamps=time_stamps,
            )

    results = model.transcribe(
        audio=audio_path,
        language=language,
        return_time_stamps=aligner_requested,
    )
    if not results:
        raise RuntimeError("Qwen3-ASR returned empty result")

    r = results[0]
    text = r.text.strip()
    detected_lang = r.language or "zh"

    time_stamps = _normalize_time_stamps(getattr(r, "time_stamps", None))
    segments: list[TranscriptSegment] = []

    if text:
        segments = [TranscriptSegment(start=0.0, end=0.0, text=text)]

    emit(
        progress,
        "task_done",
        "asr_transcribe",
        f"Qwen3-ASR 识别完成：{detected_lang}，{len(segments)} 段",
    )
    return Transcript(
        source=f"qwen3:{model_id}",
        language=detected_lang,
        segments=segments,
        time_stamps=time_stamps,
    )


def _split_audio_for_qwen(audio_path: str, chunk_seconds: int, output_dir: Path) -> list[tuple[Path, float, float]]:
    duration = _probe_duration(audio_path)
    if duration is None or duration <= chunk_seconds:
        return []

    output_pattern = output_dir / "chunk_%05d.wav"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        str(output_pattern),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("长音频 Qwen3-ASR 切片需要 ffmpeg，请先安装并加入 PATH") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = f"ffmpeg 音频切片失败：{detail}" if detail else "ffmpeg 音频切片失败"
        raise RuntimeError(message) from exc

    chunks = sorted(output_dir.glob("chunk_*.wav"))
    if not chunks:
        return []

    result: list[tuple[Path, float, float]] = []
    for index, chunk in enumerate(chunks):
        start = float(index * chunk_seconds)
        end = min(duration, float((index + 1) * chunk_seconds))
        result.append((chunk, start, end))
    return result


def _probe_duration(audio_path: str) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _merge_detected_languages(languages: list[str]) -> str:
    if not languages:
        return "zh"
    return max(set(languages), key=languages.count)


def _normalize_time_stamps(items: Any, *, offset: float = 0.0) -> list[dict[str, Any]]:
    """Convert qwen-asr aligner objects or dicts into stable JSON records."""
    normalized: list[dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, dict):
            text = item.get("text", "")
            start = item.get("start", item.get("start_time", 0.0))
            end = item.get("end", item.get("end_time", 0.0))
        else:
            text = getattr(item, "text", "")
            start = getattr(item, "start_time", getattr(item, "start", 0.0))
            end = getattr(item, "end_time", getattr(item, "end", 0.0))
        text = str(text).strip()
        if not text:
            continue
        try:
            start_value = float(start) + offset
            end_value = float(end) + offset
        except (TypeError, ValueError):
            continue
        if start_value < 0 or end_value < start_value:
            continue
        normalized.append({"text": text, "start": start_value, "end": end_value})
    return normalized


def _install_optional_runtime_shims() -> None:
    """Provide tiny fallbacks for optional qwen-asr imports we do not use.

    The upstream qwen-asr package imports demo/alignment helpers from its package
    root. Plain ASR inference only needs audio loading/resampling, while forced
    alignment pulls in nagisa/soynlp. Keep the runtime lean by stubbing the
    unused alignment import and replacing librosa with soundfile+scipy when
    librosa is not installed.
    """
    if importlib.util.find_spec("librosa") is None and "librosa" not in sys.modules:
        sys.modules["librosa"] = _build_librosa_shim()
    if importlib.util.find_spec("nagisa") is None and "nagisa" not in sys.modules:
        module = types.ModuleType("nagisa")
        module.__spec__ = importlib.machinery.ModuleSpec("nagisa", loader=None)
        module.tagging = lambda text: types.SimpleNamespace(words=list(str(text)))
        sys.modules["nagisa"] = module


def _build_librosa_shim() -> types.ModuleType:
    module = types.ModuleType("librosa")
    module.__spec__ = importlib.machinery.ModuleSpec("librosa", loader=None)

    def load(path: str, *, sr: int | None = None, mono: bool = True):
        import numpy as np
        import soundfile as sf

        audio, source_sr = sf.read(path, dtype="float32", always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            if mono:
                audio = audio.mean(axis=1)
            else:
                audio = audio.T
        if sr is not None and int(source_sr) != int(sr):
            audio = resample(audio, orig_sr=int(source_sr), target_sr=int(sr))
            source_sr = sr
        return audio.astype(np.float32, copy=False), int(source_sr)

    def resample(y, *, orig_sr: int, target_sr: int):
        import numpy as np
        from scipy.signal import resample_poly

        if int(orig_sr) == int(target_sr):
            return np.asarray(y, dtype=np.float32)
        factor = math.gcd(int(orig_sr), int(target_sr))
        up = int(target_sr) // factor
        down = int(orig_sr) // factor
        return resample_poly(y, up, down, axis=-1).astype(np.float32, copy=False)

    module.load = load
    module.resample = resample
    return module


def transcribe_url(
    url: str,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
) -> Transcript:
    import tempfile

    from .downloader import download_audio

    with tempfile.TemporaryDirectory() as tmpdir:
        emit(progress, "log", "download_audio", "准备下载音频")
        audio_path = download_audio(
            url,
            tmpdir,
            settings.cookie_file,
            sessdata=settings.bili_sessdata,
            bili_jct=settings.bili_jct,
            buvid3=settings.bili_buvid3,
            timeout=settings.download_timeout,
            chunk_size=settings.download_chunk_size,
            yt_dlp_format=settings.yt_dlp_format,
            yt_dlp_audio_format=settings.yt_dlp_audio_format,
            yt_dlp_audio_quality=settings.yt_dlp_audio_quality,
            progress=progress,
        )
        return transcribe(audio_path, settings, progress=progress)
