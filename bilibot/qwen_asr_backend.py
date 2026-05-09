"""Qwen3-ASR backend — state-of-the-art Chinese speech recognition."""

from __future__ import annotations

import logging

from .config import Settings
from .models import Transcript, TranscriptSegment
from .progress import ProgressCallback, emit

logger = logging.getLogger("bilibot.qwen_asr")

QWEN3_MODEL_IDS: dict[str, str] = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
}

QWEN3_DEFAULT_MODEL = "qwen3-asr-1.7b"


def _check_qwen_asr() -> bool:
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
    from pathlib import Path
    if Path(model).is_dir():
        return str(Path(model).resolve())
    if "/" in model or "\\" in model:
        return model
    return model


def _resolve_device(settings: Settings) -> str:
    if settings.asr_device:
        return settings.asr_device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


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
    from qwen_asr import Qwen3ASRModel

    model_id = _resolve_model(settings)
    device = _resolve_device(settings)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    emit(progress, "task_start", "asr_model_load", f"加载 Qwen3-ASR 模型：{model_id}")
    model = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=1024,
    )
    emit(progress, "task_done", "asr_model_load", f"Qwen3-ASR 模型已加载：{model_id}")

    language = settings.language
    if not language or language == "auto":
        language = None

    emit(progress, "task_start", "asr_transcribe", "Qwen3-ASR 语音识别中")
    results = model.transcribe(
        audio=audio_path,
        language=language,
    )
    if not results:
        raise RuntimeError("Qwen3-ASR returned empty result")

    r = results[0]
    text = r.text.strip()
    detected_lang = r.language or "zh"

    segments: list[TranscriptSegment] = []
    if hasattr(r, "time_stamps") and r.time_stamps:
        for ts in r.time_stamps:
            seg_text = str(ts.get("text", "")).strip()
            if not seg_text:
                continue
            start = float(ts.get("start", 0))
            end = float(ts.get("end", 0))
            if end <= 0 and start > 0:
                end = start + 1.0
            if start < 0 and end > 0:
                start = max(0.0, end - 1.0)
            if start >= 0 and end >= 0 and start <= end:
                segments.append(TranscriptSegment(start=start, end=end, text=seg_text))

    if not segments and text:
        segments = [TranscriptSegment(start=0.0, end=0.0, text=text)]

    emit(
        progress,
        "task_done",
        "asr_transcribe",
        f"Qwen3-ASR 识别完成：{detected_lang}，{len(segments)} 段",
    )
    return Transcript(source=f"qwen3/{model_id}", language=detected_lang, segments=segments)


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
