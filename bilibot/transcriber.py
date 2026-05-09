"""Transcribe audio with faster-whisper and route to other ASR backends."""

from __future__ import annotations

import tempfile
from typing import Any

from .asr import AsrPlan, detect_runtime, resolve_asr_plan, resolve_backend
from .config import Settings
from .downloader import download_audio
from .models import Transcript, TranscriptSegment
from .progress import ProgressCallback, emit


def transcribe(
    audio_path: str,
    settings: Settings,
    *,
    plan: AsrPlan | None = None,
    progress: ProgressCallback | None = None,
) -> Transcript:
    """Transcribe an audio file using faster-whisper."""
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    plan = plan or resolve_asr_plan(settings)
    emit(progress, "log", "asr_plan", plan.reason)
    emit(
        progress, "task_start", "asr_model_load",
        f"加载 ASR 模型：{plan.model} ({plan.device}, {plan.compute_type})",
    )
    model = WhisperModel(
        plan.model,
        device=plan.device,
        compute_type=plan.compute_type,
        cpu_threads=settings.asr_cpu_threads,
        num_workers=settings.asr_num_workers,
        download_root=settings.asr_download_root or None,
        local_files_only=settings.asr_local_files_only,
    )
    emit(progress, "task_done", "asr_model_load", f"ASR 模型已加载：{plan.model}")

    transcribe_kwargs: dict[str, Any] = {
        "language": settings.language or None,
        "task": settings.asr_task,
        "beam_size": settings.asr_beam_size,
        "condition_on_previous_text": plan.condition_on_previous_text,
        "vad_filter": plan.vad_filter,
        "without_timestamps": False,
    }
    if settings.asr_vad_min_silence_ms is not None:
        transcribe_kwargs["vad_parameters"] = {"min_silence_duration_ms": settings.asr_vad_min_silence_ms}
    if settings.asr_hotwords:
        transcribe_kwargs["hotwords"] = settings.asr_hotwords

    initial_prompt = settings.asr_initial_prompt
    if not initial_prompt and settings.language == "zh":
        initial_prompt = "以下是中文普通话的句子，使用简体中文。"
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt

    runner: Any = model
    if plan.batch_size > 1:
        runner = BatchedInferencePipeline(model=model)
        transcribe_kwargs["batch_size"] = plan.batch_size

    emit(progress, "task_start", "asr_transcribe", "语音识别中")
    segments, info = runner.transcribe(audio_path, **transcribe_kwargs)
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    if duration > 0:
        emit(progress, "task_update", "asr_transcribe", "语音识别中", total=duration, completed=0)

    transcript_segments = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        transcript_segments.append(TranscriptSegment(start=float(seg.start), end=float(seg.end), text=text))
        if duration > 0:
            emit(progress, "task_update", "asr_transcribe", "语音识别中", completed=min(float(seg.end), duration))
        else:
            emit(progress, "task_update", "asr_transcribe", "语音识别中", advance=1)

    detected = info.language
    emit(progress, "task_done", "asr_transcribe", f"语音识别完成：{detected}，{len(transcript_segments)} 段")
    return Transcript(source="whisper", language=detected, segments=transcript_segments)


def transcribe_url(
    url: str,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
) -> Transcript:
    """Select ASR backend, download audio, and return structured transcript."""
    runtime = detect_runtime()
    best_gpu = max(runtime.gpus, key=lambda g: g.memory_mb, default=None)
    gpu_mb = best_gpu.memory_mb if best_gpu else 0
    backend = resolve_backend(settings, gpu_mb)

    if backend == "gguf":
        from .gguf_asr_backend import transcribe_url as _gguf
        return _gguf(url, settings, progress=progress)

    if backend == "qwen3":
        from .qwen_asr_backend import transcribe_url as _qwen
        return _qwen(url, settings, progress=progress)

    # whisper path
    plan = resolve_asr_plan(settings, runtime)
    with tempfile.TemporaryDirectory() as tmpdir:
        emit(progress, "log", "download_audio", "准备下载音频")
        audio_path = download_audio(
            url, tmpdir, settings.cookie_file,
            sessdata=settings.bili_sessdata, bili_jct=settings.bili_jct,
            buvid3=settings.bili_buvid3,
            timeout=settings.download_timeout,
            chunk_size=settings.download_chunk_size,
            yt_dlp_format=settings.yt_dlp_format,
            yt_dlp_audio_format=settings.yt_dlp_audio_format,
            yt_dlp_audio_quality=settings.yt_dlp_audio_quality,
            progress=progress,
        )
        return transcribe(audio_path, settings, plan=plan, progress=progress)


def get_transcript(url: str, cookie_file: str = "") -> str:
    """Quick helper: download audio and return transcript text."""
    settings = Settings(cookie_file=cookie_file)
    return transcribe_url(url, settings).text
