"""Download audio and transcribe with faster-whisper."""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx
from bilibili_api import Credential, video

from .asr import AsrPlan, detect_runtime, resolve_asr_plan, resolve_backend
from .config import Settings
from .extractor import parse_bvid
from .models import Transcript, TranscriptSegment
from .progress import ProgressCallback, emit


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
BILI_COOKIE_NAMES = ("SESSDATA", "bili_jct", "buvid3")


def download_audio(
    url: str,
    output_dir: str,
    cookie_file: str = "",
    sessdata: str = "",
    bili_jct: str = "",
    buvid3: str = "",
    *,
    timeout: float = 60.0,
    chunk_size: int = 1024 * 1024,
    yt_dlp_format: str = "bestaudio",
    yt_dlp_audio_format: str = "mp3",
    yt_dlp_audio_quality: str = "5",
    progress: ProgressCallback | None = None,
) -> str:
    """Download the best available audio and return the local media path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    try:
        bvid = parse_bvid(url)
    except Exception as exc:
        bvid = ""
        errors.append(f"BV 解析失败：{exc}")

    if bvid:
        try:
            return asyncio.run(
                download_bilibili_audio(
                    bvid,
                    output_dir,
                    cookie_file=cookie_file,
                    sessdata=sessdata,
                    bili_jct=bili_jct,
                    buvid3=buvid3,
                    timeout=timeout,
                    chunk_size=chunk_size,
                    progress=progress,
                )
            )
        except Exception as exc:
            errors.append(f"B站直链下载失败：{exc}")

    try:
        return download_audio_with_ytdlp(
            url,
            output_dir,
            cookie_file,
            yt_dlp_format=yt_dlp_format,
            yt_dlp_audio_format=yt_dlp_audio_format,
            yt_dlp_audio_quality=yt_dlp_audio_quality,
            progress=progress,
        )
    except Exception as exc:
        message = str(exc)
        if "HTTP Error 412" in message:
            message += (
                "\nB站返回 HTTP 412，通常是 yt-dlp 的元数据请求被风控。"
                " 已优先尝试 B站 API 直链下载；如果仍失败，请配置 BILI_SESSDATA"
                " 或 --cookie-file 后重试。"
            )
        if errors:
            message += "\n\n前置尝试：\n" + "\n".join(f"- {item}" for item in errors)
        raise RuntimeError(message) from exc


async def download_bilibili_audio(
    bvid: str,
    output_dir: str,
    *,
    cookie_file: str = "",
    sessdata: str = "",
    bili_jct: str = "",
    buvid3: str = "",
    timeout: float = 60.0,
    chunk_size: int = 1024 * 1024,
    progress: ProgressCallback | None = None,
) -> str:
    """Download Bilibili audio through bilibili-api-python play URLs."""
    cookie_values = _load_bili_cookie_values(cookie_file)
    sessdata = sessdata or cookie_values.get("SESSDATA", "")
    bili_jct = bili_jct or cookie_values.get("bili_jct", "")
    buvid3 = buvid3 or cookie_values.get("buvid3", "")
    credential = (
        Credential(sessdata=sessdata, bili_jct=bili_jct, buvid3=buvid3)
        if sessdata
        else None
    )

    v = video.Video(bvid=bvid, credential=credential)
    info = await v.get_info()
    cid = _first_cid(info)
    if not cid:
        raise RuntimeError("未能获取视频 cid")

    play_url = await v.get_download_url(cid=cid, html5=False)
    audio = _select_dash_audio(play_url)
    referer = f"https://www.bilibili.com/video/{bvid}"
    headers = _bili_download_headers(
        referer,
        sessdata=sessdata,
        bili_jct=bili_jct,
        buvid3=buvid3,
    )

    if audio:
        path = Path(output_dir) / "audio.m4s"
        await _download_first_available(
            _media_urls(audio),
            path,
            headers,
            timeout,
            chunk_size,
            progress,
        )
        return str(path)

    # Some videos expose only a single progressive stream. It is larger than
    # DASH audio, but ffmpeg/faster-whisper can still read the audio track.
    html5_play_url = await v.get_download_url(cid=cid, html5=True)
    durl = (html5_play_url.get("durl") or play_url.get("durl") or [None])[0]
    if not durl:
        raise RuntimeError("播放地址中没有 DASH 音频或可用 durl")

    path = Path(output_dir) / "audio.mp4"
    await _download_first_available(
        _media_urls(durl),
        path,
        headers,
        timeout,
        chunk_size,
        progress,
    )
    return str(path)


def download_audio_with_ytdlp(
    url: str,
    output_dir: str,
    cookie_file: str = "",
    *,
    yt_dlp_format: str = "bestaudio",
    yt_dlp_audio_format: str = "mp3",
    yt_dlp_audio_quality: str = "5",
    progress: ProgressCallback | None = None,
) -> str:
    """Download best audio from URL using yt-dlp, return path to audio file."""
    out_template = os.path.join(output_dir, "audio.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--user-agent",
        DEFAULT_USER_AGENT,
        "--referer",
        "https://www.bilibili.com/",
        "-f",
        yt_dlp_format,
        "--extract-audio",
        "--audio-format",
        yt_dlp_audio_format,
        "--audio-quality",
        yt_dlp_audio_quality,
        "-o",
        out_template,
        url,
    ]
    if cookie_file and os.path.exists(cookie_file):
        cmd += ["--cookies", cookie_file]

    emit(progress, "task_start", "download_audio", "使用 yt-dlp 下载音频")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        emit(progress, "task_done", "download_audio", "yt-dlp 下载失败")
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    for f in Path(output_dir).glob("audio.*"):
        emit(progress, "task_done", "download_audio", f"yt-dlp 下载完成：{f.name}")
        return str(f)
    emit(progress, "task_done", "download_audio", "yt-dlp 未生成音频文件")
    raise FileNotFoundError("yt-dlp did not produce an audio file")


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
        progress,
        "task_start",
        "asr_model_load",
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
        transcribe_kwargs["vad_parameters"] = {
            "min_silence_duration_ms": settings.asr_vad_min_silence_ms
        }
    if settings.asr_hotwords:
        transcribe_kwargs["hotwords"] = settings.asr_hotwords
    initial_prompt = settings.asr_initial_prompt
    if not initial_prompt and settings.language == "zh":
        initial_prompt = (
            "以下是中文普通话的句子，使用简体中文。"
        )
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
        emit(
            progress,
            "task_update",
            "asr_transcribe",
            "语音识别中",
            total=duration,
            completed=0,
        )

    transcript_segments = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        transcript_segments.append(
            TranscriptSegment(start=float(seg.start), end=float(seg.end), text=text)
        )
        if duration > 0:
            emit(
                progress,
                "task_update",
                "asr_transcribe",
                "语音识别中",
                completed=min(float(seg.end), duration),
            )
        else:
            emit(progress, "task_update", "asr_transcribe", "语音识别中", advance=1)

    detected = info.language
    emit(
        progress,
        "task_done",
        "asr_transcribe",
        f"语音识别完成：{detected}，{len(transcript_segments)} 段",
    )
    return Transcript(source="whisper", language=detected, segments=transcript_segments)


def transcribe_url(
    url: str,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
) -> Transcript:
    """Download audio and return structured transcript."""
    runtime = detect_runtime()
    best_gpu = max(runtime.gpus, key=lambda g: g.memory_mb, default=None)
    gpu_mb = best_gpu.memory_mb if best_gpu else 0
    backend = resolve_backend(settings, gpu_mb)

    if backend == "gguf":
        from .gguf_asr_backend import transcribe_url as gguf_transcribe_url

        return gguf_transcribe_url(url, settings, progress=progress)

    if backend == "qwen3":
        from .qwen_asr_backend import transcribe_url as qwen_transcribe_url

        return qwen_transcribe_url(url, settings, progress=progress)

    plan = resolve_asr_plan(settings, runtime)
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
        return transcribe(
            audio_path,
            settings,
            plan=plan,
            progress=progress,
        )


def get_transcript(url: str, cookie_file: str = "") -> str:
    """Download audio and return transcript text."""
    settings = Settings(cookie_file=cookie_file)
    return transcribe_url(url, settings).text


def _first_cid(info: dict[str, Any]) -> int | None:
    pages = info.get("pages", [])
    if pages:
        return pages[0].get("cid")
    return info.get("cid")


def _select_dash_audio(play_url: dict[str, Any]) -> dict[str, Any] | None:
    audios = play_url.get("dash", {}).get("audio") or []
    if not audios:
        return None
    return max(audios, key=lambda item: int(item.get("bandwidth") or 0))


def _media_urls(media: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ("base_url", "baseUrl", "url"):
        value = media.get(key)
        if isinstance(value, str) and value:
            urls.append(value)

    for key in ("backup_url", "backupUrl", "backup_urls", "backupUrl"):
        value = media.get(key)
        if isinstance(value, str) and value:
            urls.append(value)
        elif isinstance(value, list):
            urls.extend(item for item in value if isinstance(item, str) and item)

    return list(dict.fromkeys(urls))


async def _download_first_available(
    urls: list[str],
    path: Path,
    headers: dict[str, str],
    timeout: float,
    chunk_size: int,
    progress: ProgressCallback | None,
) -> None:
    if not urls:
        raise RuntimeError("没有可下载的媒体 URL")

    last_error: Exception | None = None
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, headers=headers) as client:
        for media_url in urls:
            try:
                async with client.stream("GET", media_url) as response:
                    response.raise_for_status()
                    total = _content_length(response)
                    emit(
                        progress,
                        "task_start",
                        "download_audio",
                        f"下载音频：{path.name}",
                        total=total,
                        unit="bytes",
                    )
                    with path.open("wb") as output:
                        async for chunk in response.aiter_bytes(chunk_size):
                            if chunk:
                                output.write(chunk)
                                emit(
                                    progress,
                                    "task_update",
                                    "download_audio",
                                    f"下载音频：{path.name}",
                                    advance=len(chunk),
                                )

                if path.stat().st_size <= 0:
                    raise RuntimeError("下载文件为空")
                emit(progress, "task_done", "download_audio", f"音频下载完成：{path.name}")
                return
            except Exception as exc:
                last_error = exc
                emit(progress, "log", "download_audio", f"音频 URL 下载失败：{exc}")
                path.unlink(missing_ok=True)

    emit(progress, "task_done", "download_audio", "音频下载失败")
    raise RuntimeError(f"所有媒体 URL 下载失败，最后错误：{last_error}")


def _content_length(response: httpx.Response) -> float | None:
    raw = response.headers.get("content-length", "")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _bili_download_headers(
    referer: str,
    *,
    sessdata: str = "",
    bili_jct: str = "",
    buvid3: str = "",
) -> dict[str, str]:
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Referer": referer,
        "Origin": "https://www.bilibili.com",
        "Accept": "*/*",
    }
    cookie = _cookie_header({"SESSDATA": sessdata, "bili_jct": bili_jct, "buvid3": buvid3})
    if cookie:
        headers["Cookie"] = cookie
    return headers


def _load_bili_cookie_values(cookie_file: str) -> dict[str, str]:
    if not cookie_file or not os.path.exists(cookie_file):
        return {}

    values: dict[str, str] = {}
    for raw_line in Path(cookie_file).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#") and not line.startswith("#HttpOnly_"):
            continue

        if "\t" in line or len(line.split()) >= 7:
            fields = line.split("\t") if "\t" in line else line.split()
            if len(fields) >= 7:
                name, value = fields[-2], fields[-1]
                if name in BILI_COOKIE_NAMES and value:
                    values[name] = value
            continue

        for part in line.split(";"):
            if "=" not in part:
                continue
            name, value = (item.strip() for item in part.split("=", 1))
            if name in BILI_COOKIE_NAMES and value:
                values[name] = value

    return values


def _cookie_header(values: dict[str, str]) -> str:
    return "; ".join(f"{name}={value}" for name, value in values.items() if value)
