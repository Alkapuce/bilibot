"""Download audio and transcribe with faster-whisper."""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx
from bilibili_api import Credential, video

from .extractor import parse_bvid
from .models import Transcript, TranscriptSegment


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
                )
            )
        except Exception as exc:
            errors.append(f"B站直链下载失败：{exc}")

    try:
        return download_audio_with_ytdlp(url, output_dir, cookie_file)
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
) -> str:
    """Download Bilibili audio through bilibili-api-python play URLs."""
    cookie_values = _load_bili_cookie_values(cookie_file)
    sessdata = sessdata or cookie_values.get("SESSDATA", "")
    bili_jct = bili_jct or cookie_values.get("bili_jct", "")
    buvid3 = buvid3 or cookie_values.get("buvid3", "")
    credential = Credential(sessdata=sessdata, bili_jct=bili_jct, buvid3=buvid3) if sessdata else None

    v = video.Video(bvid=bvid, credential=credential)
    info = await v.get_info()
    cid = _first_cid(info)
    if not cid:
        raise RuntimeError("未能获取视频 cid")

    play_url = await v.get_download_url(cid=cid, html5=False)
    audio = _select_dash_audio(play_url)
    referer = f"https://www.bilibili.com/video/{bvid}"
    headers = _bili_download_headers(referer, sessdata=sessdata, bili_jct=bili_jct, buvid3=buvid3)

    if audio:
        path = Path(output_dir) / "audio.m4s"
        await _download_first_available(_media_urls(audio), path, headers)
        return str(path)

    # Some videos expose only a single progressive stream. It is larger than
    # DASH audio, but ffmpeg/faster-whisper can still read the audio track.
    html5_play_url = await v.get_download_url(cid=cid, html5=True)
    durl = (html5_play_url.get("durl") or play_url.get("durl") or [None])[0]
    if not durl:
        raise RuntimeError("播放地址中没有 DASH 音频或可用 durl")

    path = Path(output_dir) / "audio.mp4"
    await _download_first_available(_media_urls(durl), path, headers)
    return str(path)


def download_audio_with_ytdlp(url: str, output_dir: str, cookie_file: str = "") -> str:
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
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "5",
        "-o",
        out_template,
        url,
    ]
    if cookie_file and os.path.exists(cookie_file):
        cmd += ["--cookies", cookie_file]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    for f in Path(output_dir).glob("audio.*"):
        return str(f)
    raise FileNotFoundError("yt-dlp did not produce an audio file")


def transcribe(
    audio_path: str,
    language: str = "zh",
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> Transcript:
    """Transcribe an audio file using faster-whisper."""
    from faster_whisper import WhisperModel

    print(f"[transcriber] loading Whisper {model_name} model ({device}, {compute_type})...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    print(f"[transcriber] transcribing {audio_path} ...")
    segments, info = model.transcribe(audio_path, language=language, beam_size=5)

    transcript_segments = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        transcript_segments.append(
            TranscriptSegment(start=float(seg.start), end=float(seg.end), text=text)
        )

    detected = info.language
    print(f"[transcriber] detected language: {detected}, segments: {len(transcript_segments)}")
    return Transcript(source="whisper", language=detected, segments=transcript_segments)


def transcribe_url(
    url: str,
    cookie_file: str = "",
    language: str = "zh",
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    sessdata: str = "",
    bili_jct: str = "",
    buvid3: str = "",
) -> Transcript:
    """Download audio and return structured transcript."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("[transcriber] downloading audio...")
        audio_path = download_audio(
            url,
            tmpdir,
            cookie_file,
            sessdata=sessdata,
            bili_jct=bili_jct,
            buvid3=buvid3,
        )
        return transcribe(
            audio_path,
            language=language,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
        )


def get_transcript(url: str, cookie_file: str = "") -> str:
    """Download audio and return transcript text."""
    return transcribe_url(url, cookie_file=cookie_file).text


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


async def _download_first_available(urls: list[str], path: Path, headers: dict[str, str]) -> None:
    if not urls:
        raise RuntimeError("没有可下载的媒体 URL")

    last_error: Exception | None = None
    async with httpx.AsyncClient(follow_redirects=True, timeout=60, headers=headers) as client:
        for media_url in urls:
            try:
                async with client.stream("GET", media_url) as response:
                    response.raise_for_status()
                    with path.open("wb") as output:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            if chunk:
                                output.write(chunk)

                if path.stat().st_size <= 0:
                    raise RuntimeError("下载文件为空")
                return
            except Exception as exc:
                last_error = exc
                path.unlink(missing_ok=True)

    raise RuntimeError(f"所有媒体 URL 下载失败，最后错误：{last_error}")


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
