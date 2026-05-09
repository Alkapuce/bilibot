"""Download Bilibili media: video (yt-dlp) and audio (API + yt-dlp fallback)."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
from bilibili_api import Credential, video

from .extractor import parse_bvid
from .progress import ProgressCallback, emit


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
BILI_COOKIE_NAMES = ("SESSDATA", "bili_jct", "buvid3")

# Best available format without requiring login (non-premium max is 720P).
DEFAULT_FORMAT = "bestvideo[height<=720]+bestaudio/best[height<=720]/best/worst"


# ── video download ──────────────────────────────────────────────────────────

def download_video(
    url: str,
    output_dir: str = ".",
    *,
    fmt: str = DEFAULT_FORMAT,
    cookie_file: str = "",
    output_template: str = "%(title)s [%(id)s].%(ext)s",
    merge_output_format: str = "mp4",
) -> str:
    """Download a Bilibili video and return the output file path."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(str(out_dir), output_template)

    bvid = ""
    try:
        bvid = parse_bvid(url)
        canonical_url = f"https://www.bilibili.com/video/{bvid}"
    except Exception:
        canonical_url = url

    cmd = [
        "yt-dlp", "--no-playlist",
        "--user-agent", DEFAULT_USER_AGENT,
        "--referer", "https://www.bilibili.com/",
        "-f", fmt,
        "--merge-output-format", merge_output_format,
        "-o", out_path,
        "--no-simulate", "--print", "filename",
        canonical_url,
    ]
    if cookie_file and os.path.exists(cookie_file):
        cmd += ["--cookies", cookie_file]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp 下载失败：\n{result.stderr}")

    downloaded = result.stdout.strip()
    if downloaded and os.path.exists(downloaded):
        return downloaded

    for f in out_dir.glob(f"*{bvid}*"):
        return str(f)
    for f in sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        return str(f)

    raise FileNotFoundError(f"yt-dlp 完成但未找到输出文件。输出目录: {out_dir}\nstdout: {result.stdout}")


# ── audio download (orchestrator) ───────────────────────────────────────────

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
    """Download the best available audio (B站 API → yt-dlp fallback)."""
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
                _download_bilibili_audio(
                    bvid, output_dir,
                    cookie_file=cookie_file, sessdata=sessdata,
                    bili_jct=bili_jct, buvid3=buvid3,
                    timeout=timeout, chunk_size=chunk_size, progress=progress,
                )
            )
        except Exception as exc:
            errors.append(f"B站直链下载失败：{exc}")

    try:
        return _download_audio_with_ytdlp(
            url, output_dir, cookie_file,
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


# ── Bilibili API 直链音频下载 ──────────────────────────────────────────────

async def _download_bilibili_audio(
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
        await _download_first_available(_media_urls(audio), path, headers, timeout, chunk_size, progress)
        return str(path)

    html5_play_url = await v.get_download_url(cid=cid, html5=True)
    durl = (html5_play_url.get("durl") or play_url.get("durl") or [None])[0]
    if not durl:
        raise RuntimeError("播放地址中没有 DASH 音频或可用 durl")

    path = Path(output_dir) / "audio.mp4"
    await _download_first_available(_media_urls(durl), path, headers, timeout, chunk_size, progress)
    return str(path)


# ── yt-dlp 音频下载 ────────────────────────────────────────────────────────

def _download_audio_with_ytdlp(
    url: str,
    output_dir: str,
    cookie_file: str = "",
    *,
    yt_dlp_format: str = "bestaudio",
    yt_dlp_audio_format: str = "mp3",
    yt_dlp_audio_quality: str = "5",
    progress: ProgressCallback | None = None,
) -> str:
    out_template = os.path.join(output_dir, "audio.%(ext)s")
    cmd = [
        "yt-dlp", "--no-playlist",
        "--user-agent", DEFAULT_USER_AGENT,
        "--referer", "https://www.bilibili.com/",
        "-f", yt_dlp_format,
        "--extract-audio",
        "--audio-format", yt_dlp_audio_format,
        "--audio-quality", yt_dlp_audio_quality,
        "-o", out_template,
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


# ── internal helpers ────────────────────────────────────────────────────────

def _first_cid(info: dict[str, Any]) -> int | None:
    pages = info.get("pages", [])
    return pages[0].get("cid") if pages else info.get("cid")


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
                    emit(progress, "task_start", "download_audio", f"下载音频：{path.name}", total=total, unit="bytes")
                    with path.open("wb") as output:
                        async for chunk in response.aiter_bytes(chunk_size):
                            if chunk:
                                output.write(chunk)
                                emit(progress, "task_update", "download_audio", f"下载音频：{path.name}", advance=len(chunk))
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
