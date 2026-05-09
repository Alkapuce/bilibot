"""Download Bilibili videos via yt-dlp."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .extractor import parse_bvid

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Best available format without requiring login (non-premium max is 720P).
# Falls back gracefully: try 720P video+audio, then best, then worst.
DEFAULT_FORMAT = (
    "bestvideo[height<=720]+bestaudio/best[height<=720]/best/worst"
)


def download_video(
    url: str,
    output_dir: str = ".",
    *,
    fmt: str = DEFAULT_FORMAT,
    cookie_file: str = "",
    output_template: str = "%(title)s [%(id)s].%(ext)s",
    merge_output_format: str = "mp4",
) -> str:
    """Download a Bilibili video and return the output file path.

    Uses yt-dlp under the hood.  Handles both direct URLs and BV ids.

    Args:
        url: Bilibili URL, BV id, or short link (b23.tv).
        output_dir: Directory to save the downloaded file.
        fmt: yt-dlp format selector.
        cookie_file: Netscape-format cookie file for premium access.
        output_template: yt-dlp output filename template.
        merge_output_format: Container format for merged streams.

    Returns:
        Absolute path to the downloaded file.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = os.path.join(str(out_dir), output_template)

    # Resolve b23.tv / BV suffix to canonical URL
    bvid = ""
    try:
        bvid = parse_bvid(url)
        canonical_url = f"https://www.bilibili.com/video/{bvid}"
    except Exception:
        canonical_url = url

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--user-agent", DEFAULT_USER_AGENT,
        "--referer", "https://www.bilibili.com/",
        "-f", fmt,
        "--merge-output-format", merge_output_format,
        "-o", out_path,
        "--print", "filename",
        canonical_url,
    ]

    if cookie_file and os.path.exists(cookie_file):
        cmd += ["--cookies", cookie_file]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp 下载失败：\n{result.stderr}")

    # yt-dlp --print filename outputs the actual file path
    downloaded = result.stdout.strip()
    if downloaded and os.path.exists(downloaded):
        return downloaded

    # Fallback: glob for the file
    for f in out_dir.glob(f"*{bvid}*"):
        return str(f)
    for f in sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        return str(f)

    raise FileNotFoundError(
        f"yt-dlp 完成但未找到输出文件。输出目录: {out_dir}\n"
        f"stdout: {result.stdout}"
    )
