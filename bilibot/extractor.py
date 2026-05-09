"""Extract video metadata and subtitles from Bilibili."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from bilibili_api import Credential, video

from .models import optional_float
from .progress import ProgressCallback, emit


@dataclass
class VideoInfo:
    bvid: str
    title: str
    author: str
    desc: str
    duration: int  # seconds
    url: str
    subtitles: list[dict[str, Any]] = field(default_factory=list)
    cover: str = ""
    tags: list[str] = field(default_factory=list)


def parse_bvid(url: str) -> str:
    """Extract BV id from URL or return as-is if already a BV id."""
    url = _clean_video_input(url)
    bvid = _find_bvid(url)
    if bvid:
        return bvid

    if re.fullmatch(r"1[a-zA-Z0-9]{9}", url):
        return f"BV{url}"

    if re.search(r"https?://(?:b23\.tv|bili2233\.cn)/", url):
        import httpx

        resp = httpx.get(url, follow_redirects=True, timeout=10)
        # Extract BV号 from redirect URL first — the final page may return
        # 412 (tracking params expired) but the redirect URL already contains the BV号
        bvid = _find_bvid(str(resp.url))
        if bvid:
            return bvid
        resp.raise_for_status()
        bvid = _find_bvid(resp.text)
        if bvid:
            return bvid

    raise ValueError(f"Cannot parse BV id from: {url}")


def _find_bvid(text: str) -> str | None:
    m = re.search(r"\b[bB][vV]([a-zA-Z0-9]{10,})\b", text)
    if m:
        return f"BV{m.group(1)}"
    return None


def _clean_video_input(value: str) -> str:
    return str(value).strip().strip("'\"“”‘’`<>")


async def _fetch(
    bvid: str,
    credential: Credential | None = None,
    progress: ProgressCallback | None = None,
) -> VideoInfo:
    v = video.Video(bvid=bvid, credential=credential)
    info = await v.get_info()

    title = info.get("title", "")
    author = info.get("owner", {}).get("name", "")
    desc = info.get("desc", "")
    duration = info.get("duration", 0)
    cover = info.get("pic", "")
    url = f"https://www.bilibili.com/video/{bvid}"

    vi = VideoInfo(
        bvid=bvid,
        title=title,
        author=author,
        desc=desc,
        duration=duration,
        url=url,
        cover=cover,
    )

    # Try to get tags (video classification labels, helpful for domain context)
    try:
        tag_list = await v.get_tags()
        vi.tags = [item.get("tag_name", "") for item in (tag_list or []) if item.get("tag_name")]
    except Exception:
        pass

    # Try to get subtitles
    try:
        subtitles = await _fetch_subtitles(v, info, credential, progress)
        vi.subtitles = subtitles
    except Exception as e:
        emit(progress, "log", "metadata", f"字幕列表获取失败：{e}")

    return vi


async def _fetch_subtitles(
    v: video.Video,
    info: dict[str, Any],
    credential: Credential | None = None,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Fetch CC subtitles for all available languages."""
    results = []
    pages = info.get("pages", [])
    cid = pages[0]["cid"] if pages else info.get("cid")
    if not cid:
        return results

    subtitles_list = info.get("subtitle", {}).get("list", [])
    if not subtitles_list:
        if credential is None:
            return results
        sub_info = await v.get_subtitle(cid=cid)
        subtitles_list = sub_info.get("subtitles", [])

    import httpx

    async with httpx.AsyncClient() as client:
        for sub in subtitles_list:
            lan = sub.get("lan_doc", sub.get("lan", "unknown"))
            lan_code = sub.get("lan", "")
            sub_url = sub.get("subtitle_url", "")
            if not sub_url:
                continue
            if sub_url.startswith("//"):
                sub_url = "https:" + sub_url
            try:
                resp = await client.get(sub_url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                emit(progress, "log", "metadata", f"字幕轨道获取失败：{lan}，{e}")
                continue

            body = data.get("body", [])
            segments = []
            for item in body:
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                segments.append(
                    {
                        "start": float(item.get("from", 0)),
                        "end": optional_float(item.get("to")),
                        "text": content,
                    }
                )

            text = "\n".join(
                f"[{segment['start']:.1f}s] {segment['text']}"
                for segment in segments
            )
            results.append(
                {
                    "lan": lan,
                    "lan_code": lan_code,
                    "content": text,
                    "segments": segments,
                }
            )

    return results


def extract(
    url: str,
    sessdata: str = "",
    bili_jct: str = "",
    buvid3: str = "",
    progress: ProgressCallback | None = None,
) -> VideoInfo:
    """Main entry: extract video info and subtitles."""
    bvid = parse_bvid(url)
    credential = None
    if sessdata:
        credential = Credential(sessdata=sessdata, bili_jct=bili_jct, buvid3=buvid3)
    return asyncio.run(_fetch(bvid, credential, progress))
