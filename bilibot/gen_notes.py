"""Generate notes from an already-extracted transcript.

Called via ``bilibot gen-notes <bvid>`` — reuses existing *_字幕.json
and *_信息.json files in the output directory without re-extracting.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Settings, load_settings
from .extractor import VideoInfo
from .models import Transcript, TranscriptSegment
from .progress import ProgressCallback
from .storage import output_root, save_notes_artifact
from .summarizer import render_basic_notes, summarize_video


def run(bvid: str, settings: Settings | None = None, *, progress: ProgressCallback | None = None) -> Path:
    """Generate notes for *bvid* using previously saved artifacts.

    Returns the path to the generated ``_笔记.md`` file.
    """
    if settings is None:
        settings = load_settings()

    data_dir = output_root(settings.output_dir, bvid)

    # Support both new naming ({title}_字幕.json) and legacy (transcript.json)
    transcript_files = sorted(data_dir.glob("*_字幕.json"))
    meta_files = sorted(data_dir.glob("*_信息.json"))

    # Fallback: legacy filenames (metadata.json / transcript.json / notes.md)
    if not transcript_files:
        transcript_files = sorted(data_dir.glob("transcript*.json"))
    if not meta_files:
        meta_files = sorted(data_dir.glob("metadata.json"))

    if not transcript_files or not meta_files:
        raise FileNotFoundError(
            f"在 {data_dir}/ 中未找到 *_字幕.json 或 *_信息.json。\n"
            f"请先运行: bilibot summarize {bvid}"
        )

    raw = json.loads(transcript_files[-1].read_text(encoding="utf-8"))
    meta = json.loads(meta_files[-1].read_text(encoding="utf-8"))

    info = VideoInfo(
        bvid=meta.get("bvid", bvid),
        title=meta.get("title", ""),
        author=meta.get("author", ""),
        desc=meta.get("desc", ""),
        duration=meta.get("duration", 0),
        cover=meta.get("cover", ""),
        url=meta.get("url", f"https://www.bilibili.com/video/{bvid}"),
    )

    segments_raw = raw.get("segments", [])
    segments = [
        TranscriptSegment(start=s["start"], text=s["text"], end=s.get("end"))
        for s in segments_raw
    ]
    transcript = Transcript(
        source=raw.get("source", "bilibili_subtitle"),
        language=raw.get("language", "zh"),
        segments=segments,
    )

    try:
        notes = summarize_video(info, transcript, settings, progress=progress)
    except Exception as exc:
        if progress:
            from .progress import emit
            emit(progress, "log", "gen_notes", f"LLM 调用失败: {exc}，使用基础笔记")
        notes = render_basic_notes(info, transcript, f"LLM 调用失败：{exc}")

    return save_notes_artifact(settings.output_dir, bvid, notes, title=info.title)
