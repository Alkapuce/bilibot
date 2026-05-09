"""End-to-end Bilibili video analysis pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .extractor import VideoInfo, extract
from .models import Transcript, TranscriptSegment, optional_float
from .postprocessor import postprocess_transcript
from .progress import ProgressCallback, emit
from .storage import save_notes_artifact, save_transcript_artifacts
from .summarizer import render_basic_notes, summarize_video
from .transcriber import transcribe_url


@dataclass
class PipelineResult:
    info: VideoInfo
    transcript: Transcript
    raw_transcript: Transcript | None
    notes: str
    paths: dict[str, Path]


def analyze_url(
    url: str,
    settings: Settings,
    *,
    force_asr: bool = False,
    no_llm: bool = False,
    progress: ProgressCallback | None = None,
) -> PipelineResult:
    emit(progress, "task_start", "metadata", "获取视频信息和 B站字幕")
    info = extract(
        url,
        sessdata=settings.bili_sessdata,
        bili_jct=settings.bili_jct,
        buvid3=settings.bili_buvid3,
        progress=progress,
    )
    emit(
        progress,
        "task_done",
        "metadata",
        f"视频信息获取完成：{info.bvid}，字幕轨道 {len(info.subtitles)} 条",
    )

    transcript: Transcript | None = None
    if not force_asr:
        subtitle = select_subtitle(info.subtitles, preferred_language=settings.language)
        if subtitle:
            emit(
                progress,
                "log",
                "subtitle",
                f"使用 B站已有字幕：{subtitle.get('lan') or subtitle.get('lan_code') or 'unknown'}",
            )
            candidate = transcript_from_subtitle(subtitle)
            if candidate.segments:
                transcript = candidate
            else:
                emit(progress, "log", "subtitle", "B站字幕无有效段落，回退 ASR")

    if transcript is None:
        emit(progress, "log", "subtitle", "未找到可用字幕，下载音频并进行语音转文本")
        transcript = transcribe_url(info.url, settings, progress=progress)

    raw_transcript = transcript
    if settings.subtitle_postprocess:
        transcript = postprocess_transcript(info, transcript, settings, progress=progress)

    emit(progress, "task_start", "storage", "写入字幕/转录文件")
    transcript_paths = save_transcript_artifacts(
        settings.output_dir,
        info,
        transcript,
        raw_transcript=raw_transcript if transcript.postprocessed else None,
    )
    emit(progress, "task_done", "storage", "字幕/转录文件写入完成")

    if no_llm:
        emit(progress, "log", "llm_summarize", "跳过 LLM 总结")
        notes = render_basic_notes(info, transcript, "已通过 --no-llm 跳过 LLM 总结。")
    else:
        try:
            notes = summarize_video(info, transcript, settings, progress=progress)
        except Exception as exc:
            emit(progress, "log", "llm_summarize", f"LLM 总结失败: {exc}")
            notes = render_basic_notes(info, transcript, f"LLM 总结失败：{exc}")

    emit(progress, "task_start", "storage", "写入笔记文件")
    notes_path = save_notes_artifact(settings.output_dir, info.bvid, notes, title=info.title)
    emit(progress, "task_done", "storage", "笔记文件写入完成")
    paths = {**transcript_paths, "notes": notes_path}
    return PipelineResult(
        info=info,
        transcript=transcript,
        raw_transcript=raw_transcript if transcript.postprocessed else None,
        notes=notes,
        paths=paths,
    )


def select_subtitle(subtitles: list[dict[str, Any]], preferred_language: str = "zh") -> dict[str, Any] | None:
    """Select the best-matching subtitle track, returning None if nothing matches."""
    if not subtitles:
        return None

    def score(item: dict[str, Any]) -> int:
        lan = f"{item.get('lan', '')} {item.get('lan_code', '')}".lower()
        preferred = preferred_language.lower()
        if preferred and preferred in lan:
            return 100
        if any(token in lan for token in ("zh", "chinese", "中文", "简体", "繁体")):
            return 90
        return 0

    best = sorted(subtitles, key=score, reverse=True)[0]
    return best if score(best) > 0 else None


def transcript_from_subtitle(subtitle: dict[str, Any]) -> Transcript:
    segments = [
        TranscriptSegment(
            start=float(item.get("start", 0)),
            end=optional_float(item.get("end")),
            text=str(item.get("text", "")).strip(),
        )
        for item in subtitle.get("segments", [])
        if str(item.get("text", "")).strip()
    ]

    if not segments:
        segments = _parse_subtitle_content(subtitle.get("content", ""))

    return Transcript(
        source="bilibili_subtitle",
        language=str(subtitle.get("lan") or subtitle.get("lan_code") or "unknown"),
        segments=segments,
    )


def _parse_subtitle_content(content: str) -> list[TranscriptSegment]:
    pattern = re.compile(r"^\[(?P<start>\d+(?:\.\d+)?)s\]\s*(?P<text>.*)$")
    segments = []
    for line in content.splitlines():
        match = pattern.match(line.strip())
        if match:
            segments.append(
                TranscriptSegment(
                    start=float(match.group("start")),
                    text=match.group("text").strip(),
                )
            )
        elif line.strip():
            segments.append(TranscriptSegment(start=0.0, text=line.strip()))
    return segments
