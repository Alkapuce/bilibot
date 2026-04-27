"""End-to-end Bilibili video analysis pipeline."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .extractor import VideoInfo, extract
from .models import Transcript, TranscriptSegment
from .storage import save_artifacts
from .summarizer import render_basic_notes, summarize_video
from .transcriber import transcribe_url


ProgressCallback = Callable[[str], None]


@dataclass
class PipelineResult:
    info: VideoInfo
    transcript: Transcript
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
    report = progress or (lambda _message: None)

    report("获取视频信息和 B站字幕")
    info = extract(
        url,
        sessdata=settings.bili_sessdata,
        bili_jct=settings.bili_jct,
        buvid3=settings.bili_buvid3,
    )

    transcript = None
    if not force_asr:
        subtitle = select_subtitle(info.subtitles, preferred_language=settings.language)
        if subtitle:
            report("使用 B站已有字幕")
            transcript = transcript_from_subtitle(subtitle)

    if transcript is None:
        report("未找到可用字幕，下载音频并进行语音转文本")
        transcript = transcribe_url(
            info.url,
            cookie_file=settings.cookie_file,
            language=settings.language,
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            sessdata=settings.bili_sessdata,
            bili_jct=settings.bili_jct,
            buvid3=settings.bili_buvid3,
        )

    if no_llm:
        report("跳过 LLM 总结")
        notes = render_basic_notes(info, transcript)
    else:
        report("调用 LLM 生成笔记")
        notes = summarize_video(info, transcript, settings)

    report("写入输出文件")
    paths = save_artifacts(settings.output_dir, info, transcript, notes)
    return PipelineResult(info=info, transcript=transcript, notes=notes, paths=paths)


def select_subtitle(subtitles: list[dict[str, Any]], preferred_language: str = "zh") -> dict[str, Any] | None:
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

    return sorted(subtitles, key=score, reverse=True)[0]


def transcript_from_subtitle(subtitle: dict[str, Any]) -> Transcript:
    segments = [
        TranscriptSegment(
            start=float(item.get("start", 0)),
            end=_optional_float(item.get("end")),
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


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
