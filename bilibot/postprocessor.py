"""Optional LLM-based transcript post-processing."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from .config import Settings
from .extractor import VideoInfo
from .llm import LLMClient
from .models import Transcript, TranscriptSegment
from .progress import ProgressCallback, emit


def postprocess_transcript(
    info: VideoInfo,
    transcript: Transcript,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
) -> Transcript:
    text = transcript.text.strip()
    if not text:
        emit(progress, "log", "subtitle_postprocess", "字幕为空，跳过后处理。")
        return transcript

    try:
        return _postprocess_transcript_impl(info, transcript, settings, progress=progress)
    except Exception as exc:
        emit(progress, "log", "subtitle_postprocess", f"字幕后处理失败: {exc}，使用原始字幕")
        return transcript


def _postprocess_transcript_impl(
    info: VideoInfo,
    transcript: Transcript,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
) -> Transcript:
    llm_settings = replace(
        settings,
        llm_base_url=settings.subtitle_postprocess_base_url or settings.llm_base_url,
        llm_api_key=settings.subtitle_postprocess_api_key or settings.llm_api_key,
        llm_model=settings.subtitle_postprocess_model or settings.llm_model,
        llm_temperature=(
            settings.subtitle_postprocess_temperature
            if settings.subtitle_postprocess_temperature is not None
            else settings.llm_temperature
        ),
        chunk_chars=settings.subtitle_postprocess_chunk_chars,
    )
    llm = LLMClient(llm_settings)
    chunks = _split_segments(transcript.segments, settings.subtitle_postprocess_chunk_chars)
    new_segments = list(transcript.segments)
    emit(
        progress,
        "task_start",
        "subtitle_postprocess",
        f"字幕后处理：{llm_settings.llm_model}",
        total=len(chunks),
        unit="chunk",
    )
    for index, chunk in enumerate(chunks, start=1):
        emit(
            progress,
            "task_update",
            "subtitle_postprocess",
            f"字幕后处理 {index}/{len(chunks)}",
            completed=index - 1,
        )
        replacements = _postprocess_chunk(
            llm,
            info,
            transcript,
            chunk,
            settings.subtitle_postprocess_style,
            progress=progress,
        )
        for item_index, text_value in replacements.items():
            if 0 <= item_index < len(new_segments):
                original = new_segments[item_index]
                new_segments[item_index] = TranscriptSegment(
                    start=original.start,
                    end=original.end,
                    text=text_value.strip() or original.text,
                )
        emit(
            progress,
            "task_update",
            "subtitle_postprocess",
            f"字幕后处理 {index}/{len(chunks)}",
            advance=1,
        )

    emit(progress, "task_done", "subtitle_postprocess", "字幕后处理完成")
    return Transcript(
        source=transcript.source,
        language=transcript.language,
        segments=new_segments,
        postprocessed=True,
        postprocess_model=llm_settings.llm_model,
    )


def _postprocess_chunk(
    llm: LLMClient,
    info: VideoInfo,
    transcript: Transcript,
    chunk: list[tuple[int, TranscriptSegment]],
    style: str,
    *,
    progress: ProgressCallback | None = None,
) -> dict[int, str]:
    payload = [
        {
            "index": index,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        for index, segment in chunk
    ]

    # Build context: video title, description, author, and tags for domain awareness
    context_lines = [f"视频标题：{info.title}"]
    if info.author:
        context_lines.append(f"作者：{info.author}")
    if info.desc:
        context_lines.append(f"简介：{info.desc}")
    if info.tags:
        context_lines.append(f"标签：{'、'.join(info.tags)}")
    context = "\n".join(context_lines)

    prompt = f"""请对字幕文本做后处理，返回 JSON 数组。

{context}
字幕来源：{transcript.source}
处理风格：{style}

规则：
- 只优化 text 字段，可修正错别字、ASR 误识别、标点、明显口语断裂。
- 可参考标题、简介、标签中的术语来纠正 ASR 对专业名词的误识别。
- 不要新增字幕中没有的信息，不要改变说话含义。
- 必须保留输入的 index 数量与顺序。
- 不要输出 Markdown，不要输出解释。
- 返回格式：[{{"index": 0, "text": "优化后的文本"}}]

输入 JSON：
{json.dumps(payload, ensure_ascii=False)}
"""
    try:
        response = llm.complete_stream(
            [
                {"role": "system", "content": "你是字幕校对助手，只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            task_name="subtitle_postprocess",
            progress=progress,
        )
    except Exception as exc:
        emit(
            progress,
            "log",
            "subtitle_postprocess",
            f"字幕后处理分块 LLM 调用失败，保留原文 ({len(chunk)} 段): {exc}",
        )
        return {}
    items = _parse_json_array(response)
    replacements: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get("text", "")).strip()
        if text:
            replacements[index] = text
    return replacements


def _split_segments(
    segments: list[TranscriptSegment],
    max_chars: int,
) -> list[list[tuple[int, TranscriptSegment]]]:
    chunks: list[list[tuple[int, TranscriptSegment]]] = []
    current: list[tuple[int, TranscriptSegment]] = []
    current_size = 0
    limit = max(1000, max_chars)
    for index, segment in enumerate(segments):
        size = len(segment.text) + 32
        if current and current_size + size > limit:
            chunks.append(current)
            current = []
            current_size = 0
        current.append((index, segment))
        current_size += size
    if current:
        chunks.append(current)
    return chunks


def _parse_json_array(text: str) -> list[Any]:
    """Extract a JSON array from LLM response text.

    Handles various markdown code fence formats and performs fallback
    repair for common JSON syntax issues in LLM output.
    """
    stripped = text.strip()

    # ── Remove markdown code fences ──────────────────────────────
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```")
        # Skip optional language tag line (e.g. "json")
        stripped = stripped.lstrip()
        nl = stripped.find("\n")
        if nl != -1:
            tag = stripped[:nl].strip()
            if tag and len(tag) < 20 and "[" not in tag:
                stripped = stripped[nl + 1:]
        # Remove trailing closing fence
        stripped = stripped.rstrip()
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()

    # ── Locate JSON array boundaries ─────────────────────────────
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []

    json_str = stripped[start : end + 1]

    # ── Parse with fallback repair ───────────────────────────────
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        parsed = _fallback_parse_json_array(json_str)

    return parsed if isinstance(parsed, list) else []


def _fallback_parse_json_array(json_str: str) -> Any:
    """Try to repair and parse malformed JSON array from LLM output."""
    import re

    # 1. Strip trailing commas before ] or }
    repaired = re.sub(r",\s*([}\]])", r"\1", json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # 2. Extract individual objects via regex (handles severely malformed output)
    objects = re.findall(
        r'\{\s*"index"\s*:\s*(\d+)\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        repaired,
    )
    if objects:
        result: list[dict[str, Any]] = []
        for idx_str, txt in objects:
            try:
                result.append({"index": int(idx_str), "text": txt})
            except (ValueError, TypeError):
                continue
        return result

    return []
