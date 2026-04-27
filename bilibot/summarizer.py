"""Transcript chunking and LLM note generation."""

from __future__ import annotations

from .config import Settings
from .extractor import VideoInfo
from .llm import LLMClient
from .models import Transcript, format_timestamp


SYSTEM_PROMPT = """你是一个严谨的视频学习笔记助手。你的任务是根据视频信息和字幕生成中文笔记。
要求：
- 只基于给定内容总结，不编造未出现的信息。
- 保留重要时间点，方便用户回看。
- 输出结构清晰的 Markdown。
- 如果信息不足，明确写“字幕中未提供”。
"""


def split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for line in text.splitlines():
        line_size = len(line) + 1
        if current and current_size + line_size > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_size = 0

        if line_size > max_chars:
            for start in range(0, len(line), max_chars):
                piece = line[start : start + max_chars]
                if current:
                    chunks.append("\n".join(current))
                    current = []
                    current_size = 0
                chunks.append(piece)
            continue

        current.append(line)
        current_size += line_size

    if current:
        chunks.append("\n".join(current))
    return chunks


def summarize_video(info: VideoInfo, transcript: Transcript, settings: Settings) -> str:
    llm = LLMClient(settings)
    transcript_text = transcript.text.strip()
    if not transcript_text:
        return render_basic_notes(info, transcript, "未获取到可用于总结的字幕或转写文本。")

    chunks = split_text(transcript_text, settings.chunk_chars)
    if len(chunks) == 1:
        return _summarize_single(llm, info, transcript, chunks[0])

    partial_notes = []
    for index, chunk in enumerate(chunks, start=1):
        partial_notes.append(_summarize_chunk(llm, info, transcript, chunk, index, len(chunks)))
    return _merge_notes(llm, info, transcript, partial_notes)


def render_basic_notes(info: VideoInfo, transcript: Transcript, reason: str = "") -> str:
    lines = [
        f"# {info.title or info.bvid}",
        "",
        "## 视频信息",
        "",
        f"- BV号: {info.bvid}",
        f"- 作者: {info.author or '未知'}",
        f"- 时长: {format_timestamp(float(info.duration))}",
        f"- 链接: {info.url}",
        f"- 字幕来源: {transcript.source}",
        "",
        "## 说明",
        "",
        reason or "未调用 LLM，总结已跳过。",
        "",
    ]
    return "\n".join(lines)


def _video_context(info: VideoInfo, transcript: Transcript) -> str:
    return "\n".join(
        [
            f"标题：{info.title}",
            f"BV号：{info.bvid}",
            f"作者：{info.author}",
            f"简介：{info.desc}",
            f"时长：{format_timestamp(float(info.duration))}",
            f"链接：{info.url}",
            f"字幕来源：{transcript.source}",
            f"字幕语言：{transcript.language}",
        ]
    )


def _summarize_single(llm: LLMClient, info: VideoInfo, transcript: Transcript, text: str) -> str:
    prompt = f"""请根据下面的视频信息和完整字幕生成一份完整中文 Markdown 笔记。

输出结构：
# {info.title or info.bvid}
## 视频信息
## 一句话总结
## 核心内容
## 时间线
## 关键观点
## 术语和概念
## 可行动建议
## 仍需核实

视频信息：
{_video_context(info, transcript)}

完整字幕：
{text}
"""
    return llm.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )


def _summarize_chunk(
    llm: LLMClient,
    info: VideoInfo,
    transcript: Transcript,
    text: str,
    index: int,
    total: int,
) -> str:
    prompt = f"""这是同一个视频字幕的第 {index}/{total} 段。请先生成分段笔记，供后续合并。

要求：
- 提取本段出现的主题、事实、论点、例子、术语。
- 保留本段重要时间点。
- 不要写最终总标题。

视频信息：
{_video_context(info, transcript)}

字幕片段：
{text}
"""
    return llm.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )


def _merge_notes(
    llm: LLMClient,
    info: VideoInfo,
    transcript: Transcript,
    partial_notes: list[str],
) -> str:
    joined = "\n\n---\n\n".join(
        f"分段笔记 {index}：\n{note}" for index, note in enumerate(partial_notes, start=1)
    )
    prompt = f"""请将下面的分段笔记合并成一份完整中文 Markdown 视频笔记。

要求：
- 去重并重组为连贯结构，不要简单拼接。
- 保留关键时间点。
- 如果分段笔记之间存在矛盾，放到“仍需核实”。

输出结构：
# {info.title or info.bvid}
## 视频信息
## 一句话总结
## 核心内容
## 时间线
## 关键观点
## 术语和概念
## 可行动建议
## 仍需核实

视频信息：
{_video_context(info, transcript)}

分段笔记：
{joined}
"""
    return llm.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
