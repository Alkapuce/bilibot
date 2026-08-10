"""Transcript chunking and LLM note generation."""

from __future__ import annotations

from .config import Settings
from .extractor import VideoInfo
from .llm import LLMClient
from .models import Transcript, format_timestamp
from .progress import ProgressCallback, emit


SYSTEM_PROMPT = """你是一个顶级的 CS231n 课程笔记专家。你的任务是根据视频字幕生成深度结构化中文笔记。

**核心要求**：
- 逐段精读，不要跳过大段内容。每个概念都要覆盖。
- 时间线必须按课程推进顺序列出每个知识点出现的位置（精确到字幕段的时间戳）。
- 关键观点必须是课程中讲师明确阐述的论点，每条附上原文或近原文引用。
- 术语和概念必须列出课程中出现的每个专业术语，给出定义、使用场景、与其它概念的关联。
- 可行动建议必须是课程中提到的实操建议（作业要求、编程技巧、论文阅读、实验设置等）。
- 仍需核实：标记课程中讲师不确定、带条件、或需要查证补充的内容。
- 禁止写“字幕中未提供”这种偷懒表述——你应该从字幕中提取所有可用信息。
- 笔记应该丰富到能替代看视频，让读者只看笔记就能理解课程核心内容。
- 全中文输出，但专业术语保留英文原名并附中文解释。
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


def summarize_video(
    info: VideoInfo,
    transcript: Transcript,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
) -> str:
    llm = LLMClient(settings)
    transcript_text = transcript.text.strip()
    if not transcript_text:
        return render_basic_notes(info, transcript, "未获取到可用于总结的字幕或转写文本。")

    chunks = split_text(transcript_text, settings.chunk_chars)
    emit(
        progress,
        "log",
        "llm_summarize",
        f"字幕共 {len(transcript_text)} 字符，分为 {len(chunks)} 块，使用模型 {settings.llm_model}",
    )
    if len(chunks) == 1:
        emit(progress, "task_start", "llm_summarize", f"生成笔记：{settings.llm_model}", total=1)
        emit(progress, "log", "llm_summarize", f"第 1 块，{len(chunks[0])} 字符，正在调用 LLM...")
        notes = _summarize_single(llm, info, transcript, chunks[0], progress=progress)
        emit(progress, "task_done", "llm_summarize", "笔记生成完成")
        return notes

    partial_notes = []
    emit(
        progress,
        "task_start",
        "llm_summarize",
        f"分块生成笔记：{settings.llm_model}",
        total=len(chunks) + 1,
        unit="chunk",
    )
    for index, chunk in enumerate(chunks, start=1):
        emit(
            progress,
            "task_update",
            "llm_summarize",
            f"总结分块 {index}/{len(chunks)} ({len(chunk)} 字符)",
        )
        emit(progress, "log", "llm_summarize", f"第 {index}/{len(chunks)} 块，{len(chunk)} 字符，正在调用 LLM...")
        partial_notes.append(
            _summarize_chunk(llm, info, transcript, chunk, index, len(chunks), progress=progress)
        )
        emit(
            progress,
            "task_update",
            "llm_summarize",
            f"总结分块 {index}/{len(chunks)} 完成",
            advance=1,
        )
    emit(progress, "task_update", "llm_summarize", "合并分块笔记")
    emit(progress, "log", "llm_summarize", "正在合并分块笔记...")
    notes = _merge_notes(llm, info, transcript, partial_notes, progress=progress)
    emit(progress, "task_done", "llm_summarize", "笔记生成完成")
    return notes


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
    ]
    if reason:
        lines.append(reason)
    elif not transcript.text.strip():
        lines.append("未获取到可用于总结的字幕或转写文本。")
    else:
        lines.append("未调用 LLM，总结已跳过。")
    lines.append("")
    return "\n".join(lines)


def _video_context(info: VideoInfo, transcript: Transcript, *, desc_max_chars: int = 500) -> str:
    desc = info.desc
    if len(desc) > desc_max_chars:
        desc = desc[:desc_max_chars] + "…"
    return "\n".join(
        [
            f"标题：{info.title}",
            f"BV号：{info.bvid}",
            f"作者：{info.author}",
            f"简介：{desc}",
            f"时长：{format_timestamp(float(info.duration))}",
            f"链接：{info.url}",
            f"字幕来源：{transcript.source}",
            f"字幕语言：{transcript.language}",
        ]
    )


def _summarize_single(
    llm: LLMClient,
    info: VideoInfo,
    transcript: Transcript,
    text: str,
    *,
    progress: ProgressCallback | None = None,
) -> str:
    prompt = f"""请根据下面的视频字幕生成一份深度中文 Markdown 笔记。这是 CS231n 课程的正式学习笔记，不是泛泛的摘要。

**输出结构必须严格包含以下章节**：

# 第X讲：{info.title or info.bvid}
## 视频信息
（BV号、讲师、时长等基本信息）

## 课程大纲（本讲内容地图）
用 5-10 个要点列出本讲覆盖的所有主题，形成内容地图。

## 时间线（按课程推进）
按时间顺序列出每个知识点出现的位置。格式：`[时间段] 知识点：具体内容`
- 不要跳段，覆盖整节课
- 每 5-10 分钟至少一个时间点

## 关键观点（讲师核心论点）
每条观点需包含：
- **论点**：讲师明确表达的观点
- **原文引用**：字幕中接近原文的引用
- **论证逻辑**：讲师如何论证该观点

## 术语和概念（完整词汇表）
每个术语必须包含：
- **英文名** (中文译名)：定义
- 使用场景
- 与其它概念的关联（如：X 是 Y 的基础，与 Z 对比...）

## 可行动建议
- 课程中提到的作业、实验、编程练习
- 推荐阅读的论文或资料
- 学习路径建议

## 仍需核实
- 讲师含糊或不确定的内容
- 需要查阅外部资料补充的部分

**警告**：
- 禁止简短敷衍。每个章节至少 5 条以上内容。
- 禁止跳过中间时间段。
- 禁止写"字幕中未提供"而不尝试提取。
- 笔记应达到 2000 字以上。

视频信息：
{_video_context(info, transcript)}

完整字幕：
{text}
"""
    return llm.complete_stream(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        task_name="llm_summarize",
        progress=progress,
    )


def _summarize_chunk(
    llm: LLMClient,
    info: VideoInfo,
    transcript: Transcript,
    text: str,
    index: int,
    total: int,
    *,
    progress: ProgressCallback | None = None,
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
    return llm.complete_stream(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        task_name="llm_summarize",
        progress=progress,
    )


def _merge_notes(
    llm: LLMClient,
    info: VideoInfo,
    transcript: Transcript,
    partial_notes: list[str],
    *,
    progress: ProgressCallback | None = None,
) -> str:
    joined = "\n\n---\n\n".join(
        f"分段笔记 {index}：\n{note}" for index, note in enumerate(partial_notes, start=1)
    )
    prompt = f"""请将下面的分段笔记合并成一份完整中文 Markdown 视频笔记。

要求：
- 去重并重组为连贯结构，不要简单拼接。
- 保留关键时间点。
- 如果分段笔记之间存在矛盾，放到"仍需核实"。

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
    return llm.complete_stream(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        task_name="llm_summarize",
        progress=progress,
    )
