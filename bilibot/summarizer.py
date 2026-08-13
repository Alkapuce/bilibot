"""Transcript chunking and LLM note generation."""

from __future__ import annotations

from .config import Settings
from .extractor import VideoInfo
from .llm import LLMClient
from .models import Transcript, format_timestamp
from .progress import ProgressCallback, emit


SYSTEM_PROMPT = """你是一个擅长把视频内容整理成易读摘要的中文笔记助手。

你的目标不是套固定模板，而是让读者轻松、快速、完整地掌握视频信息。

核心原则：
- 先判断视频类型和叙事结构，再选择最适合阅读的 Markdown 组织方式。
- 保留视频里的重要事实、观点、步骤、例子、结论和上下文，不机械复述字幕。
- 根据内容自然分组：教程可按步骤，评测可按维度，访谈可按话题，新闻可按事件脉络，长讲解可按主题层级。
- 用更容易扫读的表达：短段落、清晰小标题、要点列表、必要时表格。
- 避免大段文字墙：每个主要部分先给一句结论，再用少量短段落或要点展开。
- 输出是中等详细摘要，不是完整讲义；压缩枝节，保留主线和高价值细节。
- 控制标题层级：优先使用 4-8 个二级标题；三级标题只在内容确实复杂时使用，避免每个小点都升成标题。
- 时间戳只在有助于定位关键片段时使用；不要为了凑格式强行做完整时间线。
- 可以概括和重组，但不要编造字幕中没有的信息，不要加入外部知识当作视频内容。
- 全中文输出；必要的英文术语可保留原文，并给出自然解释。
"""


def split_text(text: str, max_chars: int) -> list[str]:
    max_chars = max(1, int(max_chars))
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
    model_label = llm.model_sequence_label()
    transcript_text = transcript.text.strip()
    if not transcript_text:
        return render_basic_notes(info, transcript, "未获取到可用于总结的字幕或转写文本。")

    chunk_chars = effective_summary_chunk_chars(settings)
    chunks = split_text(transcript_text, chunk_chars)
    emit(
        progress,
        "log",
        "llm_summarize",
        f"字幕共 {len(transcript_text)} 字符，按 {chunk_chars} 字符上限分为 {len(chunks)} 块，模型链路 {model_label}",
    )
    if len(chunks) == 1:
        emit(progress, "task_start", "llm_summarize", f"生成笔记：{model_label}", total=1)
        emit(progress, "log", "llm_summarize", f"第 1 块，{len(chunks[0])} 字符，正在调用 LLM...")
        notes = _summarize_single(llm, info, transcript, chunks[0], progress=progress)
        emit(progress, "task_done", "llm_summarize", "笔记生成完成")
        return notes

    partial_notes = []
    emit(
        progress,
        "task_start",
        "llm_summarize",
        f"分块生成笔记：{model_label}",
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


def effective_summary_chunk_chars(settings: Settings) -> int:
    requested = max(1, int(settings.chunk_chars))
    cap = int(settings.summary_max_single_chunk_chars)
    if cap > 0:
        return min(requested, cap)
    return requested


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
    prompt = f"""请根据下面的视频字幕生成一份易读、信息完整的中文 Markdown 摘要笔记。

请先在心里判断视频类型、内容密度和叙事方式，然后自行设计输出结构。不要套用固定章节；只使用真正有助于阅读的标题和列表。

建议的阅读体验：
- 开头用 3-6 句话给出“这个视频讲了什么、最重要的信息是什么、适合谁读”。
- 正文按视频自身逻辑重组，而不是逐句复述字幕。
- 对复杂内容使用分组小标题；对步骤、对比、参数、优缺点等内容可用列表或表格。
- 对重要结论、关键数字、具体方法、案例细节要尽量保留。
- 每个主要部分建议先写 1 句概括，再用 3-6 个要点或短段落展开；避免连续大段叙述。
- 如果视频里有明显的段落转折或关键片段，可以附少量时间戳帮助定位。
- 不要强制输出术语表、行动建议、核实清单、完整时间线或固定观点列表。
- 不要写“字幕中未提供”之类的占位话；没有的信息直接不写。
- 原始字幕很短时保持简洁；信息量很大时可以充分展开，但不要灌水。

篇幅和层级控制：
- 默认写成中等长度摘要：约 2500-5000 个中文字符；高密度教程或长视频最多约 7000 字符。
- 优先使用 4-8 个二级标题；三级标题要少用，只有大段内容需要再分组时才使用。
- 避免标题过碎、段落过短、每个事实单独成节；相关事实应合并成一组。
- 不要为了“完整”把所有细节平铺出来；优先保证读者能轻松读完并掌握全貌。

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
    prompt = f"""这是同一个视频字幕的第 {index}/{total} 段。请先整理一份分段摘要，供后续合并成完整笔记。

要求：
- 提取本段出现的重要信息、事实、观点、例子、步骤、结论和上下文。
- 按本段内容自然分组，不要套固定模板。
- 记录有助于定位的少量关键时间点，但不要强制做完整时间线。
- 分段摘要要为最终压缩服务，保留主线和高价值细节，略去重复铺垫。
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
    prompt = f"""请将下面的分段摘要合并成一份完整、易读的中文 Markdown 视频摘要笔记。

要求：
- 去重、重排并压缩重复内容，不要简单拼接分段结果。
- 先判断整支视频的类型和内容架构，再设计最适合阅读的结构。
- 开头保留一个简短总览，帮助读者快速知道视频核心信息。
- 正文按主题、事件脉络、步骤、对比维度或问答话题自然组织。
- 只在有定位价值时保留少量关键时间点。
- 不要强制输出术语表、行动建议、核实清单、完整时间线或固定观点列表。
- 不要加入分段摘要里没有的内容。
- 最终输出要比各分段之和明显更短：压缩重复例子、相近论点和枝节铺垫。
- 默认目标约 2500-5000 个中文字符；高密度教程或长视频最多约 7000 字符。
- 优先使用 4-8 个二级标题，三级标题少用；不要把每个小事实单独做标题。

标题：
# {info.title or info.bvid}

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
