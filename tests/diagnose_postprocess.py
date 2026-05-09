"""Diagnose why _parse_json_array returns empty for deepseek-v4-pro-nothinking."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bilibot.config import load_settings
from bilibot.llm import LLMClient
from bilibot.models import Transcript, TranscriptSegment
from bilibot.extractor import VideoInfo
from bilibot.postprocessor import _postprocess_chunk, _parse_json_array

PROJECT_ROOT = Path(__file__).resolve().parent.parent

settings = load_settings()

# Simulate the postprocess LLM config fallback
from dataclasses import replace
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
print(f"Model: {llm_settings.llm_model}")
print(f"Base URL: {llm_settings.llm_base_url}")

# Load a few segments from the actual transcript
with open(PROJECT_ROOT / "data/BV1YkR1BXEow/transcript_raw.json") as f:
    raw = json.load(f)

with open(PROJECT_ROOT / "data/BV1YkR1BXEow/metadata.json") as f:
    meta = json.load(f)

info = VideoInfo(
    bvid=meta["bvid"],
    title=meta["title"],
    author=meta.get("author", ""),
    desc=meta.get("desc", ""),
    duration=meta.get("duration", 0),
    cover=meta.get("cover", ""),
    url=meta.get("url", ""),
)

segments = [TranscriptSegment(start=s["start"], text=s["text"], end=s.get("end")) for s in raw["segments"]]
transcript = Transcript(
    source=raw.get("source", "whisper_asr"),
    language=raw.get("language", "zh"),
    segments=segments,
)

# Take a small chunk (first 5 segments) to test
chunk = [(i, transcript.segments[i]) for i in range(min(5, len(transcript.segments)))]
print(f"\nChunk size: {len(chunk)} segments")
for idx, seg in chunk:
    print(f"  [{idx}] start={seg.start:.1f}s text={seg.text[:80]}")

# Build payload (same as _postprocess_chunk does)
payload = [
    {"index": idx, "start": seg.start, "end": seg.end, "text": seg.text}
    for idx, seg in chunk
]

prompt = f"""请对字幕文本做后处理，返回 JSON 数组。

视频标题：{info.title}
字幕来源：{transcript.source}
处理风格：{settings.subtitle_postprocess_style}

规则：
- 只优化 text 字段，可修正错别字、ASR 误识别、标点、明显口语断裂。
- 不要新增字幕中没有的信息，不要改变说话含义。
- 必须保留输入的 index 数量与顺序。
- 不要输出 Markdown，不要输出解释。
- 返回格式：[{{"index": 0, "text": "优化后的文本"}}]

输入 JSON：
{json.dumps(payload, ensure_ascii=False)}
"""

llm = LLMClient(llm_settings)

# Call with stream=False for cleaner capture
print("\n--- Calling LLM (non-streaming) ---")
response = llm.complete([
    {"role": "system", "content": "你是字幕校对助手，只输出合法 JSON。"},
    {"role": "user", "content": prompt},
])

print(f"RAW LLM RESPONSE ({len(response)} chars):")
print(repr(response[:2000]))

print("\n--- _parse_json_array result ---")
parsed = _parse_json_array(response)
print(f"Parsed items: {len(parsed)}")
for item in parsed:
    print(f"  {item}")

# Also try streaming
print("\n--- Calling LLM (streaming) ---")
response2 = llm.complete_stream([
    {"role": "system", "content": "你是字幕校对助手，只输出合法 JSON。"},
    {"role": "user", "content": prompt},
])

print(f"\nRAW STREAMING RESPONSE ({len(response2)} chars):")
print(repr(response2[:2000]))

print("\n--- _parse_json_array result (streaming) ---")
parsed2 = _parse_json_array(response2)
print(f"Parsed items: {len(parsed2)}")
for item in parsed2:
    print(f"  {item}")
