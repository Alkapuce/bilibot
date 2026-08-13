"""Diagnose postprocess JSON parsing against local artifacts.

This is a manual diagnostic script. It calls the configured real LLM and reads
local files from ``data/<bvid>/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from bilibot.config import Settings, load_settings
from bilibot.extractor import VideoInfo
from bilibot.llm import LLMClient
from bilibot.models import Transcript, TranscriptSegment
from bilibot.postprocessor import _parse_json_array

DEFAULT_BVID = "BV1YkR1BXEow"


def main() -> int:
    args = _parse_args()
    bvid = args.bvid
    artifact_dir = PROJECT_ROOT / "data" / bvid
    raw_path = artifact_dir / "transcript_raw.json"
    metadata_path = artifact_dir / "metadata.json"

    print("WARNING: this script calls the configured real LLM and may incur API cost.")
    print(f"Artifact directory: {artifact_dir}")
    settings = load_settings()
    llm_settings = _postprocess_llm_settings(settings)
    print(f"Model: {llm_settings.llm_model}")
    print(f"Base URL: {llm_settings.llm_base_url}")

    raw = _read_json(raw_path)
    meta = _read_json(metadata_path)
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

    chunk = [(i, transcript.segments[i]) for i in range(min(5, len(transcript.segments)))]
    print(f"\nChunk size: {len(chunk)} segments")
    for idx, seg in chunk:
        print(f"  [{idx}] start={seg.start:.1f}s text={seg.text[:80]}")

    payload = [
        {"index": idx, "start": seg.start, "end": seg.end, "text": seg.text}
        for idx, seg in chunk
    ]
    prompt = _build_prompt(info, transcript, settings.subtitle_postprocess_style, payload)
    llm = LLMClient(llm_settings)

    print("\n--- Calling LLM (non-streaming) ---")
    response = llm.complete([
        {"role": "system", "content": "你是字幕校对助手，只输出合法 JSON。"},
        {"role": "user", "content": prompt},
    ])
    _print_parse_result("RAW LLM RESPONSE", response)

    print("\n--- Calling LLM (streaming) ---")
    response2 = llm.complete_stream([
        {"role": "system", "content": "你是字幕校对助手，只输出合法 JSON。"},
        {"role": "user", "content": prompt},
    ])
    _print_parse_result("RAW STREAMING RESPONSE", response2)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose subtitle postprocess JSON parsing against data/<bvid>/ artifacts.",
    )
    parser.add_argument(
        "bvid",
        nargs="?",
        default=DEFAULT_BVID,
        help=f"Bilibili BV id under data/ (default: {DEFAULT_BVID})",
    )
    return parser.parse_args()


def _postprocess_llm_settings(settings: Settings) -> Settings:
    return replace(
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_prompt(info: VideoInfo, transcript: Transcript, style: str, payload: list[dict[str, Any]]) -> str:
    return f"""请对字幕文本做后处理，返回 JSON 数组。

视频标题：{info.title}
字幕来源：{transcript.source}
处理风格：{style}

规则：
- 只优化 text 字段，可修正错别字、ASR 误识别、标点、明显口语断裂。
- 不要新增字幕中没有的信息，不要改变说话含义。
- 必须保留输入的 index 数量与顺序。
- 不要输出 Markdown，不要输出解释。
- 返回格式：[{{"index": 0, "text": "优化后的文本"}}]

输入 JSON：
{json.dumps(payload, ensure_ascii=False)}
"""


def _print_parse_result(label: str, response: str) -> None:
    print(f"{label} ({len(response)} chars):")
    print(repr(response[:2000]))

    print("\n--- _parse_json_array result ---")
    parsed = _parse_json_array(response)
    print(f"Parsed items: {len(parsed)}")
    for item in parsed:
        print(f"  {item}")


if __name__ == "__main__":
    raise SystemExit(main())
