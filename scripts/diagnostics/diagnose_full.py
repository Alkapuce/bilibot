"""Reproduce a subtitle postprocess issue against local artifacts.

This is a manual diagnostic script. It calls the configured real LLM and reads
local files from ``data/<bvid>/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from bilibot.config import load_settings
from bilibot.extractor import VideoInfo
from bilibot.models import Transcript, TranscriptSegment
from bilibot.postprocessor import postprocess_transcript, _split_segments

DEFAULT_BVID = "BV1YkR1BXEow"


def main() -> int:
    args = _parse_args()
    bvid = args.bvid
    artifact_dir = PROJECT_ROOT / "data" / bvid
    raw_path = artifact_dir / "transcript_raw.json"
    metadata_path = artifact_dir / "metadata.json"

    print("WARNING: this script calls the configured real LLM and may incur API cost.")
    print(f"Artifact directory: {artifact_dir}")
    raw = _read_json(raw_path)
    meta = _read_json(metadata_path)

    settings = load_settings()
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

    total_chars = sum(len(s.text) for s in segments)
    print(f"Model: {settings.subtitle_postprocess_model or settings.llm_model}")
    print(f"Total segments: {len(segments)}")
    print(f"Total chars: {total_chars}")
    print(f"Subtitle postprocess chunk chars: {settings.subtitle_postprocess_chunk_chars}")

    chunks = _split_segments(segments, settings.subtitle_postprocess_chunk_chars)
    print(f"Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        chars = sum(len(segment.text) for _, segment in chunk)
        print(f"  Chunk {i + 1}: {len(chunk)} segments, {chars} chars")

    print("\nRunning postprocess_transcript...")
    start = time.perf_counter()
    result = postprocess_transcript(info, transcript, settings)
    elapsed = time.perf_counter() - start

    diffs = 0
    for i, (orig, proc) in enumerate(zip(transcript.segments, result.segments)):
        if orig.text != proc.text:
            diffs += 1
            if diffs <= 10:
                print(f"\nSEG[{i}] RAW:  {orig.text}")
                print(f"SEG[{i}] PROC: {proc.text}")

    print("\n--- Summary ---")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Postprocessed: {result.postprocessed}")
    print(f"Model: {result.postprocess_model}")
    print(f"Modified segments: {diffs}/{len(transcript.segments)}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full subtitle postprocess diagnostics against data/<bvid>/ artifacts.",
    )
    parser.add_argument(
        "bvid",
        nargs="?",
        default=DEFAULT_BVID,
        help=f"Bilibili BV id under data/ (default: {DEFAULT_BVID})",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
