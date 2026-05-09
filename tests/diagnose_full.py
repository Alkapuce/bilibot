"""Reproduce the BV1YkR1BXEow postprocess issue with full 152 segments."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bilibot.config import load_settings
from bilibot.models import Transcript, TranscriptSegment
from bilibot.extractor import VideoInfo
from bilibot.postprocessor import postprocess_transcript, _split_segments

PROJECT_ROOT = Path(__file__).resolve().parent.parent

settings = load_settings()

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

total_chars = sum(len(s.text) for s in segments)
print(f"Total segments: {len(segments)}")
print(f"Total chars: {total_chars}")
print(f"Subtitle postprocess chunk chars: {settings.subtitle_postprocess_chunk_chars}")

chunks = _split_segments(segments, settings.subtitle_postprocess_chunk_chars)
print(f"Chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    chars = sum(len(s.text) for _, s in c)
    print(f"  Chunk {i+1}: {len(c)} segments, {chars} chars")

print(f"\nRunning postprocess_transcript...")
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

print(f"\n--- Summary ---")
print(f"Elapsed: {elapsed:.1f}s")
print(f"Postprocessed: {result.postprocessed}")
print(f"Model: {result.postprocess_model}")
print(f"Modified segments: {diffs}/{len(transcript.segments)}")
