"""Filesystem output helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .extractor import VideoInfo
from .models import Transcript, jsonable


def output_root(base_dir: Path, bvid: str) -> Path:
    path = base_dir / bvid
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_transcript_artifacts(
    base_dir: Path,
    info: VideoInfo,
    transcript: Transcript,
    *,
    raw_transcript: Transcript | None = None,
) -> dict[str, Path]:
    root = output_root(base_dir, info.bvid)
    paths: dict[str, Path] = {
        "metadata": root / "metadata.json",
        "transcript_json": root / "transcript.json",
        "transcript_md": root / "transcript.md",
    }
    _write_json(paths["metadata"], _metadata_payload(info))
    _write_json(paths["transcript_json"], transcript.to_dict())
    paths["transcript_md"].write_text(transcript.to_markdown(), encoding="utf-8")
    if raw_transcript is not None:
        paths["transcript_raw_json"] = root / "transcript_raw.json"
        paths["transcript_raw_md"] = root / "transcript_raw.md"
        _write_json(paths["transcript_raw_json"], raw_transcript.to_dict())
        paths["transcript_raw_md"].write_text(raw_transcript.to_markdown(), encoding="utf-8")
    return paths


def save_notes_artifact(base_dir: Path, bvid: str, notes: str) -> Path:
    root = output_root(base_dir, bvid)
    path = root / "notes.md"
    path.write_text(notes.rstrip() + "\n", encoding="utf-8")
    return path


def save_artifacts(
    base_dir: Path,
    info: VideoInfo,
    transcript: Transcript,
    notes: str,
    *,
    raw_transcript: Transcript | None = None,
) -> dict[str, Path]:
    paths = save_transcript_artifacts(base_dir, info, transcript, raw_transcript=raw_transcript)
    notes_path = save_notes_artifact(base_dir, info.bvid, notes)
    paths["notes"] = notes_path
    return paths


def _metadata_payload(info: VideoInfo) -> dict[str, Any]:
    payload = asdict(info)
    subtitles = payload.pop("subtitles", [])
    payload["subtitle_tracks"] = [
        {
            "lan": item.get("lan", ""),
            "lan_code": item.get("lan_code", ""),
            "segments": len(item.get("segments", [])),
            "content_chars": len(item.get("content", "")),
        }
        for item in subtitles
    ]
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
