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


def save_artifacts(base_dir: Path, info: VideoInfo, transcript: Transcript, notes: str) -> dict[str, Path]:
    root = output_root(base_dir, info.bvid)
    paths = {
        "metadata": root / "metadata.json",
        "transcript_json": root / "transcript.json",
        "transcript_md": root / "transcript.md",
        "notes": root / "notes.md",
    }

    _write_json(paths["metadata"], _metadata_payload(info))
    _write_json(paths["transcript_json"], transcript.to_dict())
    paths["transcript_md"].write_text(transcript.to_markdown(), encoding="utf-8")
    paths["notes"].write_text(notes.rstrip() + "\n", encoding="utf-8")
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
