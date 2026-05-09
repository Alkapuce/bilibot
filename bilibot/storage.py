"""Filesystem output helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .extractor import VideoInfo
from .models import Transcript, jsonable


_MAX_TITLE_LEN = 60


def safe_filename(title: str, max_len: int = _MAX_TITLE_LEN) -> str:
    """Sanitize video title for use in a filename.

    Replaces characters unsafe on Windows/Linux filesystems with ``_``,
    collapses consecutive underscores, strips leading/trailing separators,
    and truncates to *max_len* characters.
    """
    safe = re.sub(r'[\\/:*?"<>|\n\r\t]', "_", title)
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("_. ")
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("_")
    return safe or "untitled"


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
    prefix = safe_filename(info.title)

    paths: dict[str, Path] = {
        "metadata": root / f"{prefix}_信息.json",
        "transcript_json": root / f"{prefix}_字幕.json",
        "captions_txt": root / f"{prefix}_字幕.txt",
    }
    _write_json(paths["metadata"], _metadata_payload(info))
    _write_json(paths["transcript_json"], transcript.to_dict())
    paths["captions_txt"].write_text(transcript.to_captions(), encoding="utf-8")
    if raw_transcript is not None:
        paths["transcript_raw_json"] = root / f"{prefix}_字幕原文.json"
        paths["captions_raw_txt"] = root / f"{prefix}_字幕原文.txt"
        _write_json(paths["transcript_raw_json"], raw_transcript.to_dict())
        paths["captions_raw_txt"].write_text(raw_transcript.to_captions(), encoding="utf-8")
    return paths


def save_notes_artifact(base_dir: Path, bvid: str, notes: str, *, title: str = "") -> Path:
    root = output_root(base_dir, bvid)
    prefix = safe_filename(title) if title else bvid
    path = root / f"{prefix}_笔记.md"
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
    notes_path = save_notes_artifact(base_dir, info.bvid, notes, title=info.title)
    paths["notes"] = notes_path
    return paths


def _metadata_payload(info: VideoInfo) -> dict[str, Any]:
    payload = asdict(info)
    subtitles = payload.pop("subtitles", [])
    tags = payload.pop("tags", [])
    payload["subtitle_tracks"] = [
        {
            "lan": item.get("lan", ""),
            "lan_code": item.get("lan_code", ""),
            "segments": len(item.get("segments", [])),
            "content_chars": len(item.get("content", "")),
        }
        for item in subtitles
    ]
    payload["tags"] = tags
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
