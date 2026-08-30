"""Shared data models for bilibot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


TranscriptSource = str


@dataclass
class TranscriptSegment:
    start: float
    text: str
    end: float | None = None

    def to_line(self) -> str:
        return f"[{format_timestamp(self.start)}] {self.text}"


@dataclass
class Transcript:
    source: TranscriptSource
    language: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    time_stamps: list[dict[str, Any]] = field(default_factory=list)
    postprocessed: bool = False
    postprocess_model: str = ""

    @property
    def text(self) -> str:
        return "\n".join(segment.to_line() for segment in self.segments)

    def to_markdown(self) -> str:
        if self.source == "bilibili_subtitle":
            source_label = "Bilibili subtitle"
        elif self.source.startswith("qwen3-gguf"):
            source_label = "Qwen3-ASR GGUF"
        elif self.source.startswith("qwen3:") or self.source.startswith("qwen3/"):
            source_label = "Qwen3-ASR"
        elif self.source.startswith("whisper"):
            source_label = "Whisper ASR"
        else:
            source_label = self.source or "unknown"
        lines = [
            "# Transcript",
            "",
            f"- Source: {source_label}",
            f"- Language: {self.language or 'unknown'}",
            f"- Segments: {len(self.segments)}",
            f"- Time stamps: {len(self.time_stamps)}",
            f"- Postprocessed: {'yes' if self.postprocessed else 'no'}",
            "",
            "## Full Text",
            "",
        ]
        lines.extend(segment.to_line() for segment in self.segments)
        lines.append("")
        return "\n".join(lines)

    def to_captions(self) -> str:
        import re
        # Join segments with space to avoid word-merging at boundaries,
        # then split by Chinese sentence-ending punctuation for readability.
        full = " ".join(s.text for s in self.segments if s.text.strip())
        lines = [l.strip() for l in re.split(r"(?<=[。！？；\n])", full) if l.strip()]
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
