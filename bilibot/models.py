"""Shared data models for bilibot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


TranscriptSource = Literal["bilibili_subtitle", "whisper"]


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

    @property
    def text(self) -> str:
        return "\n".join(segment.to_line() for segment in self.segments)

    def to_markdown(self) -> str:
        source_label = "Bilibili subtitle" if self.source == "bilibili_subtitle" else "Whisper ASR"
        lines = [
            "# Transcript",
            "",
            f"- Source: {source_label}",
            f"- Language: {self.language or 'unknown'}",
            f"- Segments: {len(self.segments)}",
            "",
            "## Full Text",
            "",
        ]
        lines.extend(segment.to_line() for segment in self.segments)
        lines.append("")
        return "\n".join(lines)

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
