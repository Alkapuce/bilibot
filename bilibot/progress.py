"""Progress event primitives shared by pipeline and CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


ProgressKind = Literal["log", "task_start", "task_update", "task_done"]
ProgressCallback = Callable[["ProgressEvent"], None]


@dataclass(frozen=True)
class ProgressEvent:
    kind: ProgressKind
    name: str
    message: str
    total: float | None = None
    completed: float | None = None
    advance: float | None = None
    unit: str = ""


def emit(
    progress: ProgressCallback | None,
    kind: ProgressKind,
    name: str,
    message: str,
    *,
    total: float | None = None,
    completed: float | None = None,
    advance: float | None = None,
    unit: str = "",
) -> None:
    if progress is None:
        return
    progress(
        ProgressEvent(
            kind=kind,
            name=name,
            message=message,
            total=total,
            completed=completed,
            advance=advance,
            unit=unit,
        )
    )
