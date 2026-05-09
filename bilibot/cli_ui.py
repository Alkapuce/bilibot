"""Rich terminal UI components for bilibot CLI."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .models import format_timestamp, jsonable
from .pipeline import PipelineResult
from .progress import ProgressEvent


def print_result(result: PipelineResult, console: Console | None = None) -> None:
    """Print a PipelineResult as Rich tables."""
    if console is None:
        from .cli import console as _console
        console = _console

    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="cyan", no_wrap=True)
    summary.add_column()
    summary.add_row("标题", result.info.title or result.info.bvid)
    summary.add_row("BV", result.info.bvid)
    summary.add_row("作者", result.info.author or "未知")
    summary.add_row("时长", format_timestamp(float(result.info.duration)))
    summary.add_row("字幕来源", result.transcript.source)
    summary.add_row("字幕段落", str(len(result.transcript.segments)))
    summary.add_row("字幕后处理", result.transcript.postprocess_model if result.transcript.postprocessed else "未启用")
    console.print(Panel(summary, title="完成", border_style="green"))

    artifacts = Table(title="Artifacts")
    artifacts.add_column("Name", style="cyan", no_wrap=True)
    artifacts.add_column("Path")
    for name, path in result.paths.items():
        artifacts.add_row(name, str(path))
    console.print(artifacts)


def print_result_json(result: PipelineResult, console: Console | None = None) -> None:
    """Print a PipelineResult as JSON."""
    if console is None:
        from .cli import console as _console
        console = _console

    console.print_json(
        data=jsonable(
            {
                "bvid": result.info.bvid,
                "title": result.info.title,
                "author": result.info.author,
                "duration": result.info.duration,
                "transcript_source": result.transcript.source,
                "transcript_language": result.transcript.language,
                "transcript_segments": len(result.transcript.segments),
                "transcript_chars": len(result.transcript.text),
                "postprocessed": result.transcript.postprocessed,
                "postprocess_model": result.transcript.postprocess_model or None,
                "artifacts": {name: str(path) for name, path in result.paths.items()},
            }
        )
    )


class RichProgressReporter:
    """Progress callback that renders Rich progress bars and logs."""

    def __init__(self, console: Console, *, verbose: bool = False):
        self.console = console
        self.verbose = verbose
        self.tasks: dict[str, int] = {}
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

    def __enter__(self) -> "RichProgressReporter":
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.progress.__exit__(exc_type, exc, traceback)

    def __call__(self, event: ProgressEvent) -> None:
        if event.kind == "log":
            if self.verbose:
                self.progress.log(f"[dim]{event.message}[/dim]")
            return

        if event.kind == "task_start":
            task_id = self.tasks.get(event.name)
            if task_id is None:
                self.tasks[event.name] = self.progress.add_task(event.message, total=event.total)
            else:
                self.progress.reset(
                    task_id,
                    total=event.total,
                    completed=0,
                    description=event.message,
                    visible=True,
                )
            if self.verbose:
                self.progress.log(f"[dim]{event.message}[/dim]")
            return

        task_id = self.tasks.get(event.name)
        if task_id is None:
            task_id = self.progress.add_task(event.message, total=event.total)
            self.tasks[event.name] = task_id

        update_kwargs: dict[str, object] = {"description": event.message}
        if event.total is not None:
            update_kwargs["total"] = event.total
        if event.completed is not None:
            update_kwargs["completed"] = event.completed
        if event.advance is not None:
            update_kwargs["advance"] = event.advance

        if event.kind == "task_done":
            task = self.progress.tasks[task_id]
            if task.total is not None:
                update_kwargs["completed"] = task.total
            self.progress.update(task_id, **update_kwargs)
            self.progress.stop_task(task_id)
            if self.verbose:
                self.progress.log(f"[green]{event.message}[/green]")
            return

        self.progress.update(task_id, **update_kwargs)
