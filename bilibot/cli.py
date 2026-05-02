"""Command line interface for bilibot."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from textwrap import dedent

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

from .asr import ASR_PRESETS, WHISPER_MODEL_IDS, detect_runtime, resolve_asr_plan, resolve_backend
from .config import Settings, load_settings
from .extractor import extract
from .models import format_timestamp
from .pipeline import PipelineResult, analyze_url
from .progress import ProgressEvent
from .storage import _metadata_payload


console = Console()


def main(argv: list[str] | None = None) -> int:
    argv = _normalize_argv(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "summarize":
            return run_summarize(args)
        if args.command == "info":
            return run_info(args)
        if args.command == "doctor":
            return run_doctor(args)
    except KeyboardInterrupt:
        console.print("\n[red]已中断[/red]")
        return 130
    except Exception as exc:
        console.print(f"[red]失败：{exc}[/red]")
        return 1

    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bilibot",
        description="Fetch Bilibili transcripts and generate structured notes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """\
            examples:
              bilibot BV1WdokBNEcn
              bilibot 1WdokBNEcn --no-llm
              bilibot summarize https://www.bilibili.com/video/BV1WdokBNEcn/ --asr-preset accurate
              bilibot summarize BV1WdokBNEcn --postprocess-subtitles --subtitle-postprocess-model deepseek-v4-pro
              bilibot doctor
            """
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    summarize = subparsers.add_parser("summarize", help="Analyze one Bilibili video")
    summarize.add_argument("video", nargs="+", metavar="VIDEO", help="Bilibili URL, BV id, or BV suffix")
    summarize.add_argument("--force-asr", action="store_true", help="Ignore Bilibili subtitles and run ASR")
    summarize.add_argument("--no-llm", action="store_true", help="Only save metadata and transcript")
    add_common_options(summarize)

    info = subparsers.add_parser("info", help="Fetch video metadata only")
    info.add_argument("video", nargs="+", metavar="VIDEO", help="Bilibili URL, BV id, or BV suffix")
    info.add_argument("--json", action="store_true", help="Print raw metadata JSON")

    doctor = subparsers.add_parser("doctor", help="Show local runtime and ASR recommendations")
    doctor.add_argument("--json", action="store_true", help="Print raw runtime JSON")

    return parser


def add_common_options(parser: argparse.ArgumentParser) -> None:
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--output-dir", help="Output directory, default: data")
    runtime.add_argument("--cookie-file", help="Bilibili/yt-dlp cookie file for audio download")
    runtime.add_argument("--language", help="Transcript language hint, default: zh")
    runtime.add_argument("--verbose", action="store_true", help="Show detailed progress logs")
    runtime.add_argument("--quiet", action="store_true", help="Disable progress display")

    download = parser.add_argument_group("download")
    download.add_argument("--download-timeout", type=float, help="HTTP download timeout in seconds")
    download.add_argument("--download-chunk-size", type=int, help="HTTP download chunk size in bytes")
    download.add_argument("--yt-dlp-format", help="yt-dlp format selector, default: bestaudio")
    download.add_argument("--yt-dlp-audio-format", help="yt-dlp extracted audio format, default: mp3")
    download.add_argument("--yt-dlp-audio-quality", help="yt-dlp audio quality, default: 5")

    asr = parser.add_argument_group("asr")
    asr.add_argument("--asr-backend", choices=("auto", "whisper", "qwen3", "gguf"), help="ASR backend: auto (detect), whisper, qwen3, gguf")
    asr.add_argument("--asr-gguf-model-dir", help="GGUF model directory (ONNX + GGUF files)")
    asr.add_argument("--asr-preset", choices=ASR_PRESETS, help="ASR preset, default: auto")
    asr.add_argument("--asr-model", help="ASR model name, HuggingFace path, or local dir")
    asr.add_argument("--whisper-model", dest="asr_model", help="Alias for --asr-model")
    asr.add_argument("--asr-device", help="ASR device: cpu, cuda, or auto")
    asr.add_argument("--whisper-device", dest="asr_device", help="Alias for --asr-device")
    asr.add_argument("--asr-compute-type", help="ASR compute type, e.g. int8, float16, int8_float16")
    asr.add_argument("--whisper-compute-type", dest="asr_compute_type", help="Alias for --asr-compute-type")
    asr.add_argument("--asr-task", choices=("transcribe", "translate"), help="Whisper task")
    asr.add_argument("--asr-beam-size", type=int, help="ASR beam size")
    asr.add_argument("--asr-batch-size", type=int, help="Batched inference size; 0 disables batching")
    asr.add_argument(
        "--asr-vad-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Silero VAD filtering",
    )
    asr.add_argument("--asr-vad-min-silence-ms", type=int, help="VAD min silence duration in ms")
    asr.add_argument(
        "--asr-condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Condition ASR on previous text",
    )
    asr.add_argument("--asr-cpu-threads", type=int, help="CTranslate2 CPU thread count")
    asr.add_argument("--asr-num-workers", type=int, help="CTranslate2 worker count")
    asr.add_argument("--asr-download-root", help="ASR model download/cache directory")
    asr.add_argument(
        "--asr-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Only load local ASR model files",
    )
    asr.add_argument("--asr-hotwords", help="ASR hotwords")
    asr.add_argument("--asr-initial-prompt", help="ASR initial prompt")

    llm = parser.add_argument_group("llm")
    llm.add_argument("--llm-base-url", help="OpenAI-compatible base URL")
    llm.add_argument("--llm-api-key", help="OpenAI-compatible API key")
    llm.add_argument("--llm-model", help="LLM model name")
    llm.add_argument("--llm-timeout", type=float, help="LLM timeout in seconds")
    llm.add_argument("--llm-temperature", type=float, help="LLM temperature")
    llm.add_argument("--llm-max-tokens", type=int, help="LLM max output tokens")
    llm.add_argument("--chunk-chars", type=int, help="Max transcript chars per LLM chunk")

    postprocess = parser.add_argument_group("subtitle postprocess")
    postprocess.add_argument(
        "--postprocess-subtitles",
        dest="subtitle_postprocess",
        action="store_true",
        default=None,
        help="Enable LLM subtitle cleanup before note generation",
    )
    postprocess.add_argument(
        "--no-postprocess-subtitles",
        dest="subtitle_postprocess",
        action="store_false",
        help="Disable LLM subtitle cleanup",
    )
    postprocess.add_argument("--subtitle-postprocess-base-url", help="Postprocess LLM base URL")
    postprocess.add_argument("--subtitle-postprocess-api-key", help="Postprocess LLM API key")
    postprocess.add_argument("--subtitle-postprocess-model", help="Postprocess LLM model")
    postprocess.add_argument("--subtitle-postprocess-temperature", type=float, help="Postprocess LLM temperature")
    postprocess.add_argument("--subtitle-postprocess-chunk-chars", type=int, help="Postprocess chunk size")
    postprocess.add_argument("--subtitle-postprocess-style", help="Postprocess style hint")


def run_summarize(args: argparse.Namespace) -> int:
    video = _video_input(args.video)
    settings = _settings_from_args(args)
    reporter = None if args.quiet else RichProgressReporter(console, verbose=args.verbose)

    if reporter is None:
        result = analyze_url(video, settings, force_asr=args.force_asr, no_llm=args.no_llm)
    else:
        with reporter:
            result = analyze_url(
                video,
                settings,
                force_asr=args.force_asr,
                no_llm=args.no_llm,
                progress=reporter,
            )

    _print_result(result)
    return 0


def run_info(args: argparse.Namespace) -> int:
    video = _video_input(args.video)
    settings = load_settings()
    info = extract(
        video,
        sessdata=settings.bili_sessdata,
        bili_jct=settings.bili_jct,
        buvid3=settings.bili_buvid3,
    )
    payload = _metadata_payload(info)
    if args.json:
        console.print_json(data=payload)
        return 0

    table = Table(title=info.title or info.bvid, show_header=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("BV", info.bvid)
    table.add_row("作者", info.author or "未知")
    table.add_row("时长", format_timestamp(float(info.duration)))
    table.add_row("字幕轨道", str(len(info.subtitles)))
    table.add_row("链接", info.url)
    console.print(table)
    return 0


def run_doctor(args: argparse.Namespace) -> int:
    runtime = detect_runtime()
    settings = load_settings()
    if args.json:
        console.print_json(
            data={
                "cpu_model": runtime.cpu_model,
                "cpu_count": runtime.cpu_count,
                "memory_total_gb": runtime.memory_total_gb,
                "memory_available_gb": runtime.memory_available_gb,
                "cuda_device_count": runtime.cuda_device_count,
                "gpus": [gpu.__dict__ for gpu in runtime.gpus],
                "cpu_compute_types": runtime.cpu_compute_types,
                "cuda_compute_types": runtime.cuda_compute_types,
            }
        )
        return 0

    runtime_table = Table(title="Runtime", show_header=False)
    runtime_table.add_column("Field", style="cyan", no_wrap=True)
    runtime_table.add_column("Value")
    runtime_table.add_row("CPU", runtime.cpu_model)
    runtime_table.add_row("CPU cores", str(runtime.cpu_count))
    runtime_table.add_row(
        "Memory",
        f"{runtime.memory_available_gb:g}GiB available / {runtime.memory_total_gb:g}GiB total",
    )
    runtime_table.add_row("CUDA devices", str(runtime.cuda_device_count))
    runtime_table.add_row("CPU compute", ", ".join(runtime.cpu_compute_types) or "unknown")
    runtime_table.add_row("CUDA compute", ", ".join(runtime.cuda_compute_types) or "unavailable")
    for index, gpu in enumerate(runtime.gpus, start=1):
        runtime_table.add_row(
            f"GPU {index}",
            f"{gpu.name}, {gpu.memory_mb}MiB, driver {gpu.driver_version}",
        )

    preset_table = Table(title="ASR Presets")
    preset_table.add_column("Preset", style="cyan", no_wrap=True)
    preset_table.add_column("Model")
    preset_table.add_column("Device")
    preset_table.add_column("Compute")
    preset_table.add_column("Batch")
    preset_table.add_column("VAD")
    preset_table.add_column("Reason")
    gpu_mb = max((g.memory_mb for g in runtime.gpus), default=0)
    for preset in ASR_PRESETS:
        preset_settings = replace(
            settings,
            asr_preset=preset,
            asr_model="",
            asr_device="",
            asr_compute_type="",
            asr_batch_size=None,
            asr_vad_filter=None,
        )
        plan = resolve_asr_plan(preset_settings, runtime)
        reason = plan.reason
        if preset == "auto":
            backend = resolve_backend(settings, gpu_mb)
            reason += f" || 运行时 auto 后端 = {backend}"
        preset_table.add_row(
            preset,
            plan.model,
            plan.device,
            plan.compute_type,
            str(plan.batch_size),
            "on" if plan.vad_filter else "off",
            reason,
        )

    console.print(runtime_table)
    console.print(preset_table)

    backend = resolve_backend(settings, gpu_mb)
    console.print(
        Panel(
            f"显存 {gpu_mb}MB → auto 后端 = [bold]{backend}[/bold]"
            + (f" (低于 {QWEN3_1_7B_MIN_VRAM_MB}MB, 回退 whisper)" if backend == "whisper" and gpu_mb else ""),
            title="Backend Selection",
        )
    )

    model_table = Table(title="Available Models")
    model_table.add_column("Backend", style="cyan", no_wrap=True)
    model_table.add_column("Model ID")
    model_table.add_column("Description")
    for mid, hf_id in WHISPER_MODEL_IDS.items():
        model_table.add_row("whisper", mid, hf_id)
    from .qwen_asr_backend import QWEN3_MODEL_IDS as qwen_ids
    for mid, hf_id in qwen_ids.items():
        model_table.add_row("qwen3", mid, hf_id)
    from .gguf_asr_backend import GGUF_MODEL_IDS as gguf_ids, _check_available as gguf_ok
    for mid, desc in gguf_ids.items():
        status = "OK" if gguf_ok(settings) else "需安装"
        model_table.add_row("gguf", mid, f"{desc} [{status}]")
    console.print(model_table)

    return 0


class RichProgressReporter:
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


def _settings_from_args(args: argparse.Namespace) -> Settings:
    return load_settings(
        output_dir=args.output_dir,
        cookie_file=args.cookie_file,
        language=args.language,
        download_timeout=args.download_timeout,
        download_chunk_size=args.download_chunk_size,
        yt_dlp_format=args.yt_dlp_format,
        yt_dlp_audio_format=args.yt_dlp_audio_format,
        yt_dlp_audio_quality=args.yt_dlp_audio_quality,
        asr_backend=args.asr_backend,
        asr_preset=args.asr_preset,
        asr_model=args.asr_model,
        asr_device=args.asr_device,
        asr_compute_type=args.asr_compute_type,
        asr_task=args.asr_task,
        asr_beam_size=args.asr_beam_size,
        asr_batch_size=args.asr_batch_size,
        asr_vad_filter=args.asr_vad_filter,
        asr_vad_min_silence_ms=args.asr_vad_min_silence_ms,
        asr_condition_on_previous_text=args.asr_condition_on_previous_text,
        asr_cpu_threads=args.asr_cpu_threads,
        asr_num_workers=args.asr_num_workers,
        asr_download_root=args.asr_download_root,
        asr_local_files_only=args.asr_local_files_only,
        asr_hotwords=args.asr_hotwords,
        asr_initial_prompt=args.asr_initial_prompt,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_timeout=args.llm_timeout,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        chunk_chars=args.chunk_chars,
        subtitle_postprocess=args.subtitle_postprocess,
        subtitle_postprocess_base_url=args.subtitle_postprocess_base_url,
        subtitle_postprocess_api_key=args.subtitle_postprocess_api_key,
        subtitle_postprocess_model=args.subtitle_postprocess_model,
        subtitle_postprocess_temperature=args.subtitle_postprocess_temperature,
        subtitle_postprocess_chunk_chars=args.subtitle_postprocess_chunk_chars,
        subtitle_postprocess_style=args.subtitle_postprocess_style,
    )


def _print_result(result: PipelineResult) -> None:
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


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    commands = {"summarize", "info", "doctor", "-h", "--help"}
    if argv[0] not in commands:
        return ["summarize", *argv]
    return argv


def _video_input(value: list[str]) -> str:
    import re

    raw = " ".join(value).strip().strip("'\"“”‘’`<>")
    raw = re.sub(r"[?&]spm_id_from=[^&\s]+", "", raw)
    raw = re.sub(r"[?&]vd_source=[^&\s]+", "", raw)
    raw = re.sub(r"&$", "", raw)
    raw = re.sub(r"\?$", "", raw)
    return raw
