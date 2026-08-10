"""Command handlers for each bilibot subcommand."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .asr import ASR_PRESETS, QWEN3_1_7B_MIN_VRAM_MB, WHISPER_MODEL_IDS, detect_runtime, resolve_asr_plan, resolve_backend
from .cli_ui import RichProgressReporter, print_result, print_result_json
from .config import Settings, load_settings
from .downloader import DEFAULT_FORMAT, download_video
from .extractor import extract
from .gen_notes import run as run_gen_notes
from .models import Transcript, format_timestamp, jsonable
from .pipeline import analyze_url
from .storage import _metadata_payload, save_local_transcript_artifacts


# ── exit codes (follow sysexits.h conventions loosely) ──────────────────────
EX_OK = 0
EX_ERR = 1
EX_INTERRUPT = 130


def _json_err(message: str, console: Console) -> int:
    """Print a JSON error line and return EX_ERR."""
    console.print_json(data={"error": True, "message": message})
    return EX_ERR


# ── summarize ───────────────────────────────────────────────────────────────

def run_summarize(args: argparse.Namespace) -> int:
    from .cli import _video_input, console

    video = _video_input(args.video)
    settings = _settings_from_args(args)
    reporter = None if args.quiet else RichProgressReporter(console, verbose=args.verbose)

    try:
        if reporter is None:
            result = analyze_url(video, settings, force_asr=args.force_asr, no_notes=args.no_notes)
        else:
            with reporter:
                result = analyze_url(
                    video, settings,
                    force_asr=args.force_asr, no_notes=args.no_notes,
                    progress=reporter,
                )

        if args.json:
            print_result_json(result)
        else:
            print_result(result)

        # Also print artifact paths as one-liners for easy agent consumption
        if args.quiet and not args.json:
            for path in result.paths.values():
                console.print(str(path))
        return EX_OK

    except Exception as exc:
        if args.json:
            return _json_err(str(exc), console)
        console.print(f"[red]失败：{exc}[/red]")
        return EX_ERR


# ── info ────────────────────────────────────────────────────────────────────

def run_info(args: argparse.Namespace) -> int:
    from .cli import _video_input, console

    video = _video_input(args.video)
    try:
        settings = load_settings()
        info = extract(
            video,
            sessdata=settings.bili_sessdata,
            bili_jct=settings.bili_jct,
            buvid3=settings.bili_buvid3,
        )
    except Exception as exc:
        if args.json:
            return _json_err(str(exc), console)
        console.print(f"[red]获取信息失败：{exc}[/red]")
        return EX_ERR

    payload = _metadata_payload(info)
    if args.json:
        console.print_json(data=payload)
        return EX_OK

    table = Table(title=info.title or info.bvid, show_header=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("BV", info.bvid)
    table.add_row("作者", info.author or "未知")
    table.add_row("时长", format_timestamp(float(info.duration)))
    table.add_row("字幕轨道", str(len(info.subtitles)))
    table.add_row("链接", info.url)
    console.print(table)
    return EX_OK


# ── local ASR ───────────────────────────────────────────────────────────────

def run_asr(args: argparse.Namespace) -> int:
    from .cli import console

    settings = _settings_from_args(args)
    reporter = None if args.quiet else RichProgressReporter(console, verbose=args.verbose)
    output_base = settings.output_dir / "local_asr"

    try:
        paths_by_audio: dict[Path, dict[str, Path]] = {}
        if reporter is None:
            for audio in args.audio:
                audio_path = _resolve_audio_path(audio)
                transcript = _transcribe_local_audio(audio_path, settings)
                paths_by_audio[audio_path] = save_local_transcript_artifacts(output_base, audio_path, transcript)
        else:
            with reporter:
                for audio in args.audio:
                    audio_path = _resolve_audio_path(audio)
                    transcript = _transcribe_local_audio(audio_path, settings, progress=reporter)
                    paths_by_audio[audio_path] = save_local_transcript_artifacts(output_base, audio_path, transcript)

        if args.json:
            console.print_json(
                data={
                    str(audio): {name: str(path) for name, path in paths.items()}
                    for audio, paths in paths_by_audio.items()
                }
            )
        else:
            _print_asr_result(paths_by_audio, console)
        return EX_OK
    except Exception as exc:
        if args.json:
            return _json_err(str(exc), console)
        console.print(f"[red]本地转写失败：{exc}[/red]")
        return EX_ERR


# ── doctor ──────────────────────────────────────────────────────────────────

def run_doctor(args: argparse.Namespace) -> int:
    from .cli import console

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
        return EX_OK

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

    return EX_OK


# ── gen-notes ───────────────────────────────────────────────────────────────

def run_gen_notes_cmd(args: argparse.Namespace) -> int:
    from .cli import console

    settings = load_settings(output_dir=args.output_dir) if args.output_dir else load_settings()
    try:
        path = run_gen_notes(args.bvid, settings)
        if args.json:
            console.print_json(data={"bvid": args.bvid, "notes_path": str(path)})
        else:
            console.print(f"[green]笔记已保存到：[/green]{path}")
        return EX_OK
    except FileNotFoundError as exc:
        if args.json:
            return _json_err(str(exc), console)
        console.print(f"[yellow]{exc}[/yellow]")
        return EX_ERR
    except Exception as exc:
        if args.json:
            return _json_err(str(exc), console)
        console.print(f"[red]笔记生成失败：{exc}[/red]")
        return EX_ERR


# ── download ────────────────────────────────────────────────────────────────

def run_download(args: argparse.Namespace) -> int:
    from .cli import _video_input, console

    video = _video_input(args.video)
    fmt = args.format or DEFAULT_FORMAT
    try:
        path = download_video(
            video,
            output_dir=args.output_dir,
            fmt=fmt,
            cookie_file=args.cookie_file,
            merge_output_format=args.merge_format,
        )
        if args.json:
            console.print_json(data={"file": str(path)})
        else:
            console.print(f"[green]下载完成：[/green]{path}")
        return EX_OK
    except Exception as exc:
        if args.json:
            return _json_err(str(exc), console)
        console.print(f"[red]下载失败：{exc}[/red]")
        return EX_ERR


def _print_asr_result(paths_by_audio: dict[Path, dict[str, Path]], console: Console) -> None:
    console.print(Panel(f"已转写 {len(paths_by_audio)} 个本地音频文件", title="完成", border_style="green"))
    artifacts = Table(title="Artifacts")
    artifacts.add_column("Audio", style="cyan", no_wrap=True)
    artifacts.add_column("Name", style="cyan", no_wrap=True)
    artifacts.add_column("Path")
    for audio_path, paths in paths_by_audio.items():
        for name, path in paths.items():
            artifacts.add_row(audio_path.name, name, str(path))
    console.print(artifacts)


def _resolve_audio_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"本地音频文件不存在：{path}")
    return path.resolve()


def _transcribe_local_audio(audio_path: Path, settings: Settings, *, progress=None) -> Transcript:
    runtime = detect_runtime()
    gpu_mb = max((gpu.memory_mb for gpu in runtime.gpus), default=0)
    backend = resolve_backend(settings, gpu_mb)

    if backend == "gguf":
        from .gguf_asr_backend import transcribe

        return transcribe(str(audio_path), settings, progress=progress)
    if backend == "qwen3":
        from .qwen_asr_backend import transcribe

        return transcribe(str(audio_path), settings, progress=progress)

    from .transcriber import transcribe

    plan = resolve_asr_plan(settings, runtime)
    return transcribe(str(audio_path), settings, plan=plan, progress=progress)


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
