"""Command line interface for bilibot."""

from __future__ import annotations

import argparse
import re as _re
import sys as _sys
from textwrap import dedent

from rich.console import Console

from .asr import ASR_PRESETS
from .cli_cmds import (
    EX_ERR,
    EX_INTERRUPT,
    EX_OK,
    run_doctor,
    run_download,
    run_asr,
    run_gen_notes_cmd,
    run_info,
    run_summarize,
)


console = Console()


def main(argv: list[str] | None = None) -> int:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    argv = _normalize_argv(list(_sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "summarize":
            return run_summarize(args)
        if args.command == "info":
            return run_info(args)
        if args.command == "doctor":
            return run_doctor(args)
        if args.command == "download":
            return run_download(args)
        if args.command == "asr":
            return run_asr(args)
        if args.command == "gen-notes":
            return run_gen_notes_cmd(args)
    except KeyboardInterrupt:
        console.print("\n[red]已中断[/red]")
        return EX_INTERRUPT
    except Exception as exc:
        console.print(f"[red]失败：{exc}[/red]")
        return EX_ERR

    parser.print_help()
    return EX_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bilibot",
        description="Fetch Bilibili transcripts and generate structured notes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """\
            examples:
              bilibot BVxxxx
              bilibot xxxx --no-notes
              bilibot summarize https://www.bilibili.com/video/BVxxxx/ --asr-preset accurate
              bilibot summarize BVxxxx --postprocess-subtitles --subtitle-postprocess-model deepseek-v4-pro
              bilibot asr ./meeting.m4a --asr-backend auto
              bilibot doctor
            """
        ),
    )
    parser.add_argument("-V", "--version", action="version", version="bilibot 0.2.0")
    subparsers = parser.add_subparsers(dest="command")

    # ── summarize ──────────────────────────────────────────────────────
    summarize = subparsers.add_parser("summarize", help="Analyze one Bilibili video")
    summarize.add_argument("video", nargs="+", metavar="VIDEO", help="Bilibili URL, BV id, or BV suffix")
    summarize.add_argument("--force-asr", action="store_true", help="Ignore Bilibili subtitles and run ASR")
    summarize.add_argument("--no-notes", action="store_true", help="Skip LLM note generation (subtitle postprocessing still runs)")
    summarize.add_argument("--no-llm", action="store_true", dest="no_notes", help=argparse.SUPPRESS)  # deprecated alias
    summarize.add_argument("--json", action="store_true", help="Output result summary as JSON")
    add_common_options(summarize)

    # ── info ───────────────────────────────────────────────────────────
    info = subparsers.add_parser("info", help="Fetch video metadata only")
    info.add_argument("video", nargs="+", metavar="VIDEO", help="Bilibili URL, BV id, or BV suffix")
    info.add_argument("--json", action="store_true", help="Print raw metadata JSON")

    # ── doctor ─────────────────────────────────────────────────────────
    doctor = subparsers.add_parser("doctor", help="Show local runtime and ASR recommendations")
    doctor.add_argument("--json", action="store_true", help="Print raw runtime JSON")

    # ── gen-notes ──────────────────────────────────────────────────────
    gen_notes = subparsers.add_parser("gen-notes", help="Generate notes from existing transcript (no re-extraction)")
    gen_notes.add_argument("bvid", metavar="BVID", help="BV id of a previously extracted video")
    gen_notes.add_argument("--output-dir", help="Output directory, default: data")
    gen_notes.add_argument("--json", action="store_true", help="Output result as JSON")

    # ── download ───────────────────────────────────────────────────────
    download = subparsers.add_parser("download", help="Download Bilibili video")
    download.add_argument("video", nargs="+", metavar="VIDEO", help="Bilibili URL, BV id, or BV suffix")
    download.add_argument("-o", "--output-dir", default=".", help="Output directory (default: current directory)")
    download.add_argument("-f", "--format", default=None, help="yt-dlp format selector (default: best up to 720P, no login needed)")
    download.add_argument("--cookie-file", default="", help="Netscape cookie file for premium (1080P+) access")
    download.add_argument("--merge-format", default="mp4", help="Container format after merging streams (default: mp4)")
    download.add_argument("--json", action="store_true", help="Output file path as JSON")

    # ── asr ────────────────────────────────────────────────────────────
    asr_cmd = subparsers.add_parser("asr", help="Transcribe local audio/video files")
    asr_cmd.add_argument("audio", nargs="+", metavar="AUDIO", help="Local audio/video file to transcribe")
    asr_cmd.add_argument("--json", action="store_true", help="Output artifact paths as JSON")
    add_common_options(asr_cmd)

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
    asr.add_argument("--asr-backend", choices=("auto", "whisper", "qwen3", "gguf"), help="ASR backend")
    asr.add_argument("--asr-gguf-model-dir", help="GGUF model directory (llama.cpp GGUF+mmproj or legacy ONNX+GGUF files)")
    asr.add_argument("--asr-gguf-cli", help="llama.cpp multimodal CLI path for GGUF+mmproj models")
    asr.add_argument("--asr-preset", choices=ASR_PRESETS, help="ASR preset, default: auto")
    asr.add_argument("--asr-model", help="ASR model name, HuggingFace path, or local dir")
    asr.add_argument("--whisper-model", dest="asr_model", help="Alias for --asr-model")
    asr.add_argument(
        "--forced-aligner-model",
        "--asr-forced-aligner-model",
        dest="asr_forced_aligner_model",
        help="Qwen3-ForcedAligner checkpoint path or HuggingFace model id",
    )
    asr.add_argument(
        "--timestamps",
        "--return-time-stamps",
        dest="asr_return_time_stamps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Return Qwen3 ForcedAligner word/character timestamps",
    )
    asr.add_argument("--asr-device", help="ASR device: cpu, cuda, or auto")
    asr.add_argument("--whisper-device", dest="asr_device", help="Alias for --asr-device")
    asr.add_argument("--asr-compute-type", help="ASR compute type, e.g. int8, float16, int8_float16")
    asr.add_argument("--whisper-compute-type", dest="asr_compute_type", help="Alias for --asr-compute-type")
    asr.add_argument("--asr-task", choices=("transcribe", "translate"), help="Whisper task")
    asr.add_argument("--asr-beam-size", type=int, help="ASR beam size")
    asr.add_argument("--asr-batch-size", type=int, help="Batched inference size; 0 disables batching")
    asr.add_argument("--asr-vad-filter", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable VAD filtering")
    asr.add_argument("--asr-vad-min-silence-ms", type=int, help="VAD min silence duration in ms")
    asr.add_argument("--asr-condition-on-previous-text", action=argparse.BooleanOptionalAction, default=None, help="Condition on previous text")
    asr.add_argument("--asr-cpu-threads", type=int, help="CTranslate2 CPU thread count")
    asr.add_argument("--asr-num-workers", type=int, help="CTranslate2 worker count")
    asr.add_argument("--asr-download-root", help="ASR model download/cache directory")
    asr.add_argument("--asr-local-files-only", action=argparse.BooleanOptionalAction, default=None, help="Only use local model files")
    asr.add_argument("--asr-hotwords", help="ASR hotwords")
    asr.add_argument("--asr-initial-prompt", help="ASR initial prompt")

    llm = parser.add_argument_group("llm")
    llm.add_argument("--llm-base-url", help="OpenAI-compatible base URL")
    llm.add_argument("--llm-api-key", help="OpenAI-compatible API key")
    llm.add_argument("--llm-model", help="LLM model name")
    llm.add_argument("--llm-fallback-models", help="Comma/space separated fallback model list")
    llm.add_argument("--llm-model-providers", help="Model to provider mapping, e.g. grok-4.6=grok,dsv4flash=deepseek")
    llm.add_argument("--llm-provider-base-urls", help="Provider base URL mapping, e.g. grok=https://.../v1")
    llm.add_argument("--llm-provider-api-keys", help="Provider API key mapping, e.g. grok=sk-...")
    llm.add_argument("--llm-timeout", type=float, help="LLM timeout in seconds")
    llm.add_argument("--llm-temperature", type=float, help="LLM temperature")
    llm.add_argument("--llm-max-tokens", type=int, help="LLM max output tokens")
    llm.add_argument("--llm-max-retries", type=int, help="Retry count for transient LLM/API failures")
    llm.add_argument("--llm-retry-base-delay", type=float, help="Initial retry delay in seconds")
    llm.add_argument("--llm-retry-max-delay", type=float, help="Maximum retry delay in seconds")
    llm.add_argument("--chunk-chars", type=int, help="Max transcript chars per LLM chunk")
    llm.add_argument("--summary-max-single-chunk-chars", type=int, help="Cap summary LLM chunk size; 0 disables cap")

    postprocess = parser.add_argument_group("subtitle postprocess")
    postprocess.add_argument("--postprocess-subtitles", dest="subtitle_postprocess", action="store_true", default=None, help="Enable LLM subtitle cleanup before note generation")
    postprocess.add_argument("--no-postprocess-subtitles", dest="subtitle_postprocess", action="store_false", help="Disable LLM subtitle cleanup")
    postprocess.add_argument("--subtitle-postprocess-base-url", help="Postprocess LLM base URL")
    postprocess.add_argument("--subtitle-postprocess-api-key", help="Postprocess LLM API key")
    postprocess.add_argument("--subtitle-postprocess-model", help="Postprocess LLM model")
    postprocess.add_argument("--subtitle-postprocess-temperature", type=float, help="Postprocess LLM temperature")
    postprocess.add_argument("--subtitle-postprocess-chunk-chars", type=int, help="Postprocess chunk size")
    postprocess.add_argument("--subtitle-postprocess-style", help="Postprocess style hint")


# ── internal helpers ────────────────────────────────────────────────────────

def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    commands = {"summarize", "info", "doctor", "download", "asr", "gen-notes", "-h", "--help", "-V", "--version"}
    if argv[0] not in commands:
        return ["summarize", *argv]
    return argv


def _video_input(value: list[str]) -> str:
    raw = " ".join(value).strip().strip("'\"“”‘’`<>")
    raw = _re.sub(r"[?&]spm_id_from=[^&\s]+", "", raw)
    raw = _re.sub(r"[?&]vd_source=[^&\s]+", "", raw)
    raw = _re.sub(r"&$", "", raw)
    raw = _re.sub(r"\?$", "", raw)
    return raw
