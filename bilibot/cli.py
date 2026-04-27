"""Command line interface for bilibot."""

from __future__ import annotations

import argparse
import sys
from textwrap import dedent

from rich.console import Console
from rich.panel import Panel

from .config import load_settings
from .extractor import extract
from .pipeline import analyze_url
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
        description="Fetch Bilibili video transcripts and generate LLM notes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """\
            examples:
              bilibot BV1WdokBNEcn
              bilibot 1WdokBNEcn
              bilibot summarize https://www.bilibili.com/video/BV1WdokBNEcn/
              bilibot summarize BV1WdokBNEcn --no-llm
            """
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    summarize = subparsers.add_parser("summarize", help="Analyze one Bilibili video")
    summarize.add_argument("video", nargs="+", metavar="VIDEO", help="Bilibili URL, BV id, or BV suffix")
    add_common_options(summarize)
    summarize.add_argument("--force-asr", action="store_true", help="Ignore Bilibili subtitles and run ASR")
    summarize.add_argument("--no-llm", action="store_true", help="Only save metadata and transcript")

    info = subparsers.add_parser("info", help="Fetch video metadata only")
    info.add_argument("video", nargs="+", metavar="VIDEO", help="Bilibili URL, BV id, or BV suffix")

    return parser


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", help="Output directory, default: data")
    parser.add_argument("--cookie-file", help="Bilibili/yt-dlp cookie file for audio download")
    parser.add_argument("--language", help="Transcript language hint, default: zh")
    parser.add_argument("--whisper-model", help="faster-whisper model, default: base")
    parser.add_argument("--whisper-device", help="faster-whisper device, default: cpu")
    parser.add_argument("--whisper-compute-type", help="faster-whisper compute type, default: int8")
    parser.add_argument("--llm-base-url", help="OpenAI-compatible base URL")
    parser.add_argument("--llm-api-key", help="OpenAI-compatible API key")
    parser.add_argument("--llm-model", help="LLM model name")
    parser.add_argument("--chunk-chars", type=int, help="Max transcript chars per LLM chunk")


def run_summarize(args: argparse.Namespace) -> int:
    video = _video_input(args.video)
    settings = load_settings(
        output_dir=args.output_dir,
        cookie_file=args.cookie_file,
        language=args.language,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_compute_type=args.whisper_compute_type,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        chunk_chars=args.chunk_chars,
    )

    def report(message: str) -> None:
        console.print(f"[cyan]•[/cyan] {message}")

    result = analyze_url(
        video,
        settings,
        force_asr=args.force_asr,
        no_llm=args.no_llm,
        progress=report,
    )

    lines = [
        f"[bold]{result.info.title or result.info.bvid}[/bold]",
        "",
        f"字幕来源: {result.transcript.source}",
        f"字幕段落: {len(result.transcript.segments)}",
        "",
    ]
    lines.extend(f"{name}: {path}" for name, path in result.paths.items())
    console.print(Panel("\n".join(lines), title="完成"))
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
    console.print_json(data=_metadata_payload(info))
    return 0


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    commands = {"summarize", "info", "-h", "--help"}
    if argv[0] not in commands:
        return ["summarize", *argv]
    return argv


def _video_input(value: list[str]) -> str:
    return " ".join(value).strip().strip("'\"“”‘’`<>")
