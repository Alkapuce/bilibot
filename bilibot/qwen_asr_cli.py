"""Standalone CLI entrypoint for local Qwen3-ASR transcription."""

from __future__ import annotations

import sys

from .cli import main as bilibot_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return bilibot_main(["asr", *args, "--asr-backend", "qwen3"])


if __name__ == "__main__":
    raise SystemExit(main())
