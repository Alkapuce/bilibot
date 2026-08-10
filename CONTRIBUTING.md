# Contributing to bilibot

Thanks for your interest! Here's how to contribute.

## Setup

```bash
git clone https://github.com/anomalyco/bilibot
cd bilibot
uv sync
```

## Development workflow

1. Fork the repo and create a branch from `master`.
2. Make your changes. Keep each commit focused on a single concern.
3. Run the syntax check before committing:

   ```bash
   uv run python -m compileall bilibot
   ```

4. Test manually with a public Bilibili video:

   ```bash
   uv run bilibot doctor                     # check environment
   uv run bilibot info BVxxxx                # test metadata extraction
   uv run bilibot summarize BVxxxx --no-notes  # test transcript extraction
   ```

5. Open a PR with a clear description of the change.

## Code style

- Python 3.12+, `from __future__ import annotations`
- 4-space indentation, `snake_case` functions, `PascalCase` dataclasses
- Use `pathlib.Path` for all filesystem paths (cross-platform)
- Chinese for user-facing messages, English for code identifiers

## Commit conventions

```
bilibot: fix path handling on Windows
cli: add --json flag
downloader: refactor audio download into downloader module
```

## Adding new features

- New CLI flags → add to `cli.py` parser, wire in `cli_cmds.py`
- New config keys → add to `config.py:Settings`, update `.env.example`, document in `README.md`
- New ASR backend → add module, register in `transcriber.py:transcribe_url()`
- All code must work on both Linux and Windows

## Questions?

Open an issue or start a discussion.
