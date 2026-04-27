# Repository Guidelines

## Project Structure & Module Organization

`bilibot/` contains the Python package and CLI. Important modules include `cli.py` for argument parsing, `pipeline.py` for the end-to-end workflow, `extractor.py` for Bilibili metadata/subtitles, `transcriber.py` for Whisper fallback ASR, `summarizer.py` and `llm.py` for note generation, `storage.py` for writing artifacts, and `models.py` for shared dataclasses. `README.md` documents user-facing behavior. Runtime outputs are written to `data/{bvid}/` and are ignored by git. Local configuration belongs in `.env`; keep `.env.example` as the committed template.

## Build, Test, and Development Commands

- `uv sync`: create/update the local virtual environment from `pyproject.toml` and `uv.lock`.
- `uv run bilibot --help`: verify the installed CLI entry point.
- `uv run bilibot summarize BVxxxx --no-llm`: smoke-test metadata and transcript extraction without calling an LLM.
- `uv run bilibot info BVxxxx`: fetch and print video metadata only.
- `uv build`: build the package with Hatchling.
- `uv run python -m compileall bilibot`: quick syntax check when no test suite is present.

## Coding Style & Naming Conventions

Use Python 3.12 features consistently, including `from __future__ import annotations`, dataclasses, and modern type hints such as `list[str]` and `Path | None`. Follow PEP 8 with 4-space indentation. Use `snake_case` for functions, variables, and modules; use `PascalCase` for dataclasses and other types. Keep CLI text concise and user-facing messages compatible with Rich output. Prefer small functions that keep extraction, transcription, LLM, and storage responsibilities separated.

## Testing Guidelines

There is currently no committed test suite or pytest dependency. For new behavior, add tests under `tests/` using `test_*.py` file names and focused fixtures. Prioritize deterministic unit tests for parsing, subtitle selection, timestamp formatting, and storage serialization. Avoid tests that require live Bilibili, Whisper model downloads, or LLM calls unless they are explicitly marked as integration tests and skipped by default.

## Commit & Pull Request Guidelines

This repository has no existing commit history, so use short imperative commit subjects going forward, for example `Add subtitle parsing tests` or `Handle missing audio URLs`. Keep each commit scoped to one concern. Pull requests should describe the behavior change, list verification commands run, note any configuration or dependency changes, and include sample CLI output when the user-facing command behavior changes.

## Security & Configuration Tips

Do not commit `.env`, cookies, generated audio, or `data/` artifacts. Treat `LLM_API_KEY`, Bilibili cookies, and local service URLs as secrets. When adding configuration, update `.env.example` and document the option in `README.md`.
