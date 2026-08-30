# bilibot · B站视频笔记助手

[English](#english) | [中文](#chinese)

---

<a id="english"></a>
## English

**bilibot** is a CLI tool that extracts Bilibili video metadata and subtitles, then calls an LLM to generate structured Markdown study notes.

Three local ASR backends are supported: **Qwen3-ASR GGUF**, **Qwen3-ASR (HuggingFace)**, and **Whisper (faster-whisper)**. The `auto` mode selects the best available one.

### Quick Start

```bash
git clone https://github.com/anomalyco/bilibot
cd bilibot
uv sync

# Check your environment
uv run bilibot doctor

# Analyze a video (auto selects best ASR backend)
uv run bilibot summarize https://www.bilibili.com/video/BVxxxx/
```

You can also pass just the BV ID:

```bash
uv run bilibot BVxxxx
uv run bilibot 1xxxxxxxxx
```

> Wrap URLs with `?` or `&` in quotes to prevent shell interpretation.

### Setup

Copy `.env.example` to `.env` and fill in your LLM endpoints:

```env
# GPT-5.5 uses a provider exposing the OpenAI Responses API.
# Keep this endpoint/key in the local, ignored .env file.
LLM_MODEL=gpt-5.5
LLM_FALLBACK_MODELS=
LLM_MODEL_PROVIDERS=gpt-5.5=responses_provider
LLM_PROVIDER_RESPONSES_PROVIDER_WIRE_API=responses
```

`LLM_MODEL` is the primary model, and `LLM_FALLBACK_MODELS` is an ordered comma- or space-separated fallback chain. Provider-specific endpoint/key values are read from `LLM_PROVIDER_<NAME>_BASE_URL` and `LLM_PROVIDER_<NAME>_API_KEY`; keep real values in the ignored `.env` file. Set the selected provider's `LLM_PROVIDER_<NAME>_WIRE_API=responses` for a GPT-5.5-compatible Responses provider. Other providers continue to use Chat Completions by default.

### ASR Backends

| Priority | Backend | Requirement | Size |
|----------|---------|-------------|------|
| 1 | **qwen3** (HuggingFace BF16) | ≥6 GB VRAM + safetensors model | ~4.4 GB |
| 2 | **whisper** (faster-whisper) | Always available | 0.15–3 GB |
| 3 | **gguf** (llama.cpp) | Optional legacy path: `llama-mtmd-cli` + GGUF/mmproj | ~2.4 GB |

#### Qwen3 BF16 (recommended)

```bash
uv sync --extra qwen3
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir .models/Qwen3-ASR-1.7B
uv run bilibot summarize BVxxxx --asr-backend qwen3 --asr-model .models/Qwen3-ASR-1.7B
uv run qwen3-asr ./meeting.m4a --asr-model .models/Qwen3-ASR-1.7B
```

To return word/character timestamps, download `Qwen/Qwen3-ForcedAligner-0.6B` and pass both checkpoints:

```bash
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir .models/Qwen3-ForcedAligner-0.6B
uv run qwen3-asr ./meeting.m4a \\
  --asr-model .models/Qwen3-ASR-1.7B \\
  --forced-aligner-model .models/Qwen3-ForcedAligner-0.6B \\
  --timestamps
```

The normal transcript artifacts remain available. When timestamps are returned, an additional `timestamps.json` is written next to `transcript.json`; each item has `text`, `start`, and `end` in seconds. `--timestamps` can also discover `.models/Qwen3-ForcedAligner-0.6B` from the project directory. Forced alignment supports audio chunks up to five minutes; longer files are split and timestamp offsets are restored.

#### GGUF (optional legacy path)

```bash
uv sync --extra gguf
export ASR_GGUF_MODEL_DIR=/path/to/Qwen3-ASR-1.7B-GGUF
export ASR_GGUF_CLI=/path/to/llama.cpp/build/bin/llama-mtmd-cli
uv run bilibot summarize BVxxxx --asr-backend gguf
```

#### Whisper

```bash
uv run bilibot summarize BVxxxx --asr-backend whisper --asr-preset accurate
```

### Commands

```bash
uv run bilibot summarize <url>                  # Full pipeline
uv run bilibot summarize <url> --no-notes         # Extract transcript only (skip notes, postprocess still runs)
uv run bilibot summarize <url> --force-asr      # Skip Bilibili subtitles
uv run bilibot summarize <url> --json           # JSON output (for scripts/agents)
uv run bilibot summarize <url> --quiet          # Only print file paths
uv run bilibot info <url>                       # Metadata only
uv run bilibot info <url> --json                # Metadata as JSON
uv run bilibot download <url>                   # Download video
uv run bilibot download <url> --json            # Output file path as JSON
uv run bilibot asr ./meeting.m4a                # Transcribe local audio/video
uv run qwen3-asr ./meeting.m4a                  # Transcribe with local Qwen3-ASR
uv run qwen3-asr ./meeting.m4a --timestamps    # Transcribe with word/character timestamps
uv run bilibot asr ./meeting.m4a --json         # Output artifact paths as JSON
uv run bilibot gen-notes <bvid>                 # Regenerate notes from cache
uv run bilibot doctor                           # System check
uv run bilibot doctor --json                    # System check as JSON
```

### Output

All files go to `data/{bvid}/`:

| File | Description |
|------|-------------|
| `{title}_信息.json` | Video metadata |
| `{title}_字幕.json` | Structured transcript |
| `{title}_字幕.txt` | Plain-text transcript |
| `{title}_笔记.md` | LLM-generated notes |

Local audio transcription writes to `data/local_asr/{audio-stem}/` with `transcript.json`, `transcript.md`, and `captions.txt`. With `--timestamps`, it also writes `timestamps.json` and stores the same records under `time_stamps` in `transcript.json`.

### Requirements

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) (auto-installed; system package recommended)

### Tech Stack

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper ASR
- [Qwen3-ASR GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF) — llama.cpp GGUF ASR
- [qwen-asr](https://github.com/QwenLM/Qwen3-ASR) — Qwen3 HuggingFace ASR
- [bilibili-api-python](https://github.com/Nemo2011/bilibili-api) — Bilibili API
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — Media download
- [OpenAI SDK](https://github.com/openai/openai-python) — LLM client
- [rich](https://github.com/Textualize/rich) — Terminal UI

### License

MIT — see [LICENSE](LICENSE).

---

<a id="chinese"></a>
## 中文

**bilibot** 是 B 站视频笔记助手 CLI 工具：获取视频元数据与字幕，调用大模型生成结构化 Markdown 笔记。

支持三种本地语音识别后端：**Qwen3-ASR GGUF**、**Qwen3-ASR (HuggingFace)**、**Whisper (faster-whisper)**。`auto` 模式自动选择最优方案。

### 快速开始

```bash
git clone https://github.com/anomalyco/bilibot
cd bilibot
uv sync

# 检查本机环境
uv run bilibot doctor

# 使用（auto 自动选择最适合本机的 ASR 后端）
uv run bilibot summarize https://www.bilibili.com/video/BVxxxx/
```

也支持直接传 BV 号：

```bash
uv run bilibot BVxxxx
uv run bilibot 1xxxxxxxxx
```

> 带 `?`、`&` 参数的链接请用引号包裹，避免 shell 误解析。

### 配置

复制 `.env.example` 为 `.env`，填入 LLM 接口地址和密钥：

```env
# GPT-5.5 使用支持 OpenAI Responses API 的 provider。
# 真实 endpoint/key 放在不会提交的 .env 中。
LLM_MODEL=gpt-5.5
LLM_FALLBACK_MODELS=
LLM_MODEL_PROVIDERS=gpt-5.5=responses_provider
LLM_PROVIDER_RESPONSES_PROVIDER_WIRE_API=responses
```

`LLM_MODEL` 是主模型，`LLM_FALLBACK_MODELS` 是按顺序尝试的 fallback 链路，支持逗号或空格分隔。provider 的 endpoint/key 通过 `LLM_PROVIDER_<NAME>_BASE_URL` / `LLM_PROVIDER_<NAME>_API_KEY` 配置，真实值应放在不会提交的 `.env` 中。GPT-5.5 所选 provider 必须设置对应的 `LLM_PROVIDER_<NAME>_WIRE_API=responses`，其他 provider 默认继续走 Chat Completions。

### ASR 后端

| 优先级 | 后端 | 条件 | 模型大小 |
|--------|------|------|----------|
| 1 | **qwen3** (HuggingFace BF16) | ≥6 GB 显存 + safetensors 模型 | ~4.4 GB |
| 2 | **whisper** (faster-whisper) | 始终可用 | 0.15–3 GB |
| 3 | **gguf** (llama.cpp) | 可选旧路径：`llama-mtmd-cli` + GGUF/mmproj | ~2.4 GB |

#### Qwen3 BF16 后端（推荐）

```bash
uv sync --extra qwen3
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir .models/Qwen3-ASR-1.7B
uv run bilibot summarize BVxxxx --asr-backend qwen3 --asr-model .models/Qwen3-ASR-1.7B
uv run qwen3-asr ./meeting.m4a --asr-model .models/Qwen3-ASR-1.7B
```

启用 Qwen3 ForcedAligner 后，CLI 会额外输出词级或字级时间戳：

```bash
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir .models/Qwen3-ForcedAligner-0.6B
uv run qwen3-asr ./meeting.m4a \\
  --asr-model .models/Qwen3-ASR-1.7B \\
  --forced-aligner-model .models/Qwen3-ForcedAligner-0.6B \\
  --timestamps
```

启用后，`data/local_asr/{音频文件名}/` 下会多出 `timestamps.json`，每项包含 `text`、`start`、`end`，时间单位为秒；`transcript.json` 的 `time_stamps` 字段也保存同样内容。只传 `--timestamps` 时，CLI 会优先查找项目目录下的 `.models/Qwen3-ForcedAligner-0.6B`。超过五分钟的音频会分片识别，并自动恢复为原始音频时间轴。

#### GGUF 后端（可选旧路径）

```bash
uv sync --extra gguf
export ASR_GGUF_MODEL_DIR=/path/to/Qwen3-ASR-1.7B-GGUF
export ASR_GGUF_CLI=/path/to/llama.cpp/build/bin/llama-mtmd-cli
uv run bilibot summarize BVxxxx --asr-backend gguf
```

#### Whisper 后端

```bash
uv run bilibot summarize BVxxxx --asr-backend whisper --asr-preset accurate
```

### 命令

```bash
uv run bilibot summarize <url>                  # 完整流程
uv run bilibot summarize <url> --no-notes         # 仅提取字幕（不生成笔记，后处理照跑）
uv run bilibot summarize <url> --force-asr      # 跳过 B 站字幕，强制 ASR
uv run bilibot summarize <url> --json           # JSON 输出（供脚本/Agent 使用）
uv run bilibot summarize <url> --quiet          # 仅输出文件路径
uv run bilibot info <url>                       # 仅获取元数据
uv run bilibot info <url> --json                # 元数据 JSON 输出
uv run bilibot download <url>                   # 下载视频
uv run bilibot download <url> --json            # 输出文件路径 JSON
uv run bilibot asr ./meeting.m4a                # 转写本地音频/视频
uv run qwen3-asr ./meeting.m4a                  # 使用本地 Qwen3-ASR 转写
uv run qwen3-asr ./meeting.m4a --timestamps    # 使用 ForcedAligner 输出词/字级时间戳
uv run bilibot asr ./meeting.m4a --json         # 输出产物路径 JSON
uv run bilibot gen-notes <bvid>                 # 从已有字幕重新生成笔记
uv run bilibot doctor                           # 查看本机环境
uv run bilibot doctor --json                    # 本机环境 JSON 输出
```

### 输出文件

`data/{bvid}/` 目录下生成：

| 文件 | 说明 |
|------|------|
| `{标题}_信息.json` | 视频元数据 |
| `{标题}_字幕.json` | 结构化字幕 |
| `{标题}_字幕.txt` | 纯文本字幕 |
| `{标题}_笔记.md` | LLM 生成的结构化笔记 |

本地音频转写写入 `data/local_asr/{音频文件名}/`，包含 `transcript.json`、`transcript.md` 和 `captions.txt`。启用 `--timestamps` 后还会生成 `timestamps.json`，`transcript.json` 中也会保存 `time_stamps` 字段。

### 环境要求

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)（依赖自动安装，推荐系统包管理器安装）

### 技术栈

同上 English 部分。

### 许可证

MIT — 详见 [LICENSE](LICENSE)。
