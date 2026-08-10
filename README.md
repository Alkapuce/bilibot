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

Copy `.env.example` to `.env` and fill in your LLM endpoint:

```env
LLM_BASE_URL=http://localhost:5001/v1
LLM_API_KEY=your-key-here
LLM_MODEL=deepseek-v4-pro
```

At minimum, only `LLM_BASE_URL` and `LLM_API_KEY` are required. All other settings have sensible defaults.

### ASR Backends

| Priority | Backend | Requirement | Size |
|----------|---------|-------------|------|
| 1 | **gguf** (ONNX + llama.cpp) | ONNX Runtime + llama.cpp lib | ~1.8 GB |
| 2 | **qwen3** (HuggingFace BF16) | ≥6 GB VRAM | ~3.4 GB |
| 3 | **whisper** (faster-whisper) | Always available | 0.15–3 GB |

#### GGUF (recommended)

```bash
uv sync --extra gguf
export ASR_GGUF_MODEL_DIR=/path/to/Qwen3-ASR-1.7B-GGUF
export ASR_GGUF_LLAMA_BIN=/path/to/llama.cpp/bin
uv run bilibot summarize BVxxxx
```

Download the GGUF model from [HuggingFace](https://huggingface.co/HaujetZhao/Qwen3-ASR-1.7B-GGUF) and llama.cpp binaries from [Releases](https://github.com/ggml-org/llama.cpp/releases).

#### Qwen3 BF16

```bash
uv sync --extra qwen3
uv run bilibot summarize BVxxxx --asr-backend qwen3
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

Local audio transcription writes to `data/local_asr/{audio-stem}/` with `transcript.json`, `transcript.md`, and `captions.txt`.

### Requirements

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) (auto-installed; system package recommended)

### Tech Stack

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper ASR
- [Qwen3-ASR GGUF](https://github.com/HaujetZhao/Qwen3-ASR-GGUF) — ONNX + llama.cpp ASR
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
LLM_BASE_URL=http://localhost:5001/v1
LLM_API_KEY=your-key-here
LLM_MODEL=deepseek-v4-pro
```

只需配置 `LLM_BASE_URL` 和 `LLM_API_KEY` 即可使用，其余均有合理默认值。

### ASR 后端

| 优先级 | 后端 | 条件 | 模型大小 |
|--------|------|------|----------|
| 1 | **gguf** (ONNX + llama.cpp) | ONNX Runtime + llama.cpp 库 | ~1.8 GB |
| 2 | **qwen3** (HuggingFace BF16) | ≥6 GB 显存 | ~3.4 GB |
| 3 | **whisper** (faster-whisper) | 始终可用 | 0.15–3 GB |

#### GGUF 后端（推荐）

```bash
uv sync --extra gguf
export ASR_GGUF_MODEL_DIR=/path/to/Qwen3-ASR-1.7B-GGUF
export ASR_GGUF_LLAMA_BIN=/path/to/llama.cpp/bin
uv run bilibot summarize BVxxxx
```

模型从 [HuggingFace](https://huggingface.co/HaujetZhao/Qwen3-ASR-1.7B-GGUF) 下载，llama.cpp 库从 [Releases](https://github.com/ggml-org/llama.cpp/releases) 获取。

#### Qwen3 BF16 后端

```bash
uv sync --extra qwen3
uv run bilibot summarize BVxxxx --asr-backend qwen3
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

本地音频转写写入 `data/local_asr/{音频文件名}/`，包含 `transcript.json`、`transcript.md` 和 `captions.txt`。

### 环境要求

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)（依赖自动安装，推荐系统包管理器安装）

### 技术栈

同上 English 部分。

### 许可证

MIT — 详见 [LICENSE](LICENSE)。
