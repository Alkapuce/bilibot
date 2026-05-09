# bilibot

B 站视频笔记助手：获取 Bilibili 视频元数据与字幕，调用大模型生成结构化 Markdown 笔记。  
支持三种本地语音识别后端：Qwen3-ASR GGUF、Qwen3-ASR BF16（HuggingFace）、Whisper（faster-whisper）。

## 快速开始

```bash
# 克隆 & 安装
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

> 如果复制的是带 `?`、`&` 参数的完整链接，在终端中请**用引号包裹 URL**，避免 shell 把 `?` 当通配符、`&` 当后台执行符。

## ASR 后端

bilibot 支持三层 ASR 后端，`auto` 模式按以下优先级自动选择：

| 优先级 | 后端 | 条件 | 模型大小 |
|--------|------|------|----------|
| 1 | **gguf** (ONNX + llama.cpp Q4_K_M) | ONNX Runtime + llama.dll 可用 | ~1.8 GB |
| 2 | **qwen3** (HuggingFace BF16) | ≥6 GB VRAM | ~3.4 GB |
| 3 | **whisper** (faster-whisper) | 始终可用 | 0.15–3 GB |

### GGUF 后端（推荐）

Qwen3-ASR Q4_K_M 量化 + ONNX encoder + llama.cpp decoder。当前中文 ASR 性价比最优。

```bash
# 安装依赖
uv sync --extra gguf

# 配置环境变量（模型目录 + llama.cpp DLL 目录）
export ASR_GGUF_MODEL_DIR=/path/to/Qwen3-ASR-1.7B-GGUF
export ASR_GGUF_LLAMA_BIN=/path/to/llama.cpp/bin

# 使用
uv run bilibot summarize BVxxxx
```

模型可以从 HuggingFace 下载或自行转换：

```
模型目录/
  qwen3_asr_encoder_frontend.fp16.onnx
  qwen3_asr_encoder_backend.fp16.onnx
  qwen3_asr_llm.q4_k.gguf
```

llama.cpp 预编译 DLL 可从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下载，放入 `bilibot/_gguf_core/bin/` 或通过 `ASR_GGUF_LLAMA_BIN` 指定。

### Qwen3-ASR BF16 后端

HuggingFace 全精度模型，首次运行自动下载缓存。

```bash
uv sync --extra qwen3
uv run bilibot summarize BVxxxx --asr-backend qwen3
```

### Whisper 后端

老牌方案，始终可用，无需额外安装。

```bash
uv run bilibot summarize BVxxxx --asr-backend whisper --asr-preset accurate
```

## 命令

```bash
uv run bilibot summarize <url>                  # 完整流程
uv run bilibot summarize <url> --no-llm         # 仅提取字幕（跳过 LLM 后处理与笔记）
uv run bilibot summarize <url> --force-asr      # 跳过 B 站字幕，强制 ASR
uv run bilibot summarize <url> --postprocess-subtitles  # 启用 LLM 字幕后处理
uv run bilibot summarize <url> --asr-backend qwen3
uv run bilibot summarize <url> --asr-backend gguf
uv run bilibot info <url>                       # 仅获取元数据
uv run bilibot download <url>                   # 下载视频文件
uv run bilibot doctor                           # 查看环境 & ASR 推荐
```

## 常用选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--asr-backend` | `auto` | ASR 后端：`auto`/`whisper`/`qwen3`/`gguf` |
| `--asr-preset` | `auto` | Whisper 预设：`fast`/`balanced`/`accurate`/`turbo`/`best`/`auto` |
| `--asr-model` | 自动 | 模型名或本地路径 |
| `--asr-device` | 自动 | cpu / cuda |
| `--asr-gguf-model-dir` | 环境变量 | GGUF 模型目录 |
| `--output-dir` | `data` | 输出目录 |
| `--language` | `zh` | 字幕/转写语言 |
| `--llm-base-url` | `.env` | LLM 接口地址 |
| `--llm-model` | `.env` | LLM 模型名 |
| `--postprocess-subtitles` | 关闭 | LLM 字幕后处理 |

完整选项见 `bilibot summarize --help`。

## 输出文件

`data/{bvid}/` 目录下生成，文件名以视频标题为前缀（安全截断至 60 字符，特殊字符替换为 `_`）：

| 文件 | 说明 |
|------|------|
| `{标题}_信息.json` | 视频元数据（含标签、字幕轨道摘要） |
| `{标题}_字幕.json` | 结构化字幕/转写（后处理后） |
| `{标题}_字幕.txt` | 纯文本字幕 |
| `{标题}_字幕原文.json` | 原始字幕（后处理前，仅当启用 `--postprocess-subtitles` 时） |
| `{标题}_字幕原文.txt` | 原始纯文本字幕 |
| `{标题}_笔记.md` | LLM 生成的结构化笔记 |

> 旧版本生成的 `metadata.json`、`transcript.json`、`notes.md` 等文件仍可正常读取，新运行默认生成上述命名格式。

### 辅助脚本

```bash
# 从已有字幕直接生成笔记（无需重新提取）
python gen_notes.py BV1YkR1BXEow
```

## 环境变量

```env
# 后端选择
ASR_BACKEND=auto
ASR_MODEL=

# GGUF
ASR_GGUF_MODEL_DIR=
ASR_GGUF_LLAMA_BIN=

# LLM
LLM_BASE_URL=http://localhost:5001/v1
LLM_API_KEY=replace-with-your-key
LLM_MODEL=deepseek-v4-pro

# B 站认证
BILI_SESSDATA=
BILI_JCT=
BILI_BUVID3=

# 其他见 .env.example
```

## 技术栈

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper ASR
- [Qwen3-ASR GGUF](https://github.com/HaujetZhao/Qwen3-ASR-GGUF) — ONNX + llama.cpp ASR
- [qwen-asr](https://github.com/QwenLM/Qwen3-ASR) — Qwen3 HuggingFace ASR
- [bilibili-api-python](https://github.com/Nemo2011/bilibili-api) — B 站 API
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — 视频/音频下载
- [OpenAI SDK](https://github.com/openai/openai-python) — LLM 调用
- [rich](https://github.com/Textualize/rich) — 终端输出

## 依赖

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv) 管理依赖
