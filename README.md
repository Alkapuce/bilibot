# bilibot

B 站视频笔记助手：获取 Bilibili 视频元数据与字幕（无字幕时自动回退到本地 Whisper 语音识别），并调用兼容 OpenAI 接口的大模型生成 Markdown 格式的结构化笔记。

## 快速开始

支持 Linux、macOS、Windows。

```bash
uv sync
uv run bilibot summarize https://www.bilibili.com/video/BVxxxx/
```

也支持直接传 BV 号：

```bash
uv run bilibot BVxxxx
uv run bilibot 1xxxxxxxxx
```

检查本机 ASR 环境和推荐模型：

```bash
uv run bilibot doctor
```

如果复制的是带 `?`、`&` 参数的完整链接，在 bash/zsh 中建议只保留 `/video/BV.../`
这一段，或直接传 BV 号，避免 shell 把 `?` 当通配符、把 `&` 当后台执行符。
Windows 的 cmd/PowerShell 不受此限制，可以直接粘贴完整链接。

默认大模型配置指向本地兼容 OpenAI 的服务，可通过复制 `.env.example` 为 `.env` 来覆盖：

```env
LLM_BASE_URL=http://localhost:5001/v1
LLM_API_KEY=replace-with-your-key
LLM_MODEL=deepseek-expert-reasoner
```

## 输出文件

每次运行会在 `data/{bvid}/` 目录下生成以下文件：

| 文件 | 说明 |
|------|------|
| `metadata.json` | 视频元数据及字幕轨道摘要 |
| `transcript.json` | 结构化字幕/转写段落 |
| `transcript.md` | 带时间戳的完整字幕/转写文本 |
| `transcript_raw.json` | 启用字幕后处理时保存的原始字幕/转写 |
| `transcript_raw.md` | 启用字幕后处理时保存的原始字幕/转写文本 |
| `notes.md` | 大模型生成的结构化笔记 |

## 命令

```bash
uv run bilibot summarize <url或bvid>              # 完整流程：提取字幕 → 大模型生成笔记
uv run bilibot summarize <url或bvid> --no-llm     # 仅提取字幕，不调用大模型
uv run bilibot summarize <url或bvid> --force-asr  # 跳过 B 站字幕，强制本地语音识别
uv run bilibot summarize <url或bvid> --asr-preset accurate
uv run bilibot summarize <url或bvid> --postprocess-subtitles
uv run bilibot info <url或bvid>                    # 仅获取视频元数据
uv run bilibot doctor                              # 查看本机 ASR 推荐
```

常用选项：

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `data` | 输出目录 |
| `--cookie-file` | 无 | B 站直链/yt-dlp 下载音频用的 cookie 文件 |
| `--language` | `zh` | 字幕/转写语言 |
| `--verbose` | 关闭 | 输出更详细的阶段日志 |
| `--asr-preset` | `auto` | ASR 预设：`fast`/`balanced`/`accurate`/`turbo`/`auto` |
| `--asr-model` | 自动 | faster-whisper 模型，如 `small`、`medium`、`large-v3`、`turbo`、`distil-large-v3` |
| `--asr-device` | 自动 | ASR 运行设备（cpu/cuda） |
| `--asr-compute-type` | 自动 | ASR 计算精度（int8/float16/int8_float16 等） |
| `--asr-batch-size` | 自动 | batched inference 批大小，`0` 为关闭 |
| `--asr-vad-filter` | 自动 | 启用/关闭 faster-whisper 内置 VAD 静音过滤 |
| `--postprocess-subtitles` | 关闭 | 生成笔记前调用 LLM 清理字幕 |
| `--subtitle-postprocess-model` | 同 `--llm-model` | 字幕后处理使用的模型 |
| `--llm-base-url` | 见 `.env.example` | 兼容 OpenAI 的大模型接口地址 |
| `--llm-api-key` | 见 `.env.example` | 大模型 API Key |
| `--llm-model` | 见 `.env.example` | 大模型名称 |
| `--llm-timeout` | `180` | 大模型请求超时 |
| `--llm-temperature` | 无 | 大模型温度 |
| `--chunk-chars` | `8000` | 每次发给大模型的字幕最大字符数 |

`--whisper-model`、`--whisper-device`、`--whisper-compute-type` 仍作为兼容别名保留。

## ASR 模型建议

`bilibot` 仍使用 faster-whisper 作为本地 ASR 后端。当前可选模型包括 OpenAI Whisper 系列的 `large-v3`、更快的 `turbo`，以及 Distil-Whisper 的 `distil-large-v3`。faster-whisper 基于 CTranslate2，支持 CUDA、int8/float16 量化、VAD 过滤和 batched 推理。

本机检测到 2GB 级别 GPU 时，`auto` 不会默认跑 `large-v3`/`turbo`，因为显存太小容易 OOM；会选择 CPU `small` + int8。Windows 上若 `doctor` 显示内存为 0，请手动指定 ASR 参数。想要更高准确率可以显式指定：

```bash
uv run bilibot summarize BVxxxx --force-asr --asr-preset accurate
uv run bilibot summarize BVxxxx --force-asr --asr-model medium --asr-device cpu --asr-compute-type int8
```

如果有 8GB 以上 NVIDIA GPU，可以尝试：

```bash
uv run bilibot summarize BVxxxx --force-asr --asr-model large-v3 --asr-device cuda --asr-compute-type float16 --asr-batch-size 8
uv run bilibot summarize BVxxxx --force-asr --asr-model distil-large-v3 --asr-device cuda --asr-compute-type float16 --asr-batch-size 8
```

## 字幕后处理

字幕后处理是可选步骤，发生在字幕/ASR 之后、笔记生成之前。它会调用兼容 OpenAI 的模型修正标点、错别字、ASR 误识别和明显断裂，不改变时间戳。开启后会额外保存 `transcript_raw.*`，最终 `transcript.*` 使用后处理结果。

```bash
uv run bilibot summarize BVxxxx \
  --postprocess-subtitles \
  --subtitle-postprocess-model deepseek-v4-pro \
  --subtitle-postprocess-style clean
```

## 环境变量

支持通过 `.env` 文件或环境变量配置，完整列表见 `.env.example`：

```env
LLM_BASE_URL=http://localhost:5001/v1
LLM_API_KEY=replace-with-your-key
LLM_MODEL=deepseek-expert-reasoner
LLM_TIMEOUT=180.0
LLM_TEMPERATURE=
LLM_MAX_TOKENS=
CHUNK_CHARS=8000

BILIBOT_OUTPUT_DIR=data
TRANSCRIPT_LANGUAGE=zh

ASR_PRESET=auto
ASR_MODEL=
ASR_DEVICE=
ASR_COMPUTE_TYPE=
ASR_TASK=transcribe
ASR_BEAM_SIZE=5
ASR_BATCH_SIZE=
ASR_VAD_FILTER=
ASR_VAD_MIN_SILENCE_MS=
ASR_CONDITION_ON_PREVIOUS_TEXT=
ASR_CPU_THREADS=0
ASR_NUM_WORKERS=1
ASR_DOWNLOAD_ROOT=
ASR_LOCAL_FILES_ONLY=false
ASR_HOTWORDS=
ASR_INITIAL_PROMPT=

SUBTITLE_POSTPROCESS=false
SUBTITLE_POSTPROCESS_BASE_URL=
SUBTITLE_POSTPROCESS_API_KEY=
SUBTITLE_POSTPROCESS_MODEL=
SUBTITLE_POSTPROCESS_TEMPERATURE=
SUBTITLE_POSTPROCESS_CHUNK_CHARS=6000
SUBTITLE_POSTPROCESS_STYLE=clean

DOWNLOAD_TIMEOUT=60.0
DOWNLOAD_CHUNK_SIZE=1048576
YT_DLP_FORMAT=bestaudio
YT_DLP_AUDIO_FORMAT=mp3
YT_DLP_AUDIO_QUALITY=5

# 访问限制视频时的 B 站认证信息
BILI_SESSDATA=
BILI_JCT=
BILI_BUVID3=
BILI_COOKIE_FILE=
```

## 工作流程

1. **提取视频信息**：通过 B 站 API 获取标题、简介、字幕列表等元数据
2. **获取字幕**：优先使用 B 站已有字幕；若无字幕或启用 `--force-asr`，则优先通过 B 站 API 直链下载音频，失败后再回退 yt-dlp，调用本地 faster-whisper 进行语音识别
3. **字幕后处理**（可选）：调用单独可配置的大模型清理字幕文本，保留原始字幕文件
4. **生成笔记**（可选）：将字幕分段发送给大模型，汇总生成结构化 Markdown 笔记
5. **保存输出**：将所有结果写入 `data/{bvid}/` 目录

## 技术栈

- [bilibili-api-python](https://github.com/Nemo2011/bilibili-api) — B 站 API 封装
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 本地语音识别
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — 视频/音频下载
- [OpenAI Python SDK](https://github.com/openai/openai-python) — 调用兼容接口的大模型
- [rich](https://github.com/Textualize/rich) — 终端美化输出

## 依赖

- Python >= 3.12
- 使用 [uv](https://github.com/astral-sh/uv) 管理依赖
