# bilibot

B 站视频笔记助手：获取 Bilibili 视频元数据与字幕（无字幕时自动回退到本地 Whisper 语音识别），并调用兼容 OpenAI 接口的大模型生成 Markdown 格式的结构化笔记。

## 快速开始

```bash
uv sync
uv run bilibot summarize https://www.bilibili.com/video/BVxxxx/
```

也支持直接传 BV 号：

```bash
uv run bilibot BVxxxx
uv run bilibot 1xxxxxxxxx
```

如果复制的是带 `?`、`&` 参数的完整链接，在 zsh/bash 中仍建议只保留 `/video/BV.../`
这一段，或直接传 BV 号，避免 shell 把 `?` 当通配符、把 `&` 当后台执行符。

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
| `notes.md` | 大模型生成的结构化笔记 |

## 命令

```bash
uv run bilibot summarize <url或bvid>              # 完整流程：提取字幕 → 大模型生成笔记
uv run bilibot summarize <url或bvid> --no-llm     # 仅提取字幕，不调用大模型
uv run bilibot summarize <url或bvid> --force-asr  # 跳过 B 站字幕，强制本地语音识别
uv run bilibot info <url或bvid>                    # 仅获取视频元数据
```

常用选项：

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `data` | 输出目录 |
| `--cookie-file` | 无 | B 站直链/yt-dlp 下载音频用的 cookie 文件 |
| `--language` | `zh` | 字幕/转写语言 |
| `--whisper-model` | `base` | faster-whisper 模型（base/small/medium/large） |
| `--whisper-device` | `cpu` | faster-whisper 运行设备（cpu/cuda） |
| `--whisper-compute-type` | `int8` | faster-whisper 计算精度 |
| `--llm-base-url` | 见 `.env.example` | 兼容 OpenAI 的大模型接口地址 |
| `--llm-api-key` | 见 `.env.example` | 大模型 API Key |
| `--llm-model` | 见 `.env.example` | 大模型名称 |
| `--chunk-chars` | `8000` | 每次发给大模型的字幕最大字符数 |

## 环境变量

支持通过 `.env` 文件或环境变量配置，完整列表见 `.env.example`：

```env
LLM_BASE_URL=http://localhost:5001/v1
LLM_API_KEY=replace-with-your-key
LLM_MODEL=deepseek-expert-reasoner
LLM_TIMEOUT=180.0

BILIBOT_OUTPUT_DIR=data
TRANSCRIPT_LANGUAGE=zh
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# 访问限制视频时的 B 站认证信息
BILI_SESSDATA=
BILI_JCT=
BILI_BUVID3=
BILI_COOKIE_FILE=
```

## 工作流程

1. **提取视频信息**：通过 B 站 API 获取标题、简介、字幕列表等元数据
2. **获取字幕**：优先使用 B 站已有字幕；若无字幕或启用 `--force-asr`，则优先通过 B 站 API 直链下载音频，失败后再回退 yt-dlp，调用本地 faster-whisper 进行语音识别
3. **生成笔记**（可选）：将字幕分段发送给大模型，汇总生成结构化 Markdown 笔记
4. **保存输出**：将所有结果写入 `data/{bvid}/` 目录

## 技术栈

- [bilibili-api-python](https://github.com/Nemo2011/bilibili-api) — B 站 API 封装
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 本地语音识别
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — 视频/音频下载
- [OpenAI Python SDK](https://github.com/openai/openai-python) — 调用兼容接口的大模型
- [rich](https://github.com/Textualize/rich) — 终端美化输出

## 依赖

- Python >= 3.12
- 使用 [uv](https://github.com/astral-sh/uv) 管理依赖
