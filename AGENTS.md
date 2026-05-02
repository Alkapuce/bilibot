# Repository Guidelines

## Project Overview

bilibot 是 B 站视频笔记助手 CLI 工具，通过 Python 3.12 实现。核心流程：获取 B 站视频元数据/字幕 → 字幕不存在时回退本地 faster-whisper ASR → 可选 LLM 字幕后处理 → LLM 结构化笔记生成。

**部署场景：** Linux（服务端常驻运行）和 Windows（开发 & 客户端直接使用）。所有代码、路径处理、子进程调用必须同时兼容 Linux 和 Windows。

## Project Structure

```
bilibot/
├── bilibot/           # Python 包（入口 & 核心逻辑）
│   ├── __init__.py    # 包标识
│   ├── __main__.py    # `python -m bilibot` 入口
│   ├── cli.py         # argparse CLI 定义 & Rich 输出
│   ├── config.py      # Settings dataclass + 环境变量加载（python-dotenv）
│   ├── models.py      # 共享数据模型：Transcript, TranscriptSegment 等
│   ├── extractor.py   # B 站 API 元数据/字幕提取（bilibili-api-python）
│   ├── transcriber.py # 音频下载（API 直链 → yt-dlp 回退）+ faster-whisper ASR
│   ├── asr.py         # 硬件检测、runtime 探测、ASR preset 解析
│   ├── llm.py         # OpenAI 兼容 LLM 客户端封装
│   ├── summarizer.py  # LLM 笔记生成（分 chunk 发送 + 汇总）
│   ├── postprocessor.py # 字幕后处理（LLM 清理标点/错别字）
│   ├── pipeline.py    # 端到端工作流编排（analyze_url）
│   ├── progress.py    # 进度事件协议 & callback 抽象
│   └── storage.py     # 文件写入：metadata.json / transcript.* / notes.md
├── data/              # 运行时输出（git ignored）
├── .env.example       # 可提交的配置模板
├── pyproject.toml     # 项目元数据 & 依赖（hatchling build）
├── uv.lock            # uv 锁定文件
└── README.md          # 用户文档
```

## Build, Test, and Development Commands

```bash
uv sync                          # 创建/更新虚拟环境
uv run bilibot --help            # 验证 CLI 入口
uv run bilibot summarize BVxxxx --no-llm   # 仅提取字幕，不调 LLM
uv run bilibot info BVxxxx       # 仅获取元数据
uv run bilibot doctor            # 检查本地 ASR 环境
uv run python -m compileall bilibot   # 快速语法检查（无测试套件时）
uv build                         # hatchling 构建包
```

## Coding Style & Naming Conventions

- Python 3.12+，统一使用 `from __future__ import annotations`
- 现代类型提示：`list[str]`、`Path | None`、`dict[str, Any]`
- 遵循 PEP 8，4 空格缩进
- `snake_case`：函数、变量、模块；`PascalCase`：dataclass 及其他类型
- dataclass 优先使用 `frozen=True`（immutable config/result 对象）
- CLI 输出使用 Rich 库，用户信息用中文
- 函数保持小而专注（extract / transcribe / summarize / postprocess 职责分离）

## Linux & Windows 兼容性要求

**所有代码改动必须同时保证 Linux 和 Windows 兼容性：**

- 路径处理：始终使用 `pathlib.Path`，禁止硬编码 `/` 或 `\`，禁止使用 `os.path.join(str1, str2)` 等字符串拼接
- 子进程调用：使用 `subprocess.run(..., capture_output=True)`，禁止 `shell=True` 除非必要；yt-dlp 等外部工具路径不要硬编码
- 系统调用：`os.cpu_count()`、`platform.system()` 等必须在两个平台返回合理值；系统特定操作（如 `ctypes` 读内存、NVIDIA GPU 检测）需有降级处理
- 文件权限：不要在 Python 代码中依赖 Unix 文件权限（如 `os.chmod`），除非做好降级
- 换行符：所有文件使用 LF，`.gitattributes` 保持规范
- 环境变量：大小写敏感只能在两个平台一致的部分使用；路径类型变量使用 `Path()` 包装
- 测试：如果将来添加测试，CI 应同时在 Linux 和 Windows 运行

## Testing Guidelines

当前无测试套件。新增行为时请在 `tests/` 目录添加 `test_*.py` 文件：
- 优先编写确定性单元测试：字幕选择、时间戳格式化、存储序列化、URL 解析
- 避免需要 B 站网络访问、Whisper 模型下载或 LLM 调用的测试，除非显式标记为集成测试并默认跳过
- 使用 `uv run pytest` 运行测试

## Commit & Pull Request Guidelines

- Commit 消息简短、祈使语气、加前缀，如 `bilibot: fix path handling on Windows` 或 `cli: add --quiet flag`
- 每次 commit 聚焦单一关注点
- PR 描述应包含：行为变更说明、验证步骤、配置/依赖变更说明、Windows 和 Linux 兼容性验证结果

## Security & Configuration Tips

- **绝不提交** `.env`、cookie 文件、生成的音频文件、`data/` 目录内容
- `LLM_API_KEY`、B 站 cookie（`BILI_SESSDATA`/`BILI_JCT`/`BILI_BUVID3`）属于敏感信息
- 新增配置项时：先在 `config.py:Settings` 添加字段 → 在 `config.py:load_settings` 读取环境变量 → 更新 `.env.example` → 在 `README.md` 中说明
- CLI 参数优先级高于环境变量（`_settings_from_args` 中可见）
