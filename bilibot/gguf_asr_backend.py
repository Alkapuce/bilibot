"""Qwen3-ASR GGUF backend — ONNX + llama.cpp pipeline.

Priority:
1. qwen_asr_gguf Python 模块 (若在 sys.path 中)
2. 独立的 _gguf_core 模块 (需手动配置 llama.cpp DLL)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import Settings
from .models import Transcript, TranscriptSegment
from .progress import ProgressCallback, emit

logger = logging.getLogger("bilibot.gguf_asr")

GGUF_MODEL_IDS: dict[str, str] = {
    "qwen3-asr-1.7b": "Qwen3-ASR-1.7B",
    "qwen3-asr-1.7b-q8_0": "Qwen3-ASR-1.7B GGUF Q8_0",
    "qwen3-asr-0.6b-q8_0": "Qwen3-ASR-0.6B GGUF Q8_0",
}

_HF_GGUF_ALIASES: dict[str, str] = {
    "qwen3-asr-1.7b": "ggml-org/Qwen3-ASR-1.7B-GGUF",
    "qwen3-asr-1.7b-q8_0": "ggml-org/Qwen3-ASR-1.7B-GGUF",
    "qwen3-asr-0.6b-q8_0": "ggml-org/Qwen3-ASR-0.6B-GGUF",
}


def _check_available(settings: Settings) -> bool:
    try:
        model_dir = _model_dir(settings)
        files = _find_files(model_dir)
        if files["layout"] == "mtmd":
            _resolve_mtmd_cli(settings)
        else:
            _resolve_llama_bin(settings)
        return True
    except Exception:
        return False


def _model_dir(settings: Settings) -> Path:
    candidates = [
        settings.asr_gguf_model_dir,
        settings.asr_model if settings.asr_model else "",
        os.getenv("ASR_GGUF_MODEL_DIR", ""),
    ]
    for value in candidates:
        if value:
            p = Path(value).expanduser()
            if p.is_dir():
                return p.resolve()

    for default in (
        Path(".models") / "Qwen3-ASR-1.7B-GGUF",
        Path(".models") / "Qwen3-ASR-0.6B-GGUF",
        Path("models") / "Qwen3-ASR-1.7B-GGUF",
        Path("models") / "Qwen3-ASR-0.6B-GGUF",
    ):
        if default.is_dir():
            return default.resolve()

    raise FileNotFoundError(
        "未找到 GGUF 模型目录。请设置 ASR_GGUF_MODEL_DIR 环境变量，\n"
        "或将模型放到 .models/Qwen3-ASR-1.7B-GGUF/ 目录。"
    )


def _find_files(model_dir: Path) -> dict[str, str]:
    legacy_files = _find_legacy_files(model_dir)
    if legacy_files:
        return legacy_files

    mtmd_files = _find_mtmd_files(model_dir)
    if mtmd_files:
        return mtmd_files

    raise FileNotFoundError(
        f"在 {model_dir} 中未找到可用 GGUF ASR 文件。期望官方 llama.cpp "
        "GGUF+mmproj 布局，或旧版 ONNX encoder + GGUF decoder 布局。"
    )


def _find_legacy_files(model_dir: Path) -> dict[str, str] | None:
    candidates = {
        "encoder_frontend": [
            "qwen3_asr_encoder_frontend.fp16.onnx",
            "qwen3_asr_encoder_frontend.int4.onnx",
        ],
        "encoder_backend": [
            "qwen3_asr_encoder_backend.fp16.onnx",
            "qwen3_asr_encoder_backend.int4.onnx",
        ],
        "llm": [
            "qwen3_asr_llm.q4_k.gguf",
            "qwen3_asr_llm.q8_0.gguf",
        ],
    }
    result: dict[str, str] = {}
    for key, filenames in candidates.items():
        for fn in filenames:
            p = model_dir / fn
            if p.is_file():
                result[key] = str(p)
                break
        if key not in result:
            return None
    result["layout"] = "legacy"
    return result


def _find_mtmd_files(model_dir: Path) -> dict[str, str] | None:
    ggufs = [path for path in model_dir.glob("*.gguf") if path.is_file()]
    mmproj = [path for path in ggufs if "mmproj" in path.name.casefold()]
    models = [path for path in ggufs if path not in mmproj]
    if not models or not mmproj:
        return None
    return {
        "layout": "mtmd",
        "model": str(_preferred_gguf(models)),
        "mmproj": str(_preferred_gguf(mmproj)),
    }


def _preferred_gguf(paths: list[Path]) -> Path:
    def rank(path: Path) -> tuple[int, int, str]:
        name = path.name.casefold()
        if "q8_0" in name:
            quant = 0
        elif "q4" in name:
            quant = 1
        elif "q5" in name or "q6" in name:
            quant = 2
        elif "bf16" in name or "f16" in name:
            quant = 3
        else:
            quant = 4
        return (quant, path.stat().st_size, name)

    return sorted(paths, key=rank)[0]


def _resolve_llama_bin(settings: Settings) -> str:
    env_bin = settings.asr_gguf_llama_bin or os.getenv("ASR_GGUF_LLAMA_BIN", "")
    if env_bin and Path(env_bin).is_dir():
        return env_bin

    search_paths = [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    search_paths.extend(
        [
            Path("/usr/lib/x86_64-linux-gnu/llama"),
            Path("/usr/local/lib"),
            Path("/usr/lib"),
        ]
    )
    for p in search_paths:
        if any((p / name).exists() for name in ("llama.dll", "libllama.so", "libllama.so.0")):
            return str(p)

    bundled = Path(__file__).parent / "_gguf_core" / "bin"
    if bundled.is_dir():
        if _contains_any_file(bundled, ("llama.*", "libllama.*")):
            return str(bundled)

    raise FileNotFoundError(
        "未找到 llama.cpp 动态库。请设置 ASR_GGUF_LLAMA_BIN 环境变量，\n"
        "或从 https://github.com/ggml-org/llama.cpp/releases 下载预编译包放入 _gguf_core/bin/"
    )


def _resolve_mtmd_cli(settings: Settings) -> str:
    configured = settings.asr_gguf_cli or os.getenv("ASR_GGUF_CLI", "") or "llama-mtmd-cli"
    p = Path(configured).expanduser()
    if p.is_file():
        return str(p.resolve())
    found = shutil.which(configured)
    if found:
        return found
    raise FileNotFoundError("未找到 llama-mtmd-cli，请安装 llama.cpp-tools-extra 或设置 ASR_GGUF_CLI")


def transcribe(
    audio_path: str,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
) -> Transcript:
    import io, contextlib

    model_dir = _model_dir(settings)
    files = _find_files(model_dir)
    if files["layout"] == "mtmd":
        return _transcribe_with_mtmd(audio_path, settings, files, progress=progress)

    emit(progress, "task_start", "asr_model_load", f"加载 Qwen3-ASR GGUF 模型: {model_dir}")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        engine = _create_engine_qwen_asr_gguf(model_dir, files, settings)
        if not engine:
            engine = _create_engine_standalone(model_dir, files, settings)
        if not engine:
            raise RuntimeError("无法初始化 GGUF 引擎")

    emit(progress, "task_done", "asr_model_load", f"Qwen3-ASR GGUF 已加载")

    language = _map_lang(settings.language)

    emit(progress, "task_start", "asr_transcribe", "Qwen3-ASR GGUF 识别中", total=None)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = engine.transcribe(audio_file=audio_path, language=language, temperature=0.4)
    segments = _build_segments(result, audio_path)
    text = _trim_repetition(result.text.strip())
    emit(progress, "task_done", "asr_transcribe", f"Qwen3-ASR GGUF 完成: {len(text)} 字符, {len(segments)} 段")

    return Transcript(
        source="qwen3-gguf",
        language=language or "zh",
        segments=segments or [TranscriptSegment(start=0.0, end=0.0, text=text)],
    )


def _transcribe_with_mtmd(
    audio_path: str,
    settings: Settings,
    files: dict[str, str],
    *,
    progress: ProgressCallback | None = None,
) -> Transcript:
    cli = _resolve_mtmd_cli(settings)
    language = _map_lang(settings.language)
    prompt = _mtmd_prompt(language)
    threads = str(settings.asr_cpu_threads or max(1, min(os.cpu_count() or 1, 16)))
    cmd = [
        cli,
        "-m",
        files["model"],
        "--mmproj",
        files["mmproj"],
        "--audio",
        audio_path,
        "-p",
        prompt,
        "-n",
        "512",
        "--temp",
        "0",
        "--threads",
        threads,
        "--ctx-size",
        "4096",
        "-lv",
        "1",
        "--no-perf",
    ]
    if settings.asr_device == "cpu":
        cmd.extend(["--device", "none", "--gpu-layers", "0", "--no-mmproj-offload"])

    emit(progress, "task_start", "asr_transcribe", f"Qwen3-ASR GGUF 识别中：{Path(files['model']).name}")
    env = os.environ.copy()
    env.setdefault("LLAMA_LOG_COLORS", "off")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        raise RuntimeError(f"llama-mtmd-cli 转写失败：{detail}")

    text = _clean_mtmd_output(result.stdout, prompt)
    if not text:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Qwen3-ASR GGUF returned empty result: {detail[-800:]}")

    emit(progress, "task_done", "asr_transcribe", f"Qwen3-ASR GGUF 完成：{len(text)} 字符")
    duration = _audio_duration(audio_path)
    return Transcript(
        source=f"qwen3-gguf/{Path(files['model']).name}",
        language=language or "zh",
        segments=[TranscriptSegment(start=0.0, end=round(duration, 1) if duration > 0 else 0.0, text=text)],
    )


def _mtmd_prompt(language: str | None) -> str:
    if language:
        return f"Transcribe the audio in {language}. Output only the transcript text."
    return "Transcribe the audio. Output only the transcript text."


def _clean_mtmd_output(output: str, prompt: str) -> str:
    import re

    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output).strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    text = re.sub(r"^(assistant|Assistant|ASSISTANT)\s*[:：]\s*", "", text).strip()
    text = re.sub(r"^<\|im_start\|>assistant\s*", "", text).strip()
    text = re.sub(r"^language\s+[A-Za-z-]+\s*<asr_text>\s*", "", text).strip()
    text = text.replace("<|im_end|>", "").strip()
    for marker in ("[end of text]", "<|endoftext|>"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def transcribe_url(
    url: str,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
) -> Transcript:
    import tempfile

    from .downloader import download_audio

    with tempfile.TemporaryDirectory() as tmpdir:
        emit(progress, "log", "download_audio", "准备下载音频")
        audio_path = download_audio(
            url, tmpdir, settings.cookie_file,
            sessdata=settings.bili_sessdata, bili_jct=settings.bili_jct,
            buvid3=settings.bili_buvid3, timeout=settings.download_timeout,
            chunk_size=settings.download_chunk_size,
            yt_dlp_format=settings.yt_dlp_format,
            yt_dlp_audio_format=settings.yt_dlp_audio_format,
            yt_dlp_audio_quality=settings.yt_dlp_audio_quality,
            progress=progress,
        )
        return transcribe(audio_path, settings, progress=progress)


def _create_engine_qwen_asr_gguf(
    model_dir: Path,
    files: dict[str, str],
    settings: Settings,
) -> object | None:
    mod_root = _find_qwen_asr_gguf_module()
    if not mod_root:
        return None
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        util_path = str(mod_root / "util")
        if util_path not in sys.path:
            sys.path.insert(0, util_path)

        onnx_path = str(mod_root / "internal" / "onnxruntime" / "capi")
        if onnx_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = onnx_path + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(onnx_path)

        from qwen_asr_gguf import create_asr_engine

        wrapper = create_asr_engine(
            model_dir=str(model_dir),
            encoder_frontend_fn=Path(files["encoder_frontend"]).name,
            encoder_backend_fn=Path(files["encoder_backend"]).name,
            llm_fn=Path(files["llm"]).name,
            vulkan_enable=True, chunk_size=25.0, n_ctx=2048, memory_num=2, verbose=False,
        )
        return wrapper.engine
    except Exception as exc:
        logger.debug("qwen_asr_gguf engine init failed: %s", exc)
        return None


def _create_engine_standalone(
    model_dir: Path,
    files: dict[str, str],
    settings: Settings,
) -> object | None:
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import gguf  # noqa: F401
    except ImportError:
        return None

    from ._gguf_core import QwenASREngine, ASREngineConfig, set_lib_dir, init_llama_lib

    try:
        set_lib_dir(_resolve_llama_bin(settings))
        init_llama_lib()
    except Exception:
        return None

    try:
        config = ASREngineConfig(
            model_dir=str(model_dir),
            encoder_frontend_fn=Path(files["encoder_frontend"]).name,
            encoder_backend_fn=Path(files["encoder_backend"]).name,
            llm_fn=Path(files["llm"]).name,
            vulkan_enable=True, chunk_size=25.0, n_ctx=2048, memory_num=2, verbose=False,
        )
        return QwenASREngine(config)
    except Exception:
        return None


def _find_qwen_asr_gguf_module() -> Path | None:
    candidates = [
        Path("C:/Program Files (x)/CapsWriter-Offline"),
        Path("C:/Program Files/CapsWriter-Offline"),
        Path.home() / "CapsWriter-Offline",
    ]
    for p in candidates:
        util = p / "util" / "qwen_asr_gguf"
        if util.is_dir():
            return p
    return None


def _contains_any_file(directory: Path, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if next(directory.glob(pattern), None) is not None:
            return True
    return False


def _map_lang(language: str | None) -> str | None:
    if not language:
        return None
    lang_map = {
        "zh": "Chinese", "en": "English", "ja": "Japanese",
        "ko": "Korean", "auto": None, "": None,
    }
    if language in lang_map:
        return lang_map[language]
    return language.capitalize()


def _trim_repetition(text: str) -> str:
    import re

    lines = re.split(r'([。！？\n.!?])', text)
    if len(lines) < 5:
        return text
    chunks = [lines[i] + (lines[i + 1] if i + 1 < len(lines) else "") for i in range(0, len(lines) - 1, 2)]
    chunks = [c for c in chunks if len(c.strip()) >= 4]

    for i in range(len(chunks) - 4, -1, -1):
        head = chunks[i].strip()
        count = 0
        for j in range(i, len(chunks)):
            if chunks[j].strip() == head:
                count += 1
            else:
                break
        if count >= 3:
            text = ""
            for k in range(i):
                text += chunks[k]
            return text
    return text


def _build_segments(result, audio_path: str) -> list[TranscriptSegment]:
    segs: list[TranscriptSegment] = []
    if hasattr(result, "segments") and result.segments:
        for s in result.segments:
            if isinstance(s, dict):
                text = s.get("text", "").strip()
                if text:
                    segs.append(TranscriptSegment(start=s.get("start", 0), end=s.get("end", 0), text=text))
    if not segs:
        duration = _audio_duration(audio_path)
        full_text = result.text.strip()
        if duration > 0 and full_text:
            chars_per_sec = max(len(full_text) / duration, 1)
            chunk = 0.0
            chunk_size = 30.0
            remaining = full_text
            while remaining and chunk < duration:
                end = min(chunk + chunk_size, duration)
                chars = int((end - chunk) * chars_per_sec)
                seg_text = remaining[:chars]
                remaining = remaining[chars:]
                if seg_text.strip():
                    segs.append(TranscriptSegment(start=round(chunk, 1), end=round(end, 1), text=seg_text.strip()))
                chunk = end
            if remaining.strip():
                segs.append(TranscriptSegment(start=round(duration, 1), end=round(duration + 1, 1), text=remaining.strip()))
    return segs


def _audio_duration(audio_path: str) -> float:
    try:
        import soundfile as sf
        return sf.info(audio_path).duration
    except Exception:
        pass
    try:
        import subprocess
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0
