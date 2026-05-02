"""Qwen3-ASR GGUF backend — ONNX + llama.cpp pipeline.

Priority:
1. qwen_asr_gguf Python 模块 (若在 sys.path 中)
2. 独立的 _gguf_core 模块 (需手动配置 llama.cpp DLL)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .config import Settings
from .models import Transcript, TranscriptSegment
from .progress import ProgressCallback, emit

logger = logging.getLogger("bilibot.gguf_asr")

GGUF_MODEL_IDS: dict[str, str] = {
    "qwen3-asr-1.7b": "Qwen3-ASR-1.7B",
}

_HF_GGUF_ALIASES: dict[str, str] = {
    "qwen3-asr-1.7b": "HaujetZhao/Qwen3-ASR-1.7B-GGUF",
}


def _check_available(settings: Settings) -> bool:
    try:
        _resolve_llama_bin(settings)
        return True
    except Exception:
        return False


def _model_dir(settings: Settings) -> Path:
    model = settings.asr_model

    if model:
        p = Path(model)
        if p.is_dir():
            return p.resolve()

    env_dir = os.getenv("ASR_GGUF_MODEL_DIR", "")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir)

    default = Path("models") / "Qwen3-ASR-1.7B-GGUF"
    if default.is_dir():
        return default.resolve()

    raise FileNotFoundError(
        "未找到 GGUF 模型目录。请设置 ASR_GGUF_MODEL_DIR 环境变量，\n"
        "或将模型放到 models/Qwen3-ASR-1.7B-GGUF/ 目录。"
    )


def _find_files(model_dir: Path) -> dict[str, str]:
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
            raise FileNotFoundError(
                f"在 {model_dir} 中未找到 {key} 文件，期望: {filenames}"
            )
    return result


def _resolve_llama_bin(settings: Settings) -> str:
    env_bin = os.getenv("ASR_GGUF_LLAMA_BIN", "")
    if env_bin and Path(env_bin).is_dir():
        return env_bin

    for p in os.environ.get("PATH", "").split(os.pathsep):
        if (Path(p) / "llama.dll").exists() or (Path(p) / "libllama.so").exists():
            return p

    bundled = Path(__file__).parent / "_gguf_core" / "bin"
    if bundled.is_dir():
        if any(bundled.glob("llama.*") or bundled.glob("libllama.*")):
            return str(bundled)

    raise FileNotFoundError(
        "未找到 llama.cpp 动态库。请设置 ASR_GGUF_LLAMA_BIN 环境变量，\n"
        "或从 https://github.com/ggml-org/llama.cpp/releases 下载预编译包放入 _gguf_core/bin/"
    )


def transcribe(
    audio_path: str,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
) -> Transcript:
    import io, contextlib

    model_dir = _model_dir(settings)
    files = _find_files(model_dir)

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
        source=f"gguf/{model_dir}",
        language=language or "zh",
        segments=segments or [TranscriptSegment(start=0.0, end=0.0, text=text)],
    )


def transcribe_url(
    url: str,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
) -> Transcript:
    import tempfile

    from .transcriber import download_audio

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
        os.add_dll_directory(onnx_path)

        from qwen_asr_gguf import create_asr_engine, ASREngineConfig
        config = ASREngineConfig(
            model_dir=str(model_dir),
            encoder_frontend_fn=Path(files["encoder_frontend"]).name,
            encoder_backend_fn=Path(files["encoder_backend"]).name,
            llm_fn=Path(files["llm"]).name,
            vulkan_enable=True, chunk_size=25.0, n_ctx=2048, memory_num=2, verbose=False,
        )
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


def _build_segments(result, audio_path: str) -> list:
    segs = []
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
