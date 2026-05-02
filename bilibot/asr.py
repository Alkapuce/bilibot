"""ASR runtime detection and preset resolution."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from dataclasses import dataclass

from .config import Settings


ASR_PRESETS = ("auto", "fast", "balanced", "accurate", "turbo", "best")

WHISPER_MODEL_IDS: dict[str, str] = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}

QWEN3_1_7B_MIN_VRAM_MB = 6000
QWEN3_0_6B_MIN_VRAM_MB = 4000


def resolve_backend(settings: Settings, gpu_mb: int = 0) -> str:
    if settings.asr_backend not in ("auto", ""):
        return settings.asr_backend
    if _gguf_available(settings):
        return "gguf"
    if gpu_mb >= QWEN3_1_7B_MIN_VRAM_MB:
        return "qwen3"
    return "whisper"


def _gguf_available(settings: Settings) -> bool:
    try:
        from .gguf_asr_backend import _check_available
        return _check_available(settings)
    except Exception:
        return False


@dataclass(frozen=True)
class GpuInfo:
    name: str
    memory_mb: int
    driver_version: str = ""


@dataclass(frozen=True)
class RuntimeInfo:
    cpu_model: str
    cpu_count: int
    memory_total_gb: float
    memory_available_gb: float
    cuda_device_count: int
    gpus: list[GpuInfo]
    cpu_compute_types: list[str]
    cuda_compute_types: list[str]


@dataclass(frozen=True)
class AsrPlan:
    preset: str
    model: str
    device: str
    compute_type: str
    batch_size: int
    vad_filter: bool
    condition_on_previous_text: bool
    reason: str


def detect_runtime() -> RuntimeInfo:
    """Collect lightweight local hardware/runtime details for ASR decisions."""
    mem_total, mem_available = _read_meminfo()
    return RuntimeInfo(
        cpu_model=_cpu_model(),
        cpu_count=os.cpu_count() or 1,
        memory_total_gb=mem_total,
        memory_available_gb=mem_available,
        cuda_device_count=_cuda_device_count(),
        gpus=_nvidia_gpus(),
        cpu_compute_types=_supported_compute_types("cpu"),
        cuda_compute_types=_supported_compute_types("cuda"),
    )


def resolve_asr_plan(settings: Settings, runtime: RuntimeInfo | None = None) -> AsrPlan:
    runtime = runtime or detect_runtime()
    preset = settings.asr_preset if settings.asr_preset in ASR_PRESETS else "auto"
    best_gpu = max(runtime.gpus, key=lambda gpu: gpu.memory_mb, default=None)
    gpu_mb = best_gpu.memory_mb if best_gpu else 0
    has_usable_cuda = runtime.cuda_device_count > 0 and gpu_mb >= 4096

    model, device, compute_type, batch_size, vad_filter, reason = _preset_plan(
        preset,
        runtime,
        gpu_mb,
        has_usable_cuda,
    )

    if settings.asr_model:
        model = settings.asr_model
        reason = f"使用显式 ASR 模型 {model}。"
    if settings.asr_device:
        device = settings.asr_device
    if settings.asr_compute_type:
        compute_type = settings.asr_compute_type
    if settings.asr_batch_size is not None:
        batch_size = max(0, settings.asr_batch_size)
    if settings.asr_vad_filter is not None:
        vad_filter = settings.asr_vad_filter

    condition_on_previous_text = not model.startswith("distil")
    if settings.asr_condition_on_previous_text is not None:
        condition_on_previous_text = settings.asr_condition_on_previous_text

    return AsrPlan(
        preset=preset,
        model=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
        reason=reason,
    )


def _preset_plan(
    preset: str,
    runtime: RuntimeInfo,
    gpu_mb: int,
    has_usable_cuda: bool,
) -> tuple[str, str, str, int, bool, str]:
    if preset == "best":
        if gpu_mb >= 11000:
            return (
                "large-v3",
                "cuda",
                "float16",
                16,
                True,
                "best 预设：11GB+ GPU，large-v3 float16 batch=16 极致质量。",
            )
        if gpu_mb >= 8192:
            return (
                "large-v3",
                "cuda",
                "float16",
                8,
                True,
                "best 预设：8GB+ GPU，large-v3 float16 高质量模式。",
            )
        if gpu_mb >= 4096:
            return (
                "turbo",
                "cuda",
                "float16",
                4,
                True,
                "best 预设：4GB+ GPU，turbo float16 高质量模式。",
            )
        return ("medium", "cpu", "int8", 0, True, "best 预设：无 4GB+ GPU，使用 CPU medium。")

    if preset == "fast":
        if has_usable_cuda:
            return ("small", "cuda", "int8_float16", 4, True, "fast 预设：CUDA 可用，优先速度。")
        return ("base", "cpu", "int8", 0, True, "fast 预设：CPU int8，优先速度。")

    if preset == "accurate":
        if gpu_mb >= 11000:
            return (
                "large-v3",
                "cuda",
                "float16",
                12,
                True,
                "accurate 预设：11GB+ GPU，large-v3 float16 batch=12。",
            )
        if gpu_mb >= 8192:
            return ("large-v3", "cuda", "float16", 8, True, "accurate 预设：8GB+ GPU，使用 large-v3。")
        if gpu_mb >= 4096:
            return ("turbo", "cuda", "int8_float16", 4, True, "accurate 预设：4GB+ GPU，使用 turbo。")
        return ("medium", "cpu", "int8", 0, True, "accurate 预设：显存不足，使用 CPU medium。")

    if preset == "turbo":
        if has_usable_cuda:
            return ("turbo", "cuda", "int8_float16", 4, True, "turbo 预设：使用 large-v3 turbo 系列。")
        return ("turbo", "cpu", "int8", 0, True, "turbo 预设：无 4GB+ GPU，改用 CPU int8。")

    if preset == "balanced":
        if gpu_mb >= 11000:
            return (
                "distil-large-v3",
                "cuda",
                "float16",
                16,
                True,
                "balanced 预设：11GB+ GPU，distil-large-v3 float16 batch=16。",
            )
        if gpu_mb >= 8192:
            return (
                "distil-large-v3",
                "cuda",
                "float16",
                8,
                True,
                "balanced 预设：8GB+ GPU，使用 distil-large-v3。",
            )
        if has_usable_cuda:
            return ("small", "cuda", "int8_float16", 4, True, "balanced 预设：4GB+ GPU，使用 small。")
        return ("small", "cpu", "int8", 0, True, "balanced 预设：CPU small，兼顾速度和准确率。")

    if gpu_mb >= 11000:
        return (
            "large-v3",
            "cuda",
            "float16",
            12,
            True,
            "auto：检测到 11GB+ GPU，使用 large-v3 float16 batch=12。",
        )
    if gpu_mb >= 8192:
        return ("large-v3", "cuda", "float16", 8, True, "auto：检测到 8GB+ GPU，使用 large-v3。")
    if gpu_mb >= 4096:
        return ("turbo", "cuda", "int8_float16", 4, True, "auto：检测到 4GB+ GPU，使用 turbo。")
    if runtime.cuda_device_count > 0 and gpu_mb:
        return (
            "small",
            "cpu",
            "int8",
            0,
            True,
            f"auto：检测到 {gpu_mb}MB GPU，显存偏小，默认改用 CPU small 以避免 OOM。",
        )
    return ("small", "cpu", "int8", 0, True, "auto：未检测到可用 CUDA GPU，使用 CPU small。")


def _cpu_model() -> str:
    # Linux: /proc/cpuinfo
    try:
        for line in _read_text("/proc/cpuinfo").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass

    # Windows: wmic
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "name"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped and not stripped.casefold().startswith("name"):
                    return stripped
    except (FileNotFoundError, OSError):
        pass

    # Generic fallback
    cpu = platform.processor()
    if cpu and cpu != "unknown":
        return cpu

    return "unknown"


def _read_meminfo() -> tuple[float, float]:
    """Read total and available memory in GB. Returns (0.0, 0.0) on failure."""
    # Linux: /proc/meminfo
    try:
        total = avl = 0.0
        for line in _read_text("/proc/meminfo").splitlines():
            if line.startswith("MemTotal:"):
                total = round(int(line.split()[1]) / 1024 / 1024, 1)
            elif line.startswith("MemAvailable:"):
                avl = round(int(line.split()[1]) / 1024 / 1024, 1)
        if total:
            return total, avl
    except (OSError, ValueError, IndexError):
        pass

    # Windows: GlobalMemoryStatusEx
    if os.name == "nt":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return (
                round(status.ullTotalPhys / (1024**3), 1),
                round(status.ullAvailPhys / (1024**3), 1),
            )
        except Exception:
            pass

    return (0.0, 0.0)


def _cuda_device_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _supported_compute_types(device: str) -> list[str]:
    try:
        import ctranslate2

        return sorted(ctranslate2.get_supported_compute_types(device))
    except Exception:
        return []


def _nvidia_gpus() -> list[GpuInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []

    gpus: list[GpuInfo] = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            memory_mb = int(fields[1])
        except ValueError:
            memory_mb = 0
        gpus.append(
            GpuInfo(
                name=fields[0],
                memory_mb=memory_mb,
                driver_version=fields[2] if len(fields) > 2 else "",
            )
        )
    return gpus


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as handle:
        return handle.read()
