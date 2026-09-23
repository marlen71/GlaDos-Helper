"""Определение железа и состояния установки.

Используется установщиком (setup_glados.py) и диагностикой (check.py).
Модуль намеренно не зависит от numpy/torch — работает на голом Python,
чтобы его можно было вызвать ДО установки зависимостей.
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WIN = os.name == "nt"

# Размеры моделей Whisper на диске (приблизительно, МБ)
WHISPER_SIZES = {
    "tiny": 75, "base": 145, "small": 460, "medium": 1500, "large-v3": 3000,
}
WHISPER_SPEED_CPU = {
    "tiny": "~0.4 с", "base": "~0.7 с", "small": "~1.5 с",
    "medium": "~4 с", "large-v3": "~10 с и больше",
}


# ---------------------------------------------------------------- видеокарта
@dataclass
class GpuInfo:
    present: bool = False
    name: str = ""
    memory_mb: int = 0
    driver_cuda: str = ""        # максимальная версия CUDA по драйверу, напр. "12.4"
    error: str = ""

    @property
    def cuda_major(self) -> int:
        m = re.match(r"(\d+)", self.driver_cuda or "")
        return int(m.group(1)) if m else 0

    @property
    def supports_cu12(self) -> bool:
        return self.cuda_major >= 12

    def recommended_models(self) -> list[str]:
        """Какие модели Whisper потянет видеопамять."""
        if not self.present:
            return []
        gb = self.memory_mb / 1024
        if gb >= 8:
            return ["tiny", "base", "small", "medium", "large-v3"]
        if gb >= 5:
            return ["tiny", "base", "small", "medium"]
        if gb >= 3:
            return ["tiny", "base", "small"]
        return ["tiny", "base"]


def parse_nvidia_smi(text: str) -> GpuInfo:
    """Разбирает вывод `nvidia-smi`. Вынесено отдельно ради тестов."""
    info = GpuInfo()
    if not text:
        return info

    m = re.search(r"CUDA Version:\s*([\d.]+)", text)
    if m:
        info.driver_cuda = m.group(1)

    # Строка вида: "| 0  NVIDIA GeForce RTX 3060    WDDM | ... |"
    m = re.search(r"\|\s*\d+\s+(NVIDIA[^|]*?)\s{2,}", text)
    if m:
        info.name = re.sub(r"\s+(On|Off|WDDM|TCC).*$", "", m.group(1)).strip()

    # Память: "8192MiB / 12288MiB"
    m = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", text)
    if m:
        info.memory_mb = int(m.group(2))

    info.present = bool(info.name or info.driver_cuda)
    if info.present and not info.name:
        info.name = "NVIDIA GPU"
    return info


def parse_nvidia_smi_query(text: str) -> GpuInfo:
    """Разбирает вывод `nvidia-smi --query-gpu=name,memory.total,driver_version`."""
    info = GpuInfo()
    line = (text or "").strip().splitlines()
    if not line:
        return info
    parts = [p.strip() for p in line[0].split(",")]
    if not parts or not parts[0]:
        return info
    info.present = True
    info.name = parts[0]
    if len(parts) > 1:
        m = re.search(r"(\d+)", parts[1])
        if m:
            info.memory_mb = int(m.group(1))
    return info


def detect_gpu() -> GpuInfo:
    """Ищет видеокарту NVIDIA через nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe and IS_WIN:
        guess = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(guess) if guess.exists() else None
    if not exe:
        return GpuInfo(error="nvidia-smi не найден (нет видеокарты NVIDIA или драйвера)")

    try:
        full = subprocess.run([exe], capture_output=True, text=True, timeout=20)
        info = parse_nvidia_smi(full.stdout)
        if not info.name or not info.memory_mb:
            q = subprocess.run(
                [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20)
            q_info = parse_nvidia_smi_query(q.stdout)
            if q_info.present:
                info.present = True
                info.name = info.name or q_info.name
                info.memory_mb = info.memory_mb or q_info.memory_mb
        return info
    except Exception as e:
        return GpuInfo(error=f"не удалось опросить nvidia-smi: {e}")


# ------------------------------------------------------------ пакеты и DLL
def module_installed(module: str, python: Path | None = None) -> bool:
    """Установлен ли модуль. Если указан python — проверяем в том окружении."""
    if python is None:
        try:
            return importlib.util.find_spec(module) is not None
        except Exception:
            return False
    try:
        r = subprocess.run(
            [str(python), "-c",
             f"import importlib.util,sys; "
             f"sys.exit(0 if importlib.util.find_spec('{module}') else 1)"],
            capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


CUDA_DLLS = ("cublas64_12.dll", "cudnn_ops64_9.dll")


def cuda_libs_loadable(python: Path | None = None) -> bool:
    """Реально ли загружаются cuBLAS и cuDNN (а не просто стоят пакеты)."""
    if not IS_WIN:
        return module_installed("nvidia.cublas", python)

    code = (
        "import ctypes, os, sys\n"
        "try:\n"
        "    import nvidia, pathlib\n"
        "    base = pathlib.Path(nvidia.__file__).parent\n"
        "    for sub in ('cublas/bin', 'cudnn/bin'):\n"
        "        p = base / sub\n"
        "        if p.is_dir():\n"
        "            os.add_dll_directory(str(p))\n"
        "except Exception:\n"
        "    pass\n"
        "ok = True\n"
        f"for d in {CUDA_DLLS!r}:\n"
        "    try:\n"
        "        ctypes.CDLL(d)\n"
        "    except OSError:\n"
        "        ok = False\n"
        "sys.exit(0 if ok else 1)\n"
    )
    try:
        if python is None:
            for d in CUDA_DLLS:
                ctypes.CDLL(d)
            return True
        r = subprocess.run([str(python), "-c", code], capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


# ------------------------------------------------------------------ модели
def silero_model_present() -> bool:
    return (ROOT / "models" / "v4_ru.pt").exists()


def whisper_cache_dir() -> Path:
    env = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "huggingface" / "hub"


def whisper_model_present(model: str) -> bool:
    """Скачана ли модель Whisper в кэш Hugging Face."""
    cache = whisper_cache_dir()
    if not cache.exists():
        return False
    needle = f"faster-whisper-{model}".lower()
    try:
        for p in cache.iterdir():
            if p.is_dir() and needle in p.name.lower():
                # в папке должны быть веса
                if any(f.suffix == ".bin" or f.name == "model.bin"
                       for f in p.rglob("*")):
                    return True
    except Exception:
        pass
    return False


def installed_whisper_models() -> list[str]:
    return [m for m in WHISPER_SIZES if whisper_model_present(m)]


# -------------------------------------------------------------- диск и ОЗУ
def free_disk_gb(path: Path | None = None) -> float:
    try:
        usage = shutil.disk_usage(str(path or ROOT))
        return usage.free / (1024 ** 3)
    except Exception:
        return -1.0


def total_ram_gb() -> float:
    try:
        if IS_WIN:
            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMSTAT()
            st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / (1024 ** 3)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except Exception:
        return -1.0


# ------------------------------------------------------------- микрофоны
def list_microphones(python: Path | None = None) -> list[str]:
    code = (
        "import sounddevice as sd\n"
        "for i, d in enumerate(sd.query_devices()):\n"
        "    if d['max_input_channels'] > 0:\n"
        "        print(d['name'])\n"
    )
    try:
        if python is None:
            import sounddevice as sd  # type: ignore

            return [d["name"] for d in sd.query_devices()
                    if d["max_input_channels"] > 0]
        r = subprocess.run([str(python), "-c", code],
                           capture_output=True, text=True, timeout=60)
        return [l.strip() for l in r.stdout.splitlines() if l.strip()]
    except Exception:
        return []


# ---------------------------------------------------------------- сводка
@dataclass
class SystemReport:
    python_version: str = ""
    python_ok: bool = False
    disk_free_gb: float = 0.0
    ram_gb: float = 0.0
    gpu: GpuInfo = field(default_factory=GpuInfo)
    venv_exists: bool = False
    base_installed: bool = False
    torch_installed: bool = False
    whisper_installed: bool = False
    cuda_installed: bool = False
    silero_present: bool = False
    whisper_models: list[str] = field(default_factory=list)
    microphones: list[str] = field(default_factory=list)


def collect(venv_python: Path | None = None) -> SystemReport:
    """Полный отчёт о состоянии системы и установки."""
    r = SystemReport()
    v = sys.version_info
    r.python_version = f"{v.major}.{v.minor}.{v.micro}"
    r.python_ok = (3, 9) <= (v.major, v.minor)
    r.disk_free_gb = free_disk_gb()
    r.ram_gb = total_ram_gb()
    r.gpu = detect_gpu()

    py = venv_python if (venv_python and Path(venv_python).exists()) else None
    r.venv_exists = py is not None
    if py:
        r.base_installed = all(module_installed(m, py)
                               for m in ("numpy", "scipy", "yaml", "sounddevice"))
        r.torch_installed = module_installed("torch", py)
        r.whisper_installed = module_installed("faster_whisper", py)
        r.cuda_installed = cuda_libs_loadable(py) if r.gpu.present else False
        r.microphones = list_microphones(py)
    r.silero_present = silero_model_present()
    r.whisper_models = installed_whisper_models()
    return r
