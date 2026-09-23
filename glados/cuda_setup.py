"""Подключение CUDA-библиотек, установленных через pip.

Пакеты nvidia-cublas-cu12 / nvidia-cudnn-cu12 кладут DLL не в системные папки,
а внутрь site-packages/nvidia/<модуль>/bin. Windows их там не ищет, поэтому
ctranslate2 падает с "Library cublas64_12.dll is not found".

Этот модуль находит такие папки и регистрирует их через os.add_dll_directory
ДО первого обращения к faster-whisper. Вызывается автоматически при импорте
glados.stt и из диагностики.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("glados.cuda")

IS_WIN = sys.platform.startswith("win")

#: Подпапки внутри пакета nvidia, где лежат библиотеки
_NVIDIA_SUBDIRS = ("bin", "lib")

#: Маски нужных файлов (cuDNN 9.x разбит на несколько DLL)
_WANTED = ("cublas64_*.dll", "cublasLt64_*.dll", "cudnn*64_*.dll",
           "cudart64_*.dll", "cufft64_*.dll")

_applied = False
_registered_dirs: list[str] = []


def nvidia_dll_dirs() -> list[Path]:
    """Все папки с DLL из pip-пакетов nvidia-*."""
    dirs: list[Path] = []
    try:
        import nvidia  # type: ignore

        roots = [Path(p) for p in getattr(nvidia, "__path__", [])]
    except Exception:
        # nvidia — namespace-пакет; поищем вручную рядом с site-packages
        roots = []
        for entry in sys.path:
            cand = Path(entry) / "nvidia"
            if cand.is_dir():
                roots.append(cand)

    for root in roots:
        if not root.is_dir():
            continue
        for component in sorted(root.iterdir()):
            if not component.is_dir():
                continue
            for sub in _NVIDIA_SUBDIRS:
                d = component / sub
                if not d.is_dir():
                    continue
                # На Windows нужны .dll, на Linux .so; при тестировании
                # раскладки Windows на другой ОС учитываем оба варианта.
                if any(d.glob("*.dll")) or any(d.glob("*.so*")):
                    dirs.append(d)
    return dirs


def apply(force: bool = False) -> list[str]:
    """Зарегистрировать папки CUDA. Возвращает список подключённых путей."""
    global _applied, _registered_dirs
    if _applied and not force:
        return _registered_dirs

    found: list[str] = []
    for d in nvidia_dll_dirs():
        s = str(d)
        try:
            if IS_WIN and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(s)
            # PATH нужен ctranslate2, который грузит библиотеки по имени
            if s not in os.environ.get("PATH", ""):
                os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")
            if not IS_WIN:
                ld = os.environ.get("LD_LIBRARY_PATH", "")
                if s not in ld:
                    os.environ["LD_LIBRARY_PATH"] = s + os.pathsep + ld
            found.append(s)
        except Exception as e:
            log.debug("Не удалось подключить %s: %s", s, e)

    _registered_dirs = found
    _applied = True
    if found:
        log.debug("Подключены папки CUDA: %s", found)
    return found


def missing_libraries() -> list[str]:
    """Какие из ключевых библиотек не загружаются. Пустой список = всё хорошо."""
    apply()
    if not IS_WIN:
        return []

    import ctypes

    required = ("cublas64_12.dll", "cudnn_ops64_9.dll")
    missing = []
    for name in required:
        try:
            ctypes.CDLL(name)
        except OSError:
            # cuDNN 9.x может называть файл иначе — ищем по маске
            if name.startswith("cudnn") and _find_any("cudnn*64_9.dll"):
                continue
            missing.append(name)
    return missing


def _find_any(pattern: str) -> Path | None:
    for d in nvidia_dll_dirs():
        for f in d.glob(pattern):
            try:
                import ctypes

                ctypes.CDLL(str(f))
                return f
            except OSError:
                continue
    return None


def describe() -> str:
    """Человекочитаемый статус — для диагностики."""
    dirs = apply()
    if not dirs:
        return "папки с библиотеками CUDA не найдены"
    miss = missing_libraries()
    if miss:
        return f"найдено {len(dirs)} папок, но не загружаются: {', '.join(miss)}"
    return f"подключено {len(dirs)} папок с библиотеками"
