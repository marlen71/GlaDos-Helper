#!/usr/bin/env python3
"""Проверка окружения GLaDOS Helper: что установлено, что нет."""
from __future__ import annotations

import importlib
import sys

REQUIRED = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("yaml", "PyYAML"),
    ("sounddevice", "sounddevice"),
    ("soundfile", "soundfile"),
    ("faster_whisper", "faster-whisper"),
    ("torch", "torch"),
]
OPTIONAL = [("pyperclip", "pyperclip"), ("psutil", "psutil")]


def _enable_utf8_console() -> None:
    """В Windows-консоли stdout часто cp866 — переключаем на UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_enable_utf8_console()



def check(mod: str, pkg: str) -> bool:
    try:
        importlib.import_module(mod)
        print(f"  [OK]   {pkg}")
        return True
    except Exception as e:
        print(f"  [НЕТ]  {pkg}  ({type(e).__name__}: {e})")
        return False


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Интерпретатор: {sys.executable}\n")

    print("Обязательные пакеты:")
    missing = [pkg for mod, pkg in REQUIRED if not check(mod, pkg)]

    print("\nДополнительные:")
    for mod, pkg in OPTIONAL:
        check(mod, pkg)

    print("\nАудиоустройства:")
    try:
        import sounddevice as sd

        default_in, default_out = sd.default.device
        for i, d in enumerate(sd.query_devices()):
            mark = ""
            if i == default_in and d["max_input_channels"] > 0:
                mark = "  <- микрофон по умолчанию"
            elif i == default_out and d["max_output_channels"] > 0:
                mark = "  <- динамики по умолчанию"
            if d["max_input_channels"] or d["max_output_channels"]:
                print(f"  {i}: {d['name']} "
                      f"(in {d['max_input_channels']} / out {d['max_output_channels']}){mark}")
    except Exception as e:
        print(f"  не удалось получить список: {e}")

    print("\nУскорение (GPU):")
    try:
        from glados import sysinfo
        from glados.stt import cuda_is_usable

        gpu = sysinfo.detect_gpu()
        if gpu.present:
            mem = f", {gpu.memory_mb / 1024:.0f} ГБ" if gpu.memory_mb else ""
            print(f"  [OK]   Видеокарта: {gpu.name}{mem}")
            if gpu.driver_cuda:
                print(f"         Драйвер поддерживает CUDA {gpu.driver_cuda}")
        else:
            print("  [нет]  Видеокарта NVIDIA не найдена")

        from glados import cuda_setup
        from glados.stt import cuda_device_count, cuda_diagnose

        ok, reason = cuda_diagnose()
        if ok:
            print("  [OK]   CUDA работает — можно ставить stt.device: cuda")
            print(f"         {cuda_setup.describe()}")
        else:
            print(f"  [НЕТ]  {reason}")
            dirs = cuda_setup.nvidia_dll_dirs()
            if dirs:
                print(f"         Найдены папки библиотек ({len(dirs)}):")
                for d in dirs[:4]:
                    print(f"           {d}")
            else:
                print("         Пакеты nvidia-* не установлены.")
                print("         Установить: install-gpu.bat")
            if cuda_device_count() == 0 and gpu.present:
                print("         Видеокарта есть, но CUDA её не видит —")
                print("         обновите драйвер NVIDIA и перезагрузите компьютер.")
    except Exception as e:
        print(f"  не удалось проверить: {e}")

    if missing:
        print("\nНе хватает пакетов: " + ", ".join(missing))
        print("Установите их командой:")
        print(f'  "{sys.executable}" -m pip install ' + " ".join(missing))
        return 1

    print("\nВсё на месте. Можно запускать run.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
