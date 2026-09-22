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

    if missing:
        print("\nНе хватает пакетов: " + ", ".join(missing))
        print("Установите их командой:")
        print(f'  "{sys.executable}" -m pip install ' + " ".join(missing))
        return 1

    print("\nВсё на месте. Можно запускать run.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
