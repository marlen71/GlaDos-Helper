#!/usr/bin/env python3
"""Установщик GLaDOS Helper. Вызывается из install.bat, но работает и сам по себе.

Создаёт .venv, ставит torch (CPU) и остальные зависимости, затем проверяет окружение.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
IS_WIN = os.name == "nt"


def _enable_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_enable_utf8_console()


def _out(s: str = "") -> None:
    """Печать, устойчивая к консоли cp866."""
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(s.encode(enc, "replace").decode(enc, "replace"), flush=True)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def run(cmd: list[str], title: str, required: bool = True) -> bool:
    _out()
    _out(f"--- {title} ---")
    _out("  " + " ".join(str(c) for c in cmd))
    code = subprocess.call([str(c) for c in cmd])
    if code != 0:
        _out(f"[{'ОШИБКА' if required else 'ВНИМАНИЕ'}] шаг завершился с кодом {code}")
        return False
    return True


def main() -> int:
    _out("=" * 60)
    _out("  Установка GLaDOS Helper")
    _out("=" * 60)
    _out(f"Папка проекта: {ROOT}")
    _out(f"Python: {sys.version.split()[0]} ({sys.executable})")

    if sys.version_info < (3, 9):
        _out("[ОШИБКА] Нужен Python 3.9 или новее.")
        return 1

    vpy = venv_python()
    if vpy.exists():
        _out("\nОкружение .venv уже существует — использую его.")
    else:
        if not run([sys.executable, "-m", "venv", str(VENV)],
                   "Создаю виртуальное окружение .venv"):
            return 1
    if not vpy.exists():
        _out(f"[ОШИБКА] Не найден интерпретатор окружения: {vpy}")
        return 1

    run([vpy, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        "Обновляю pip", required=False)

    ok = run([vpy, "-m", "pip", "install", "torch", "torchaudio",
              "--index-url", "https://download.pytorch.org/whl/cpu"],
             "Ставлю PyTorch (CPU, ~200 МБ)", required=False)
    if not ok:
        _out("Пробую обычную сборку torch с PyPI...")
        run([vpy, "-m", "pip", "install", "torch", "torchaudio"],
            "PyTorch (PyPI)", required=False)

    if not run([vpy, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
               "Ставлю остальные зависимости"):
        _out("\nУстановка зависимостей не удалась. Скопируйте текст ошибки выше.")
        return 1

    _out()
    _out("--- Проверка окружения ---")
    code = subprocess.call([str(vpy), str(ROOT / "check.py")])

    _out()
    _out("=" * 60)
    if code == 0:
        _out("  Готово! Запускайте run.bat")
    else:
        _out("  Установка завершена с замечаниями — смотрите список выше.")
    _out("=" * 60)
    return code


if __name__ == "__main__":
    sys.exit(main())
