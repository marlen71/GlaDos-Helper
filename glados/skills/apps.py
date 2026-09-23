"""Запуск приложений, игр и сайтов."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import webbrowser

log = logging.getLogger("glados.apps")


def _expand(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def _spawn_flags() -> dict:
    """Аргументы Popen, полностью отвязывающие приложение от нашей консоли.

    Без этого запущенная программа наследует наше окно и её собственные логи
    (например, служебные сообщения Electron у Яндекс Музыки или Discord)
    сыплются поверх диалога с помощником.
    """
    kwargs: dict = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform.startswith("win"):
        # DETACHED_PROCESS — у приложения не будет нашей консоли
        # CREATE_NEW_PROCESS_GROUP — Ctrl+C в нашем окне его не убьёт
        flags = 0x00000008 | 0x00000200
        flags |= getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    return kwargs


def open_target(target: str) -> bool:
    """Универсальный запуск: URL, команда start, exe-путь, имя в PATH.

    Приложение запускается отвязанным от консоли помощника, чтобы его
    служебный вывод не мешал диалогу.
    """
    target = _expand(str(target)).strip()
    if not target:
        return False

    flags = _spawn_flags()
    try:
        if target.startswith(("http://", "https://", "steam://")):
            webbrowser.open(target)
            return True

        if sys.platform.startswith("win"):
            if target.lower().startswith("start "):
                subprocess.Popen(target, shell=True, **flags)
            elif os.path.exists(_executable_part(target)):
                subprocess.Popen(target, shell=True, **flags)
            else:
                os.startfile(target)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(target, shell=True, **flags)
        return True
    except Exception as e:
        log.error("Не удалось запустить %r: %s", target, e)
        return False


def _executable_part(target: str) -> str:
    """Путь к файлу без аргументов командной строки."""
    return target.split(" --")[0].strip('"')


def open_app(cfg, key: str) -> bool:
    target = cfg.get_path(f"apps.{key}")
    if not target:
        return False
    return open_target(target)


def find_game(cfg, spoken: str):
    """Находит игру по нечёткому совпадению названия."""
    games = cfg.get("games", {}) or {}
    spoken = spoken.lower().strip(" .!?»«\"'")
    if not spoken:
        return None, None
    best = None
    for name, val in games.items():
        n = str(name).lower()
        if n == spoken:
            return name, val
        if n in spoken or spoken in n:
            if best is None or len(n) > len(str(best[0])):
                best = (name, val)
    return best if best else (None, None)


def launch_game(cfg, spoken: str):
    """Возвращает (успех, отображаемое_имя)."""
    name, val = find_game(cfg, spoken)
    if val is None:
        # Пробуем запустить в Steam по поиску не можем — сообщаем наверх
        return False, spoken
    if isinstance(val, int) or str(val).isdigit():
        return open_target(f"steam://rungameid/{val}"), name
    return open_target(str(val)), name
