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


def open_target(target: str) -> bool:
    """Универсальный запуск: URL, команда start, exe-путь, имя в PATH."""
    target = _expand(str(target)).strip()
    if not target:
        return False
    try:
        if target.startswith(("http://", "https://", "steam://")):
            webbrowser.open(target)
            return True
        if sys.platform.startswith("win"):
            if target.lower().startswith("start "):
                subprocess.Popen(target, shell=True)
            elif os.path.exists(target.split(" --")[0]):
                subprocess.Popen(target, shell=True)
            else:
                os.startfile(target)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(target, shell=True)
        return True
    except Exception as e:
        log.error("Не удалось запустить %r: %s", target, e)
        return False


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
