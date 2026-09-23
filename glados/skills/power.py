"""Выключение и перезагрузка компьютера (с возможностью отмены)."""
from __future__ import annotations

import subprocess
import sys


def shutdown(delay: int = 20) -> bool:
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["shutdown", "/s", "/t", str(delay)], check=False)
        else:
            subprocess.run(["shutdown", "-h", f"+{max(1, delay // 60)}"], check=False)
        return True
    except Exception:
        return False


def reboot(delay: int = 15) -> bool:
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["shutdown", "/r", "/t", str(delay)], check=False)
        else:
            subprocess.run(["shutdown", "-r", f"+{max(1, delay // 60)}"], check=False)
        return True
    except Exception:
        return False


def cancel() -> bool:
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["shutdown", "/a"], check=False)
        else:
            subprocess.run(["shutdown", "-c"], check=False)
        return True
    except Exception:
        return False
