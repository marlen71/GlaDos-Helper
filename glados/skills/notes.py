"""Заметки: запись текста или содержимого буфера обмена в markdown-файл."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def clipboard_text() -> str:
    try:
        import pyperclip

        return (pyperclip.paste() or "").strip()
    except Exception:
        return ""


def add_note(path: Path, text: str) -> None:
    stamp = datetime.now().strftime("%d.%m.%Y %H:%M")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Заметки GLaDOS\n\n", encoding="utf-8")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- [{stamp}] {text}\n")


def last_notes(path: Path, n: int = 5) -> list[str]:
    if not path.exists():
        return []
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
             if l.strip().startswith("- [")]
    return lines[-n:]
