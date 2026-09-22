"""Напоминания: разбор русских дат/времени, хранение в JSON, фоновый будильник."""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from ..textnum import MONTHS_NOM, WORD_NUMS

_TIME_RE = re.compile(r"\b(\d{1,2})[:.\-](\d{2})\b")
_DATE_RE = re.compile(r"\b(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?\b")
_DATE_WORD_RE = re.compile(r"\b(\d{1,2})\s+([а-яё]+)\b")
_HOUR_RE = re.compile(r"\bв\s+(\d{1,2})\s*(?:часов|часа|час)?\b")


def parse_when(text: str, now: datetime | None = None) -> tuple[datetime | None, str]:
    """Возвращает (время_срабатывания, текст_без_временной_части)."""
    now = now or datetime.now()
    low = text.lower()
    rest = text
    day = month = year = None
    hour = minute = None

    m = _DATE_RE.search(low)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        if m.group(3):
            year = int(m.group(3))
            if year < 100:
                year += 2000
        rest = rest.replace(m.group(0), " ")
        low = low.replace(m.group(0), " ")
    else:
        m = _DATE_WORD_RE.search(low)
        if m and m.group(2) in MONTHS_NOM:
            day, month = int(m.group(1)), MONTHS_NOM[m.group(2)]
            rest = rest.replace(m.group(0), " ")
            low = low.replace(m.group(0), " ")

    if "завтра" in low:
        t = now + timedelta(days=1)
        day, month, year = t.day, t.month, t.year
        rest = re.sub("завтра", " ", rest, flags=re.I)
    elif "послезавтра" in low:
        t = now + timedelta(days=2)
        day, month, year = t.day, t.month, t.year
        rest = re.sub("послезавтра", " ", rest, flags=re.I)
    elif "сегодня" in low:
        day, month, year = now.day, now.month, now.year
        rest = re.sub("сегодня", " ", rest, flags=re.I)

    m = _TIME_RE.search(low)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        rest = rest.replace(m.group(0), " ")
    else:
        m = _HOUR_RE.search(low)
        if m:
            hour, minute = int(m.group(1)), 0
            rest = rest.replace(m.group(0), " ")

    # "через 10 минут / через час / через два часа"
    m = re.search(r"через\s+([а-яё]+|\d+)\s*(минут\w*|час\w*|секунд\w*)", low)
    if m:
        raw = m.group(1)
        qty = int(raw) if raw.isdigit() else WORD_NUMS.get(raw, 1)
        unit = m.group(2)
        delta = (timedelta(minutes=qty) if unit.startswith("минут")
                 else timedelta(hours=qty) if unit.startswith("час")
                 else timedelta(seconds=qty))
        rest = rest.replace(m.group(0), " ")
        return now + delta, _clean(rest)

    if hour is None and day is None:
        return None, _clean(rest)

    target = now.replace(second=0, microsecond=0)
    if day:
        target = target.replace(day=day, month=month or now.month, year=year or now.year)
    if hour is not None:
        target = target.replace(hour=hour % 24, minute=minute or 0)
    if target <= now:
        target += timedelta(days=1)
    return target, _clean(rest)


def _clean(text: str) -> str:
    text = re.sub(r"\b(на|в|по|московскому|москве|времени|напоминалку|напоминание|"
                  r"заметку|напомни|запиши|поставь|мне|будильник)\b", " ", text,
                  flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:!?-«»\"'")
    return text


class ReminderStore:
    def __init__(self, path: Path):
        self.path = path
        self.items: list[dict] = []
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.items = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.items = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    def add(self, when: datetime, text: str) -> dict:
        item = {"at": when.isoformat(timespec="minutes"), "text": text, "done": False}
        with self._lock:
            self.items.append(item)
            self.items.sort(key=lambda x: x["at"])
            self.save()
        return item

    def pending(self) -> list[dict]:
        return [i for i in self.items if not i["done"]]

    def due(self, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now()
        out = []
        with self._lock:
            for i in self.items:
                if not i["done"] and datetime.fromisoformat(i["at"]) <= now:
                    i["done"] = True
                    out.append(i)
            if out:
                self.save()
        return out


class ReminderWatcher(threading.Thread):
    """Фоновая проверка напоминаний раз в 20 секунд."""

    def __init__(self, store: ReminderStore, notify):
        super().__init__(daemon=True)
        self.store = store
        self.notify = notify
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(20):
            for item in self.store.due():
                self.notify(item["text"])

    def stop(self) -> None:
        self._stop.set()
