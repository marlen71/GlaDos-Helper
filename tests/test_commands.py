"""Тесты разбора команд — работают без микрофона, torch и интернета."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glados.commands import CommandRouter, strip_wake  # noqa: E402
from glados.config import Config  # noqa: E402
from glados.persona import Persona  # noqa: E402
from glados.skills.reminders import ReminderStore, parse_when  # noqa: E402
from glados.textnum import date_to_words, number_to_words, time_to_words  # noqa: E402


@pytest.fixture()
def router(tmp_path, monkeypatch):
    cfg = Config.load()
    cfg["storage"] = {"notes_file": str(tmp_path / "notes.md"),
                      "reminders_file": str(tmp_path / "rem.json")}
    store = ReminderStore(Path(cfg["storage"]["reminders_file"]))
    return CommandRouter(cfg, Persona(cfg), store)


def test_wake_word():
    body, ok = strip_wake("Гладос, привет!", ["гладос"])
    assert ok and body == "привет"
    _, ok2 = strip_wake("просто фраза", ["гладос"])
    assert not ok2


def test_greeting(router):
    assert "Марлен" in router.handle("привет").reply


def test_time(router):
    assert "Время" in router.handle("сколько время").reply


def test_date(router):
    assert "Сегодня" in router.handle("какая сегодня дата").reply or \
           "рождения" in router.handle("какая сегодня дата").reply


def test_open_discord(router, monkeypatch):
    called = {}
    monkeypatch.setattr("glados.skills.apps.open_target",
                        lambda t: called.setdefault("t", t) or True)
    reply = router.handle("включи дискорд").reply
    assert "Discord" in called["t"] or "discord" in called["t"].lower()
    assert reply


def test_open_game(router, monkeypatch):
    monkeypatch.setattr("glados.skills.apps.open_target", lambda t: True)
    assert "портал 2" in router.handle("открой игру портал 2").reply.lower()


def test_note_text(router):
    reply = router.handle("запиши в заметки купить молоко").reply
    assert "купить молоко" in reply


def test_note_clipboard(router, monkeypatch):
    monkeypatch.setattr("glados.skills.notes.clipboard_text",
                        lambda: "https://example.com")
    assert "буфера" in router.handle("запиши в заметки ссылку из буфера обмена").reply


def test_power_confirm(router):
    r1 = router.handle("выключи компьютер")
    assert "Подтвердите" in r1.reply
    r2 = router.handle("нет")
    assert "Отменяю" in r2.reply


def test_stop_assistant(router):
    assert router.handle("выключись").stop is True


def test_parse_when_explicit():
    when, body = parse_when("напомни принять таблетку и выпить водички "
                            "на 23.09.2026 в 08:00", datetime(2026, 9, 22, 10, 0))
    assert when == datetime(2026, 9, 23, 8, 0)
    assert "таблетку" in body


def test_parse_when_relative():
    now = datetime(2026, 9, 22, 10, 0)
    when, _ = parse_when("напомни через 10 минут выпить воды", now)
    assert when == datetime(2026, 9, 22, 10, 10)


def test_reminder_command(router):
    reply = router.handle("напомни принять таблетку 31.12.2030 в 08:00").reply
    assert "31.12.2030" in reply and "восемь часов" in reply


def test_explicit_past_date_is_kept():
    """Явно названную дату не переносим на следующий день."""
    when, _ = parse_when("напомни выпить воды 23.09.2026 в 08:00",
                         datetime(2026, 9, 23, 20, 0))
    assert when == datetime(2026, 9, 23, 8, 0)


def test_time_only_rolls_to_tomorrow():
    """Если названо только время и оно прошло — значит, завтра."""
    when, _ = parse_when("напомни выпить воды в 08:00", datetime(2026, 9, 23, 20, 0))
    assert when == datetime(2026, 9, 24, 8, 0)


def test_numbers():
    assert number_to_words(19) == "девятнадцать"
    assert time_to_words(19, 15) == "девятнадцать часов пятнадцать минут"
    assert "сентября" in date_to_words(datetime(2026, 9, 22).date())


def test_open_yandex_music(router, monkeypatch):
    called = {}
    monkeypatch.setattr("glados.skills.apps.open_target",
                        lambda t: called.setdefault("t", t) or True)
    router.handle("открой яндекс музыку")
    assert "YandexMusic" in called["t"]


def test_open_music_shortcut(router, monkeypatch):
    called = {}
    monkeypatch.setattr("glados.skills.apps.open_target",
                        lambda t: called.setdefault("t", t) or True)
    router.handle("включи музыку")
    assert "YandexMusic" in called["t"]


@pytest.mark.parametrize("phrase,appid", [
    ("открой игру гаррис мод", "4000"),
    ("запусти игру гмод", "4000"),
    ("открой игру ксго", "4465480"),
    ("включи игру кс 1.6", "10"),
])
def test_new_games(router, monkeypatch, phrase, appid):
    called = {}
    monkeypatch.setattr("glados.skills.apps.open_target",
                        lambda t: called.setdefault("t", t) or True)
    router.handle(phrase)
    assert called["t"] == f"steam://rungameid/{appid}"


def test_recognizer_compute_type_autofix():
    from glados.stt import Recognizer
    assert Recognizer._fix_compute("cuda", "int8") == "float16"
    assert Recognizer._fix_compute("cpu", "float16") == "int8"
    assert Recognizer._fix_compute("cpu", "int8") == "int8"


def test_cuda_error_detection():
    from glados.stt import Recognizer
    r = Recognizer.__new__(Recognizer)
    assert r._is_cuda_error(RuntimeError("Library cublas64_12.dll is not found"))
    assert r._is_cuda_error(RuntimeError("cudnn_ops64_9.dll missing"))
    assert not r._is_cuda_error(ValueError("что-то другое"))
