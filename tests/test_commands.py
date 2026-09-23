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


# ------------------------------------------- нечёткое распознавание имени
@pytest.mark.parametrize("heard", [
    "гладос", "глэдос", "гладис", "ладос", "ладас", "лада", "владас",
    "складас", "сладость", "кладос", "клады", "адос", "глада",
])
def test_fuzzy_wake_variants(heard):
    """Всё, что звучит похоже на имя, должно срабатывать."""
    body, ok = strip_wake(f"{heard} привет", ["гладос"])
    assert ok, f"«{heard}» не распознано как имя"
    assert body == "привет"


@pytest.mark.parametrize("phrase", [
    "открой браузер", "включи музыку", "сколько времени",
    "запусти игру", "напомни выпить воды", "что там по погоде",
])
def test_fuzzy_no_false_positives(phrase):
    """Обычные команды без имени не должны считаться обращением."""
    _, ok = strip_wake(phrase, ["гладос"])
    assert not ok, f"«{phrase}» ошибочно принято за имя"


def test_fuzzy_can_be_disabled():
    _, ok = strip_wake("ладас привет", ["гладос"], fuzzy=False)
    assert not ok


def test_wake_word_glued_to_command():
    body, ok = strip_wake("гладосвключи дискорд", ["гладос"])
    assert ok
    assert "включи" in body


def test_custom_assistant_name():
    """Имя настраивается — проверяем на «Джарвис»."""
    body, ok = strip_wake("джарвис открой стим", ["джарвис"])
    assert ok and body == "открой стим"
    body, ok = strip_wake("жарвис открой стим", ["джарвис"])
    assert ok


def test_full_command_after_fuzzy_wake():
    """Имя распозналось неточно, но команда должна выполниться."""
    body, ok = strip_wake("складас открой игру гмод", ["гладос"])
    assert ok
    assert body == "открой игру гмод"


@pytest.mark.parametrize("day,expected", [
    (1, "первое"), (3, "третье"), (10, "десятое"), (20, "двадцатое"),
    (21, "двадцать первое"), (23, "двадцать третье"), (30, "тридцатое"),
    (31, "тридцать первое"),
])
def test_day_spoken_as_ordinal(day, expected):
    """День месяца должен звучать «двадцать третье», а не «двадцать три»."""
    from glados.textnum import day_to_words

    assert day_to_words(day) == expected


# ------------------------------------------------- живые реакции (small talk)
@pytest.mark.parametrize("phrase", [
    "спасибо", "спасибо большое", "благодарю", "спс",
    "извини", "прости", "сорри",
    "как дела", "как ты", "чем занимаешься",
    "молодец", "умница", "хорошая работа",
    "спокойной ночи", "доброй ночи",
    "кто ты", "как тебя зовут", "ты робот",
    "что ты умеешь", "какие команды",
    "расскажи анекдот", "пошути",
    "мне скучно", "я устал", "хочу есть",
    "ты меня слышишь", "что делаешь",
    "ты живая", "люблю тебя", "ты тупая",
])
def test_smalltalk_gives_real_answer(router, phrase):
    """На бытовые фразы не должно быть «не поняла команду»."""
    reply = router.handle(phrase).reply
    assert reply
    assert "не поняла" not in reply.lower()
    assert "не распознала" not in reply.lower()
    assert "не входит в мой протокол" not in reply.lower()


def test_smalltalk_can_be_disabled(tmp_path):
    from glados.commands import CommandRouter
    from glados.config import Config
    from glados.persona import Persona
    from glados.skills.reminders import ReminderStore

    cfg = Config.load()
    cfg["persona"] = {"smalltalk": False}
    cfg["storage"] = {"notes_file": str(tmp_path / "n.md"),
                      "reminders_file": str(tmp_path / "r.json")}
    r = CommandRouter(cfg, Persona(cfg), ReminderStore(tmp_path / "r.json"))
    reply = r.handle("спасибо").reply.lower()
    # При выключенной болтовне должен сработать обычный ответ «не понял»
    assert any(k in reply for k in ("не поняла", "не расслышала",
                                    "не распознала", "не знаю", "иначе"))


def test_smalltalk_does_not_hijack_commands(router):
    """Бытовые слова внутри команды не должны ломать саму команду."""
    assert "Записала" in router.handle("запиши в заметки спасибо за помощь").reply
    assert "Время" in router.handle("сколько время").reply


def test_replies_vary(router):
    """Ответы не должны повторяться подряд — иначе звучит как автоответчик."""
    seen = {router.handle("спасибо").reply for _ in range(12)}
    assert len(seen) >= 3


# ------------------------------------------------ регрессии на целые слова
def test_show_notes_does_not_shut_down(router):
    """«покажи заметки» содержит «пока» — не должно завершать работу."""
    router.handle("запиши в заметки тест")
    res = router.handle("покажи заметки")
    assert not res.stop
    assert "тест" in res.reply


@pytest.mark.parametrize("phrase", [
    "прочитай заметки", "мои заметки", "покажи заметки", "какие заметки",
])
def test_read_notes_variants(router, phrase):
    router.handle("запиши в заметки молоко")
    assert "молоко" in router.handle(phrase).reply


@pytest.mark.parametrize("phrase", [
    "какие напоминания", "покажи напоминания", "показать напоминания",
    "список напоминаний",
])
def test_list_reminders_variants(router, phrase):
    router.handle("напомни выпить воды завтра в 9")
    assert "выпить воды" in router.handle(phrase).reply


@pytest.mark.parametrize("phrase,should_stop", [
    ("пока", True), ("выключись", True), ("до свидания", True),
    ("отбой", True), ("покажи заметки", False), ("показать напоминания", False),
])
def test_shutdown_word_boundaries(router, phrase, should_stop):
    assert router.handle(phrase).stop is should_stop


# --------------------------- пунктуация от распознавания речи (регрессия)
@pytest.mark.parametrize("phrase", [
    "открой яндекс, музыка.",
    "открой яндекс музыку",
    "включи музыку",
    "запусти яндекс мьюзик",
    "открой яндекс-музыка",
    "включи музыка!",
])
def test_yandex_music_with_any_punctuation(router, monkeypatch, phrase):
    """Распознавание ставит запятые и точки как попало — это не должно мешать."""
    called = {}
    monkeypatch.setattr("glados.skills.apps.open_target",
                        lambda t: called.setdefault("t", t) or True)
    router.handle(phrase)
    assert "YandexMusic" in called.get("t", "")


def test_normalize_strips_punctuation():
    from glados.commands import normalize

    assert normalize("открой яндекс, музыка.") == "открой яндекс музыка"
    assert normalize("спасибо!") == "спасибо"
    assert normalize("открой   яндекс    музыку") == "открой яндекс музыку"


def test_normalize_keeps_dots_in_numbers():
    """Точки в датах и «кс 1.6» терять нельзя."""
    from glados.commands import normalize

    assert "1.6" in normalize("открой игру кс 1.6")
    assert "23.09.2026" in normalize("напомни 23.09.2026 в 08:00")


def test_command_with_trailing_period(router, monkeypatch):
    monkeypatch.setattr("glados.skills.apps.open_target", lambda t: True)
    assert "Время" in router.handle("сколько время?").reply


# ------------------------------------ запуск не должен засорять консоль
def test_spawn_flags_silence_child_output():
    """Логи запущенной программы не должны сыпаться в окно помощника."""
    import subprocess

    from glados.skills.apps import _spawn_flags

    flags = _spawn_flags()
    assert flags["stdout"] == subprocess.DEVNULL
    assert flags["stderr"] == subprocess.DEVNULL
    assert flags["stdin"] == subprocess.DEVNULL


def test_executable_part_strips_arguments():
    from glados.skills.apps import _executable_part

    got = _executable_part("C:/App/Update.exe --processStart App.exe")
    assert got == "C:/App/Update.exe"
