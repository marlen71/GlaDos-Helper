"""Тесты подготовки текста к произнесению.

Регрессия: Silero пропускала латиницу, поэтому «Система Aperture Science
активна» звучало как «Система активна».
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glados.speech_text import (latin_word_to_russian, prepare,  # noqa: E402
                                spell_letters, transliterate)


def _has_latin(text: str) -> bool:
    return any("a" <= c.lower() <= "z" for c in text)


@pytest.mark.parametrize("text", [
    "Система Aperture Science активна. Я слушаю вас, сэр Марлен.",
    "Запускаю Discord",
    "Открываю YouTube",
    "Открываю Steam и Telegram",
    "Запускаю Garry's Mod",
    "Открываю Yandex Music",
    "Проверьте GPU",
    "Запускаю CS2",
    "Открываю Chrome",
])
def test_no_latin_left_for_speech(text):
    """После подготовки латиницы остаться не должно — её модель не произносит."""
    assert not _has_latin(prepare(text))


def test_aperture_science_is_spoken():
    """Главная регрессия из отчёта пользователя."""
    spoken = prepare("Система Aperture Science активна")
    assert "апертура" in spoken
    assert "сайенс" in spoken
    assert not _has_latin(spoken)


@pytest.mark.parametrize("word,expected", [
    ("Discord", "дискорд"),
    ("Steam", "стим"),
    ("YouTube", "ютуб"),
    ("Telegram", "телеграм"),
    ("Chrome", "хром"),
    ("Yandex", "яндекс"),
    ("Portal", "портал"),
    ("Dota", "дота"),
])
def test_known_words_use_dictionary(word, expected):
    assert latin_word_to_russian(word) == expected


def test_possessive_not_split():
    """«Garry's Mod» не должно превращаться в «гэрри с мод»."""
    spoken = prepare("Запускаю Garry's Mod")
    assert " с " not in spoken
    assert "мод" in spoken


def test_letters_and_digits_separated():
    """«CS2» слитно читается неправильно."""
    assert "2" in prepare("Запускаю CS2")
    assert not _has_latin(prepare("Запускаю CS2"))


def test_abbreviations_spelled_out():
    assert spell_letters("gpu") == "джи пи ю"


def test_unknown_word_is_transliterated():
    """Незнакомое слово всё равно должно быть произносимым."""
    spoken = latin_word_to_russian("Notepad")
    assert spoken
    assert not _has_latin(spoken)


def test_russian_text_unchanged():
    text = "Время девятнадцать часов пятнадцать минут, сэр."
    assert prepare(text) == text


def test_numbers_preserved():
    assert "1.6" in prepare("Запускаю CS 1.6")
    assert "23.09.2026" in prepare("Напомню 23.09.2026")


def test_symbols_spoken():
    assert "и" in prepare("Steam & Discord")
    assert "&" not in prepare("Steam & Discord")


def test_quotes_removed():
    assert "«" not in prepare("Запускаю «Портал 2»")
    assert '"' not in prepare('Запускаю "Портал"')


def test_empty_input():
    assert prepare("") == ""


def test_no_double_spaces():
    assert "  " not in prepare("Открываю   Steam   и   Discord")


def test_transliterate_produces_cyrillic():
    assert not _has_latin(transliterate("something"))


# ------------------------------------------------ фраза запуска
def test_startup_phrase_is_fully_speakable():
    from glados.config import Config
    from glados.persona import Persona

    p = Persona(Config.load())
    for _ in range(20):
        phrase = p.startup()
        assert not _has_latin(prepare(phrase)), f"латиница в: {phrase}"


def test_startup_uses_configured_name():
    from glados.config import Config
    from glados.persona import Persona

    cfg = Config.load()
    cfg["wake"]["words"] = ["пятница"]
    p = Persona(cfg)
    phrases = {p.startup() for _ in range(30)}
    assert any("Пятница" in f for f in phrases)
    assert not any("Гладос" in f for f in phrases)


def test_greeting_has_no_latin():
    from glados.config import Config
    from glados.persona import Persona

    p = Persona(Config.load())
    for _ in range(20):
        assert not _has_latin(prepare(p.greeting()))
