"""Нечёткое сопоставление слова активации.

Распознавание почти никогда не выдаёт имя ассистента точно: «Гладос»
превращается в «Лада», «Ладас», «Складас», «Сладость». Перечислять все
варианты руками бесполезно — их десятки.

Здесь два механизма:
  1. фонетический код (упрощённый Soundex для кириллицы) — слова, которые
     звучат похоже, получают одинаковый код;
  2. расстояние Левенштейна — ловит опечатки распознавания.
"""
from __future__ import annotations

from functools import lru_cache

#: Группы согласных, неразличимые на слух в быстрой речи
_CONSONANT_GROUPS = {
    "б": "1", "п": "1", "в": "1", "ф": "1",
    "г": "2", "к": "2", "х": "2",
    "д": "3", "т": "3",
    "ж": "4", "ш": "4", "щ": "4", "ч": "4", "ц": "4", "з": "5", "с": "5",
    "л": "6", "р": "6",
    "м": "7", "н": "7",
}
_VOWELS = set("аеиоуыэюяё")


def normalize_word(word: str) -> str:
    w = word.lower().replace("ё", "е").strip()
    return "".join(c for c in w if c.isalnum())


@lru_cache(maxsize=4096)
def phonetic_code(word: str) -> str:
    """Грубый фонетический код: согласные по группам, гласные отброшены."""
    w = normalize_word(word)
    if not w:
        return ""
    code = []
    prev = ""
    for ch in w:
        if ch in _VOWELS:
            prev = ""
            continue
        c = _CONSONANT_GROUPS.get(ch, "")
        if c and c != prev:
            code.append(c)
        prev = c
    return "".join(code)


@lru_cache(maxsize=8192)
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """0.0 — совсем разные, 1.0 — одинаковые."""
    a, b = normalize_word(a), normalize_word(b)
    if not a or not b:
        return 0.0
    longest = max(len(a), len(b))
    return 1.0 - levenshtein(a, b) / longest


def sounds_like(word: str, target: str, threshold: float = 0.6) -> bool:
    """Похоже ли `word` на `target` по звучанию."""
    word_n, target_n = normalize_word(word), normalize_word(target)
    if not word_n or not target_n:
        return False
    if word_n == target_n:
        return True

    # Слишком короткие слова не сравниваем — много ложных срабатываний
    if len(word_n) < 3:
        return False

    if similarity(word_n, target_n) >= threshold:
        return True

    # Совпадение фонетических кодов + близкая длина
    wc, tc = phonetic_code(word_n), phonetic_code(target_n)
    if wc and wc == tc and abs(len(word_n) - len(target_n)) <= 3:
        return True

    # Код имени целиком «сидит» внутри кода услышанного слова
    # («складас» -> 52635 содержит 2635 от «гладос»), либо наоборот —
    # распознаватель проглотил согласную («лада» -> 63 внутри 2635).
    if len(tc) >= 3 and len(wc) >= 2:
        if tc in wc and len(wc) - len(tc) <= 2:
            return True
        # Обратный случай — распознаватель проглотил согласную («лада» от
        # «гладос»). Требуем совпадения ядра кода, иначе к «гладос» начинают
        # липнуть случайные короткие слова вроде «игру».
        if (wc in tc and len(tc) - len(wc) <= 2 and len(word_n) >= 4
                and _shares_core(wc, tc)
                and similarity(word_n, target_n) >= 0.35):
            return True

    # Имя «проглочено» внутри слова: «сладость» содержит «ладос»
    if len(target_n) >= 4:
        for size in (len(target_n), len(target_n) - 1):
            for i in range(len(word_n) - size + 1):
                if similarity(word_n[i:i + size], target_n) >= 0.75:
                    return True
    return False


def _shares_core(short: str, full: str) -> bool:
    """Начинается ли ядро длинного кода с короткого (без первой согласной).

    «лада» -> 63, «гладос» -> 2635: отбросив начальную 2, получаем 635,
    которое начинается на 63. А «игру» -> 26 такой проверки не проходит.
    """
    if full.startswith(short):
        return True
    return len(full) > 1 and full[1:].startswith(short)


def match_any(word: str, targets, threshold: float = 0.6) -> str | None:
    """Возвращает первый подходящий вариант из списка или None."""
    for t in targets:
        if sounds_like(word, t, threshold):
            return t
    return None
