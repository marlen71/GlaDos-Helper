"""Подготовка текста к произнесению.

Silero — русская модель: латиницу, цифры и символы она просто пропускает.
Из-за этого «Система Aperture Science активна» звучала как «Система активна»,
а «Запускаю Discord» — как «Запускаю».

Здесь латиница переводится в кириллицу так, как слово читается вслух.
Известные названия берутся из словаря, остальное транслитерируется по правилам.
"""
from __future__ import annotations

import re

#: Как произносятся известные названия. Ключ — в нижнем регистре.
KNOWN_WORDS = {
    # Программы и сервисы
    "discord": "дискорд",
    "steam": "стим",
    "youtube": "ютуб",
    "telegram": "телеграм",
    "spotify": "спотифай",
    "chrome": "хром",
    "firefox": "файрфокс",
    "windows": "виндоус",
    "explorer": "эксплорер",
    "google": "гугл",
    "yandex": "яндекс",
    "music": "мьюзик",
    "opera": "опера",
    "edge": "эдж",
    "skype": "скайп",
    "zoom": "зум",
    "obs": "о би эс",
    "vlc": "ви эл си",
    "pc": "пи си",
    "ok": "окей",
    "wifi": "вайфай",
    "email": "имейл",
    "internet": "интернет",
    "online": "онлайн",
    "offline": "офлайн",

    # Игры
    "portal": "портал",
    "dota": "дота",
    "garry": "гэрри",
    "mod": "мод",
    "counter": "каунтер",
    "strike": "страйк",
    "terraria": "террария",
    "witcher": "ведьмак",
    "minecraft": "майнкрафт",
    "fortnite": "фортнайт",
    "valorant": "валорант",
    "roblox": "роблокс",
    "among": "эмонг",
    "us": "ас",
    "go": "го",
    "gta": "джи ти эй",
    "rdr": "эр ди эр",
    "cs": "ка эс",
    "csgo": "ка эс го",
    "cs2": "ка эс два",

    # Вымышленные названия
    "aperture": "апертура",
    "garrys": "гэрриз",
    "gmod": "гмод",
    "science": "сайенс",
    "glados": "гладос",
    "jarvis": "джарвис",
    "friday": "пятница",

    # Часто встречающиеся слова
    "ready": "реди",
    "start": "старт",
    "stop": "стоп",
    "play": "плей",
    "error": "ошибка",
    "update": "апдейт",
    "app": "эпп",
    "web": "веб",
}

#: Как произносятся отдельные латинские буквы (для аббревиатур)
LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф",
    "g": "джи", "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл",
    "m": "эм", "n": "эн", "o": "оу", "p": "пи", "q": "кью", "r": "ар",
    "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл ю", "x": "экс",
    "y": "уай", "z": "зед",
}

#: Буквосочетания — проверяются раньше одиночных букв
_DIGRAPHS = [
    ("sch", "ш"), ("tch", "ч"), ("sh", "ш"), ("ch", "ч"), ("ph", "ф"),
    ("th", "т"), ("ck", "к"), ("qu", "кв"), ("wh", "в"), ("gh", "г"),
    ("oo", "у"), ("ee", "и"), ("ea", "и"), ("ou", "ау"), ("ow", "ау"),
    ("ay", "эй"), ("ai", "эй"), ("ey", "эй"), ("oy", "ой"), ("oi", "ой"),
    ("au", "о"), ("aw", "о"), ("ie", "и"), ("igh", "ай"),
]

#: Одиночные буквы
_SINGLES = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "й", "z": "з",
}

#: Символы, которые надо проговаривать
SYMBOLS = {
    "&": " и ", "@": " собака ", "%": " процентов ", "+": " плюс ",
    "=": " равно ", "№": " номер ", "€": " евро ", "$": " долларов ",
    "₽": " рублей ", "°": " градусов ",
}


def _is_latin(word: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z'’\-]*", word))


def _looks_like_abbreviation(word: str) -> bool:
    """CS, GPU, VLC — читаются по буквам."""
    if len(word) <= 4 and word.isupper():
        return True
    # слово без гласных: «cs», «pc», «js»
    return len(word) <= 4 and not set(word.lower()) & set("aeiouy")


def spell_letters(word: str) -> str:
    return " ".join(LETTER_NAMES.get(c, c) for c in word.lower())


def transliterate(word: str) -> str:
    """Побуквенная транслитерация по правилам чтения."""
    w = word.lower()
    out = []
    i = 0
    while i < len(w):
        # непроизносимая «e» на конце: «game» -> «гейм», а не «гейме»
        if i == len(w) - 1 and w[i] == "e" and len(w) > 3:
            break
        for src, dst in _DIGRAPHS:
            if w.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            out.append(_SINGLES.get(w[i], ""))
            i += 1
    return "".join(out)


def latin_word_to_russian(word: str) -> str:
    """Одно латинское слово -> как оно читается по-русски."""
    clean = word.strip("'’-")
    if not clean:
        return word

    known = KNOWN_WORDS.get(clean.lower())
    if known:
        return known

    # Составное название из известных частей: «CounterStrike»
    parts = re.findall(r"[A-Z][a-z]+|[a-z]+|[A-Z]+", clean)
    if len(parts) > 1:
        pieces = [KNOWN_WORDS.get(p.lower()) or transliterate(p) for p in parts]
        if all(pieces):
            return " ".join(pieces)

    if _looks_like_abbreviation(clean):
        return spell_letters(clean)

    return transliterate(clean) or clean


def _split_letters_digits(text: str) -> str:
    """Разделяет буквы и цифры: «CS2» -> «CS 2», иначе читается слитно."""
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    return re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)


def prepare(text: str) -> str:
    """Готовит любую строку к произнесению русской моделью.

    Латиница переводится в кириллицу, спецсимволы проговариваются,
    лишние пробелы убираются.
    """
    if not text:
        return text

    for sym, spoken in SYMBOLS.items():
        text = text.replace(sym, spoken)

    # Убираем кавычки — модель произносит их как паузы
    text = text.replace("«", "").replace("»", "").replace('"', "")
    text = _split_letters_digits(text)

    # Притяжательный апостроф: Garry's -> Garrys (иначе «гэрри с мод»)
    text = re.sub(r"([A-Za-z])['’]s\b", r"\1s", text)

    def repl(match: re.Match) -> str:
        return latin_word_to_russian(match.group(0))

    text = re.sub(r"[A-Za-z][A-Za-z'’]*", repl, text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
