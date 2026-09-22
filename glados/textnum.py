"""Числа словами по-русски + парсинг чисел из речи (нужно и для TTS, и для STT)."""
from __future__ import annotations

ONES_M = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь",
          "восемь", "девять", "десять", "одиннадцать", "двенадцать",
          "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
          "семнадцать", "восемнадцать", "девятнадцать"]
ONES_F = ONES_M.copy()
ONES_F[1], ONES_F[2] = "одна", "две"
TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят",
        "семьдесят", "восемьдесят", "девяносто"]
HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот",
            "семьсот", "восемьсот", "девятьсот"]

MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
              "августа", "сентября", "октября", "ноября", "декабря"]
MONTHS_NOM = {
    "январь": 1, "января": 1, "февраль": 2, "февраля": 2, "март": 3, "марта": 3,
    "апрель": 4, "апреля": 4, "май": 5, "мая": 5, "июнь": 6, "июня": 6,
    "июль": 7, "июля": 7, "август": 8, "августа": 8, "сентябрь": 9, "сентября": 9,
    "октябрь": 10, "октября": 10, "ноябрь": 11, "ноября": 11, "декабрь": 12, "декабря": 12,
}
WORD_NUMS = {w: i for i, w in enumerate(ONES_M)}
WORD_NUMS.update({"одна": 1, "две": 2, "первого": 1, "второго": 2, "третьего": 3})
for i, t in enumerate(TENS):
    if t:
        WORD_NUMS[t] = i * 10


def number_to_words(n: int, feminine: bool = False) -> str:
    """0..999 словами (нам хватает для времени и дат)."""
    if n < 0:
        return "минус " + number_to_words(-n, feminine)
    if n >= 1000:
        th, rest = divmod(n, 1000)
        head = f"{number_to_words(th, True)} {plural(th, 'тысяча', 'тысячи', 'тысяч')}"
        return head if rest == 0 else f"{head} {number_to_words(rest, feminine)}"
    words = []
    h, n = divmod(n, 100)
    if h:
        words.append(HUNDREDS[h])
    if n < 20:
        if n or not words:
            words.append((ONES_F if feminine else ONES_M)[n])
    else:
        t, o = divmod(n, 10)
        words.append(TENS[t])
        if o:
            words.append((ONES_F if feminine else ONES_M)[o])
    return " ".join(words)


def plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n) % 100
    if 11 <= n <= 14:
        return many
    n %= 10
    if n == 1:
        return one
    if 2 <= n <= 4:
        return few
    return many


def time_to_words(hour: int, minute: int) -> str:
    h = f"{number_to_words(hour)} {plural(hour, 'час', 'часа', 'часов')}"
    if minute == 0:
        return f"{h} ровно"
    m = f"{number_to_words(minute, True)} {plural(minute, 'минута', 'минуты', 'минут')}"
    return f"{h} {m}"


def date_to_words(d) -> str:
    year = d.year
    return (f"{number_to_words(d.day)} {MONTHS_GEN[d.month - 1]} "
            f"{year_to_words(year)} года")


def year_to_words(year: int) -> str:
    th, rest = divmod(year, 1000)
    words = f"{ONES_F[th] if th < 20 else number_to_words(th, True)} тысяч"
    words = {1: "одна тысяча", 2: "две тысячи"}.get(th, words)
    if rest == 0:
        return words
    h, r = divmod(rest, 100)
    tail = []
    if h:
        tail.append(HUNDREDS[h])
    if r:
        if r < 20:
            tail.append(_ordinal(r))
        else:
            t, o = divmod(r, 10)
            if o:
                tail.append(TENS[t])
                tail.append(_ordinal(o))
            else:
                tail.append(_ordinal_tens(t))
    else:
        tail[-1] = _ordinal_hundreds(h)
    return " ".join([words] + tail)


_ORD = {1: "первого", 2: "второго", 3: "третьего", 4: "четвёртого", 5: "пятого",
        6: "шестого", 7: "седьмого", 8: "восьмого", 9: "девятого", 10: "десятого",
        11: "одиннадцатого", 12: "двенадцатого", 13: "тринадцатого",
        14: "четырнадцатого", 15: "пятнадцатого", 16: "шестнадцатого",
        17: "семнадцатого", 18: "восемнадцатого", 19: "девятнадцатого"}
_ORD_TENS = {2: "двадцатого", 3: "тридцатого", 4: "сорокового", 5: "пятидесятого",
             6: "шестидесятого", 7: "семидесятого", 8: "восьмидесятого", 9: "девяностого"}
_ORD_HUND = {1: "сотого", 2: "двухсотого", 3: "трёхсотого", 4: "четырёхсотого",
             5: "пятисотого", 6: "шестисотого", 7: "семисотого", 8: "восьмисотого",
             9: "девятисотого"}


def _ordinal(n: int) -> str:
    return _ORD.get(n, str(n))


def _ordinal_tens(n: int) -> str:
    return _ORD_TENS.get(n, str(n * 10))


def _ordinal_hundreds(n: int) -> str:
    return _ORD_HUND.get(n, str(n * 100))
