"""Маршрутизация голосовых команд GLaDOS."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .fuzzy import match_any, sounds_like
from .persona import Persona
from .smalltalk import SmallTalk, maybe_banter
from .skills import apps, notes, power
from .skills.reminders import ReminderStore, parse_when
from .textnum import date_to_words, time_to_words


@dataclass
class Result:
    reply: str
    stop: bool = False           # завершить работу ассистента
    pending_confirm: str | None = None   # ждём "да"/"нет"


def normalize(text: str) -> str:
    """Приводит фразу к единому виду перед разбором.

    Распознавание расставляет запятые и точки как ему вздумается
    («открой яндекс, музыка.»), поэтому пунктуацию убираем, а лишние
    пробелы схлопываем — иначе составные названия не совпадают.
    """
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s:./-]", " ", text, flags=re.U)
    # Точку оставляем только между цифрами («кс 1.6», «23.09.2026»),
    # в остальных случаях это конец предложения.
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_wake(text: str, wake_words, fuzzy: bool = True) -> tuple[str, bool]:
    """Убирает слово активации. Возвращает (остаток, было_ли_обращение).

    Сначала ищем точное совпадение, затем — похожее по звучанию.
    Распознавание редко выдаёт имя точно: «Гладос» легко превращается
    в «Лада», «Складас» или «Сладость».
    """
    t = normalize(text)
    if not t:
        return "", False

    # 1) точное совпадение в начале фразы
    for w in wake_words:
        w = normalize_word_simple(w)
        if not w:
            continue
        if t.startswith(w):
            return t[len(w):].strip(" ,.!?"), True
        pos = t.find(w)
        if pos != -1 and pos <= 12:
            return (t[:pos] + t[pos + len(w):]).strip(" ,.!?"), True

    if not fuzzy:
        return t, False

    # 2) похожее по звучанию — проверяем только первые слова фразы
    words = t.split()
    for i, word in enumerate(words[:3]):
        if match_any(word, wake_words):
            rest = " ".join(words[:i] + words[i + 1:])
            return rest.strip(" ,.!?"), True

    # 3) имя могло слипнуться с соседним словом: «гладосвключи»
    if words:
        first = words[0]
        for w in wake_words:
            wn = normalize_word_simple(w)
            if len(wn) >= 4 and len(first) > len(wn) and sounds_like(first[:len(wn)], w):
                rest = first[len(wn):] + " " + " ".join(words[1:])
                return rest.strip(" ,.!?"), True

    return t, False


def normalize_word_simple(word: str) -> str:
    return word.lower().replace("ё", "е").strip()


def _has(text: str, *words) -> bool:
    return any(w in text for w in words)


def _word(text: str, *words) -> bool:
    """Совпадение по ЦЕЛОМУ слову.

    Нужно там, где подстрока даёт ложные срабатывания: «покажи» содержит
    «пока», «выходные» содержит «выход».
    """
    tokens = set(re.findall(r"\w+", text, flags=re.U))
    return any(w in tokens for w in words)


class CommandRouter:
    OPEN_VERBS = ("открой", "открыть", "включи", "включить", "запусти",
                  "запустить", "врубай", "врубить", "стартуй")

    def __init__(self, cfg, persona: Persona, reminders: ReminderStore):
        self.cfg = cfg
        self.p = persona
        self.reminders = reminders
        self.notes_path = cfg.data_file("storage.notes_file", "data/notes.md")
        self._confirm: str | None = None
        self.smalltalk = (SmallTalk(persona, cfg)
                          if cfg.get_path("persona.smalltalk", True) else None)
        self.banter = bool(cfg.get_path("persona.banter", True))
        self.banter_chance = float(cfg.get_path("persona.banter_chance", 0.25))

    # ---------------------------------------------------------------- main
    def handle(self, raw: str) -> Result:
        text = normalize(raw)
        if not text:
            return Result("")

        # Ожидание подтверждения (выключение/перезагрузка)
        if self._confirm:
            action, self._confirm = self._confirm, None
            if _has(text, "да", "подтверждаю", "давай", "конечно", "ага"):
                return self._do_power(action)
            return Result(f"Отменяю, {self.p.short}.")

        if _has(text, "отмена", "отмени"):
            power.cancel()
            return Result(f"Отменено, {self.p.short}.")

        # --- Приветствие ---
        if _has(text, "привет", "здравствуй", "здорово", "доброе утро",
                "добрый день", "добрый вечер", "хай"):
            return Result(self.p.greeting())

        # --- Завершение работы ассистента ---
        if (_word(text, "выключись", "отключись", "спи", "выход", "пока",
                  "отбой", "выключайся")
                or _has(text, "до свидания", "до встречи")
                or re.search(r"(выключи|отключи|вырубай|заверши)\w*\s+(гладос|тебя|себя)", text)
                or _has(text, "выключить гладос", "выключи гладос")):
            return Result(self.p.farewell(), stop=True)

        # --- Питание ПК ---
        if re.search(r"(выключи|выключить|отключи)\w*\s+(компьютер|комп|пк|систему)", text):
            return self._ask_power("shutdown")
        if re.search(r"(перезагрузи|перезагрузить|ребут|рестарт)\w*\s*(компьютер|комп|пк|систему)?", text):
            return self._ask_power("reboot")

        # --- Время / дата ---
        if _has(text, "сколько время", "сколько времени", "который час",
                "текущее время", "время сейчас"):
            now = datetime.now()
            return Result(f"Время {time_to_words(now.hour, now.minute)}, {self.p.short}.")
        if _has(text, "какая сегодня дата", "какое сегодня число", "какой сегодня день",
                "сегодняшняя дата", "какая дата"):
            today = datetime.now().date()
            reply = date_to_words(today)
            if self.p.is_birthday(today):
                return Result(f"{reply}. {self.p.birthday_note()}")
            return Result(f"Сегодня {reply}, {self.p.short}.")

        # --- Заметки ---
        # Чтение проверяем ПЕРВЫМ: «прочитай заметки» иначе попадает в запись
        if _has(text, "прочитай заметк", "прочти заметк", "мои заметк",
                "последние заметк", "покажи заметк", "какие заметк",
                "зачитай заметк"):
            return self._read_notes()
        if _has(text, "заметк", "запиши в заметки", "заметку"):
            return self._note(text)


        # --- Напоминания ---
        if _has(text, "напомни", "напоминалк", "напоминание", "будильник", "таймер"):
            return self._reminder(raw)
        if _has(text, "какие напоминан", "мои напоминан", "что запланировано",
                "покажи напоминан", "показать напоминан", "список напоминан",
                "прочитай напоминан"):
            return self._list_reminders()

        # --- Запуск приложений и игр ---
        if text.startswith(self.OPEN_VERBS) or _has(text, *self.OPEN_VERBS):
            return self._open(text)

        # Бытовая фраза — отвечаем по-человечески, а не «не поняла команду»
        if self.smalltalk:
            reply = self.smalltalk.handle(text)
            if reply:
                return Result(reply)

        return Result(self.p.unknown())

    # ------------------------------------------------------------- helpers
    def _with_banter(self, reply: str) -> str:
        """Иногда добавляет короткую реплику «от себя» — звучит живее."""
        if not self.banter:
            return reply
        extra = maybe_banter(self.banter_chance)
        return f"{reply} {extra}".strip() if extra else reply

    def _ask_power(self, action: str) -> Result:
        if self.cfg.get_path("power.confirm", True):
            self._confirm = action
            word = "выключение" if action == "shutdown" else "перезагрузку"
            return Result(f"Подтвердите {word} компьютера, {self.p.short}. Скажите да или нет.")
        return self._do_power(action)

    def _do_power(self, action: str) -> Result:
        if action == "shutdown":
            delay = int(self.cfg.get_path("power.shutdown_delay", 20))
            power.shutdown(delay)
            return Result(f"Выключаю компьютер через {delay} секунд, {self.p.short}. "
                          f"Скажите отмена, если передумаете.")
        delay = int(self.cfg.get_path("power.reboot_delay", 15))
        power.reboot(delay)
        return Result(f"Перезагружаю систему через {delay} секунд, {self.p.short}.")

    def _open(self, text: str) -> Result:
        # игра "Название"
        m = re.search(r"(?:игру|игра|поиграть в)\s+(.+)", text)
        if m:
            spoken = m.group(1).strip()
            ok, name = apps.launch_game(self.cfg, spoken)
            if ok:
                return Result(f"{self.p.ack()} Запускаю {name}.")
            return Result(f"Не нашла игру «{spoken}» в списке, {self.p.short}. "
                          f"Добавьте её в config.yaml, раздел games.")

        aliases = {
            "browser": ("браузер", "хром", "chrome", "интернет", "гугл хром"),
            "steam": ("стим", "steam"),
            "discord": ("дискорд", "discord", "диск"),
            "youtube": ("ютуб", "youtube", "ютьюб"),
            "telegram": ("телеграм", "telegram", "тг"),
            "yandex_music": ("яндекс музык", "яндекс мьюзик", "яндексмузык",
                             "yandex music", "яндекс-музык", "яндекс",
                             "музык"),
            "spotify": ("спотифай", "spotify"),
            "explorer": ("проводник", "папку", "файлы"),
        }
        # Сначала проверяем более длинные/специфичные названия
        for key, words in sorted(aliases.items(),
                                 key=lambda kv: -max(len(w) for w in kv[1])):
            if _has(text, *words):
                if apps.open_app(self.cfg, key):
                    return Result(self._with_banter(self.p.ack()))
                return Result(f"Не удалось запустить {words[0]}, {self.p.short}. "
                              f"Проверьте путь в config.yaml.")

        # может быть это игра без слова "игру"
        name, val = apps.find_game(self.cfg, re.sub("|".join(self.OPEN_VERBS), "", text).strip())
        if val is not None:
            ok, name = apps.launch_game(self.cfg, str(name))
            if ok:
                return Result(f"{self.p.ack()} Запускаю {name}.")
        return Result(f"Не знаю такого приложения, {self.p.short}.")

    def _read_notes(self) -> Result:
        items = notes.last_notes(self.notes_path)
        if not items:
            return Result(f"Заметок пока нет, {self.p.short}.")
        body = "; ".join(i.split("] ", 1)[-1] for i in items)
        return Result(f"Последние заметки: {body}.")

    def _note(self, text: str) -> Result:
        if _has(text, "буфер", "из буфера", "ссылку"):
            content = notes.clipboard_text()
            if not content:
                return Result(f"Буфер обмена пуст, {self.p.short}.")
            notes.add_note(self.notes_path, content)
            return Result(f"Хорошо, {self.p.short}! Записала из буфера обмена.")
        m = re.search(r"заметки?\w*\s+(.+)", text)
        content = m.group(1).strip() if m else ""
        if not content:
            return Result(f"Что именно записать, {self.p.short}?")
        notes.add_note(self.notes_path, content)
        return Result(f"Хорошо, {self.p.short}! Записала: {content}.")

    def _reminder(self, raw: str) -> Result:
        when, body = parse_when(raw)
        body = re.sub(r"^\s*(гладос|глэдос)\b", "", body, flags=re.I).strip(" ,.-")
        if not body:
            body = "напоминание"
        if when is None:
            return Result(f"Не поняла время, {self.p.short}. "
                          f"Скажите, например: напомни выпить воды завтра в 08:00.")
        self.reminders.add(when, body)
        return Result(f"Хорошо, слушаюсь, {self.p.short}. Напомню "
                      f"{when.strftime('%d.%m.%Y')} в "
                      f"{time_to_words(when.hour, when.minute)}: {body}.")

    def _list_reminders(self) -> Result:
        items = self.reminders.pending()
        if not items:
            return Result(f"Активных напоминаний нет, {self.p.short}.")
        parts = []
        for i in items[:5]:
            dt = datetime.fromisoformat(i["at"])
            parts.append(f"{dt.strftime('%d.%m')} в {dt.strftime('%H:%M')} — {i['text']}")
        return Result("Запланировано: " + "; ".join(parts) + ".")
