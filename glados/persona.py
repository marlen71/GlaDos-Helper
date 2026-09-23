"""Личность GLaDOS: обращения, фразы, лёгкий сарказм Aperture Science."""
from __future__ import annotations

import random
from datetime import date


class Persona:
    def __init__(self, cfg):
        self.name = cfg.get_path("user.name", "")
        self.title = cfg.get_path("user.title", "сэр")
        self.birthday = cfg.get_path("user.birthday", "")
        words = cfg.get_path("wake.words", []) or []
        self.assistant_name = str(words[0]).capitalize() if words else ""

    # --- обращения ---
    @property
    def address(self) -> str:
        """'сэр Марлен' / 'сэр' / 'Марлен'."""
        parts = [p for p in (self.title, self.name) if p]
        return " ".join(parts)

    @property
    def short(self) -> str:
        return self.title or self.name or ""

    def is_birthday(self, d: date | None = None) -> bool:
        if not self.birthday:
            return False
        d = d or date.today()
        return self.birthday.strip() == f"{d.day:02d}.{d.month:02d}"

    # --- реплики ---
    def greeting(self) -> str:
        hour = __import__("datetime").datetime.now().hour
        if 5 <= hour < 12:
            part = "Доброе утро"
        elif 12 <= hour < 18:
            part = "Добрый день"
        elif 18 <= hour < 23:
            part = "Добрый вечер"
        else:
            part = "Доброй ночи"
        return random.choice([
            f"{part}, {self.address}! Чем могу помочь?",
            f"Приветствую вас, {self.address}! Все системы Aperture в норме. Чем могу помочь?",
            f"{part}, {self.address}. Я к вашим услугам.",
        ])

    def ack(self) -> str:
        return random.choice([
            f"Хорошо, сию секунду, {self.short}, запускается.",
            f"Слушаюсь, {self.short}.",
            f"Выполняю, {self.short}.",
            f"Разумеется, {self.short}. Уже делаю.",
            f"Один момент, {self.short}.",
            f"Считайте, что готово, {self.short}.",
            "Запускаю.",
            f"Как пожелаете, {self.short}.",
        ])

    def done(self) -> str:
        return random.choice([
            f"Готово, {self.short}.",
            f"Сделано, {self.short}.",
            f"Записала, {self.short}.",
        ])

    def unknown(self) -> str:
        return random.choice([
            f"Простите, {self.short}, я не поняла команду.",
            f"Не расслышала, {self.short}. Повторите, пожалуйста.",
            f"Не распознала задачу, {self.short}. Попробуйте сформулировать иначе.",
            f"Такой команды я пока не знаю, {self.short}.",
            f"Можете сказать иначе, {self.short}? Я не уловила смысл.",
        ])

    def farewell(self) -> str:
        return random.choice([
            f"Отключаюсь, {self.address}. До встречи.",
            f"Перехожу в спящий режим. Было приятно работать, {self.short}.",
            f"Завершаю работу. Не скучайте, {self.short}.",
        ])

    def birthday_note(self) -> str:
        return f"Ваш день рождения, {self.address}. Поздравляю вас."
