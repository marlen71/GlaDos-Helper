"""Главный цикл ассистента GLaDOS."""
from __future__ import annotations

import logging
import time

from .commands import CommandRouter, strip_wake
from .config import Config
from .persona import Persona
from .skills.reminders import ReminderStore, ReminderWatcher
from .tts import SpeechQueue, build_tts

log = logging.getLogger("glados")


class Assistant:
    def __init__(self, cfg: Config, voice: bool = True):
        self.cfg = cfg
        self.persona = Persona(cfg)
        self.reminders = ReminderStore(cfg.data_file("storage.reminders_file",
                                                     "data/reminders.json"))
        self.router = CommandRouter(cfg, self.persona, self.reminders)
        self.wake_words = cfg.get_path("wake.words", ["гладос"])
        self.followup = float(cfg.get_path("wake.followup_seconds", 10))
        self.running = True

        self.speech = SpeechQueue(build_tts(cfg)) if voice else None
        self.watcher = ReminderWatcher(self.reminders, self._remind)
        self.watcher.start()

    # ------------------------------------------------------------------
    def say(self, text: str) -> None:
        if not text:
            return
        if self.speech:
            self.speech.say(text)
        else:
            print(f"GLaDOS: {text}")

    def _remind(self, text: str) -> None:
        self.say(f"Напоминание, {self.persona.short}: {text}.")

    def process(self, phrase: str, require_wake: bool = True) -> bool:
        """Обрабатывает фразу. Возвращает True, если команда была принята."""
        body, woken = strip_wake(phrase, self.wake_words)
        if require_wake and not woken:
            return False
        if not body:
            self.say(self.persona.greeting())
            return True
        result = self.router.handle(body)
        self.say(result.reply)
        if result.stop:
            self.running = False
        return True

    # ------------------------------------------------------------- режимы
    def run_text(self) -> None:
        """Текстовый режим — для отладки без микрофона."""
        print("Текстовый режим. Пишите команды (Ctrl+C для выхода).")
        greeted = False
        while self.running:
            try:
                line = input("Вы: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if not greeted:
                greeted = True
            if not self.process(line, require_wake=False):
                print("(нет слова активации)")
        self.shutdown()

    def run_voice(self) -> None:
        from .stt import Microphone, Recognizer

        mic = Microphone(self.cfg)
        rec = Recognizer(self.cfg)
        mic.start()
        self.say(f"Система Aperture Science активна. Я слушаю вас, {self.persona.address}.")
        if self.speech:
            self.speech.wait()

        last_reply = 0.0
        try:
            while self.running:
                audio = mic.listen_phrase()
                if audio is None:
                    continue
                if self.speech and self.speech.speaking.is_set():
                    continue  # не слушаем саму себя
                text = rec.transcribe(audio)
                if not text:
                    continue
                print(f"Вы: {text}")
                require = (time.time() - last_reply) > self.followup
                if self.process(text, require_wake=require):
                    if self.speech:
                        self.speech.wait()
                    last_reply = time.time()
        except KeyboardInterrupt:
            pass
        finally:
            mic.stop()
            self.shutdown()

    def shutdown(self) -> None:
        self.watcher.stop()
        if self.speech:
            self.speech.wait()
            self.speech.shutdown()
