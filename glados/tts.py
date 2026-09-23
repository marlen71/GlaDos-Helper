"""Синтез речи GLaDOS: Silero TTS (offline, ru) + эффекты робота.

Запасной вариант — Windows SAPI через PowerShell, если torch недоступен.
"""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np

from .audio_fx import apply_glados

log = logging.getLogger("glados.tts")

SILERO_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"


class BaseTTS:
    def say(self, text: str) -> None:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def stop(self) -> None:
        pass


class SileroTTS(BaseTTS):
    """Silero v4 (русский) + пост-обработка под GLaDOS."""

    def __init__(self, cfg):
        import torch  # локальный импорт: тяжёлый

        self.torch = torch
        self.sample_rate = int(cfg.get_path("tts.sample_rate", 48000))
        self.speaker = cfg.get_path("tts.speaker", "baya")
        self.fx_enabled = bool(cfg.get_path("tts.glados_effect.enabled", True))
        self.fx = cfg.get_path("tts.glados_effect", {}) or {}

        models_dir = cfg.root / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "v4_ru.pt"
        if not model_path.exists():
            log.info("Скачиваю модель Silero (~60 МБ)...")
            torch.hub.download_url_to_file(SILERO_URL, str(model_path))

        torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
        self.model = torch.package.PackageImporter(str(model_path)) \
            .load_pickle("tts_models", "model")
        self.model.to(torch.device("cpu"))
        log.info("Silero TTS готов (голос: %s)", self.speaker)

    def synth(self, text: str) -> np.ndarray:
        with self.torch.no_grad():
            wav = self.model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=self.sample_rate,
                put_accent=True,
                put_yo=True,
            )
        audio = wav.numpy().astype(np.float32)
        if self.fx_enabled:
            audio = apply_glados(audio, self.sample_rate, self.fx)
        return audio

    def say(self, text: str) -> None:
        import sounddevice as sd

        audio = self.synth(text)
        sd.play(audio, self.sample_rate)
        sd.wait()

    def save(self, text: str, path: str | Path) -> Path:
        import soundfile as sf

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), self.synth(text), self.sample_rate)
        return path

    def stop(self) -> None:
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass


class SapiTTS(BaseTTS):
    """Запасной TTS на встроенном синтезаторе Windows."""

    def say(self, text: str) -> None:
        safe = text.replace("'", "''")
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Add-Type -AssemblyName System.Speech; "
             "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
             f"$s.Rate=-1; $s.Speak('{safe}')"],
            check=False,
        )


class SpeechQueue:
    """Очередь реплик — Гладос не перебивает сама себя."""

    def __init__(self, engine: BaseTTS):
        self.engine = engine
        self._q: "queue.Queue[str | None]" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self.speaking = threading.Event()
        self._thread.start()

    def _worker(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                break
            self.speaking.set()
            try:
                self.engine.say(text)
            except Exception as e:  # синтез не должен ронять ассистента
                log.error("Ошибка синтеза: %s", e)
            finally:
                self.speaking.clear()
                self._q.task_done()

    def say(self, text: str) -> None:
        if text:
            print(f"GLaDOS: {text}")
            self._q.put(text)

    def wait(self) -> None:
        self._q.join()

    def shutdown(self) -> None:
        self._q.put(None)


def build_tts(cfg) -> BaseTTS:
    engine = (cfg.get_path("tts.engine", "silero") or "silero").lower()
    if engine == "silero":
        try:
            return SileroTTS(cfg)
        except Exception as e:
            log.warning("Silero недоступен (%s) — переключаюсь на SAPI", e)
    return SapiTTS()
