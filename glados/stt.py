"""Распознавание речи: микрофон -> faster-whisper.

Простой энергетический VAD: копим кадры, пока громкость выше порога,
останавливаемся после N секунд тишины и отдаём фразу в Whisper.
"""
from __future__ import annotations

import logging
import queue
import threading

import numpy as np

log = logging.getLogger("glados.stt")

SR = 16000
BLOCK = 1024


class Microphone:
    """Потоковый захват микрофона с выделением фраз."""

    def __init__(self, cfg):
        self.threshold = float(cfg.get_path("stt.vad_threshold", 0.015))
        self.silence = float(cfg.get_path("stt.silence_seconds", 0.9))
        self.max_len = float(cfg.get_path("stt.max_phrase_seconds", 15))
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None
        self.muted = threading.Event()

    def _callback(self, indata, frames, time_info, status):  # noqa: ANN001
        if status:
            log.debug("audio status: %s", status)
        self._q.put(indata[:, 0].copy())

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=SR, channels=1, dtype="float32",
            blocksize=BLOCK, callback=self._callback,
        )
        self._stream.start()
        log.info("Микрофон активен")

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def listen_phrase(self) -> np.ndarray | None:
        """Блокирующе ждёт фразу и возвращает её как float32 моно 16 кГц."""
        buf: list[np.ndarray] = []
        silent_blocks = 0
        needed_silence = int(self.silence * SR / BLOCK)
        max_blocks = int(self.max_len * SR / BLOCK)
        started = False

        while True:
            block = self._q.get()
            if self.muted.is_set():
                buf.clear()
                started = False
                continue
            level = float(np.sqrt(np.mean(block ** 2)))
            if level > self.threshold:
                started = True
                silent_blocks = 0
                buf.append(block)
            elif started:
                silent_blocks += 1
                buf.append(block)
                if silent_blocks >= needed_silence:
                    break
            if started and len(buf) > max_blocks:
                break

        audio = np.concatenate(buf) if buf else None
        if audio is None or len(audio) < SR * 0.3:
            return None
        return audio


class Recognizer:
    def __init__(self, cfg):
        from faster_whisper import WhisperModel

        device = cfg.get_path("stt.device", "auto")
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        compute = cfg.get_path("stt.compute_type", "int8")
        if device == "cuda" and compute == "int8":
            compute = "float16"
        model_name = cfg.get_path("stt.model", "small")
        log.info("Загружаю Whisper '%s' (%s/%s)...", model_name, device, compute)
        self.model = WhisperModel(model_name, device=device, compute_type=compute)
        self.language = cfg.get_path("stt.language", "ru")

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()
