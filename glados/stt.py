"""Распознавание речи: микрофон -> faster-whisper.

Простой энергетический VAD: копим кадры, пока громкость выше порога,
останавливаемся после N секунд тишины и отдаём фразу в Whisper.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading

import numpy as np

from . import cuda_setup

log = logging.getLogger("glados.stt")

# Подключаем pip-пакеты nvidia-* до первого импорта ctranslate2
cuda_setup.apply()

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
    """faster-whisper с автоматическим откатом на CPU, если CUDA недоступна."""

    #: Ошибки, означающие «CUDA/cuDNN/cuBLAS не работает»
    _CUDA_ERRORS = ("cublas", "cudnn", "cuda", "libcublas", "no kernel image",
                    "out of memory", "cuda_error")

    def __init__(self, cfg):
        self.cfg = cfg
        self.language = cfg.get_path("stt.language", "ru")
        self.beam_size = int(cfg.get_path("stt.beam_size", 1))
        self.model_name = cfg.get_path("stt.model", "small")
        self._fallen_back = False

        device = str(cfg.get_path("stt.device", "auto")).lower()
        compute = str(cfg.get_path("stt.compute_type", "int8"))

        if device == "auto":
            device = "cuda" if cuda_is_usable() else "cpu"
        elif device == "cuda":
            ok, reason = cuda_diagnose()
            if not ok:
                log.warning("Не удалось включить GPU: %s", reason)
                log.warning("Переключаюсь на CPU. Проверить подробности: check.bat")
                device = "cpu"

        self.model = self._load(device, compute)

    # ------------------------------------------------------------------
    def _load(self, device: str, compute: str):
        from faster_whisper import WhisperModel

        compute = self._fix_compute(device, compute)
        self.device, self.compute = device, compute
        log.info("Загружаю Whisper '%s' (%s/%s)...", self.model_name, device, compute)
        try:
            return WhisperModel(self.model_name, device=device, compute_type=compute)
        except Exception as e:
            if device == "cuda":
                log.warning("Не удалось поднять модель на GPU (%s). Перехожу на CPU.", e)
                return self._load("cpu", "int8")
            raise

    @staticmethod
    def _fix_compute(device: str, compute: str) -> str:
        """int8 на GPU и float16 на CPU работают плохо/никак — правим молча."""
        if device == "cuda" and compute in ("int8", "int8_float32"):
            return "float16"
        if device == "cpu" and compute in ("float16", "int8_float16"):
            return "int8"
        return compute

    def _is_cuda_error(self, err: Exception) -> bool:
        msg = f"{type(err).__name__} {err}".lower()
        return any(k in msg for k in self._CUDA_ERRORS)

    # ------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray) -> str:
        try:
            return self._run(audio)
        except Exception as e:
            if self.device == "cuda" and not self._fallen_back and self._is_cuda_error(e):
                log.error("Ошибка CUDA во время распознавания: %s", e)
                log.warning("Переключаюсь на CPU и продолжаю работу без перезапуска. "
                            "Чтобы убрать задержку, поставьте stt.device: cpu в config.yaml "
                            "или доустановьте CUDA-библиотеки (см. README).")
                self._fallen_back = True
                self.model = self._load("cpu", "int8")
                try:
                    return self._run(audio)
                except Exception as e2:
                    log.error("Распознавание не удалось и на CPU: %s", e2)
                    return ""
            log.error("Ошибка распознавания: %s", e)
            return ""

    def _run(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
            beam_size=self.beam_size,
            condition_on_previous_text=False,
        )
        # segments — ленивый генератор: материализуем здесь, чтобы поймать ошибки
        return " ".join(s.text.strip() for s in segments).strip()


def cuda_device_count() -> int:
    """Сколько CUDA-устройств видит система."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        pass
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def cuda_diagnose() -> tuple[bool, str]:
    """Проверяет CUDA и объясняет причину отказа.

    Возвращает (работает, причина). Причина пустая, если всё хорошо.
    """
    cuda_setup.apply()

    if cuda_device_count() == 0:
        return False, ("видеокарта NVIDIA не обнаружена библиотекой ctranslate2 "
                       "(проверьте драйвер командой nvidia-smi)")

    missing = cuda_setup.missing_libraries()
    if missing:
        return False, (f"не загружаются библиотеки: {', '.join(missing)}. "
                       f"Установите их: install-gpu.bat")
    return True, ""


def cuda_is_usable() -> bool:
    """CUDA считается рабочей, только если есть и устройство, и cuBLAS/cuDNN."""
    ok, _ = cuda_diagnose()
    return ok
