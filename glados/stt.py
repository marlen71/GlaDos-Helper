"""Распознавание речи: микрофон -> faster-whisper.

Простой энергетический VAD: копим кадры, пока громкость выше порога,
останавливаемся после N секунд тишины и отдаём фразу в Whisper.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
from collections import deque

import numpy as np

from . import cuda_setup

log = logging.getLogger("glados.stt")

# Подключаем pip-пакеты nvidia-* до первого импорта ctranslate2
cuda_setup.apply()

SR = 16000
BLOCK = 512                 # 32 мс — мельче шаг, точнее засечка начала речи

#: сколько звука ДО срабатывания сохранять (чтобы не терять первый слог)
PREROLL_SECONDS = 0.5
#: минимальная длительность осмысленной фразы
MIN_PHRASE_SECONDS = 0.35


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64)) + 1e-12))


class NoiseTracker:
    """Следит за уровнем фонового шума и подстраивает порог срабатывания.

    Порог = шум * множитель. Благодаря этому одна и та же настройка работает
    и на тихом USB-микрофоне, и на шумной гарнитуре, и когда рядом гудит кулер.
    """

    def __init__(self, multiplier: float = 3.0, floor: float = 0.004):
        self.multiplier = multiplier
        self.floor = floor
        self.noise = floor
        self._warm = 0

    def update(self, level: float) -> None:
        """Медленно поднимаем оценку шума, быстро опускаем."""
        if self._warm < 40:                  # первые ~1.3 с — быстрая калибровка
            self.noise = (self.noise * self._warm + level) / (self._warm + 1)
            self._warm += 1
            return
        rate = 0.02 if level > self.noise else 0.15
        self.noise = (1 - rate) * self.noise + rate * level

    @property
    def threshold(self) -> float:
        return max(self.floor, self.noise * self.multiplier)

    @property
    def release(self) -> float:
        """Порог отпускания ниже порога срабатывания — гистерезис.

        Без него тихие окончания слов («-ос» в «Гладос») обрубаются.
        """
        return self.threshold * 0.55


class Microphone:
    """Потоковый захват микрофона с выделением фраз."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.silence = float(cfg.get_path("stt.silence_seconds", 0.8))
        self.max_len = float(cfg.get_path("stt.max_phrase_seconds", 15))
        self.device = cfg.get_path("stt.device_index", None)
        self.gain = float(cfg.get_path("stt.gain", 1.0))

        self.auto = bool(cfg.get_path("stt.auto_threshold", True))
        self.manual_threshold = float(cfg.get_path("stt.vad_threshold", 0.015))
        self.noise = NoiseTracker(
            multiplier=float(cfg.get_path("stt.threshold_multiplier", 3.0)))

        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None
        self.muted = threading.Event()
        self.last_level = 0.0

    # ------------------------------------------------------------------
    def _callback(self, indata, frames, time_info, status):  # noqa: ANN001
        if status:
            log.debug("audio status: %s", status)
        self._q.put(indata[:, 0].copy())

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=SR, channels=1, dtype="float32",
            blocksize=BLOCK, device=self.device, callback=self._callback,
        )
        self._stream.start()
        name = "по умолчанию"
        try:
            info = sd.query_devices(self._stream.device, "input")
            name = info["name"]
        except Exception:
            pass
        log.info("Микрофон активен: %s", name)
        if self.auto:
            log.info("Порог срабатывания подстраивается автоматически")

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def drain(self) -> None:
        """Выбросить накопленное — чтобы не слушать собственный голос."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    @property
    def threshold(self) -> float:
        return self.noise.threshold if self.auto else self.manual_threshold

    @property
    def release_threshold(self) -> float:
        return self.noise.release if self.auto else self.manual_threshold * 0.55

    # ------------------------------------------------------------------
    def listen_phrase(self) -> np.ndarray | None:
        """Ждёт фразу и возвращает её как float32 моно 16 кГц.

        Ключевые отличия от наивного детектора:
          * кольцевой пред-буфер — начало слова не теряется;
          * гистерезис — тихий конец слова не обрубается;
          * автоподстройка под уровень шума;
          * нормализация громкости перед распознаванием.
        """
        preroll_blocks = max(1, int(PREROLL_SECONDS * SR / BLOCK))
        preroll: deque[np.ndarray] = deque(maxlen=preroll_blocks)

        buf: list[np.ndarray] = []
        silent_blocks = 0
        needed_silence = max(1, int(self.silence * SR / BLOCK))
        max_blocks = int(self.max_len * SR / BLOCK)
        started = False

        while True:
            block = self._q.get()
            if self.gain != 1.0:
                block = block * self.gain
            level = rms(block)
            self.last_level = level

            if self.muted.is_set():
                buf.clear()
                preroll.clear()
                started = False
                continue

            if not started:
                self.noise.update(level)
                preroll.append(block)
                if level > self.threshold:
                    started = True
                    buf.extend(preroll)     # забираем звук ДО срабатывания
                    buf.append(block)
                    silent_blocks = 0
                continue

            buf.append(block)
            if level > self.release_threshold:
                silent_blocks = 0
            else:
                silent_blocks += 1
                if silent_blocks >= needed_silence:
                    break
            if len(buf) > max_blocks:
                break

        if not buf:
            return None
        audio = np.concatenate(buf)

        # Длительность считаем по самой речи, без пред-буфера и хвоста тишины,
        # иначе короткий щелчок выглядит как полноценная фраза.
        speech_len = len(audio) - len(preroll) * BLOCK - silent_blocks * BLOCK
        if speech_len < SR * MIN_PHRASE_SECONDS:
            log.debug("Слишком короткий звук (%.2f с) — пропускаю",
                      max(0, speech_len) / SR)
            return None
        return normalize(audio)


def normalize(audio: np.ndarray, target_peak: float = 0.85) -> np.ndarray:
    """Приводит громкость к рабочему уровню — тихая речь распознаётся плохо.

    Опираемся на 99-й перцентиль, а не на максимум: одиночный щелчок не должен
    мешать усилению всей фразы.
    """
    if audio.size == 0:
        return audio
    mag = np.abs(audio)
    loud = float(np.percentile(mag, 99))
    if loud < 1e-5:
        return audio
    gain = min(target_peak / loud, 30.0)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def build_prompt(cfg) -> str:
    """Словарь-подсказка для Whisper.

    Модель склонна заменять незнакомое имя на похожее обычное слово
    («Гладос» -> «Лада»). Если показать ей контекст со списком ожидаемых слов,
    вероятность правильного разбора заметно растёт.
    """
    custom = cfg.get_path("stt.prompt", None)
    if custom:
        return str(custom)

    names = [str(w) for w in (cfg.get_path("wake.words", []) or [])][:4]
    name = names[0].capitalize() if names else "Гладос"
    games = [str(g) for g in (cfg.get("games", {}) or {})][:8]

    parts = [
        f"Голосовые команды ассистенту по имени {name}.",
        f"{name}, привет. {name}, открой браузер. {name}, включи Discord.",
        f"{name}, запусти Steam. {name}, открой YouTube. {name}, включи музыку.",
        f"{name}, сколько времени. {name}, какая сегодня дата.",
        f"{name}, запиши в заметки. {name}, напомни выпить воды.",
        f"{name}, выключи компьютер. {name}, перезагрузи компьютер.",
    ]
    if games:
        parts.append("Игры: " + ", ".join(games) + ".")
    return " ".join(parts)


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
        self.prompt = build_prompt(cfg)

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
            # Свой VAD уже отработал; повторная обрезка глотала короткие слова
            vad_filter=False,
            beam_size=self.beam_size,
            condition_on_previous_text=False,
            # Подсказка словаря: резко повышает шанс услышать имя и команды
            initial_prompt=self.prompt,
            temperature=[0.0, 0.2, 0.4],
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.5,
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
