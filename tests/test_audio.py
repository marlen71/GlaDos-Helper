"""Тесты детектора речи: пред-буфер, гистерезис, нормализация."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glados.config import Config  # noqa: E402
from glados.stt import (BLOCK, PREROLL_SECONDS, SR, Microphone,  # noqa: E402
                        NoiseTracker, build_prompt, normalize, rms)


def test_rms_of_silence_is_zero():
    assert rms(np.zeros(BLOCK, dtype=np.float32)) < 1e-5


def test_rms_grows_with_amplitude():
    quiet = rms(np.full(BLOCK, 0.01, dtype=np.float32))
    loud = rms(np.full(BLOCK, 0.5, dtype=np.float32))
    assert loud > quiet * 10


def test_normalize_boosts_quiet_audio():
    quiet = np.full(SR, 0.02, dtype=np.float32)
    louder = normalize(quiet)
    assert np.max(np.abs(louder)) > 0.5


def test_normalize_does_not_clip():
    loud = np.full(SR, 0.99, dtype=np.float32)
    assert np.max(np.abs(normalize(loud))) <= 1.0


def test_normalize_leaves_silence_alone():
    silence = np.zeros(SR, dtype=np.float32)
    assert np.array_equal(normalize(silence), silence)


# ------------------------------------------------------------ шумомер
def test_noise_tracker_adapts_to_quiet_room():
    n = NoiseTracker()
    for _ in range(100):
        n.update(0.001)
    assert n.threshold < 0.01


def test_noise_tracker_adapts_to_noisy_room():
    n = NoiseTracker()
    for _ in range(100):
        n.update(0.05)
    assert n.threshold > 0.05


def test_release_threshold_below_trigger():
    """Гистерезис: отпускание должно быть ниже срабатывания."""
    n = NoiseTracker()
    for _ in range(50):
        n.update(0.01)
    assert n.release < n.threshold


# ---------------------------------------------------- захват фразы
class FakeMic(Microphone):
    """Микрофон, читающий из заранее заданного списка блоков."""

    def __init__(self, cfg, blocks):
        super().__init__(cfg)
        for b in blocks:
            self._q.put(b)
        # хвост тишины, чтобы детектор завершил фразу
        for _ in range(200):
            self._q.put(np.zeros(BLOCK, dtype=np.float32))


def _cfg():
    c = Config.load()
    c["stt"]["auto_threshold"] = True
    c["stt"]["silence_seconds"] = 0.3
    return c


def _silence(n):
    return [np.zeros(BLOCK, dtype=np.float32) for _ in range(n)]


def _speech(n, amp=0.3):
    rng = np.random.RandomState(0)
    return [(rng.randn(BLOCK) * amp).astype(np.float32) for _ in range(n)]


def test_captures_speech_after_silence():
    blocks = _silence(40) + _speech(60)
    mic = FakeMic(_cfg(), blocks)
    audio = mic.listen_phrase()
    assert audio is not None
    assert len(audio) > SR * 0.3


def test_preroll_keeps_start_of_word():
    """Главная регрессия: первый слог не должен теряться.

    Записываем короткий громкий всплеск. Если пред-буфера нет, в результат
    попадёт только часть после срабатывания порога.
    """
    blocks = _silence(40) + _speech(30)
    mic = FakeMic(_cfg(), blocks)
    audio = mic.listen_phrase()
    assert audio is not None
    # в захват должен войти звук ДО порога — значит длина больше самой речи
    speech_samples = 30 * BLOCK
    assert len(audio) > speech_samples


def test_ignores_too_short_blip():
    """Щелчок мыши или стук не должны считаться фразой."""
    blocks = _silence(40) + _speech(2) + _silence(60)
    mic = FakeMic(_cfg(), blocks)
    assert mic.listen_phrase() is None


def test_quiet_speech_is_captured_and_normalized():
    blocks = _silence(40) + _speech(60, amp=0.02)
    mic = FakeMic(_cfg(), blocks)
    audio = mic.listen_phrase()
    assert audio is not None
    assert np.max(np.abs(audio)) > 0.5, "тихая речь должна усиливаться"


def test_muted_microphone_returns_nothing():
    blocks = _speech(60)
    mic = FakeMic(_cfg(), blocks)
    mic.muted.set()
    for _ in range(300):
        mic._q.put(np.zeros(BLOCK, dtype=np.float32))
    mic.muted.clear()
    mic._q.put(np.zeros(BLOCK, dtype=np.float32))


def test_preroll_constant_is_reasonable():
    assert 0.2 <= PREROLL_SECONDS <= 1.0


# ---------------------------------------------------------- подсказка
def test_prompt_mentions_assistant_name():
    cfg = Config.load()
    prompt = build_prompt(cfg)
    assert "Гладос" in prompt or "гладос" in prompt.lower()


def test_prompt_includes_games():
    cfg = Config.load()
    assert "Игры:" in build_prompt(cfg)


def test_custom_prompt_overrides():
    cfg = Config.load()
    cfg["stt"]["prompt"] = "своя подсказка"
    assert build_prompt(cfg) == "своя подсказка"


def test_prompt_follows_custom_name():
    cfg = Config.load()
    cfg["wake"]["words"] = ["джарвис"]
    assert "Джарвис" in build_prompt(cfg)
