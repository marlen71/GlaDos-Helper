"""Эффекты, превращающие обычный синтезированный голос в голос GLaDOS.

Цепочка: pitch-shift -> ring modulation -> flanger -> soft drive -> reverb.
Всё на numpy/scipy, без внешних бинарников.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def _pitch_shift(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Простой phase-vocoder-free сдвиг: resample + time-stretch через OLA."""
    if abs(semitones) < 0.01:
        return x
    rate = 2.0 ** (semitones / 12.0)
    # 1) time-stretch на 1/rate методом overlap-add
    win = 1024
    hop_in = win // 4
    hop_out = int(round(hop_in / rate))
    if hop_out < 1:
        hop_out = 1
    window = np.hanning(win).astype(np.float32)
    n_frames = max(1, (len(x) - win) // hop_in)
    out = np.zeros(n_frames * hop_out + win, dtype=np.float32)
    norm = np.zeros_like(out)
    for i in range(n_frames):
        seg = x[i * hop_in: i * hop_in + win]
        if len(seg) < win:
            break
        o = i * hop_out
        out[o:o + win] += seg * window
        norm[o:o + win] += window
    norm[norm < 1e-6] = 1e-6
    stretched = out / norm
    # 2) resample обратно -> меняется высота тона
    idx = np.arange(0, len(stretched), rate)
    idx = idx[idx < len(stretched) - 1]
    lo = idx.astype(np.int32)
    frac = (idx - lo).astype(np.float32)
    return (stretched[lo] * (1 - frac) + stretched[lo + 1] * frac).astype(np.float32)


def _ring_mod(x: np.ndarray, sr: int, depth: float, freq: float) -> np.ndarray:
    if depth <= 0:
        return x
    t = np.arange(len(x), dtype=np.float32) / sr
    carrier = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return ((1.0 - depth) * x + depth * x * carrier).astype(np.float32)


def _flanger(x: np.ndarray, sr: int, amount: float, rate_hz: float = 0.25) -> np.ndarray:
    if amount <= 0:
        return x
    max_delay = int(0.004 * sr)          # 4 мс
    t = np.arange(len(x), dtype=np.float32) / sr
    lfo = (1 + np.sin(2 * np.pi * rate_hz * t)) * 0.5
    delays = (lfo * max_delay).astype(np.int32)
    idx = np.arange(len(x)) - delays
    idx[idx < 0] = 0
    return ((1 - amount) * x + amount * x[idx]).astype(np.float32)


def _soft_drive(x: np.ndarray, drive: float) -> np.ndarray:
    if drive <= 1.0:
        return x
    return np.tanh(x * drive).astype(np.float32) / np.tanh(drive)


def _reverb(x: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Короткий «лабораторный» реверб — свёртка с затухающим шумом."""
    if amount <= 0:
        return x
    length = int(0.35 * sr)
    noise = np.random.RandomState(42).randn(length).astype(np.float32)
    ir = noise * np.exp(-np.linspace(0, 7, length)).astype(np.float32)
    ir /= np.abs(ir).sum() + 1e-9
    wet = fftconvolve(x, ir)[: len(x)].astype(np.float32)
    return ((1 - amount) * x + amount * wet * 3.0).astype(np.float32)


def apply_glados(audio: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """Применить полную цепочку эффектов."""
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x
    x = _pitch_shift(x, sr, float(params.get("pitch_semitones", -1.5)))
    x = _ring_mod(x, sr, float(params.get("robot_depth", 0.35)),
                  float(params.get("robot_freq", 62)))
    x = _flanger(x, sr, float(params.get("flanger", 0.25)))
    x = _soft_drive(x, float(params.get("drive", 1.15)))
    x = _reverb(x, sr, float(params.get("reverb", 0.22)))
    peak = float(np.max(np.abs(x))) or 1.0
    return (x / peak * 0.92).astype(np.float32)
