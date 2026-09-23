#!/usr/bin/env python3
"""Проверка микрофона: слышит ли система вас исправно.

Запуск:  mictest.bat
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _utf8() -> None:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_utf8()


def out(s: str = "") -> None:
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(s.encode(enc, "replace").decode(enc, "replace"), flush=True)


def rule(c: str = "=") -> None:
    out(c * 64)


def bar(level: float, width: int = 40) -> str:
    """Полоска громкости."""
    filled = min(width, int(level * width * 12))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------------------------------------------------------- шаги
def step_devices():
    import sounddevice as sd

    rule()
    out("  ШАГ 1. Устройства записи")
    rule()
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None

    found = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "  <-- используется по умолчанию" if i == default_in else ""
            found.append((i, d["name"]))
            out(f"  {i:>3}: {d['name']}{mark}")

    if not found:
        out()
        out("  [ПЛОХО] Микрофонов не найдено.")
        out("  Подключите микрофон и проверьте: Параметры -> Система -> Звук -> Ввод")
        return None

    out()
    name = "неизвестно"
    try:
        name = sd.query_devices(default_in, "input")["name"]
    except Exception:
        pass
    out(f"  [ХОРОШО] Найдено устройств записи: {len(found)}")
    out(f"  Будет использован: {name}")
    return default_in


def step_noise(seconds: float = 3.0):
    """Замер фонового шума в тишине."""
    import numpy as np
    import sounddevice as sd

    from glados.stt import BLOCK, SR, rms

    out()
    rule()
    out("  ШАГ 2. Замер фонового шума")
    rule()
    out(f"  ПОМОЛЧИТЕ {seconds:.0f} секунды. Идёт замер тишины...")
    out()

    levels = []
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=BLOCK) as stream:
        end = time.time() + seconds
        while time.time() < end:
            block, _ = stream.read(BLOCK)
            lvl = rms(block[:, 0])
            levels.append(lvl)
            print(f"\r  {bar(lvl)} {lvl:.4f}", end="", flush=True)
    out()
    out()

    noise = float(np.median(levels))
    peak = float(np.max(levels))
    out(f"  Средний уровень шума: {noise:.4f}")
    out(f"  Пиковый уровень:      {peak:.4f}")
    out()

    if noise < 0.002:
        out("  [ОТЛИЧНО] Очень тихо. Идеальные условия.")
    elif noise < 0.01:
        out("  [ХОРОШО] Обычный фоновый шум, помех не будет.")
    elif noise < 0.03:
        out("  [ТЕРПИМО] Шумновато. Помогут наушники вместо колонок.")
    else:
        out("  [ПЛОХО] Очень шумно. Возможны ложные срабатывания.")
        out("  Причины: колонки играют музыку, кулер рядом с микрофоном,")
        out("  включено усиление микрофона в настройках Windows.")
    return noise


def step_speech(noise: float, seconds: float = 5.0):
    """Замер громкости речи."""
    import numpy as np
    import sounddevice as sd

    from glados.stt import BLOCK, SR, rms

    out()
    rule()
    out("  ШАГ 3. Проверка громкости речи")
    rule()
    out(f"  ГОВОРИТЕ обычным голосом {seconds:.0f} секунд.")
    out("  Например, считайте вслух: раз, два, три, четыре...")
    out()

    levels = []
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=BLOCK) as stream:
        end = time.time() + seconds
        while time.time() < end:
            block, _ = stream.read(BLOCK)
            lvl = rms(block[:, 0])
            levels.append(lvl)
            print(f"\r  {bar(lvl)} {lvl:.4f}", end="", flush=True)
    out()
    out()

    arr = np.array(levels)
    loud = float(np.percentile(arr, 90))
    out(f"  Громкость речи:  {loud:.4f}")
    out(f"  Уровень шума:    {noise:.4f}")

    ratio = loud / max(noise, 1e-5)
    out(f"  Превышение над шумом: в {ratio:.0f} раз")
    out()

    if loud < 0.01:
        out("  [ПЛОХО] Слишком тихо. Ассистент вас не услышит.")
        out("  Что сделать:")
        out("    1. Параметры -> Система -> Звук -> Ввод -> Громкость на 80-100%")
        out("    2. Говорите ближе к микрофону (20-40 см)")
        out("    3. В config.yaml увеличьте stt.gain до 2.0")
        return False
    if ratio < 3:
        out("  [ПЛОХО] Голос почти не отличается от шума.")
        out("  Уберите источник шума или используйте гарнитуру.")
        return False
    if loud > 0.6:
        out("  [ВНИМАНИЕ] Очень громко, возможны искажения.")
        out("  Снизьте громкость микрофона в настройках Windows до 70%.")
        return True

    out("  [ХОРОШО] Уровень речи в норме.")
    return True


def step_recognition(cfg):
    """Финальная проверка: что именно слышит распознавание."""
    out()
    rule()
    out("  ШАГ 4. Проверка распознавания")
    rule()

    from glados.config import Config
    from glados.fuzzy import match_any
    from glados.stt import Microphone, Recognizer

    wake = cfg.get_path("wake.words", ["гладос"])
    name = str(wake[0]).capitalize() if wake else "Гладос"

    out(f"  Загружаю модель «{cfg.get_path('stt.model', 'small')}»...")
    rec = Recognizer(cfg)
    mic = Microphone(cfg)
    mic.start()
    out()
    out(f"  Скажите три раза, делая паузу между фразами:")
    out(f"      «{name}, привет»")
    out()
    out("  Для выхода нажмите Ctrl+C.")
    out()

    heard, ok_wake = 0, 0
    try:
        while heard < 3:
            audio = mic.listen_phrase()
            if audio is None:
                continue
            text = rec.transcribe(audio)
            if not text:
                out("  (тишина или неразборчиво)")
                continue
            heard += 1
            words = text.lower().split()
            matched = any(match_any(w, wake) for w in words[:3])
            ok_wake += matched
            status = "[РАСПОЗНАНО]" if matched else "[имя не найдено]"
            out(f"  {heard}. Услышано: «{text}»   {status}")
    except KeyboardInterrupt:
        out()
    finally:
        mic.stop()

    out()
    if heard == 0:
        out("  [ПЛОХО] Не удалось расслышать ни одной фразы.")
        return False
    if ok_wake == heard:
        out(f"  [ОТЛИЧНО] Имя распознано во всех {heard} попытках.")
        return True
    if ok_wake > 0:
        out(f"  [ТЕРПИМО] Имя распознано в {ok_wake} из {heard} попыток.")
        out("  Совет: используйте модель покрупнее (stt.model: small или medium).")
        return True
    out(f"  [ПЛОХО] Имя не распознано ни разу.")
    out("  Что услышала система — показано выше. Добавьте эти варианты")
    out("  в config.yaml -> wake.words")
    return False


def main() -> int:
    rule()
    out("  ПРОВЕРКА МИКРОФОНА")
    rule()

    try:
        import numpy  # noqa: F401
        import sounddevice  # noqa: F401
    except ImportError as e:
        out()
        out(f"  [ОШИБКА] Не установлен пакет: {e.name}")
        out("  Запустите install.bat")
        return 1

    from glados.config import Config

    cfg = Config.load()

    if step_devices() is None:
        return 1

    noise = step_noise()
    speech_ok = step_speech(noise)

    out()
    if not speech_ok:
        out("  Исправьте замечания выше и запустите проверку снова.")
        rule()
        return 1

    try:
        answer = input("  Проверить распознавание речи? (д/н) [д]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "н"
    if answer in ("", "д", "да", "y", "yes"):
        step_recognition(cfg)

    # --- итог ---
    out()
    rule()
    out("  ИТОГ")
    rule()
    suggested = round(max(0.006, noise * 3), 4)
    out(f"  Рекомендуемый порог для config.yaml:")
    out(f"      stt.vad_threshold: {suggested}")
    out()
    out("  Но по умолчанию включена автоподстройка (stt.auto_threshold: true),")
    out("  и порог выставляется сам. Менять вручную обычно не нужно.")
    out()
    out("  Система слышит вас исправно." if speech_ok
        else "  Есть проблемы со слышимостью — смотрите замечания выше.")
    rule()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        out()
        out("Прервано.")
        sys.exit(1)
