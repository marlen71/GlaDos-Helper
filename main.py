#!/usr/bin/env python3
"""GLaDOS Helper — голосовой ассистент в стиле Portal 2.

Примеры запуска:
    python main.py                 # голосовой режим
    python main.py --text          # текстовый режим (без микрофона и TTS)
    python main.py --say "Привет"  # просто озвучить фразу
    python main.py --demo out.wav  # сохранить пример голоса в файл
    python main.py --devices       # список аудиоустройств
"""
from __future__ import annotations

import argparse
import logging
import sys

from glados.assistant import Assistant
from glados.config import Config


def main() -> int:
    ap = argparse.ArgumentParser(description="GLaDOS Helper")
    ap.add_argument("-c", "--config", default=None, help="путь к config.yaml")
    ap.add_argument("--text", action="store_true", help="текстовый режим без голоса")
    ap.add_argument("--voice-text", action="store_true",
                    help="ввод текстом, ответы голосом")
    ap.add_argument("--say", metavar="ТЕКСТ", help="озвучить фразу и выйти")
    ap.add_argument("--demo", metavar="ФАЙЛ.wav", help="сохранить демо-фразу в WAV")
    ap.add_argument("--devices", action="store_true", help="показать аудиоустройства")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    logging.basicConfig(
        level=getattr(logging, str(cfg.get_path("log_level", "INFO")).upper(), 20),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    if args.say or args.demo:
        from glados.tts import build_tts

        tts = build_tts(cfg)
        text = args.say or ("Приветствую вас, сэр Марлен. Все системы Aperture Science "
                            "функционируют нормально. Чем могу помочь?")
        if args.demo:
            path = tts.save(text, args.demo)  # type: ignore[attr-defined]
            print(f"Сохранено: {path}")
        else:
            tts.say(text)
        return 0

    assistant = Assistant(cfg, voice=not args.text)
    try:
        if args.text or args.voice_text:
            assistant.run_text()
        else:
            assistant.run_voice()
    except ModuleNotFoundError as e:
        _missing_module_help(e)
        return 1
    return 0


_PKG_HINTS = {
    "faster_whisper": "faster-whisper",
    "sounddevice": "sounddevice",
    "soundfile": "soundfile",
    "torch": "torch",
    "yaml": "PyYAML",
    "pyperclip": "pyperclip",
}


def _missing_module_help(err: ModuleNotFoundError) -> None:
    name = err.name or "?"
    pkg = _PKG_HINTS.get(name, name)
    print("\n" + "=" * 62)
    print(f"  Не установлен пакет: {pkg}")
    print("=" * 62)
    print("  Похоже, установка не завершилась. Что делать:\n")
    print("  1) Запустите install.bat ещё раз и дождитесь надписи 'Готово'.")
    print("  2) Или установите вручную этой командой:\n")
    print(f'       "{sys.executable}" -m pip install {pkg}\n')
    print("  3) Проверить окружение целиком:  python check.py")
    print("=" * 62)


if __name__ == "__main__":
    sys.exit(main())
