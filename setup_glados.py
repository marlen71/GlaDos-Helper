#!/usr/bin/env python3
"""Установщик GLaDOS Helper.

Сканирует систему, показывает что уже установлено, и позволяет выбрать
компоненты: базовые библиотеки, PyTorch, ускорение GPU (CUDA), модели.

Запуск:
    install.bat                 интерактивно
    install.bat --auto          без вопросов, оптимально под ваше железо
    install.bat --check         только показать отчёт, ничего не менять
    install.bat --repair        переустановить сломанное
    install.bat --gpu           сразу с ускорением GPU
    install.bat --cpu           сразу без GPU
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
IS_WIN = os.name == "nt"

sys.path.insert(0, str(ROOT))


def _enable_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_enable_utf8_console()


def out(s: str = "") -> None:
    """Печать, устойчивая к консоли cp866."""
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(s.encode(enc, "replace").decode(enc, "replace"), flush=True)


def rule(char: str = "=") -> None:
    out(char * 64)


def title(text: str) -> None:
    out()
    rule()
    out(f"  {text}")
    rule()


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


# ----------------------------------------------------------------- ввод
def ask(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{question}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        out()
        return default
    return answer or default


def ask_yes(question: str, default: bool = True) -> bool:
    d = "д" if default else "н"
    while True:
        a = ask(f"{question} (д/н)", d).lower()
        if a in ("д", "да", "y", "yes", "1"):
            return True
        if a in ("н", "нет", "n", "no", "0"):
            return False
        out("  Введите 'д' или 'н'.")


def ask_choice(question: str, options: list[str], default: int = 1) -> int:
    out()
    out(question)
    for i, o in enumerate(options, 1):
        out(f"  {i}. {o}")
    while True:
        a = ask("Ваш выбор", str(default))
        if a.isdigit() and 1 <= int(a) <= len(options):
            return int(a)
        out(f"  Введите число от 1 до {len(options)}.")


# ------------------------------------------------------------- установка
def pip_install(args: list[str], title_text: str, python: Path) -> bool:
    out()
    out(f"--- {title_text} ---")
    cmd = [str(python), "-m", "pip", "install", *args]
    out("  " + " ".join(cmd[3:]))
    out()
    code = subprocess.call(cmd)
    if code != 0:
        out(f"  [ОШИБКА] шаг завершился с кодом {code}")
        return False
    out(f"  [ГОТОВО] {title_text}")
    return True


def ensure_venv() -> Path | None:
    vpy = venv_python()
    if vpy.exists():
        out("  Окружение .venv уже существует — использую его.")
        return vpy
    out("  Создаю виртуальное окружение .venv ...")
    if subprocess.call([sys.executable, "-m", "venv", str(VENV)]) != 0:
        out("  [ОШИБКА] не удалось создать .venv")
        return None
    vpy = venv_python()
    return vpy if vpy.exists() else None


# ------------------------------------------------------------ отчёт
def print_report(rep, sysinfo) -> None:
    out()
    out("Проверяю систему...")
    out()

    mark = "[OK]  " if rep.python_ok else "[!]   "
    out(f"  {mark} Python {rep.python_version}")

    if rep.disk_free_gb >= 0:
        need = 3.0
        ok = rep.disk_free_gb >= need
        out(f"  {'[OK]  ' if ok else '[МАЛО]'} Свободно на диске: "
            f"{rep.disk_free_gb:.0f} ГБ" + ("" if ok else f" (нужно ~{need:.0f} ГБ)"))
    if rep.ram_gb > 0:
        out(f"  [OK]   Оперативная память: {rep.ram_gb:.0f} ГБ")

    g = rep.gpu
    if g.present:
        mem = f", {g.memory_mb / 1024:.0f} ГБ" if g.memory_mb else ""
        out(f"  [OK]   Видеокарта: {g.name}{mem}")
        if g.driver_cuda:
            note = "" if g.supports_cu12 else "  <- нужен драйвер новее для CUDA 12"
            out(f"  {'[OK]  ' if g.supports_cu12 else '[!]   '} "
                f"Драйвер поддерживает CUDA {g.driver_cuda}{note}")
    else:
        out("  [нет]  Видеокарта NVIDIA не найдена — будет работать процессор")

    if rep.microphones:
        out(f"  [OK]   Микрофон: {rep.microphones[0]}"
            + (f" (и ещё {len(rep.microphones) - 1})" if len(rep.microphones) > 1 else ""))

    out()
    out("Установленные компоненты:")
    out()

    def row(done: bool, name: str, note_yes: str, note_no: str) -> None:
        out(f"  [{'✓' if done else ' '}] {name:<32} {note_yes if done else note_no}")

    if not rep.venv_exists:
        out("  Окружение .venv ещё не создано — будет создано при установке.")
    else:
        row(rep.base_installed, "Базовые библиотеки", "уже стоят", "не установлены")
        row(rep.torch_installed, "PyTorch (для синтеза голоса)", "уже стоит", "не установлен")
        row(rep.whisper_installed, "Распознавание речи", "уже стоит", "не установлено")
        if rep.gpu.present:
            row(rep.cuda_installed, "Ускорение GPU (CUDA)",
                "работает", "не установлено")
        else:
            out(f"  [-] {'Ускорение GPU (CUDA)':<32} недоступно (нет карты NVIDIA)")

    row(rep.silero_present, "Модель голоса Silero", "скачана (60 МБ)", "не скачана (60 МБ)")

    if rep.whisper_models:
        out(f"  [✓] {'Модели распознавания':<32} " + ", ".join(rep.whisper_models))
    else:
        out(f"  [ ] {'Модели распознавания':<32} ни одна не скачана")


# ------------------------------------------------------- план установки
class Plan:
    def __init__(self):
        self.base = False
        self.torch_cpu = False
        self.torch_gpu = False
        self.whisper = False
        self.cuda = False
        self.silero = False
        self.whisper_model: str | None = None
        self.write_config = True
        self.cuda_works = False

    def empty(self) -> bool:
        return not any([self.base, self.torch_cpu, self.torch_gpu, self.whisper,
                        self.cuda, self.silero, self.whisper_model])

    def describe(self) -> list[str]:
        items = []
        if self.base:
            items.append("базовые библиотеки (~150 МБ)")
        if self.torch_gpu:
            items.append("PyTorch с поддержкой GPU (~2.5 ГБ)")
        elif self.torch_cpu:
            items.append("PyTorch CPU (~200 МБ)")
        if self.whisper:
            items.append("распознавание речи (~50 МБ)")
        if self.cuda:
            items.append("библиотеки CUDA: cuBLAS + cuDNN (~1.1 ГБ)")
        if self.silero:
            items.append("модель голоса Silero (60 МБ)")
        if self.whisper_model:
            from glados import sysinfo

            mb = sysinfo.WHISPER_SIZES.get(self.whisper_model, 0)
            items.append(f"модель распознавания «{self.whisper_model}» ({mb} МБ)")
        return items


def plan_full(rep, gpu: bool, model: str | None) -> Plan:
    p = Plan()
    p.base = not rep.base_installed
    p.whisper = not rep.whisper_installed
    p.silero = not rep.silero_present
    if gpu:
        p.cuda = not rep.cuda_installed
    p.torch_cpu = not rep.torch_installed
    if model and model not in rep.whisper_models:
        p.whisper_model = model
    return p


def plan_interactive(rep, sysinfo) -> Plan:
    p = Plan()

    out()
    rule("-")
    gpu_possible = rep.gpu.present and rep.gpu.supports_cu12

    options = ["Базовая установка (процессор)            ~2 ГБ    рекомендуется"]
    if gpu_possible:
        options.append("Установка с ускорением GPU               ~3.2 ГБ  у вас есть NVIDIA")
    options.append("Выборочно (выбрать компоненты вручную)")
    options.append("Только проверить, ничего не устанавливать")
    if rep.venv_exists:
        options.append("Починить: переустановить сломанное")

    choice = ask_choice("Что установить?", options, default=2 if gpu_possible else 1)
    picked = options[choice - 1]

    if picked.startswith("Только проверить"):
        return p  # пустой план

    if picked.startswith("Починить"):
        p.base = True
        p.whisper = True
        p.torch_cpu = not rep.torch_installed
        if rep.cuda_installed or (gpu_possible and ask_yes("  Переустановить и CUDA?", False)):
            p.cuda = True
        return p

    if picked.startswith("Базовая"):
        return plan_full(rep, gpu=False, model=choose_model(rep, sysinfo, gpu=False))

    if picked.startswith("Установка с ускорением"):
        return plan_full(rep, gpu=True, model=choose_model(rep, sysinfo, gpu=True))

    # --- выборочно ---
    out()
    out("Выбор компонентов. Enter — оставить значение по умолчанию.")
    out()

    if rep.base_installed:
        p.base = ask_yes("  Базовые библиотеки уже стоят. Переустановить?", False)
    else:
        out("  Базовые библиотеки (numpy, звук, конфиг) — 150 МБ. Обязательны.")
        p.base = True

    if rep.torch_installed:
        p.torch_cpu = ask_yes("  PyTorch уже стоит. Переустановить?", False)
    else:
        out()
        out("  PyTorch нужен для синтеза голоса Silero — 200 МБ.")
        out("  Без него голос будет системным (Windows SAPI), без тембра GLaDOS.")
        p.torch_cpu = ask_yes("  Установить PyTorch?", True)

    if rep.whisper_installed:
        p.whisper = ask_yes("  Распознавание речи уже стоит. Переустановить?", False)
    else:
        out()
        out("  Распознавание речи (faster-whisper) — 50 МБ. Обязательно.")
        p.whisper = True

    if gpu_possible:
        out()
        out(f"  Ускорение GPU (CUDA) — 1.1 ГБ.")
        out(f"  Видеокарта {rep.gpu.name} подходит.")
        out("  Ускорит распознавание в 10-20 раз: можно взять большую модель")
        out("  и всё равно получать ответ мгновенно.")
        if rep.cuda_installed:
            p.cuda = ask_yes("  CUDA уже работает. Переустановить?", False)
        else:
            p.cuda = ask_yes("  Установить ускорение GPU?", True)
    elif rep.gpu.present and not rep.gpu.supports_cu12:
        out()
        out(f"  [!] Видеокарта есть, но драйвер поддерживает только "
            f"CUDA {rep.gpu.driver_cuda}.")
        out("      Обновите драйвер NVIDIA, чтобы включить ускорение.")

    out()
    if rep.silero_present:
        p.silero = ask_yes("  Модель голоса Silero уже скачана. Скачать заново?", False)
    else:
        p.silero = ask_yes("  Скачать модель голоса Silero (60 МБ)?", True)

    p.whisper_model = choose_model(rep, sysinfo, gpu=p.cuda or rep.cuda_installed)
    return p


def choose_model(rep, sysinfo, gpu: bool) -> str | None:
    """Выбор модели распознавания с подсказкой по скорости."""
    out()
    out("  Модель распознавания речи. Чем больше — тем точнее, но медленнее.")
    if rep.whisper_models:
        out(f"  Уже скачано: {', '.join(rep.whisper_models)}")
    out()

    names = ["tiny", "base", "small", "medium", "large-v3"]
    labels = []
    for n in names:
        mb = sysinfo.WHISPER_SIZES[n]
        size = f"{mb} МБ" if mb < 1000 else f"{mb / 1000:.1f} ГБ"
        speed = "мгновенно" if gpu else sysinfo.WHISPER_SPEED_CPU[n]
        note = ""
        if n == "small":
            note = "  <- рекомендуется"
        if n in ("medium", "large-v3") and not gpu:
            note = "  <- медленно без GPU"
        if n in rep.whisper_models:
            note += "  (уже скачана)"
        labels.append(f"{n:<9} {size:>8}   ответ {speed}{note}")

    default = 5 if gpu else 3
    labels.append("не скачивать сейчас (загрузится при первом запуске)")
    idx = ask_choice("  Какую скачать?", labels, default=default)
    if idx == len(labels):
        return None
    return names[idx - 1]


# ------------------------------------------------------------ выполнение
def execute(plan: Plan, rep, vpy: Path) -> bool:
    ok = True

    if plan.base or plan.whisper:
        pip_install(["--upgrade", "pip", "setuptools", "wheel"],
                    "Обновляю pip", vpy)

    if plan.torch_gpu:
        if not pip_install(["torch", "torchaudio", "--index-url",
                            "https://download.pytorch.org/whl/cu124"],
                           "PyTorch с поддержкой GPU (~2.5 ГБ)", vpy):
            out("  Пробую CPU-версию...")
            ok &= pip_install(["torch", "torchaudio", "--index-url",
                               "https://download.pytorch.org/whl/cpu"],
                              "PyTorch CPU", vpy)
    elif plan.torch_cpu:
        if not pip_install(["torch", "torchaudio", "--index-url",
                            "https://download.pytorch.org/whl/cpu"],
                           "PyTorch CPU (~200 МБ)", vpy):
            out("  Пробую обычную сборку с PyPI...")
            ok &= pip_install(["torch", "torchaudio"], "PyTorch (PyPI)", vpy)

    if plan.base or plan.whisper:
        ok &= pip_install(["-r", str(ROOT / "requirements.txt")],
                          "Основные зависимости", vpy)

    if plan.cuda:
        ok &= pip_install(["nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*"],
                          "Библиотеки CUDA: cuBLAS + cuDNN (~1.1 ГБ)", vpy)
        plan.cuda_works = verify_cuda(vpy)

    if plan.silero:
        ok &= download_silero(vpy)

    if plan.whisper_model:
        ok &= download_whisper(vpy, plan.whisper_model, use_gpu=plan.cuda or rep.cuda_installed)

    return ok


def verify_cuda(vpy: Path) -> bool:
    """Проверяет, что CUDA реально заработала, а не просто скачались пакеты."""
    out()
    out("--- Проверяю, работает ли ускорение ---")
    code = (
        "import sys\n"
        f"sys.path.insert(0, r'{ROOT}')\n"
        "from glados.stt import cuda_diagnose\n"
        "ok, reason = cuda_diagnose()\n"
        "print('OK' if ok else 'FAIL: ' + reason)\n"
        "sys.exit(0 if ok else 1)\n"
    )
    r = subprocess.run([str(vpy), "-c", code], capture_output=True, text=True)
    output = (r.stdout + r.stderr).strip()
    if r.returncode == 0:
        out("  [OK] Ускорение GPU работает")
        return True

    out(f"  [ВНИМАНИЕ] ускорение пока не работает")
    if output:
        out(f"  {output.splitlines()[-1]}")
    out()
    out("  Гладос будет работать на процессоре — это не помешает запуску.")
    out("  Частые причины:")
    out("    • нужен драйвер NVIDIA, поддерживающий CUDA 12 (проверьте: nvidia-smi)")
    out("    • после установки драйвера нужно перезагрузить компьютер")
    return True  # не считаем фатальной ошибкой


def download_silero(vpy: Path) -> bool:
    out()
    out("--- Скачиваю модель голоса Silero (60 МБ) ---")
    code = (
        "import torch, pathlib\n"
        f"d = pathlib.Path(r'{ROOT}') / 'models'\n"
        "d.mkdir(exist_ok=True)\n"
        "f = d / 'v4_ru.pt'\n"
        "if not f.exists():\n"
        "    torch.hub.download_url_to_file("
        "'https://models.silero.ai/models/tts/ru/v4_ru.pt', str(f))\n"
        "print('Модель голоса на месте:', f)\n"
    )
    if subprocess.call([str(vpy), "-c", code]) != 0:
        out("  [ВНИМАНИЕ] не удалось скачать сейчас — загрузится при первом запуске")
        return True  # не критично
    return True


def download_whisper(vpy: Path, model: str, use_gpu: bool) -> bool:
    out()
    out(f"--- Скачиваю модель распознавания «{model}» ---")
    out("    Это может занять несколько минут.")
    device = "cuda" if use_gpu else "cpu"
    compute = "float16" if use_gpu else "int8"
    code = (
        "from faster_whisper import WhisperModel\n"
        f"m = WhisperModel('{model}', device='{device}', compute_type='{compute}')\n"
        f"print('Модель {model} готова')\n"
    )
    if subprocess.call([str(vpy), "-c", code]) != 0:
        out("  [ВНИМАНИЕ] скачать не удалось — модель загрузится при первом запуске")
        return True
    return True


# --------------------------------------------------------- правка конфига
def update_config(plan: Plan, rep) -> None:
    """Прописывает в config.yaml выбранные устройство и модель."""
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return

    # Пишем "cuda" только если ускорение реально заработало.
    # Иначе "auto": Гладос сама включит GPU, когда он появится.
    gpu_on = plan.cuda_works or (rep.cuda_installed and not plan.cuda)
    device = "cuda" if gpu_on else "auto"
    compute = "float16" if gpu_on else "int8"
    model = plan.whisper_model

    try:
        text = cfg_path.read_text(encoding="utf-8")
    except Exception:
        return

    import re

    changed = []
    new = re.sub(r'(?m)^(\s*device:\s*)"[^"]*"',
                 lambda m: m.group(1) + f'"{device}"', text, count=1)
    if new != text:
        changed.append(f"device: {device}")
        text = new

    new = re.sub(r'(?m)^(\s*compute_type:\s*)"[^"]*"',
                 lambda m: m.group(1) + f'"{compute}"', text, count=1)
    if new != text:
        changed.append(f"compute_type: {compute}")
        text = new

    if model:
        new = re.sub(r'(?m)^(\s*model:\s*)"[^"]*"',
                     lambda m: m.group(1) + f'"{model}"', text, count=1)
        if new != text:
            changed.append(f"model: {model}")
            text = new

    if changed:
        try:
            cfg_path.write_text(text, encoding="utf-8")
            out()
            out("  config.yaml обновлён: " + ", ".join(changed))
        except Exception as e:
            out(f"  Не удалось обновить config.yaml: {e}")

    # Предупредим, если выбранная модель не влезет в видеопамять
    if gpu_on and model and rep.gpu.present:
        allowed = rep.gpu.recommended_models()
        if allowed and model not in allowed:
            out()
            out(f"  [ВНИМАНИЕ] модель «{model}» может не поместиться в "
                f"{rep.gpu.memory_mb / 1024:.0f} ГБ видеопамяти.")
            out(f"  Для вашей карты подойдёт: {allowed[-1]}")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("-h", "--help", action="store_true")
    args = ap.parse_args()

    if args.help:
        out(__doc__)
        return 0

    title("Установка GLaDOS Helper")
    out(f"Папка проекта: {ROOT}")
    out(f"Python: {sys.version.split()[0]}")

    if sys.version_info < (3, 9):
        out()
        out("[ОШИБКА] Нужен Python 3.9 или новее.")
        out("Скачайте с https://www.python.org/downloads/")
        return 1

    if sys.version_info >= (3, 13):
        out()
        out("[ВНИМАНИЕ] У вас Python "
            f"{sys.version_info.major}.{sys.version_info.minor}.")
        out("  Это очень свежая версия, и часть библиотек для неё пока не собрана.")
        out("  Если установка будет падать, поставьте Python 3.12 — он проверен.")

    from glados import sysinfo

    vpy_existing = venv_python() if venv_python().exists() else None
    rep = sysinfo.collect(vpy_existing)
    print_report(rep, sysinfo)

    if args.check:
        out()
        out("Режим проверки — ничего не менялось.")
        return 0

    # --- строим план ---
    if args.repair:
        plan = Plan()
        plan.base = plan.whisper = True
        plan.torch_cpu = not rep.torch_installed
        plan.cuda = rep.cuda_installed
    elif args.auto or args.gpu or args.cpu:
        use_gpu = args.gpu or (args.auto and rep.gpu.present and rep.gpu.supports_cu12)
        if args.cpu:
            use_gpu = False
        model = args.model or ("medium" if use_gpu else "small")
        plan = plan_full(rep, gpu=use_gpu, model=model)
    else:
        plan = plan_interactive(rep, sysinfo)

    if plan.empty():
        out()
        out("Нечего устанавливать — всё уже на месте.")
        out("Запускайте run.bat")
        return 0

    # --- подтверждение ---
    out()
    rule("-")
    out("Будет установлено:")
    for item in plan.describe():
        out(f"  • {item}")
    rule("-")

    if not (args.auto or args.gpu or args.cpu or args.repair):
        if not ask_yes("Продолжить?", True):
            out("Отменено.")
            return 0

    # --- окружение ---
    out()
    vpy = ensure_venv()
    if not vpy:
        return 1

    ok = execute(plan, rep, vpy)
    update_config(plan, rep)

    # --- итог ---
    out()
    out("--- Итоговая проверка ---")
    code = subprocess.call([str(vpy), str(ROOT / "check.py")])

    title("Готово! Запускайте run.bat" if ok and code == 0
          else "Установка завершена с замечаниями — смотрите выше")

    if ok and code == 0:
        out()
        out("Полезное:")
        out("  run.bat            запустить Гладос")
        out("  run.bat --text     проверить команды без микрофона")
        out("  check.bat          диагностика")
        out("  config.yaml        настройки: имя, пути к играм, голос")
    return 0 if (ok and code == 0) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        out()
        out("Прервано пользователем.")
        sys.exit(1)
