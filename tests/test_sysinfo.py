"""Тесты определения железа и планирования установки."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glados import sysinfo  # noqa: E402

NVIDIA_SMI_OUTPUT = """\
Mon Sep 23 21:40:11 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 551.86                 Driver Version: 551.86       CUDA Version: 12.4       |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060      WDDM  |   00000000:01:00.0  On |                  N/A |
|  0%   45C    P8             12W /  170W |    1024MiB /  12288MiB |      2%      Default |
+-----------------------------------------+------------------------+----------------------+
"""

OLD_DRIVER_OUTPUT = """\
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 470.82       Driver Version: 470.82       CUDA Version: 11.4               |
|   0  NVIDIA GeForce GTX 1060      WDDM  |   00000000:01:00.0  On |                  N/A |
|  0%   40C    P8              9W /  120W |     512MiB /   6144MiB |      0%      Default |
+---------------------------------------------------------------------------------------+
"""


def test_parse_nvidia_smi_full():
    info = sysinfo.parse_nvidia_smi(NVIDIA_SMI_OUTPUT)
    assert info.present
    assert "RTX 3060" in info.name
    assert info.memory_mb == 12288
    assert info.driver_cuda == "12.4"
    assert info.cuda_major == 12
    assert info.supports_cu12


def test_parse_old_driver_not_cu12():
    info = sysinfo.parse_nvidia_smi(OLD_DRIVER_OUTPUT)
    assert info.present
    assert info.cuda_major == 11
    assert not info.supports_cu12


def test_parse_empty_output():
    info = sysinfo.parse_nvidia_smi("")
    assert not info.present


def test_parse_query_format():
    info = sysinfo.parse_nvidia_smi_query("NVIDIA GeForce RTX 4070, 12282 MiB")
    assert info.present
    assert info.name == "NVIDIA GeForce RTX 4070"
    assert info.memory_mb == 12282


@pytest.mark.parametrize("mem_mb,expected_max", [
    (12288, "large-v3"),
    (6144, "medium"),
    (4096, "small"),
    (2048, "base"),
])
def test_recommended_models_by_vram(mem_mb, expected_max):
    gpu = sysinfo.GpuInfo(present=True, name="test", memory_mb=mem_mb)
    assert gpu.recommended_models()[-1] == expected_max


def test_no_gpu_recommends_nothing():
    assert sysinfo.GpuInfo(present=False).recommended_models() == []


def test_whisper_sizes_cover_all_models():
    for m in ("tiny", "base", "small", "medium", "large-v3"):
        assert m in sysinfo.WHISPER_SIZES
        assert m in sysinfo.WHISPER_SPEED_CPU


def test_module_installed_detects_stdlib():
    assert sysinfo.module_installed("json")
    assert not sysinfo.module_installed("этого_модуля_точно_нет_12345")


def test_collect_runs_without_venv():
    rep = sysinfo.collect(None)
    assert rep.python_version
    assert isinstance(rep.whisper_models, list)
    assert not rep.venv_exists


# ----------------------------------------------------------- план установки
def _import_setup():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "setup_glados", Path(__file__).resolve().parent.parent / "setup_glados.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plan_skips_installed_components():
    setup = _import_setup()
    rep = sysinfo.SystemReport(
        base_installed=True, torch_installed=True, whisper_installed=True,
        silero_present=True, whisper_models=["small"])
    plan = setup.plan_full(rep, gpu=False, model="small")
    assert plan.empty(), "ничего не должно ставиться повторно"


def test_plan_installs_missing_parts():
    setup = _import_setup()
    rep = sysinfo.SystemReport()
    plan = setup.plan_full(rep, gpu=False, model="small")
    assert plan.base and plan.whisper and plan.silero
    assert plan.whisper_model == "small"
    assert not plan.cuda


def test_plan_with_gpu_adds_cuda():
    setup = _import_setup()
    rep = sysinfo.SystemReport(gpu=sysinfo.GpuInfo(present=True, driver_cuda="12.4"))
    plan = setup.plan_full(rep, gpu=True, model="medium")
    assert plan.cuda
    assert "CUDA" in " ".join(plan.describe())


def test_plan_describe_is_human_readable():
    setup = _import_setup()
    rep = sysinfo.SystemReport()
    plan = setup.plan_full(rep, gpu=True, model="small")
    text = " ".join(plan.describe())
    assert "МБ" in text or "ГБ" in text


# ---------------------------------------- регрессия: GTX 1660 у пользователя
SMI_NO_CUDA_VERSION = """\
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 551.86                                                                     |
|-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA GeForce GTX 1660 SUPER  WDDM |   00000000:01:00.0  On |                  N/A |
|  0%   38C    P8              8W /  125W |     512MiB /   6144MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
"""


def test_gpu_without_cuda_version_still_supported():
    """nvidia-smi не всегда печатает 'CUDA Version'.

    Раньше это приводило к тому, что установщик не предлагал GPU вообще.
    """
    info = sysinfo.parse_nvidia_smi(SMI_NO_CUDA_VERSION)
    assert info.present
    assert info.memory_mb == 6144
    assert info.cuda_major == 0
    assert info.supports_cu12, "неизвестная версия не должна блокировать GPU"


def test_gtx_1660_name_not_truncated():
    info = sysinfo.parse_nvidia_smi(SMI_NO_CUDA_VERSION)
    assert "GTX 1660 SUPER" in info.name
    assert "WDDM" not in info.name


def test_rtx_name_still_parsed():
    info = sysinfo.parse_nvidia_smi(NVIDIA_SMI_OUTPUT)
    assert "RTX 3060" in info.name
    assert "WDDM" not in info.name


def test_6gb_card_recommends_medium():
    info = sysinfo.parse_nvidia_smi(SMI_NO_CUDA_VERSION)
    assert "medium" in info.recommended_models()
    assert "large-v3" not in info.recommended_models()


def test_gpu_menu_offered_for_this_card():
    """Пункт «Установка с ускорением GPU» должен появляться для GTX 1660."""
    info = sysinfo.parse_nvidia_smi(SMI_NO_CUDA_VERSION)
    rep = sysinfo.SystemReport(gpu=info)
    gpu_possible = rep.gpu.present and rep.gpu.supports_cu12
    assert gpu_possible


def test_old_driver_still_rejected():
    """А вот честно старый драйвер CUDA 11 по-прежнему не подходит."""
    info = sysinfo.parse_nvidia_smi(OLD_DRIVER_OUTPUT)
    assert info.cuda_major == 11
    assert not info.supports_cu12


# ------------------------------------------------ подключение DLL из pip
def test_cuda_setup_finds_dirs_without_crashing():
    from glados import cuda_setup

    dirs = cuda_setup.nvidia_dll_dirs()
    assert isinstance(dirs, list)


def test_cuda_setup_apply_idempotent():
    from glados import cuda_setup

    first = cuda_setup.apply()
    second = cuda_setup.apply()
    assert first == second


def test_cuda_setup_describe_returns_text():
    from glados import cuda_setup

    assert isinstance(cuda_setup.describe(), str)


def test_cuda_diagnose_gives_reason():
    from glados.stt import cuda_diagnose

    ok, reason = cuda_diagnose()
    assert isinstance(ok, bool)
    if not ok:
        assert reason, "при отказе должна быть указана причина"


def test_finds_nvidia_dlls_in_pip_layout(tmp_path, monkeypatch):
    """Главная регрессия: DLL лежат в site-packages/nvidia/<пакет>/bin.

    Именно из-за того, что эти папки не подключались, faster-whisper падал
    с 'cublas64_12.dll is not found', хотя пакеты были установлены.
    """
    from glados import cuda_setup

    root = tmp_path / "nvidia"
    for pkg, files in {
        "cublas": ["cublas64_12.dll", "cublasLt64_12.dll"],
        "cudnn": ["cudnn_ops64_9.dll", "cudnn64_9.dll"],
        "cuda_nvrtc": ["nvrtc64_120_0.dll"],
    }.items():
        d = root / pkg / "bin"
        d.mkdir(parents=True)
        for f in files:
            (d / f).touch()

    monkeypatch.syspath_prepend(str(tmp_path))
    found = cuda_setup.nvidia_dll_dirs()
    names = {Path(p).parent.name for p in found}
    assert {"cublas", "cudnn", "cuda_nvrtc"} <= names


def test_apply_adds_dirs_to_path(tmp_path, monkeypatch):
    from glados import cuda_setup

    d = tmp_path / "nvidia" / "cublas" / "bin"
    d.mkdir(parents=True)
    (d / "cublas64_12.dll").touch()

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PATH", "")
    applied = cuda_setup.apply(force=True)
    assert any("cublas" in a for a in applied)
    import os

    assert "cublas" in os.environ["PATH"]
