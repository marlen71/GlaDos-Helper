"""Загрузка конфигурации."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


class Config(dict):
    """Словарь конфига с доступом через точку: cfg.get_path('tts.speaker')."""

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        p = Path(path) if path else DEFAULT_CONFIG
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls(data)
        cfg.path = p
        return cfg

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # --- удобные ярлыки ---
    @property
    def root(self) -> Path:
        return ROOT

    def data_file(self, dotted: str, fallback: str) -> Path:
        rel = self.get_path(dotted, fallback)
        p = Path(rel)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
