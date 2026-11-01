"""Configuration and prompt loading for FactShield v2."""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def cfg() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def prompt(name: str) -> tuple[str, str]:
    """(text, sha256) of fsv2/prompts/<name>.<version>.txt."""
    p = HERE / "prompts" / f"{name}.{cfg()['prompt_version']}.txt"
    text = p.read_text(encoding="utf-8").rstrip("\n")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()
