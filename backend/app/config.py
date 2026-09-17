"""Configuration loading.

Anything that could change without code changing lives in YAML: model choice,
source packs, tier weights, prompts. Prompts are versioned files and the
version is logged per request — a prompt changed mid-benchmark otherwise means
half the results came from a different system with no record of it.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

try:  # .env is read at import time; without this the file is ignored
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

PIPELINE_VERSION = "2.0.0"


@functools.lru_cache(maxsize=None)
def _load(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


@functools.lru_cache(maxsize=None)
def models() -> dict:
    return _load("models.yaml")


@functools.lru_cache(maxsize=None)
def source_packs() -> dict:
    return _load("source_packs.yaml")


@functools.lru_cache(maxsize=None)
def tiers() -> dict:
    return _load("tiers.yaml")


@functools.lru_cache(maxsize=None)
def prompt(name: str) -> tuple[str, str]:
    """Return (text, version). Version is the filename stem after the dot."""
    pdir = CONFIG_DIR / "prompts"
    matches = list(pdir.glob(f"{name}.v*.txt"))
    if not matches:
        raise FileNotFoundError(f"No prompt found for {name!r} in {pdir}")

    def _version_number(path) -> int:
        # Sort NUMERICALLY. A plain string sort puts v9 after v10, which would
        # silently pin the pipeline to an old prompt with no visible symptom.
        digits = "".join(ch for ch in path.stem.split(".")[-1] if ch.isdigit())
        return int(digits or 0)

    latest = max(matches, key=_version_number)
    version = latest.stem.split(".")[-1]
    return latest.read_text(), version


def pack_for(category: str, jurisdiction: str | None = None) -> dict:
    """Authority domains and expected tiers for a category.

    One search call per category, not one per site: the whole pack is passed
    as a domain filter and the search provider ranks across it. No per-claim
    source selection is needed.
    """
    packs = source_packs()
    entry = packs.get(category) or packs.get("general", {})
    if jurisdiction and "jurisdictions" in entry:
        entry = {**entry, **entry["jurisdictions"].get(jurisdiction, {})}
    return entry


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def flag(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
