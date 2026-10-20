"""Shared plumbing for the study: config, provenance, durable JSONL.

Nothing here knows about verdicts. It exists so every output file carries the
same provenance (seed, git commit) without each script reimplementing it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

from . import _BACKEND  # noqa: F401  (ensures backend/ is importable)

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parent
CONFIG_PATH = STUDY_DIR / "config.yaml"


@functools.lru_cache(maxsize=None)
def config(path: str | None = None) -> dict:
    """study/config.yaml, or the file named by STUDY_CONFIG (for dry runs
    against fixtures; the real study always uses the committed file)."""
    p = Path(path or os.environ.get("STUDY_CONFIG") or CONFIG_PATH)
    with p.open() as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_file"] = str(p)
    return cfg


def resolve(rel: str) -> Path:
    """Paths in config are relative to the repository root."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def path(key: str) -> Path:
    return resolve(config()["paths"][key])


@functools.lru_cache(maxsize=1)
def git_commit() -> str:
    """HEAD commit, suffixed `-dirty` when the tree has uncommitted changes.

    A result produced from a dirty tree cannot be reproduced from the commit
    alone, so it is marked rather than passed off as clean.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def provenance() -> dict[str, Any]:
    prov = {"seed": config()["seed"], "git_commit": git_commit()}
    if config()["_config_file"] != str(CONFIG_PATH):
        prov["config_file"] = config()["_config_file"]   # a non-default config is never silent
    return prov


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ JSONL

def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    with p.open() as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A crash mid-write leaves at most one truncated last line.
                # Anything else is corruption and must not be skipped quietly.
                remaining = fh.read().strip()
                if remaining:
                    raise ValueError(f"{p}:{n} is not valid JSON and is not the last line")
                print(f"[study] ignoring truncated final line {n} in {p}")
    return rows


class DurableAppender:
    """Append-only JSONL writer that fsyncs every row.

    `eval/snapshot.py` appends and flushes; flush alone leaves the row in the
    OS cache, where a power cut or killed VM can still lose it. fsync is what
    makes "resume rather than restart" true.
    """

    def __init__(self, p: Path):
        p.parent.mkdir(parents=True, exist_ok=True)
        self.path = p
        self._fh = p.open("a", encoding="utf-8")

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "DurableAppender":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(p)


def write_meta(for_file: Path, **extra: Any) -> None:
    """Sidecar provenance for files (CSV) that cannot carry it inline."""
    write_json(for_file.with_suffix(for_file.suffix + ".meta.json"),
               {**provenance(), "written_at": now_iso(), "file": for_file.name, **extra})


# ------------------------------------------------------------------ claims

REQUIRED_CLAIM_FIELDS = ("id", "claim", "set", "label", "domain", "source_id")
GOLD_LABELS = ("supported", "refuted", "not_enough_evidence")


def load_claims(paths: Iterable[Path] | None = None) -> list[dict]:
    if paths is None:
        paths = sorted(path("claims").glob("*.jsonl"))
    claims: list[dict] = []
    seen: set[str] = set()
    for p in paths:
        for row in read_jsonl(Path(p)):
            missing = [f for f in REQUIRED_CLAIM_FIELDS if f not in row]
            if missing:
                raise ValueError(f"{p}: claim {row.get('id')} missing {missing}")
            if row["label"] not in GOLD_LABELS:
                raise ValueError(
                    f"{p}: claim {row['id']} label {row['label']!r} not in {GOLD_LABELS}")
            if row["id"] in seen:
                raise ValueError(f"duplicate claim id {row['id']}")
            seen.add(row["id"])
            claims.append(row)
    return claims


def iter_unique(xs: Iterable[str]) -> Iterator[str]:
    seen = set()
    for x in xs:
        if x not in seen:
            seen.add(x)
            yield x
