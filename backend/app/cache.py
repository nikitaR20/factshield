"""Caching and request logging.

The cache key includes the pipeline version, so changing a prompt or a source
pack invalidates cached results automatically rather than silently serving
output from an older system.

Keying on the RESOLVED claim rather than the raw selection is what makes the
cache useful: "he said it would double" collides across unrelated articles,
while "[Name] said [thing] would double in 2026" does not.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .config import PIPELINE_VERSION, env, flag

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

_memory: dict[str, str] = {}
_redis = None


def _client():
    global _redis
    if _redis is not None:
        return _redis
    url = env("REDIS_URL")
    if not url:
        return None
    try:
        import redis.asyncio as aioredis

        _redis = aioredis.from_url(url, decode_responses=True)
        return _redis
    except Exception:
        return None


def normalise(claim: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", claim.lower())).strip()


def key_for(claim: str) -> str:
    digest = hashlib.sha256(normalise(claim).encode()).hexdigest()[:32]
    return f"fs:v{PIPELINE_VERSION}:{digest}"


def ttl_seconds() -> int:
    # A permanent dev keyspace means repeated benchmark runs cost nothing after
    # the first pass. Never enable this in a participant-facing deployment.
    if flag("FACTSHIELD_DEV_CACHE"):
        return 0
    return int(env("CACHE_TTL_SECONDS", "172800"))  # 48h


async def get(claim: str) -> dict | None:
    k = key_for(claim)
    client = _client()
    if client is not None:
        try:
            raw = await client.get(k)
            return json.loads(raw) if raw else None
        except Exception:
            pass
    raw = _memory.get(k)
    return json.loads(raw) if raw else None


async def put(claim: str, payload: dict) -> None:
    k = key_for(claim)
    blob = json.dumps(payload, default=str)
    client = _client()
    if client is not None:
        try:
            ttl = ttl_seconds()
            if ttl:
                await client.setex(k, ttl, blob)
            else:
                await client.set(k, blob)
            return
        except Exception:
            pass
    _memory[k] = blob


# ------------------------------------------------------------------ logging

LOG_PATH = Path(env("FACTSHIELD_LOG", "./logs/requests.jsonl"))


def write_log(record) -> None:
    """One JSON line per request.

    This schema IS the evaluation dataset. Benchmark ablations become filters
    over this file; the circularity analysis is already available because
    publication dates are recorded per document. That avoids the usual failure
    where a separate evaluation script quietly diverges from what production
    actually does.
    """
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
        with LOG_PATH.open("a") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass  # logging must never break a request
