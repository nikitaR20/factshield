"""Provider model discovery.

Model names go stale. Providers deprecate and rename aggressively, and a stale
name returns 404 — which looks identical to a broken key. Rather than
hardcoding names that will rot, ask the provider what this key can actually
call, and put valid names into config.

    GET /v1/models
"""

from __future__ import annotations

import httpx

from ..config import env


async def gemini_models() -> dict:
    key = env("GEMINI_API_KEY")
    if not key:
        return {"ok": False, "error": "GEMINI_API_KEY not set"}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": key},
            )
            if r.status_code != 200:
                return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
            data = r.json()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    usable = [
        m["name"].removeprefix("models/")
        for m in data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]
    return {
        "ok": bool(usable),
        "usable_for_generateContent": sorted(usable),
        "hint": "Put one of these in config/models.yaml under decide.model",
    }


async def groq_models() -> dict:
    key = env("GROQ_API_KEY")
    if not key:
        return {"ok": False, "error": "GROQ_API_KEY not set"}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            if r.status_code != 200:
                return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
            data = r.json()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    ids = [m["id"] for m in data.get("data", []) if m.get("active", True)]
    return {
        "ok": bool(ids),
        "usable": sorted(ids),
        "hint": "Put one of these in config/models.yaml under triage.model",
    }


async def probe() -> dict:
    """Actually call each configured model once. Discovery says what exists;
    this says what works."""
    from .llm import LLMError, complete_json

    results = {}
    for stage in ("triage", "decide"):
        try:
            await complete_json(
                stage,
                'Reply with JSON only: {"ok": true}',
                "Return the object.",
            )
            results[stage] = "ok"
        except LLMError as exc:
            results[stage] = str(exc)
    return results


def nli_benchmark(n: int = 24) -> dict:
    """Time the loaded NLI model on a representative batch.

    Grading dominates request time, so this is the number to watch when
    choosing between models. Model choice is an empirical question — run this,
    then decide.
    """
    import time

    from . import nli

    premise = (
        "2019 Novel Coronavirus (2019-nCoV), Wuhan, China. CDC is closely "
        "monitoring an outbreak of respiratory illness caused by a novel "
        "coronavirus that was first detected in Wuhan City, Hubei Province, China."
    )
    hypothesis = "The COVID-19 outbreak was first detected in Wuhan, China."

    nli.classify([(premise, hypothesis)])  # warm

    start = time.perf_counter()
    probs = nli.classify([(premise, hypothesis)] * n)
    elapsed = time.perf_counter() - start

    return {
        "backend": nli.backend_name(),
        "pairs": n,
        "total_seconds": round(elapsed, 2),
        "seconds_per_pair": round(elapsed / n, 4),
        "estimated_grading_seconds_per_request": round(elapsed / n * 24, 1),
        "sample_probabilities": probs[0],
        "note": "A request grades roughly 24 pairs. Under ~5s is comfortable.",
    }
