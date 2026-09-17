"""Provider-agnostic LLM calls.

Model choice is a config value, never hardcoded. Which model to use at the
verdict step is an empirical question — run the claim set through several and
report the spread. A provider abstraction is what makes that a config edit
rather than a rewrite.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from ..config import env, models

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = _FENCE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Model wrapped the object in prose. Take the outermost braces.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"No JSON object in response: {text[:200]!r}")
        return json.loads(cleaned[start : end + 1])


async def _call_groq(model: str, system: str, user: str, cfg: dict) -> str:
    key = env("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY not set")
    async with httpx.AsyncClient(timeout=cfg.get("timeout_s", 10)) as c:
        r = await c.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": cfg.get("temperature", 0),
                "max_tokens": cfg.get("max_tokens", 600),
                "response_format": {"type": "json_object"},
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def _call_gemini(model: str, system: str, user: str, cfg: dict) -> str:
    key = env("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    async with httpx.AsyncClient(timeout=cfg.get("timeout_s", 12)) as c:
        r = await c.post(
            url,
            params={"key": key},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": cfg.get("temperature", 0),
                    "maxOutputTokens": cfg.get("max_tokens", 700),
                    "responseMimeType": "application/json",
                },
            },
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]


_PROVIDERS = {"groq": _call_groq, "gemini": _call_gemini}


async def complete_json(stage: str, system: str, user: str) -> dict[str, Any]:
    """Call the model configured for `stage`, returning parsed JSON.

    Tries the primary provider, then each fallback. One JSON-repair retry per
    provider before moving on. Raises LLMError only when everything fails —
    callers treat that as abstention, never as licence to guess.
    """
    cfg_all = models()
    stage_cfg = cfg_all.get(stage, {})
    chain = [stage_cfg] + list(cfg_all.get("fallbacks", {}).get(stage, []))

    last: list[str] = []
    for entry in chain:
        provider = entry.get("provider")
        model = entry.get("model")
        fn = _PROVIDERS.get(provider)
        if not fn or not model:
            last.append(f"{provider}/{model}: not configured")
            continue
        merged = {**stage_cfg, **entry}
        for attempt in range(2):
            try:
                raw = await fn(model, system, user, merged)
                return _extract_json(raw)
            except (LLMError, json.JSONDecodeError) as exc:
                if attempt == 0:
                    user = (
                        f"{user}\n\nYour previous reply was not valid JSON "
                        f"({exc}). Reply with the JSON object only."
                    )
                    continue
                last.append(f"{provider}/{model}: {exc}")
                break
            except httpx.HTTPStatusError as exc:
                body = ""
                try:
                    body = exc.response.text[:200]
                except Exception:
                    pass
                last.append(f"{provider}/{model}: HTTP {exc.response.status_code} {body}")
                break
            except httpx.HTTPError as exc:
                last.append(f"{provider}/{model}: {type(exc).__name__} {exc}")
                break

    # Report EVERY provider's failure. Reporting only the last one hides the
    # real cause when the primary fails and the fallback fails differently.
    detail = "; ".join(last) if last else "no providers attempted"
    raise LLMError(f"All providers failed for stage {stage!r}: {detail}")