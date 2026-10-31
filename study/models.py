"""Provider adapters for the study: OpenAI, Gemini, an open-weight model on Groq.

One entry point, `complete()`, with the same signature for every provider. It
never raises for an API problem: failures come back as a `CallResult` with
`error` set, so a run records them instead of dying.

Why not `app.providers.llm`? That client forces JSON output, falls through to
other providers on error, and returns only parsed content. The study needs free
prose, needs one provider per call with no silent substitution, and needs the
full raw response. The two have different contracts, so they are separate.

Search is OFF unless `live_search=True`:
  * no `tools` field is sent;
  * model ids that are themselves search products are refused;
  * every response is scanned for any field that indicates a tool, search or
    grounding step, and what was found is returned in `tool_calls`. An empty
    list on every row is the evidence that search was off.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from app.config import env

from .common import config, now_iso

# ------------------------------------------------------------------ result


@dataclass
class CallResult:
    provider: str
    model_requested: str
    model_version: str | None = None      # what the RESPONSE says it was
    text: str | None = None
    reasoning_text: str | None = None     # provider-exposed thinking, if any
    finish_reason: str | None = None
    tools_attached: list[dict] = field(default_factory=list)   # what we SENT
    tool_calls: list[dict] = field(default_factory=list)       # what we SAW
    temperature_sent: float | None = None
    params_sent: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    latency_ms: float | None = None       # final attempt only
    latency_total_ms: float | None = None # including retries and backoff
    tokens: dict[str, int | None] = field(default_factory=dict)
    http_status: int | None = None
    attempts: int = 0
    error: str | None = None
    raw: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class _Fatal(Exception):
    """Do not retry: wrong model, bad parameter, bad key."""


# ------------------------------------------------------- tool-use detection

# Any of these keys, holding a non-empty value anywhere in a response, means a
# tool, search or grounding step may have run. Deliberately broad: a false
# alarm costs a look at one row; a miss costs the study's main control.
_TOOL_KEYS = {
    "tool_calls", "executed_tools", "function_call", "functionCall",
    "executableCode", "codeExecutionResult", "groundingMetadata",
    "groundingChunks", "searchEntryPoint", "webSearchQueries",
    "annotations", "url_citation", "search_results", "citations",
    "server_tool_use", "web_search_call", "search_context",
}
_TOOL_TYPES = {"web_search_call", "web_search", "browser_search", "file_search_call",
               "code_interpreter_call", "function_call", "mcp_call", "computer_call"}


def find_tool_use(obj: Any, path: str = "$") -> list[dict]:
    hits: list[dict] = []
    if isinstance(obj, dict):
        t = obj.get("type")
        if isinstance(t, str) and t in _TOOL_TYPES:
            hits.append({"path": path, "type": t, "detail": _short(obj)})
        for k, v in obj.items():
            p = f"{path}.{k}"
            if k in _TOOL_KEYS and v not in (None, [], {}, ""):
                hits.append({"path": p, "type": k, "detail": _short(v)})
            hits.extend(find_tool_use(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(find_tool_use(v, f"{path}[{i}]"))
    # one hit per path is enough
    uniq, seen = [], set()
    for h in hits:
        key = (h["path"], h["type"])
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    return uniq


def _short(v: Any, n: int = 300) -> str:
    s = repr(v)
    return s if len(s) <= n else s[:n] + "…"


# ------------------------------------------------------------------ helpers

def model_cfg(model_key: str) -> dict:
    models = config()["models"]
    if model_key not in models:
        raise KeyError(f"unknown model key {model_key!r}; configured: {sorted(models)}")
    return models[model_key]


def is_search_model(model_id: str) -> bool:
    mid = model_id.lower()
    return any(m in mid for m in config()["search_model_markers"])


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


async def _post_with_retry(
    client: httpx.AsyncClient, url: str, *, headers: dict, json: dict,
    timeout: float, result: CallResult,
) -> dict:
    """POST with exponential backoff and full jitter on 408/429/5xx and
    transport errors. Any other 4xx raises _Fatal at once.

    Full jitter (sleep uniformly in [0, cap]) rather than a fixed doubling, so
    concurrent workers that hit a 429 together do not retry together.
    """
    rc = config()["retry"]
    retry_statuses = set(rc["retry_statuses"])
    jitter = random.Random()  # unseeded on purpose: must not disturb the study seed
    start_all = time.perf_counter()
    last_err = "no attempt made"

    for attempt in range(1, rc["max_attempts"] + 1):
        result.attempts = attempt
        t0 = time.perf_counter()
        wait: float | None = None
        try:
            resp = await client.post(url, headers=headers, json=json, timeout=timeout)
            result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            result.http_status = resp.status_code
            if resp.status_code < 400:
                result.latency_total_ms = round((time.perf_counter() - start_all) * 1000, 1)
                return resp.json()
            body = resp.text[:500]
            last_err = f"HTTP {resp.status_code}: {body}"
            if resp.status_code not in retry_statuses:
                raise _Fatal(last_err)
            wait = _retry_after(resp)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            last_err = f"{type(exc).__name__}: {exc}"

        if attempt == rc["max_attempts"]:
            break
        cap = min(rc["max_delay_s"], rc["base_delay_s"] * 2 ** (attempt - 1))
        delay = jitter.uniform(0, cap)
        if wait is not None:
            delay = max(delay, min(wait, rc["max_delay_s"]))
        await asyncio.sleep(delay)

    result.latency_total_ms = round((time.perf_counter() - start_all) * 1000, 1)
    raise _Fatal(f"gave up after {result.attempts} attempts; last: {last_err}")


# ------------------------------------------------------------------ OpenAI

async def _openai(cfg: dict, system: str, user: str, live: bool,
                  client: httpx.AsyncClient, r: CallResult) -> None:
    key = env("OPENAI_API_KEY")
    if not key:
        raise _Fatal("OPENAI_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}"}

    if not live:
        # Plain Chat Completions. Only *search* model ids search on this
        # endpoint, and those are refused before we get here.
        body: dict[str, Any] = {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_completion_tokens": cfg["max_output_tokens"],
        }
        if cfg.get("temperature") is not None:
            body["temperature"] = cfg["temperature"]
        if cfg.get("reasoning_effort") is not None:
            body["reasoning_effort"] = cfg["reasoning_effort"]
        r.params_sent = {k: v for k, v in body.items() if k != "messages"}
        data = await _post_with_retry(client, "https://api.openai.com/v1/chat/completions",
                                      headers=headers, json=body, timeout=cfg["timeout_s"], result=r)
        r.raw = data
        r.model_version = data.get("model")
        if data.get("system_fingerprint"):
            r.params_sent["_system_fingerprint"] = data["system_fingerprint"]
        ch = (data.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        r.text = msg.get("content")
        r.finish_reason = ch.get("finish_reason")
        u = data.get("usage") or {}
        r.tokens = {"input": u.get("prompt_tokens"), "output": u.get("completion_tokens"),
                    "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    "total": u.get("total_tokens")}
        return

    # Live search: Responses API with the web_search tool attached.
    tools = [{"type": "web_search"}]
    r.tools_attached = tools
    body = {"model": cfg["model"],
            "input": [{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            "tools": tools, "max_output_tokens": cfg["max_output_tokens"]}
    if cfg.get("temperature") is not None:
        body["temperature"] = cfg["temperature"]
    if cfg.get("reasoning_effort") is not None:
        body["reasoning"] = {"effort": cfg["reasoning_effort"]}
    r.params_sent = {k: v for k, v in body.items() if k != "input"}
    data = await _post_with_retry(client, "https://api.openai.com/v1/responses",
                                  headers=headers, json=body, timeout=cfg["timeout_s"], result=r)
    r.raw = data
    r.model_version = data.get("model")
    texts = []
    for item in data.get("output") or []:
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") in ("output_text", "text"):
                    texts.append(c.get("text", ""))
    r.text = "\n".join(texts) if texts else None
    r.finish_reason = data.get("status")
    if data.get("incomplete_details"):
        r.finish_reason = f"{data.get('status')}:{data['incomplete_details'].get('reason')}"
    u = data.get("usage") or {}
    r.tokens = {"input": u.get("input_tokens"), "output": u.get("output_tokens"),
                "reasoning": (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
                "total": u.get("total_tokens")}


# ------------------------------------------------------------------ Gemini

async def _gemini(cfg: dict, system: str, user: str, live: bool,
                  client: httpx.AsyncClient, r: CallResult) -> None:
    key = env("GEMINI_API_KEY")
    if not key:
        raise _Fatal("GEMINI_API_KEY not set")
    gen: dict[str, Any] = {"maxOutputTokens": cfg["max_output_tokens"]}
    if cfg.get("temperature") is not None:
        gen["temperature"] = cfg["temperature"]
    if cfg.get("thinking_level") is not None:
        gen["thinkingConfig"] = {"thinkingLevel": cfg["thinking_level"]}
    body: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen,
    }
    if live:
        body["tools"] = [{"google_search": {}}]
        r.tools_attached = body["tools"]
    r.params_sent = {"model": cfg["model"], "generationConfig": gen, "tools": body.get("tools", [])}
    # Key in a header, not the query string: URLs end up in proxy and error logs.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg['model']}:generateContent"
    data = await _post_with_retry(client, url, headers={"x-goog-api-key": key},
                                  json=body, timeout=cfg["timeout_s"], result=r)
    r.raw = data
    r.model_version = data.get("modelVersion")
    cand = (data.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    answer = [p.get("text", "") for p in parts if "text" in p and not p.get("thought")]
    thought = [p.get("text", "") for p in parts if "text" in p and p.get("thought")]
    r.text = "".join(answer) if answer else None
    r.reasoning_text = "".join(thought) if thought else None
    r.finish_reason = cand.get("finishReason")
    block = (data.get("promptFeedback") or {}).get("blockReason")
    if block and not r.finish_reason:
        r.finish_reason = f"prompt_blocked:{block}"
    u = data.get("usageMetadata") or {}
    r.tokens = {"input": u.get("promptTokenCount"), "output": u.get("candidatesTokenCount"),
                "reasoning": u.get("thoughtsTokenCount"), "total": u.get("totalTokenCount")}


# ------------------------------------------------------------------ Groq

async def _groq(cfg: dict, system: str, user: str, live: bool,
                client: httpx.AsyncClient, r: CallResult) -> None:
    key = env("GROQ_API_KEY")
    if not key:
        raise _Fatal("GROQ_API_KEY not set")
    body: dict[str, Any] = {
        "model": cfg["model"],
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_completion_tokens": cfg["max_output_tokens"],
    }
    if cfg.get("temperature") is not None:
        body["temperature"] = cfg["temperature"]
    if cfg.get("reasoning_effort") is not None:
        body["reasoning_effort"] = cfg["reasoning_effort"]
    if live:
        if "gpt-oss" not in cfg["model"]:
            raise _Fatal(f"live search on Groq needs a gpt-oss model; {cfg['model']} has no browser_search")
        body["tools"] = [{"type": "browser_search"}]
        r.tools_attached = body["tools"]
    r.params_sent = {k: v for k, v in body.items() if k != "messages"}
    data = await _post_with_retry(client, "https://api.groq.com/openai/v1/chat/completions",
                                  headers={"Authorization": f"Bearer {key}"},
                                  json=body, timeout=cfg["timeout_s"], result=r)
    r.raw = data
    r.model_version = data.get("model")
    if data.get("system_fingerprint"):
        r.params_sent["_system_fingerprint"] = data["system_fingerprint"]
    ch = (data.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    r.text = msg.get("content")
    r.reasoning_text = msg.get("reasoning")
    r.finish_reason = ch.get("finish_reason")
    u = data.get("usage") or {}
    r.tokens = {"input": u.get("prompt_tokens"), "output": u.get("completion_tokens"),
                "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "total": u.get("total_tokens")}


# ------------------------------------------------------------------ Anthropic

async def _anthropic(cfg: dict, system: str, user: str, live: bool,
                     client: httpx.AsyncClient, r: CallResult) -> None:
    """Claude Messages API. Added 2026-10-31 for the exploratory fourth panel
    model (fsv2/fourth_model.py); not one of `study_models`.

    Claude Haiku 5.5 returns HTTP 400 for any non-default temperature, so its
    config has temperature: null and none is sent (like Gemini here). Its
    thinking is on by default; `thinking: disabled` turns it off at effort
    `high` or below. Effort goes in `output_config.effort`.
    """
    key = env("ANTHROPIC_API_KEY")
    if not key:
        raise _Fatal("ANTHROPIC_API_KEY not set")
    if live:
        raise _Fatal("live search is not implemented for the anthropic provider")
    body: dict[str, Any] = {
        "model": cfg["model"],
        "max_tokens": cfg["max_output_tokens"],
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if cfg.get("temperature") is not None:
        body["temperature"] = cfg["temperature"]
    if cfg.get("thinking") is not None:
        body["thinking"] = {"type": cfg["thinking"]}
    if cfg.get("effort") is not None:
        body["output_config"] = {"effort": cfg["effort"]}
    r.params_sent = {k: v for k, v in body.items() if k not in ("messages", "system")}
    headers = {"x-api-key": key,
               "anthropic-version": cfg.get("api_version") or "2023-06-01",
               "content-type": "application/json"}
    data = await _post_with_retry(client, "https://api.anthropic.com/v1/messages",
                                  headers=headers, json=body, timeout=cfg["timeout_s"], result=r)
    r.raw = data
    r.model_version = data.get("model")
    blocks = data.get("content") or []
    text = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    thought = [b.get("thinking", "") for b in blocks
               if b.get("type") == "thinking" and b.get("thinking")]
    r.text = "".join(text) if text else None
    r.reasoning_text = "".join(thought) if thought else None
    r.finish_reason = data.get("stop_reason")
    u = data.get("usage") or {}
    inp, out = u.get("input_tokens"), u.get("output_tokens")
    r.tokens = {"input": inp, "output": out, "reasoning": None,
                "total": (inp or 0) + (out or 0) if (inp is not None or out is not None) else None}


# ------------------------------------------------------------------ stub

async def _stub(cfg: dict, system: str, user: str, live: bool,
                client: httpx.AsyncClient, r: CallResult) -> None:
    """Offline stand-in. Deterministic in its input; makes no network call.

    It produces text in whichever shape the prompt asks for, cites passage ids
    it finds in the prompt, and occasionally cites one that does not exist, so
    every downstream branch (parsing, citation checks) is exercised. Its
    verdicts are a hash of the prompt — they mean nothing.
    """
    import re

    h = int(hashlib.sha256((cfg["model"] + system + user).encode()).hexdigest(), 16)
    labels = list(config()["labels"].values())
    verdict = labels[h % 3]
    ids = sorted(set(re.findall(r"\[(P\d+)\]", user)), key=lambda s: int(s[1:]))
    cite = ids[h % len(ids)] if ids else None
    fabricated = "P99" if (h >> 8) % 5 == 0 else None

    if "Confidence:" in user:
        text = (f"Reasoning: The passages were compared with the claim (stub output).\n"
                f"Verdict: {verdict}\nConfidence: {['low', 'medium', 'high'][(h >> 4) % 3]}")
    elif "end with the identifier" in user and (cite or fabricated):
        c1, c2 = cite or fabricated, fabricated or cite
        text = (f"Verdict: {verdict} [{c1}]\n"
                f"The stub compared the claim with the passage [{c1}]. "
                f"This sentence exists to test citation checks [{c2}].")
    else:
        text = f"Verdict: {verdict}\nThis is offline stub output, not a model's judgement."

    r.raw = {"stub": True, "note": "offline stand-in; not model output",
             "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
    r.model_version = "stub-offline-v1"
    r.text = text
    r.finish_reason = "stop"
    r.params_sent = {"model": cfg["model"], "temperature": cfg.get("temperature")}
    r.tokens = {"input": len(user.split()), "output": len(text.split()), "reasoning": 0,
                "total": len(user.split()) + len(text.split())}
    r.http_status = 200
    r.attempts = 1
    r.latency_ms = r.latency_total_ms = 0.0


_PROVIDERS = {"openai": _openai, "gemini": _gemini, "groq": _groq,
              "anthropic": _anthropic, "stub": _stub}


# ------------------------------------------------------------------ entry

async def complete(model_key: str, system: str, user: str, *, live_search: bool = False,
                   client: httpx.AsyncClient | None = None) -> CallResult:
    """Call one configured model once. Same signature for every provider.

    Never raises for an API failure; check `result.error`.
    """
    cfg = model_cfg(model_key)
    r = CallResult(provider=cfg["provider"], model_requested=cfg["model"], timestamp=now_iso())

    if cfg["model"] == "CHANGE_ME":
        r.error = f"model for {model_key!r} is still CHANGE_ME in study/config.yaml"
        return r
    if not live_search and is_search_model(cfg["model"]):
        r.error = f"{cfg['model']!r} is a search product; refused in closed-book mode"
        return r

    fn = _PROVIDERS[cfg["provider"]]
    own = client is None
    client = client or httpx.AsyncClient()
    try:
        await fn(cfg, system, user, live_search, client, r)
    except _Fatal as exc:
        r.error = str(exc)
    except Exception as exc:  # malformed JSON, unexpected shape: record, don't die
        r.error = f"{type(exc).__name__}: {exc}"
    finally:
        if own:
            await client.aclose()

    if r.raw is not None:
        r.tool_calls = find_tool_use(r.raw)
    # Derived only from what was actually put in the request. No request, or
    # no temperature field in it, means None.
    sent = r.params_sent.get("generationConfig", r.params_sent) if r.params_sent else {}
    r.temperature_sent = sent.get("temperature")
    if r.error is None and not r.text:
        r.error = f"empty response text (finish_reason={r.finish_reason})"
    return r
