"""Calls to the panel models, and reading their JSON replies.

The provider code is the study's own (study/models.py), so the panel sends the
same requests the comparison models receive: same model names, token limits,
timeouts, reasoning settings and retry rules. The one thing v2 may change is
the temperature of the EXTRA samples drawn when the panel disagrees.

`study.models.complete()` takes its settings from study/config.yaml only, so
this module repeats its few lines with a settings override. It uses the
study's private provider table on purpose rather than copying the providers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re

import httpx

from study.common import config as study_config, now_iso
from study.models import CallResult, _Fatal, _PROVIDERS, find_tool_use, is_search_model

VERDICTS = ("supported", "refuted", "conflicting_evidence", "not_enough_evidence")


def panel_cfg(model_key: str, *, temperature: float | None | str = "study",
              stub: bool = False) -> dict:
    base = copy.deepcopy(study_config()["models"][model_key])
    if stub:
        base.update(provider="fsv2_stub", model=f"stub-{model_key}")
    # A model the study runs WITHOUT a temperature field (temperature: null —
    # e.g. OpenAI reasoning models that reject anything but the default) keeps
    # sending none. Its default sampling is already random, which is what the
    # extra samples need; forcing a value would make the API refuse the call.
    if temperature != "study" and base.get("temperature") is not None:
        base["temperature"] = temperature
    return base


async def call(cfg: dict, system: str, user: str, client: httpx.AsyncClient) -> CallResult:
    r = CallResult(provider=cfg["provider"], model_requested=cfg["model"], timestamp=now_iso())
    if cfg["model"] == "CHANGE_ME":
        r.error = "model is still CHANGE_ME in study/config.yaml"
        return r
    if is_search_model(cfg["model"]):
        r.error = f"{cfg['model']!r} is a search product; refused"
        return r
    fn = _stub if cfg["provider"] == "fsv2_stub" else _PROVIDERS[cfg["provider"]]
    try:
        await fn(cfg, system, user, False, client, r)
    except _Fatal as exc:
        r.error = str(exc)
    except Exception as exc:  # record, don't die
        r.error = f"{type(exc).__name__}: {exc}"
    if r.raw is not None:
        r.tool_calls = find_tool_use(r.raw)
    sent = r.params_sent.get("generationConfig", r.params_sent) if r.params_sent else {}
    r.temperature_sent = sent.get("temperature")
    if r.error is None and not r.text:
        r.error = f"empty response text (finish_reason={r.finish_reason})"
    return r


# ----------------------------------------------------------------- parsing

def extract_json(text: str | None) -> dict | None:
    """First balanced {...} object in the text (code fences tolerated)."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = s.find("{", start + 1)
    return None


def parse_verdict(text: str | None, valid_pids: set[str]) -> dict:
    """Normalise one panel reply. Anything malformed becomes 'unparseable',
    which can never be part of a unanimous verdict."""
    obj = extract_json(text)
    if obj is None:
        return {"verdict": "unparseable", "passages": {}, "statements": [], "reason": None}
    v = str(obj.get("verdict", "")).strip().lower().replace(" ", "_")
    if v not in VERDICTS:
        v = "unparseable"
    marks = {}
    for p in obj.get("passages") or []:
        if isinstance(p, dict) and str(p.get("id")) in valid_pids:
            marks[str(p["id"])] = {
                "same_subject": str(p.get("same_subject", "")).lower(),
                "stance": str(p.get("stance", "")).lower(),
            }
    statements = []
    for s in obj.get("key_statements") or []:
        if not isinstance(s, dict) or not str(s.get("text", "")).strip():
            continue
        pids = [str(x) for x in (s.get("passages") or [])]
        statements.append({
            "text": str(s["text"]).strip(),
            "passages": [x for x in pids if x in valid_pids],
            "fabricated_ids": [x for x in pids if x not in valid_pids],
        })
    return {"verdict": v, "passages": marks, "statements": statements,
            "reason": obj.get("reason")}


# -------------------------------------------------------------------- stub

async def _stub(cfg: dict, system: str, user: str, live: bool,
                client: httpx.AsyncClient, r: CallResult) -> None:
    """Offline stand-in. Deterministic in (model, temperature, prompt); no
    network. Its verdicts are a hash and MEAN NOTHING — it exists to exercise
    every branch: agreement, disagreement, copies, fabricated ids, bad JSON."""
    key = f"{cfg['model']}|{cfg.get('temperature')}|{system}|{user}"
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    pids = sorted(set(re.findall(r"^\[(P\d+)\]", user, flags=re.M)), key=lambda x: int(x[1:]))
    if '"checking_as"' in user:  # claim extraction prompt
        sel = user.split("SELECTION:\n", 1)[-1].strip()
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", sel) if len(s.strip()) > 15][:8]
        claims = [{"original": s, "checking_as": s, "resolved": i % 4 != 3,
                   "unclear": None if i % 4 != 3 else "stub: unclear reference (test)",
                   "queries": [s[:60], " ".join(s.split()[:5])]} for i, s in enumerate(sents)]
        text = json.dumps({"claims": claims, "skipped": [{"text": "stub opinion", "reason": "opinion"}]})
    elif '"same"' in user:  # fact-check match prompt
        text = json.dumps({"same": [1]})
    elif '"sentences"' in user:  # explanation prompt
        cite = pids[h % len(pids)] if pids else "P1"
        # Quote the start of the cited passage, so the stub checker can confirm it
        # and the explanation screen can be seen offline.
        m = re.search(rf"^\[{cite}\].*?\nURL: [^\n]*\n([^\n]+)", user, flags=re.M | re.S)
        first = re.split(r"(?<=[.!?])\s+", m.group(1))[0].rstrip(".") if m else "The passage discusses it"
        text = json.dumps({"sentences": [f"{first} [{cite}].",
                                         "Summary: no passage gives a date."]})
    elif h % 13 == 0:
        text = "I think the claim is probably true."          # not JSON
    else:
        # Mostly decided by the claim alone, so the three stub models usually
        # agree and every result screen can be seen offline; one claim in five
        # gets model-dependent (disagreeing) answers.
        claim_only = user.split("CLAIM:\n", 1)[-1].split("\n", 1)[0]
        hc = int(hashlib.sha256(claim_only.encode()).hexdigest(), 16)
        verdict = VERDICTS[(h >> 3) % 4] if hc % 5 == 0 else VERDICTS[hc % 4]
        if "supported" in cfg["model"]:  # used by tests to force a verdict
            verdict = "supported"
        cited = [pids[(h >> 11) % len(pids)]] if pids else []
        if (h >> 17) % 7 == 0:
            cited.append("P99")
        claim = user.split("CLAIM:\n", 1)[-1].split("\n", 1)[0]
        text = json.dumps({
            "passages": [{"id": p, "same_subject": "yes" if (h >> (i + 20)) % 2 else "unclear",
                          "stance": "neither"} for i, p in enumerate(pids)],
            "verdict": verdict,
            "key_statements": ([{"text": claim, "passages": cited}]
                               if verdict in ("supported", "refuted") else []),
            "reason": "offline stub output, not a model's judgement",
        })
    r.raw = {"stub": True, "choices": [{"message": {"content": text}}]}
    r.model_version = "fsv2-stub-offline"
    r.text = text
    r.finish_reason = "stop"
    r.params_sent = {"model": cfg["model"], "temperature": cfg.get("temperature")}
    r.tokens = {"input": len(user.split()), "output": len(text.split()), "reasoning": 0,
                "total": len(user.split()) + len(text.split())}
    r.http_status, r.attempts, r.latency_ms, r.latency_total_ms = 200, 1, 0.0, 0.0
