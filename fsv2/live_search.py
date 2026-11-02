"""Live search on the Set B claims: each model checks the claim with its provider's own
web search, no evidence pack. Plan: fsv2/LIVE_SEARCH_PLAN.md. Settings: fsv2/live_search.yaml.

    python -m fsv2.live_search run --smoke                                     # plumbing test
    python -m fsv2.live_search run --models openai,claude,gemini               # full run
    python -m fsv2.live_search analyse --labels fsv2/live_world_labels.csv \
        --results study/results/factshield_v2p2_dev.jsonl study/results/factshield_v2p2_test.jsonl \
                  study/results/factshield_v2p2_new.jsonl

The run refuses to start until every claim has a WORLD_LABEL in the sheet, so the labels are
fixed before any answer is seen. Verdicts are never printed. Re-running resumes: claims with
a successful row for a model are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import httpx  # noqa: E402
import yaml  # noqa: E402

from app.config import env  # noqa: E402
from study.common import now_iso  # noqa: E402
from study.models import CallResult, _Fatal, _post_with_retry, find_tool_use  # noqa: E402

from . import fourth_model as fm  # noqa: E402
from .llm import extract_json  # noqa: E402

HERE = Path(__file__).resolve().parent
NEE = "not_enough_evidence"
DEFAULT_OUT = "study/results/live_search_setB.jsonl"
SMOKE_CLAIM = "The Eiffel Tower is located in Paris, France."
LABELS = ("supported", "refuted", NEE)


def settings(path: Path | None = None) -> dict:
    return yaml.safe_load((path or HERE / "live_search.yaml").read_text(encoding="utf-8"))


def prompt_text(name: str) -> tuple[str, str]:
    t = (HERE / "prompts" / f"{name}.txt").read_text(encoding="utf-8").rstrip("\n")
    return t, hashlib.sha256(t.encode("utf-8")).hexdigest()


def build_user(template: str, claim: str, today: str) -> str:
    return template.replace("{date}", today).replace("{claim}", claim)


def host(url: str | None) -> str:
    try:
        h = (urlparse(url or "").hostname or "").lower()
    except ValueError:
        return ""
    return h[4:] if h.startswith("www.") else h


# ------------------------------------------------------------------ providers
# Each returns (CallResult, search_info) where search_info = {"queries": [...],
# "sources": [{"url", "title"}], "n_searches": int, "search_errors": [...]}.

async def _openai(cfg: dict, system: str, user: str, client: httpx.AsyncClient, r: CallResult):
    key = env("OPENAI_API_KEY")
    if not key:
        raise _Fatal("OPENAI_API_KEY not set")
    tool = {"type": "web_search"}
    if cfg.get("search_context_size"):
        tool["search_context_size"] = cfg["search_context_size"]
    body = {"model": cfg["model"],
            "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "tools": [tool], "tool_choice": "auto",
            "include": ["web_search_call.action.sources"],
            "max_output_tokens": cfg["max_output_tokens"]}
    if cfg.get("max_tool_calls") is not None:
        body["max_tool_calls"] = cfg["max_tool_calls"]
    if cfg.get("temperature") is not None:
        body["temperature"] = cfg["temperature"]
    if cfg.get("reasoning_effort") is not None:
        body["reasoning"] = {"effort": cfg["reasoning_effort"]}
    r.tools_attached = body["tools"]
    r.params_sent = {k: v for k, v in body.items() if k != "input"}
    data = await _post_with_retry(client, "https://api.openai.com/v1/responses",
                                  headers={"Authorization": f"Bearer {key}"}, json=body,
                                  timeout=cfg["timeout_s"], result=r)
    r.raw = data
    r.model_version = data.get("model")
    info = {"queries": [], "sources": [], "n_searches": 0, "search_errors": []}
    texts = []
    for item in data.get("output") or []:
        t = item.get("type")
        if t == "web_search_call":
            act = item.get("action") or {}
            if act.get("type", "search") == "search":
                info["n_searches"] += 1
                qs = act.get("queries") or ([act["query"]] if act.get("query") else [])
                info["queries"] += qs
            for s in act.get("sources") or []:
                if s.get("url"):
                    info["sources"].append({"url": s["url"], "title": s.get("title")})
            if item.get("status") == "failed":
                info["search_errors"].append("failed")
        elif t == "message":
            for c in item.get("content") or []:
                if c.get("type") in ("output_text", "text"):
                    texts.append(c.get("text", ""))
                    for a in c.get("annotations") or []:
                        if a.get("type") == "url_citation" and a.get("url"):
                            info["sources"].append({"url": a["url"], "title": a.get("title")})
    r.text = "\n".join(texts) if texts else None
    r.finish_reason = data.get("status")
    if data.get("incomplete_details"):
        r.finish_reason = f"{data.get('status')}:{data['incomplete_details'].get('reason')}"
    u = data.get("usage") or {}
    r.tokens = {"input": u.get("input_tokens"), "output": u.get("output_tokens"),
                "reasoning": (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
                "total": u.get("total_tokens")}
    return info


async def _anthropic(cfg: dict, system: str, user: str, client: httpx.AsyncClient, r: CallResult):
    key = env("ANTHROPIC_API_KEY")
    if not key:
        raise _Fatal("ANTHROPIC_API_KEY not set")
    tool = {"type": cfg.get("tool_version") or "web_search_20250305", "name": "web_search"}
    if cfg.get("max_uses") is not None:
        tool["max_uses"] = cfg["max_uses"]
    messages = [{"role": "user", "content": user}]
    body = {"model": cfg["model"], "max_tokens": cfg["max_output_tokens"], "system": system,
            "messages": messages, "tools": [tool]}
    if cfg.get("thinking") is not None:
        body["thinking"] = {"type": cfg["thinking"]}
    if cfg.get("effort") is not None:
        body["output_config"] = {"effort": cfg["effort"]}
    r.tools_attached = [tool]
    r.params_sent = {k: v for k, v in body.items() if k not in ("messages", "system")}
    headers = {"x-api-key": key, "anthropic-version": cfg.get("api_version") or "2023-06-01",
               "content-type": "application/json"}
    info = {"queries": [], "sources": [], "n_searches": 0, "search_errors": []}
    blocks, raws = [], []
    tin = tout = 0
    for _ in range(4):  # first call + up to 3 continuations after pause_turn
        data = await _post_with_retry(client, "https://api.anthropic.com/v1/messages",
                                      headers=headers, json=body, timeout=cfg["timeout_s"], result=r)
        raws.append(data)
        content = data.get("content") or []
        blocks += content
        u = data.get("usage") or {}
        tin += u.get("input_tokens") or 0
        tout += u.get("output_tokens") or 0
        info["n_searches"] += ((u.get("server_tool_use") or {}).get("web_search_requests") or 0)
        r.finish_reason = data.get("stop_reason")
        r.model_version = data.get("model")
        if data.get("stop_reason") != "pause_turn":
            break
        messages = messages + [{"role": "assistant", "content": content}]
        body = {**body, "messages": messages}
    r.raw = raws[0] if len(raws) == 1 else {"continued_responses": raws}
    texts = []
    for b in blocks:
        t = b.get("type")
        if t == "server_tool_use" and b.get("name") == "web_search":
            q = (b.get("input") or {}).get("query")
            if q:
                info["queries"].append(q)
        elif t == "web_search_tool_result":
            c = b.get("content")
            if isinstance(c, dict):
                info["search_errors"].append(c.get("error_code") or c.get("type"))
            for s in (c if isinstance(c, list) else []):
                if s.get("url"):
                    info["sources"].append({"url": s["url"], "title": s.get("title")})
        elif t == "text":
            texts.append(b.get("text", ""))
            for cit in b.get("citations") or []:
                if cit.get("url"):
                    info["sources"].append({"url": cit["url"], "title": cit.get("title")})
    r.text = "".join(texts) if texts else None
    r.tokens = {"input": tin, "output": tout, "reasoning": None, "total": tin + tout}
    return info


async def _gemini(cfg: dict, system: str, user: str, client: httpx.AsyncClient, r: CallResult):
    key = env("GEMINI_API_KEY")
    if not key:
        raise _Fatal("GEMINI_API_KEY not set")
    gen = {"maxOutputTokens": cfg["max_output_tokens"]}
    if cfg.get("temperature") is not None:
        gen["temperature"] = cfg["temperature"]
    if cfg.get("thinking_level") is not None:
        gen["thinkingConfig"] = {"thinkingLevel": cfg["thinking_level"]}
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen, "tools": [{"google_search": {}}]}
    r.tools_attached = body["tools"]
    r.params_sent = {"model": cfg["model"], "generationConfig": gen, "tools": body["tools"]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg['model']}:generateContent"
    data = await _post_with_retry(client, url, headers={"x-goog-api-key": key}, json=body,
                                  timeout=cfg["timeout_s"], result=r)
    r.raw = data
    r.model_version = data.get("modelVersion")
    cand = (data.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    ans = [p.get("text", "") for p in parts if "text" in p and not p.get("thought")]
    r.text = "".join(ans) if ans else None
    r.finish_reason = cand.get("finishReason")
    gm = cand.get("groundingMetadata") or {}
    info = {"queries": list(gm.get("webSearchQueries") or []), "sources": [],
            "n_searches": len(gm.get("webSearchQueries") or []), "search_errors": []}
    for ch in gm.get("groundingChunks") or []:
        w = ch.get("web") or {}
        if w.get("uri"):
            # uri is a Google redirect link; title is normally the site's domain
            info["sources"].append({"url": w["uri"], "title": w.get("title")})
    u = data.get("usageMetadata") or {}
    r.tokens = {"input": u.get("promptTokenCount"), "output": u.get("candidatesTokenCount"),
                "reasoning": u.get("thoughtsTokenCount"), "total": u.get("totalTokenCount")}
    return info


_PROVIDERS = {"openai": _openai, "anthropic": _anthropic, "gemini": _gemini}


async def ask(model_key: str, cfg: dict, system: str, user: str,
              client: httpx.AsyncClient) -> tuple[CallResult, dict]:
    r = CallResult(provider=cfg["provider"], model_requested=cfg["model"], timestamp=now_iso())
    info = {"queries": [], "sources": [], "n_searches": 0, "search_errors": []}
    try:
        info = await _PROVIDERS[cfg["provider"]](cfg, system, user, client, r)
    except _Fatal as exc:
        r.error = str(exc)
    except Exception as exc:  # record, don't die
        r.error = f"{type(exc).__name__}: {exc}"
    if r.raw is not None:
        r.tool_calls = find_tool_use(r.raw)
    if r.error is None and not r.text:
        r.error = f"empty response text (finish_reason={r.finish_reason})"
    return r, info


def parse_answer(text: str | None) -> dict:
    obj = extract_json(text)
    if obj is None:
        return {"verdict": "unparseable", "sources_listed": [], "reason": None}
    v = str(obj.get("verdict", "")).strip().lower().replace(" ", "_")
    if v == "conflicting_evidence":
        v = NEE
    if v not in LABELS:
        v = "unparseable"
    src = [str(s) for s in (obj.get("sources") or []) if isinstance(s, str)][:10]
    return {"verdict": v, "sources_listed": src, "reason": obj.get("reason")}


# ------------------------------------------------------------------ fact-check detection

def factcheck_rules() -> tuple[set[str], list[str]]:
    from study.common import config
    pk = config()["packs"]
    return set(pk.get("exclude_domains") or []), list(pk.get("exclude_url_patterns") or [])


def is_factcheck(url: str | None, title: str | None, rules) -> bool:
    domains, patterns = rules
    h = host(url)
    t = (title or "").lower().strip()
    t = t[4:] if t.startswith("www.") else t
    for d in domains:
        if h == d or h.endswith("." + d) or t == d or t.endswith("." + d):
            return True
    low = (url or "").lower()
    # Gemini's source URLs are redirects; their path says nothing about the page.
    if "grounding-api-redirect" in low:
        return False
    return any(p in low for p in patterns)


# ------------------------------------------------------------------ labels

def load_labels(path) -> dict:
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            lab_key = next(k for k in row if k.startswith("WORLD_LABEL"))
            org_key = next(k for k in row if k.startswith("ORIGIN"))
            lab = (row[lab_key] or "").strip().lower().replace(" ", "_")
            if lab in ("nee", "not_enough", "unproven"):
                lab = NEE
            out[row["claim_id"]] = {"label": lab, "origin": (row[org_key] or "").strip().lower(),
                                    "claim": row["claim"]}
    return out


def check_labels(labels: dict) -> list[str]:
    return [cid for cid, v in labels.items() if v["label"] not in LABELS]


# ------------------------------------------------------------------ run

def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {(r["claim_id"], r["model_key"]) for r in fm.read_jsonl(path) if not r.get("error")}


async def run_async(a) -> None:
    conf = settings(Path(a.settings) if a.settings else None)
    if a.smoke:
        # Plumbing and settings check on a claim that is NOT in the study, so no study
        # claim is answered before the world labels are fixed.
        labels = {"smoke-0001": {"label": "supported", "origin": "smoke", "claim": SMOKE_CLAIM}}
        lsha = None
        a.out = a.out if a.out != DEFAULT_OUT else "study/results/live_search_smoke.jsonl"
    else:
        labels = load_labels(a.labels)
        bad = check_labels(labels)
        if bad:
            raise SystemExit(f"{len(bad)} claims have no valid WORLD_LABEL yet (e.g. {bad[:3]}). "
                             "Fill fsv2/live_world_labels.csv first: labels must be fixed before any "
                             "live answer exists. For a plumbing test use --smoke.")
        lsha = hashlib.sha256(Path(a.labels).read_bytes()).hexdigest()
    template, psha = prompt_text(conf["prompt"])
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    for m in models:
        if m not in conf["models"]:
            raise SystemExit(f"unknown model key {m!r} in live_search.yaml")
    ids = sorted(labels)
    if a.ids:
        want = set(a.ids.split(","))
        ids = [c for c in ids if c in want]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    today = a.date or date.today().isoformat()
    todo = [(c, m) for c in ids for m in models if (c, m) not in done]
    if a.limit:
        kept, per = [], Counter()
        for c, m in todo:
            if per[m] < a.limit:
                kept.append((c, m))
                per[m] += 1
        todo = kept
    print(f"{len(ids)} claims x {len(models)} models; {len(done)} already done; running {len(todo)}")
    tot = Counter()
    async with httpx.AsyncClient() as client:
        with out.open("a", encoding="utf-8") as fh:
            for cid, m in todo:
                cfg = conf["models"][m]
                user = build_user(template, labels[cid]["claim"], today)
                t0 = time.perf_counter()
                r, info = await ask(m, cfg, conf["system"], user, client)
                parsed = parse_answer(r.text) if r.error is None else \
                    {"verdict": None, "sources_listed": [], "reason": None}
                row = {"kind": "live_search_v1", "claim_id": cid, "model_key": m,
                       "provider": r.provider, "model_requested": r.model_requested,
                       "model_version": r.model_version, "params_sent": r.params_sent,
                       "tools_attached": r.tools_attached, "prompt": conf["prompt"],
                       "prompt_sha": psha, "labels_sha": lsha, "run_date": today,
                       "error": r.error, **parsed, "queries": info["queries"],
                       "tool_sources": info["sources"], "n_searches": info["n_searches"],
                       "search_errors": info["search_errors"],
                       "searched": info["n_searches"] > 0 or bool(info["queries"]),
                       "finish_reason": r.finish_reason, "tokens": r.tokens, "text": r.text,
                       "timestamp": r.timestamp, "attempts": r.attempts,
                       "latency_total_ms": r.latency_total_ms,
                       "seconds": round(time.perf_counter() - t0, 2), "raw": r.raw}
                fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
                fh.flush()
                if r.error:
                    tot["errors"] += 1
                    shown = "ERROR " + r.error[:160]
                else:
                    tot["ok"] += 1
                    tot[f"in_{m}"] += (r.tokens or {}).get("input") or 0
                    tot[f"out_{m}"] += (r.tokens or {}).get("output") or 0
                    tot[f"searches_{m}"] += info["n_searches"]
                    shown = (f"ok  searches {info['n_searches']}  sources {len(info['sources'])}"
                             + (f"  verdict {parsed['verdict']}" if a.smoke else "")
                             + ("" if parsed["verdict"] != "unparseable" else "  (reply could not be parsed)")
                             + ("" if (r.tokens or {}).get("input", 0) < 100_000 else "  (input over 100K tokens)"))
                print(f"{cid:<26}{m:<8}{row['seconds']:>7.1f}s  {shown}")
    prices = conf.get("prices") or {}
    print(f"\nwritten to {out}; ok {tot['ok']}, errors {tot['errors']}")
    for m in models:
        p = prices.get(m) or {}
        cost = (tot[f"in_{m}"] / 1e6 * p.get("input_per_m", 0) + tot[f"out_{m}"] / 1e6 * p.get("output_per_m", 0)
                + tot[f"searches_{m}"] * p.get("per_search", 0))
        print(f"  {m:<7} tokens in {tot[f'in_{m}']:,} / out {tot[f'out_{m}']:,}; searches {tot[f'searches_{m}']}; "
              f"estimated cost ${cost:.3f} (list prices in live_search.yaml)")
    if tot["errors"]:
        print("Run the same command again to retry the failed calls.")


# ------------------------------------------------------------------ analyse

def analyse(a) -> dict:
    labels = load_labels(a.labels)
    bad = check_labels(labels)
    if bad:
        raise SystemExit(f"claims without a valid WORLD_LABEL: {bad}")
    gold = {c: v["label"] for c, v in labels.items()}
    origin = {c: v["origin"] for c, v in labels.items()}
    latest = {}
    for r in fm.read_jsonl(a.live):
        if r.get("kind") == "live_search_v1" and not r.get("error"):
            latest[(r["claim_id"], r["model_key"])] = r
    models = sorted({m for _, m in latest})
    ids = sorted(gold)
    rules = factcheck_rules()
    rep = {"n_claims": len(ids), "models": {}, "agreement": {}, "frozen_pack_vs_world": {}}

    def p(*x):
        print(*x)

    p(f"{len(ids)} claims; world labels: " + ", ".join(f"{l} {sum(gold[c] == l for c in ids)}" for l in LABELS)
      + f"; origin: {dict(Counter(origin[c] or 'unknown' for c in ids))}")
    p(f"{'model':<8}{'n':>4}{'correct':>8}{'answered':>9}{'CW':>4}{'unparse':>8}  acc|ans  95% CI       "
      f"searched  searches/claim  fact-check source")
    for m in models:
        have = [c for c in ids if (c, m) in latest]
        pred = {c: latest[(c, m)]["verdict"] for c in have}
        s = fm.score(pred, gold, have)
        rows = [latest[(c, m)] for c in have]
        searched = sum(bool(r.get("searched")) for r in rows)
        nsearch = sum(r.get("n_searches") or 0 for r in rows)
        fc = {c: any(is_factcheck(x.get("url"), x.get("title"), rules) for x in latest[(c, m)].get("tool_sources") or [])
              or any(is_factcheck(u, None, rules) for u in latest[(c, m)].get("sources_listed") or [])
              for c in have}
        with_fc = [c for c in have if fc[c]]
        without = [c for c in have if not fc[c]]
        s_fc = fm.score(pred, gold, with_fc) if with_fc else None
        s_no = fm.score(pred, gold, without) if without else None
        by_origin = {o: fm.score(pred, gold, [c for c in have if origin[c] == o])
                     for o in sorted({origin[c] for c in have}) if o}
        ci = s["accuracy_on_answered_ci95"]
        acc = f"{s['accuracy_on_answered']:.0%}" if s["accuracy_on_answered"] is not None else "-"
        p(f"{m:<8}{len(have):>4}{s['correct']:>8}{s['answered']:>9}{s['confidently_wrong']:>4}"
          f"{s['unparseable']:>8}  {acc:<7}  {(f'{ci[0]:.2f}-{ci[1]:.2f}' if ci else '-'):<11}"
          f"{searched:>6}/{len(have):<3}{nsearch / max(1, len(have)):>10.1f}{len(with_fc):>12}")
        p(f"         recall: " + ", ".join(f"{l} {s['per_class'][l]['recall']:.2f}" if s['per_class'][l]['recall'] is not None
                                          else f"{l} -" for l in LABELS))
        if s_fc and s_no:
            p(f"         with a fact-check source: correct {s_fc['correct']}/{s_fc['n']}; "
              f"without: {s_no['correct']}/{s_no['n']}")
        for o, so in by_origin.items():
            p(f"         origin {o}: correct {so['correct']}/{so['n']}, CW {so['confidently_wrong']}")
        rep["models"][m] = {"score": s, "searched": searched, "searches": nsearch,
                            "with_factcheck_source": len(with_fc),
                            "score_with_factcheck": s_fc, "score_without_factcheck": s_no,
                            "by_origin": by_origin}
    full = [c for c in ids if all((c, m) in latest for m in models)]
    if len(models) > 1 and full:
        allagree = [c for c in full if len({latest[(c, m)]["verdict"] for m in models}) == 1]
        allwrong_same = [c for c in allagree if latest[(c, models[0])]["verdict"] != gold[c]]
        p(f"\nall {len(models)} gave the same answer on {len(allagree)} of {len(full)}; "
          f"all the same WRONG answer on {len(allwrong_same)}: {allwrong_same}")
        rep["agreement"] = {"claims": len(full), "all_same": len(allagree), "all_same_wrong": allwrong_same}
    if a.results:
        saved = fm.load_saved(a.results)
        claude_v2, _ = fm.load_fourth(a.claude_v2, "claude") if a.claude_v2 and Path(a.claude_v2).exists() else ({}, {})
        p("\nfrozen-pack v2 answers scored against the WORLD labels (descriptive; different evidence and date):")
        for m in models:
            pred = ({c: saved["single"].get((c, m)) for c in ids} if m != "claude" else claude_v2)
            have = [c for c in ids if pred.get(c)]
            if not have:
                continue
            s = fm.score({c: fm.norm(pred[c]) for c in have}, gold, have)
            p(f"  {m:<8} n {len(have):>3}  correct {s['correct']:>3}  answered {s['answered']:>3}  CW {s['confidently_wrong']:>2}")
            rep["frozen_pack_vs_world"][m] = s
        have = [c for c in ids if c in saved["final"]]
        s = fm.score({c: saved["final"][c] for c in have}, gold, have)
        p(f"  {'FactShield':<8} n {len(have):>3}  correct {s['correct']:>3}  answered {s['answered']:>3}  CW {s['confidently_wrong']:>2}")
        rep["frozen_pack_vs_world"]["factshield"] = s
    if a.summary:
        Path(a.summary).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        p(f"\nsummary JSON: {a.summary}")
    return rep


class _Tee:
    """Write everything printed to the console AND to a UTF-8 file (PowerShell's
    Tee-Object writes UTF-16, which GitHub shows as binary)."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="live calls (costs money)")
    r.add_argument("--models", default="openai,claude,gemini")
    r.add_argument("--labels", default=str(HERE / "live_world_labels.csv"))
    r.add_argument("--settings")
    r.add_argument("--out", default=DEFAULT_OUT)
    r.add_argument("--ids")
    r.add_argument("--limit", type=int, help="at most this many claims per model (smoke test)")
    r.add_argument("--date", help="date written into the prompt (default: today)")
    r.add_argument("--smoke", action="store_true",
                   help="plumbing/settings check on one dummy claim outside the study; "
                        "written to study/results/live_search_smoke.jsonl")
    s = sub.add_parser("analyse", help="score the live answers (no calls)")
    s.add_argument("--labels", default=str(HERE / "live_world_labels.csv"))
    s.add_argument("--live", default="study/results/live_search_setB.jsonl")
    s.add_argument("--results", nargs="*", help="saved FactShield v2 files (optional, for the frozen-pack comparison)")
    s.add_argument("--claude-v2", default="study/results/fsv2_r1_claude.jsonl")
    s.add_argument("--summary", default="study/results/live_search_summary.json")
    s.add_argument("--report", default="live_report.txt", help="UTF-8 copy of the printed report")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        asyncio.run(run_async(a))
        return
    old, tee = sys.stdout, _Tee(a.report)
    sys.stdout = tee
    try:
        analyse(a)
    finally:
        sys.stdout = old
        tee.f.close()
    print(f"report saved: {a.report}")


if __name__ == "__main__":
    main()
