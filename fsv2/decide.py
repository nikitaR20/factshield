"""Stages 4–6 for one claim: panel verdict, commit rule, optional explanation.

The commit rule (design 5, fixed 2026-10-25 before any run):

  1. Round 1: each panel model once, under its study settings.
  2. All three give the same valid verdict -> that is the panel verdict.
     Otherwise each model gives `extra_samples_per_model` more answers at
     `extra_sample_temperature`; each model's label is the majority of its
     samples; all three majorities equal -> panel verdict. Still split ->
     not_enough_evidence, reason "models_disagreed".
  3. A supported/refuted panel verdict stands only if at least one key
     statement, from a sample that gave that verdict, cites a passage that
     sample marked as about the same subject, AND the checker confirms the
     cited text supports the statement. Otherwise not_enough_evidence, reason
     "citation_check_failed". For refuted, the statement is the counter-fact,
     so a yes/no support checker is enough; that it contradicts the claim is
     the panel's unanimous judgement, not the checker's.
  4. conflicting_evidence needs no check (it commits to nothing). In thesis
     scoring it counts as not_enough_evidence.

Ablations computed from the SAME calls (declared before any run, no extra
model calls): fs2_no_check (step 3 skipped), fs2_majority_r1 (2-of-3 in round
1, then step 3), fs2_single:<model> (one model's round-1 verdict, no check).
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter

import httpx

from study.prompts import load as study_prompt

from . import evidence as ev
from . import llm
from .checker import supported as check_ok
from .settings import cfg, prompt

COMMITTED = ("supported", "refuted")
TO_STUDY = {"supported": "supported", "refuted": "refuted",
            "not_enough_evidence": "not_enough_evidence",
            "conflicting_evidence": "not_enough_evidence", "unparseable": "unparseable"}


def build_prompt(claim: str, items: list[ev.Prepared]) -> tuple[str, str, dict]:
    system, sys_sha = study_prompt("system")        # same system message as the comparison
    tmpl, v_sha = prompt("verdict")
    user = tmpl.replace("{claim}", claim).replace("{passages}", ev.render(items))
    return system, user, {"system_sha": sys_sha, "verdict_sha": v_sha}


def _sample_row(key: str, r, parsed: dict, temperature) -> dict:
    return {"model_key": key, "model_version": r.model_version, "temperature": temperature,
            "error": r.error, "verdict": parsed["verdict"], "passages": parsed["passages"],
            "statements": parsed["statements"], "reason": parsed["reason"],
            "tokens": r.tokens, "tool_calls": r.tool_calls, "text": r.text}


async def _ask(key, temperature, system, user, valid, stub, client):
    c = llm.panel_cfg(key, temperature=temperature, stub=stub)
    r = await llm.call(c, system, user, client)
    parsed = llm.parse_verdict(r.text, valid) if r.error is None else \
        {"verdict": "unparseable", "passages": {}, "statements": [], "reason": None}
    return _sample_row(key, r, parsed, c.get("temperature"))


def _unanimous(labels: list[str | None]) -> str | None:
    if labels and all(l is not None and l != "unparseable" for l in labels) and len(set(labels)) == 1:
        return labels[0]
    return None


def _majority(labels: list[str]) -> str | None:
    if not labels:
        return None
    lab, n = Counter(labels).most_common(1)[0]
    return lab if n * 2 > len(labels) and lab != "unparseable" else None


def _candidates(samples: list[dict], verdict: str) -> list[tuple[str, list[str], str]]:
    """(statement, eligible pids, model_key) from samples that gave `verdict`.
    Eligible = cited AND marked same_subject 'yes' by that same sample."""
    out = []
    for s in samples:
        if s["verdict"] != verdict:
            continue
        for st in s["statements"]:
            pids = [p for p in st["passages"]
                    if s["passages"].get(p, {}).get("same_subject") == "yes"]
            if pids:
                out.append((st["text"], pids, s["model_key"]))
    return out


async def _check(candidates, items, checker, cache: dict) -> list[dict]:
    todo = []
    for text, pids, _ in candidates:
        doc = ev.text_for_check(items, pids)
        if (doc, text) not in cache:
            todo.append((doc, text))
    todo = list(dict.fromkeys(todo))
    if todo:
        probs = await asyncio.to_thread(checker.score, [d for d, _ in todo], [t for _, t in todo])
        cache.update({k: p for k, p in zip(todo, probs)})
    res = []
    for text, pids, key in candidates:
        p = cache[(ev.text_for_check(items, pids), text)]
        res.append({"statement": text, "passages": pids, "model_key": key,
                    "prob": round(float(p), 4), "confirmed": check_ok(p)})
    return res


async def _apply_check(label, samples, items, checker, cache):
    if label not in COMMITTED:
        return label, None, []
    checks = await _check(_candidates(samples, label), items, checker, cache)
    if any(c["confirmed"] for c in checks):
        return label, None, checks
    return "not_enough_evidence", "citation_check_failed", checks


async def check_claim(claim: str, pack: dict, *, checker, stub: bool,
                      client: httpx.AsyncClient, explain: bool = False) -> dict:
    conf = cfg()
    items = ev.prepare(pack)
    valid = {it.pid for it in items}
    system, user, shas = build_prompt(claim, items)
    panel = conf["panel"]

    r1 = list(await asyncio.gather(*[_ask(k, "study", system, user, valid, stub, client)
                                     for k in panel]))
    calls_failed = [s for s in r1 if s["error"]]
    if calls_failed:
        return {"error": "; ".join(f"{s['model_key']}: {s['error']}" for s in calls_failed),
                "samples": r1}

    samples = list(r1)
    panel_label = _unanimous([s["verdict"] for s in r1])
    path = "unanimous_round1" if panel_label else None
    if panel_label is None:
        n, t = conf["extra_samples_per_model"], conf["extra_sample_temperature"]
        extra = list(await asyncio.gather(*[_ask(k, t, system, user, valid, stub, client)
                                            for k in panel for _ in range(n)]))
        failed = [s for s in extra if s["error"]]
        if failed:
            return {"error": "; ".join(f"{s['model_key']}: {s['error']}" for s in failed),
                    "samples": samples + extra}
        samples += extra
        per_model = {k: _majority([s["verdict"] for s in samples if s["model_key"] == k])
                     for k in panel}
        panel_label = _unanimous(list(per_model.values()))
        path = "unanimous_after_extra" if panel_label else "models_disagreed"

    cache: dict = {}
    if panel_label is None:
        final, reason, checks = "not_enough_evidence", "models_disagreed", []
    else:
        final, reason, checks = await _apply_check(panel_label, samples, items, checker, cache)

    # ---- ablations from the same calls
    no_check = panel_label or "not_enough_evidence"
    maj = _majority([s["verdict"] for s in r1])
    if maj is None:
        majority_r1, maj_reason = "not_enough_evidence", "no_majority"
    else:
        majority_r1, maj_reason, _ = await _apply_check(maj, r1, items, checker, cache)

    confirmed = [c for c in checks if c["confirmed"]]
    cited = list(dict.fromkeys(p for c in confirmed for p in c["passages"]))
    out = {
        "error": None, "claim": claim, "final": final, "reason": reason, "path": path,
        "panel_label": panel_label, "checks": checks, "cited_pids": cited,
        "ablations": {"fs2_no_check": no_check, "fs2_majority_r1": majority_r1,
                      "fs2_majority_r1_reason": maj_reason,
                      **{f"fs2_single:{s['model_key']}": s["verdict"] for s in r1}},
        "n_calls": len(samples), "samples": samples, "prompt_sha": shas,
        "evidence": [{"pid": it.pid, "domain": it.domain, "source_type": it.source_type,
                      "rating": it.rating, "same_text_as": it.same_text_as,
                      "same_org_as": it.same_org_as, "credits_wire": it.credits_wire}
                     for it in items],
    }
    if explain:
        out["explanation"] = await explain_verdict(claim, final, reason, items, samples,
                                                   cited, checker, stub, client, cache)
    return out


# ---------------------------------------------------------------- stage 6

# A citation is a bracket holding only passage ids: [P1], [P1, P2], and the
# full-width forms some models emit (【P1】, ［P1］, （P1）). Seen in the
# 2026-10-26 smoke test: gpt-oss wrote 【P1】【P2】 and every cited sentence was
# dropped as uncited. Ordinary parentheses count only when they hold ids alone.
_CITE = re.compile(r"[\[【［(（]\s*(P\d+(?:\s*[,;]\s*P\d+)*)\s*[\]】］)）]")


def _cited_ids(sentence: str) -> list[str]:
    return [p for grp in _CITE.findall(sentence) for p in re.findall(r"P\d+", grp)]


async def explain_verdict(claim, verdict, reason, items, samples, cited, checker, stub,
                          client, cache) -> dict:
    conf = cfg()
    if cited:
        pids = cited
    else:  # non-committed: passages any sample judged on-subject, else all
        pids = sorted({p for s in samples for p, m in s["passages"].items()
                       if m.get("same_subject") == "yes"}, key=lambda x: int(x[1:])) \
            or [it.pid for it in items]
    shown = [it for it in items if it.pid in pids]
    tmpl, sha = prompt("explain")
    label = verdict if not reason else f"{verdict} ({reason.replace('_', ' ')})"
    user = (tmpl.replace("{max_sentences}", str(conf["explanation"]["max_sentences"]))
                .replace("{claim}", claim).replace("{verdict}", label)
                .replace("{passages}", ev.render(shown)))
    system, _ = study_prompt("system")
    c = llm.panel_cfg(conf["explainer"], stub=stub)
    r = await llm.call(c, system, user, client)
    obj = llm.extract_json(r.text) if r.error is None else None
    sentences = [str(s).strip() for s in (obj or {}).get("sentences", []) if str(s).strip()]
    sentences = sentences[: conf["explanation"]["max_sentences"]]

    valid = {it.pid for it in shown}
    slots: list[dict | None] = [None] * len(sentences)   # keeps the writer's order
    dropped, todo = [], []
    for i, s in enumerate(sentences):
        if s.lower().startswith("summary:"):
            slots[i] = {"text": s, "kind": "summary", "passages": []}
            continue
        ids = list(dict.fromkeys(p for p in _cited_ids(s) if p in valid))
        if not ids:
            dropped.append({"text": s, "why": "no valid citation"})
            continue
        todo.append((i, s, ids))
    if todo:
        docs = [ev.text_for_check(items, ids) for _, _, ids in todo]
        plain = [_CITE.sub("", s).strip() for _, s, _ in todo]
        probs = await asyncio.to_thread(checker.score, docs, plain)
        for (i, s, ids), p in zip(todo, probs):
            if check_ok(p):
                slots[i] = {"text": s, "kind": "fact", "passages": ids, "prob": round(float(p), 4)}
            else:
                dropped.append({"text": s, "why": f"checker {p:.2f}", "passages": ids})
    kept = [x for x in slots if x is not None]
    factual = [k for k in kept if k["kind"] == "fact"]
    # Every non-summary sentence counts, including ones dropped for having no
    # usable citation; otherwise an explanation whose facts were ALL dropped
    # could pass as complete with only its "Summary:" lines left.
    n_fact = len(factual) + len(dropped)
    quotes_only = n_fact > 0 and len(factual) / n_fact < conf["explanation"]["min_surviving_share"]
    return {"error": r.error, "model_version": r.model_version, "prompt_sha": sha,
            "sentences": [] if quotes_only else kept, "dropped": dropped,
            "quotes_only": quotes_only or (r.error is not None) or not sentences,
            "passages_shown": pids}
