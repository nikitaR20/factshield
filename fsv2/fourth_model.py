"""Fourth panel model (Claude) on the frozen packs — EXPLORATORY, added 2026-10-31.

Chosen after the main FactShield v2 results were seen. It never changes the
frozen design or its results. The comparisons are fixed in
fsv2/FOURTH_MODEL_PLAN.md, written before any Claude call.

Two steps:

1. run — one round-1 answer per claim from the fourth model, on the same
   pack, the same system + verdict prompt (v2) and the same parser as the
   panel. Verdicts are never printed. Safe to re-run: claims with a
   successful row are skipped, so errors (rate limits, 529 overloaded) are
   retried by running the same command again.

       python -m fsv2.fourth_model run --results study/results/factshield_v2p2_dev.jsonl \
           study/results/factshield_v2p2_test.jsonl study/results/<new setB file>.jsonl --limit 3
       (check the 3 rows, then the same command without --limit)

2. analyse — offline, no model calls. Combines the saved round-1 answers of
   openai / gemini / groq (fs2_single rows) with the fourth model's answers.

       python -m fsv2.fourth_model analyse --results <same files> \
           --fourth study/results/fsv2_r1_claude.jsonl

Every panel here is SIMULATED from round-1 answers: no extra samples on
disagreement and no citation check. That is why "3 models, unanimous" is
recomputed rather than taken from the FactShield rows; the FactShield result
as run is printed beside it for reference. (In the main run the citation check
changed 0 of 100 verdicts; extra samples settled 3 claims.)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

HERE = Path(__file__).resolve().parent
NEE = "not_enough_evidence"
COMMITTED = ("supported", "refuted")
LABELS = ("supported", "refuted", NEE)
BASE3 = ("openai", "gemini", "groq")


def norm(v):
    """Study scoring: conflicting_evidence counts as not_enough_evidence."""
    if v is None:
        return None
    return NEE if v == "conflicting_evidence" else v


# ------------------------------------------------------------------ loading

def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


FROZEN_VERSION = "fsv2-0.2"


def load_saved(paths, version: str = FROZEN_VERSION) -> dict:
    """Saved FactShield rows -> round-1 answers, final verdicts, pack hashes,
    round-1 reasons. Later rows win (as study.score keeps the last successful row).

    Only rows of the frozen version are read (model_version starts with
    "fsv2-0.2/"), so a file that also holds dev run 1 (fsv2-0.1, prompt v1) or
    stub rows can be listed safely: those rows are skipped and counted."""
    single, final, packh, reasons, psha = {}, {}, {}, {}, {}
    skipped = Counter()
    for p in paths:
        for r in read_jsonl(p):
            cid = r.get("claim_id")
            if r.get("error"):
                continue
            mv = str(r.get("model_version") or "")
            if r.get("condition") in ("fs2", "fs2_single") and (
                    not mv.startswith(version + "/") or "STUB" in mv):
                skipped[mv.split("/")[0] or "none"] += 1
                continue
            if r.get("condition") == "fs2_single":
                single[(cid, r["model"].split(":", 1)[1])] = norm(r.get("parsed_verdict"))
            elif r.get("condition") == "fs2":
                fv = r.get("factshield_v2") or {}
                final[cid] = norm(fv.get("final"))
                packh[cid] = r.get("pack_hash")
                psha[cid] = fv.get("prompt_sha")
                for s in (fv.get("samples") or [])[:3]:  # round 1 only
                    reasons[(cid, s.get("model_key"))] = s.get("reason")
    if skipped:
        print(f"[fourth_model] skipped {sum(skipped.values())} rows not from {version}: "
              f"{dict(skipped)}")
    return {"single": single, "final": final, "pack_hash": packh, "reasons": reasons,
            "prompt_sha": psha}


def load_fourth(path, key: str) -> tuple[dict, dict]:
    """Fourth model rows -> {claim: verdict}, {claim: reason}. Last successful row wins."""
    verdicts, reasons = {}, {}
    if not Path(path).exists():
        return verdicts, reasons
    for r in read_jsonl(path):
        if r.get("model_key") != key or r.get("error"):
            continue
        verdicts[r["claim_id"]] = norm(r.get("verdict"))
        reasons[r["claim_id"]] = r.get("reason")
    return verdicts, reasons


def load_gold(gold_json: str | None) -> tuple[dict, str]:
    """Same source as fsv2/overlap.py: claim CSVs + label_overrides, unless a
    claim_id -> label JSON is given (use the audit file the main results used)."""
    if gold_json:
        return {k: norm(v) for k, v in json.load(open(gold_json, encoding="utf-8")).items()}, gold_json
    from study.claimsets import claims_folder, load_csv_claims
    from study.common import _apply_label_overrides, config
    folder = claims_folder()
    claims, _ = load_csv_claims(folder)
    _apply_label_overrides(claims)
    n_ov = len(config()["claims"]["csv"].get("label_overrides") or {})
    return {c["id"]: norm(c["label"]) for c in claims}, f"{folder} + {n_ov} label overrides"


def read_ids(path) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    ids = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]          # drop comments BEFORE splitting on commas
        ids.update(x.strip() for x in line.split(",") if x.strip())
    return ids


# ------------------------------------------------------------------ rules

def unanimous(labels) -> str | None:
    """All valid and equal (same rule as fsv2.decide._unanimous)."""
    labels = list(labels)
    if labels and all(l in LABELS for l in labels) and len(set(labels)) == 1:
        return labels[0]
    return None


def at_least(labels, k: int) -> str | None:
    """A valid label given by at least k of the models."""
    c = Counter(l for l in labels if l in LABELS)
    if not c:
        return None
    lab, n = c.most_common(1)[0]
    return lab if n >= k else None


def rule_unanimous(models):
    return lambda answers: unanimous(answers[m] for m in models) or NEE


def rule_majority(models):
    need = len(models) // 2 + 1
    return lambda answers: at_least([answers[m] for m in models], need) or NEE


def pattern(labels) -> str:
    """Vote split over the four answers, e.g. '4', '3-1', '2-2', '2-1-1', '1-1-1-1'.
    Unparseable answers count as their own group."""
    return "-".join(str(n) for n in sorted(Counter(labels).values(), reverse=True))


# ------------------------------------------------------------------ metrics

def wilson(k: int, n: int) -> tuple[float, float] | None:
    if n == 0:
        return None
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def score(pred: dict, gold: dict, ids) -> dict:
    ids = list(ids)
    n = len(ids)
    correct = sum(pred[c] == gold[c] for c in ids)
    answered = [c for c in ids if pred[c] in COMMITTED]
    cw = sum(pred[c] != gold[c] for c in answered)
    abst = [c for c in ids if pred[c] == NEE]
    per = {}
    for lab in LABELS:
        tp = sum(pred[c] == lab and gold[c] == lab for c in ids)
        npred = sum(pred[c] == lab for c in ids)
        ngold = sum(gold[c] == lab for c in ids)
        prec = tp / npred if npred else None
        rec = tp / ngold if ngold else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else 0.0
        per[lab] = {"precision": prec, "recall": rec, "f1": f1, "n_pred": npred, "n_gold": ngold}
    return {
        "n": n, "correct": correct, "accuracy": correct / n if n else None,
        "answered": len(answered), "coverage": len(answered) / n if n else None,
        "confidently_wrong": cw,
        "accuracy_on_answered": (len(answered) - cw) / len(answered) if answered else None,
        "accuracy_on_answered_ci95": wilson(len(answered) - cw, len(answered)),
        "abstained": len(abst),
        "abstention_accuracy": (sum(gold[c] == NEE for c in abst) / len(abst)) if abst else None,
        "unparseable": sum(pred[c] not in LABELS for c in ids),
        "macro_f1": sum(per[l]["f1"] for l in LABELS) / len(LABELS),
        "per_class": per,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(pred_a: dict, pred_b: dict, gold: dict, ids, what: str) -> dict:
    """b = A has the event and B does not; c = the reverse."""
    if what == "confidently_wrong":
        ev = lambda p, c: p[c] in COMMITTED and p[c] != gold[c]
    else:  # correct
        ev = lambda p, c: p[c] == gold[c]
    b = sum(ev(pred_a, c) and not ev(pred_b, c) for c in ids)
    c_ = sum(ev(pred_b, c) and not ev(pred_a, c) for c in ids)
    return {"a_only": b, "b_only": c_, "p_exact": mcnemar_exact(b, c_)}


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def phi(a: list[bool], b: list[bool]) -> float:
    n11 = sum(x and y for x, y in zip(a, b)); n00 = sum((not x) and (not y) for x, y in zip(a, b))
    n10 = sum(x and not y for x, y in zip(a, b)); n01 = sum((not x) and y for x, y in zip(a, b))
    d = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / d if d else float("nan")


def effective_models(answers: dict, gold: dict, ids, models) -> float:
    """Kish k/(1+(k-1)*mean pairwise phi of 'wrong'), as in fsv2/overlap.py."""
    ids = list(ids)
    ps = [phi([answers[c][a] != gold[c] for c in ids], [answers[c][b] != gold[c] for c in ids])
          for a, b in itertools.combinations(models, 2)]
    if any(math.isnan(p) for p in ps):
        return float("nan")
    k, rho = len(models), sum(ps) / len(ps)
    return k / (1 + (k - 1) * rho)


def bootstrap_effective(answers, gold, ids, models, reps=2000, seed=0):
    rnd, ids, vals = random.Random(seed), list(ids), []
    for _ in range(reps):
        s = [rnd.choice(ids) for _ in ids]
        v = effective_models(answers, gold, s, models)
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    if not vals:
        return None
    return round(vals[int(.025 * len(vals))], 2), round(vals[int(.975 * len(vals)) - 1], 2)


# ------------------------------------------------------------------ run

async def run_async(a) -> None:
    import httpx

    from study.packs import load_pack

    from . import evidence as ev
    from . import llm
    from .decide import build_prompt

    saved = load_saved(a.results)
    ids = sorted(saved["final"])
    if a.ids:
        want = set(a.ids.split(","))
        ids = [c for c in ids if c in want]
    if not ids:
        raise SystemExit("no claims found in --results (need fs2 rows without errors)")
    out = Path(a.out or f"study/results/fsv2_r1_{a.model_key}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done, _ = load_fourth(out, a.model_key)
    todo = [c for c in ids if c not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(ids)} claims in the saved results; {len(done)} already done; running {len(todo)}")
    tok_in = tok_out = n_err = 0
    async with httpx.AsyncClient() as client:
        with out.open("a", encoding="utf-8") as fh:
            for cid in todo:
                t0 = time.perf_counter()
                row = {"kind": "fourth_model_r1", "claim_id": cid, "model_key": a.model_key}
                try:
                    pack = load_pack(cid)
                    if pack.get("pack_hash") != saved["pack_hash"].get(cid):
                        raise ValueError(f"pack hash {pack.get('pack_hash')} differs from the "
                                         f"saved FactShield row ({saved['pack_hash'].get(cid)})")
                    items = ev.prepare(pack)
                    valid = {it.pid for it in items}
                    system, user, shas = build_prompt(pack["claim"], items)
                    want = saved["prompt_sha"].get(cid)
                    if want and want != shas:
                        raise ValueError(f"prompt differs from the saved FactShield run: "
                                         f"now {shas}, saved {want}")
                    if not want:
                        raise ValueError("saved FactShield row has no prompt_sha to compare")
                    cfg = llm.panel_cfg(a.model_key)      # study settings, as round 1
                    r = await llm.call(cfg, system, user, client)
                    parsed = (llm.parse_verdict(r.text, valid) if r.error is None else
                              {"verdict": None, "passages": {}, "statements": [], "reason": None})
                    row.update({
                        "pack_hash": pack.get("pack_hash"), "prompt_sha": shas,
                        "model_requested": r.model_requested, "model_version": r.model_version,
                        "params_sent": r.params_sent, "temperature_sent": r.temperature_sent,
                        "error": r.error, "verdict": parsed["verdict"],
                        "passages": parsed["passages"], "statements": parsed["statements"],
                        "reason": parsed["reason"], "finish_reason": r.finish_reason,
                        "tokens": r.tokens, "tool_calls": r.tool_calls, "text": r.text,
                        "timestamp": r.timestamp, "attempts": r.attempts,
                        "latency_total_ms": r.latency_total_ms})
                    tok_in += (r.tokens or {}).get("input") or 0
                    tok_out += (r.tokens or {}).get("output") or 0
                except Exception as exc:  # recorded, not hidden
                    row["error"] = f"{type(exc).__name__}: {exc}"
                secs = round(time.perf_counter() - t0, 2)
                row["seconds"] = secs
                fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
                fh.flush()
                # Verdicts are never printed: reading them before the analysis is fixed would steer it.
                if row.get("error"):
                    n_err += 1
                    shown = "ERROR " + row["error"][:140]
                else:
                    shown = "ok" + ("  (reply could not be parsed)" if row["verdict"] == "unparseable" else "")
                print(f"{cid:<26}{secs:>7.1f}s  {shown}")
    print(f"\nwritten to {out}; errors this run: {n_err}; tokens in {tok_in:,} / out {tok_out:,}")
    if n_err:
        print("Run the same command again to retry the failed claims.")


# ------------------------------------------------------------------ analyse

def analyse(a) -> dict:
    saved = load_saved(a.results)
    fourth, fourth_reason = load_fourth(a.fourth, a.model_key)
    gold, gold_src = load_gold(a.gold)
    models = (*BASE3, a.model_key)
    new_ids = read_ids(a.new_ids)
    excl = read_ids(a.exclude)

    from .run_packs import split_of

    all_ids = sorted(saved["final"])
    have = [c for c in all_ids
            if all(saved["single"].get((c, m)) is not None for m in BASE3) and c in fourth]
    missing = sorted(set(all_ids) - set(have))
    no_gold = [c for c in have if c not in gold]
    if no_gold:
        raise SystemExit(f"no gold label for {no_gold}")
    answers = {c: {**{m: saved["single"][(c, m)] for m in BASE3}, a.model_key: fourth[c]}
               for c in have}

    subsets = {
        "all": have,
        "unseen (test + new)": [c for c in have if c in new_ids or split_of(c) == "test"],
        f"without {len(excl & set(have))} audited-label claims": [c for c in have if c not in excl],
    }

    # conditions (pre-specified in FOURTH_MODEL_PLAN.md)
    k = a.model_key
    conds = {
        "FactShield v2 as run (reference)": {c: saved["final"][c] for c in have},
        "U3 openai+gemini+groq (simulated)": None,
        f"U3' openai+gemini+{k}": None,
        "U4 all four agree": None,
        "M4 at least 3 of 4 agree": None,
        **{f"single {m}": None for m in models},
    }
    rules = {
        "U3 openai+gemini+groq (simulated)": rule_unanimous(BASE3),
        f"U3' openai+gemini+{k}": rule_unanimous(("openai", "gemini", k)),
        "U4 all four agree": rule_unanimous(models),
        "M4 at least 3 of 4 agree": lambda ans: at_least([ans[m] for m in models], 3) or NEE,
        **{f"single {m}": (lambda m: lambda ans: ans[m])(m) for m in models},
    }
    for name, rule in rules.items():
        conds[name] = {c: rule(answers[c]) for c in have}

    report = {"gold_source": gold_src, "n_claims": len(have), "missing_claims": missing,
              "fourth_model_key": k, "subsets": {}, "combinations": {}, "agreement": {},
              "overlap": {}, "comparisons": {}, "shared_errors": []}

    def p(*x):
        print(*x)

    p(f"gold: {gold_src}")
    p(f"claims with all four answers: {len(have)}" + (f"; missing {missing}" if missing else ""))
    sim_diff = [c for c in have if conds["U3 openai+gemini+groq (simulated)"][c]
                != conds["FactShield v2 as run (reference)"][c]]
    p(f"simulated U3 differs from FactShield as run on {len(sim_diff)} claims "
      f"(extra samples / citation check): {sim_diff}")

    for sname, ids in subsets.items():
        p(f"\n=== {sname}: {len(ids)} claims; gold "
          + ", ".join(f"{l} {sum(gold[c] == l for c in ids)}" for l in LABELS) + " ===")
        p(f"{'condition':<38}{'correct':>8}{'answered':>9}{'CW':>4}{'acc|ans':>9}  95% CI      macroF1")
        rows = {}
        for name, pred in conds.items():
            s = score(pred, gold, ids)
            rows[name] = s
            ci = s["accuracy_on_answered_ci95"]
            acc = f"{s['accuracy_on_answered']:.0%}" if s["accuracy_on_answered"] is not None else "-"
            p(f"{name:<38}{s['correct']:>8}{s['answered']:>9}{s['confidently_wrong']:>4}{acc:>9}  "
              f"{(f'{ci[0]:.2f}-{ci[1]:.2f}' if ci else '-'):<11} {s['macro_f1']:.2f}"
              + (f"  unparseable {s['unparseable']}" if s["unparseable"] else ""))
        report["subsets"][sname] = rows

        # pre-specified paired tests; primary first, the rest Holm-corrected
        base = conds["U3 openai+gemini+groq (simulated)"]
        tests = [("PRIMARY  CW: U4 vs U3", conds["U4 all four agree"], base, "confidently_wrong"),
                 ("CW: M4 vs U3", conds["M4 at least 3 of 4 agree"], base, "confidently_wrong"),
                 (f"CW: U3' (groq->{k}) vs U3", conds[f"U3' openai+gemini+{k}"], base, "confidently_wrong"),
                 (f"CW: single {k} vs single openai", conds[f"single {k}"], conds["single openai"],
                  "confidently_wrong"),
                 ("correct: U4 vs U3", conds["U4 all four agree"], base, "correct"),
                 ("correct: M4 vs U3", conds["M4 at least 3 of 4 agree"], base, "correct")]
        res = [{"test": t, **paired(x, y, gold, ids, w)} for t, x, y, w in tests]
        adj = holm([r["p_exact"] for r in res[1:]])
        res[0]["p_holm"] = None
        for r, q in zip(res[1:], adj):
            r["p_holm"] = q
        lost = sum(base[c] in COMMITTED and base[c] == gold[c]
                   and conds["U4 all four agree"][c] not in COMMITTED for c in ids)
        p("paired tests (A_only = A has the event and B does not; exact McNemar; "
          "secondary p Holm-adjusted):")
        for r in res:
            p(f"  {r['test']:<38} A_only {r['a_only']:>2}  B_only {r['b_only']:>2}  "
              f"p {r['p_exact']:.3f}" + (f"  Holm {r['p_holm']:.3f}" if r["p_holm"] is not None else ""))
        p(f"  correct verdicts U3 gave that U4 no longer gives: {lost}")
        report["comparisons"][sname] = {"tests": res, "correct_lost_U4_vs_U3": lost}

    ids = subsets["all"]
    # ---------------- all 15 combinations (exploratory, descriptive only)
    p("\n=== every combination of the four (all claims; round-1 answers; exploratory) ===")
    p(f"{'models':<36}{'rule':<10}{'correct':>8}{'answered':>9}{'CW':>4}")
    for size in range(1, len(models) + 1):
        for combo in itertools.combinations(models, size):
            rlist = [("single", lambda ans, m=combo[0]: ans[m])] if size == 1 else \
                [("unanimous", rule_unanimous(combo))] + \
                ([("majority", rule_majority(combo))] if size >= 3 else [])
            for rname, rule in rlist:
                s = score({c: rule(answers[c]) for c in ids}, gold, ids)
                report["combinations"][f"{'+'.join(combo)} / {rname}"] = {
                    k2: s[k2] for k2 in ("correct", "answered", "confidently_wrong", "macro_f1")}
                p(f"{'+'.join(combo):<36}{rname:<10}{s['correct']:>8}{s['answered']:>9}"
                  f"{s['confidently_wrong']:>4}")

    # ---------------- agreement level vs correctness
    p("\n=== agreement among the four vs gold (all claims) ===")
    p(f"{'split':<10}{'claims':>7}{'plurality right':>17}{'all wrong':>11}")
    by = {}
    for c in ids:
        labs = [answers[c][m] for m in models]
        top = Counter(labs).most_common(1)[0][0]
        d = by.setdefault(pattern(labs), {"claims": 0, "plurality_right": 0, "all_wrong": 0})
        d["claims"] += 1
        d["plurality_right"] += top == gold[c]
        d["all_wrong"] += all(l != gold[c] for l in labs)
    for pat in sorted(by, key=lambda s: [-int(x) for x in s.split("-")]):
        d = by[pat]
        p(f"{pat:<10}{d['claims']:>7}{d['plurality_right']:>17}{d['all_wrong']:>11}")
    p("  ('2-2' plurality = whichever tied label Counter lists first; read those rows as ties)")
    report["agreement"] = by

    # ---------------- error overlap
    p("\n=== error overlap (wrong = any answer that is not gold) ===")
    for name, ms in (("3 models (as run)", BASE3), (f"openai+gemini+{k}", ("openai", "gemini", k)),
                     ("all four", models)):
        eff = effective_models(answers, gold, ids, ms)
        ci = bootstrap_effective(answers, gold, ids, ms)
        rates = [sum(answers[c][m] != gold[c] for c in ids) / len(ids) for m in ms]
        allw = sum(all(answers[c][m] != gold[c] for m in ms) for c in ids)
        same = sum(all(answers[c][m] != gold[c] for m in ms) and len({answers[c][m] for m in ms}) == 1
                   for c in ids)
        p(f"{name:<24} effective models {eff:.2f} of {len(ms)} (95% {ci})  all wrong {allw} "
          f"(independence {len(ids) * math.prod(rates):.1f}); all the SAME wrong answer {same}")
        report["overlap"][name] = {"effective": eff, "ci95": ci, "all_wrong": allw,
                                   "all_same_wrong": same,
                                   "expected_if_independent": len(ids) * math.prod(rates)}
    for m in BASE3:
        a_ = [answers[c][m] != gold[c] for c in ids]
        b_ = [answers[c][k] != gold[c] for c in ids]
        p(f"  phi(wrong) {m}-{k}: {phi(a_, b_):.2f}")

    # ---------------- what the fourth model said on the known shared errors
    fs = saved["final"]
    focus = sorted({c for c in ids if fs[c] in COMMITTED and fs[c] != gold[c]}
                   | {c for c in ids if all(answers[c][m] != gold[c] for m in BASE3)
                      and len({answers[c][m] for m in BASE3}) == 1})
    p(f"\n=== FactShield's confident errors and the 3-model shared errors: what {k} said ===")
    for c in focus:
        p(f"  {c}: gold {gold[c]}; openai/gemini/groq {[answers[c][m] for m in BASE3]}; "
          f"{k} {answers[c][k]}; FactShield {fs[c]}")
        report["shared_errors"].append({"claim_id": c, "gold": gold[c], "factshield": fs[c],
                                        **{m: answers[c][m] for m in models}})

    # ---------------- error-coding worksheet (for Nikita, by hand)
    if a.worksheet:
        texts = {}
        try:
            from study.packs import load_pack
            for c in ids:
                try:
                    texts[c] = load_pack(c)["claim"]
                except Exception:
                    texts[c] = ""
        except Exception:
            pass
        reasons = {**saved["reasons"], **{(c, k): r for c, r in fourth_reason.items()}}
        rows = []
        for c in ids:
            wrong = [m for m in models if answers[c][m] != gold[c]]
            if len(wrong) >= 2:
                rows.append({"claim_id": c, "claim": texts.get(c, ""), "gold": gold[c],
                             "n_wrong_of_4": len(wrong),
                             **{f"{m}_verdict": answers[c][m] for m in models},
                             **{f"{m}_reason": reasons.get((c, m)) or "" for m in models},
                             "error_type": "", "notes": ""})
        rows.sort(key=lambda r: -r["n_wrong_of_4"])
        with open(a.worksheet, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["claim_id"])
            w.writeheader()
            w.writerows(rows)
        p(f"\nerror-coding worksheet ({len(rows)} claims with >=2 of 4 wrong): {a.worksheet}")

    if a.summary:
        Path(a.summary).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        p(f"summary JSON: {a.summary}")
    return report


# ------------------------------------------------------------------ cli

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="round-1 answers from the fourth model (makes API calls)")
    r.add_argument("--results", nargs="+", required=True, help="saved FactShield v2 jsonl files")
    r.add_argument("--model-key", default="claude", help="key in study/config.yaml models")
    r.add_argument("--out", help="default study/results/fsv2_r1_<key>.jsonl")
    r.add_argument("--ids", help="comma-separated claim ids")
    r.add_argument("--limit", type=int, help="run only this many (smoke test)")

    s = sub.add_parser("analyse", help="4-model comparison from saved answers (no API calls)")
    s.add_argument("--results", nargs="+", required=True)
    s.add_argument("--fourth", default="study/results/fsv2_r1_claude.jsonl")
    s.add_argument("--model-key", default="claude")
    s.add_argument("--gold", help="JSON claim_id -> label (default: CSVs + label_overrides)")
    s.add_argument("--new-ids", default=str(HERE / "test_ids_new_setB.txt"))
    s.add_argument("--exclude", default=str(HERE / "claude_influenced_ids.txt"),
                   help="claims whose labels were changed in audits Claude took part in")
    s.add_argument("--worksheet", default="study/results/fourth_model_error_worksheet.csv")
    s.add_argument("--summary", default="study/results/fourth_model_summary.json")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        asyncio.run(run_async(a))
    else:
        analyse(a)


if __name__ == "__main__":
    main()
