"""Pre-study checks: what each model knows unaided, and whether anything
retrieves server-side.

    python -m study.probe closed-book [--models ...] [--only ...]
    python -m study.probe leak-check  [--models ...]
    python -m study.probe summarise

CLOSED-BOOK. Every claim to every model with no evidence, repeated. Produces a
per-claim, per-model `knew_unaided` flag (correct and committed on EVERY
repeat) for use as a covariate, and per-set accuracy.

Reading the closed-book numbers. The spec's rule of thumb — "near 33% means
guessing" — assumes a model picks uniformly among three labels. It does not:
without evidence, most models answer "not enough evidence" much of the time.
So this reports three things side by side and lets you judge:
    commit rate         share answered supported/refuted
    accuracy            over all claims with a supported/refuted gold label
    committed accuracy  among committed answers, tested against 50%, the rate
                        of a coin flip between supported and refuted
and the majority-class rate of each set as a second baseline.

LEAK CHECK. A few post-cutoff claims with an empty pack. If a model answers
them correctly well above a coin flip, or any tool/grounding field appears in a
response, something retrieved server-side. Claims whose alteration is visible
from the text alone (an implausible number) are excluded, because those can be
refuted with no knowledge at all.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
from collections import defaultdict
from math import comb
from pathlib import Path

from .common import config, load_claims, path, read_jsonl, write_meta, provenance
from .run import execute

CLOSED = "closed_book.jsonl"
LEAK = "leak_check.jsonl"
COMMITTED = ("supported", "refuted")


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact; no scipy needed."""
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _latest(rows: list[dict]) -> list[dict]:
    """Last successful row per key, re-parsed with the CURRENT parser.

    The verdict stored at call time used the parser of that day; summaries
    re-read the saved response text so a parser fix (parser-v4, 2026-10-29)
    applies to every row alike. The stored verdict is kept as
    parsed_verdict_at_call."""
    from .prompts import PARSER_VERSION, parse
    best = {}
    for r in rows:
        if r.get("error") is None:
            best[(r["claim_id"], r["model"], r["run_index"])] = r
    out = []
    for r in best.values():
        if "response_text" not in r:  # nothing saved to re-read (e.g. test rows)
            out.append(r)
            continue
        p = parse(r.get("response_text"), r.get("condition") or "closed_book")
        out.append({**r, "parsed_verdict_at_call": r.get("parsed_verdict"),
                    "parsed_verdict": p["parsed_verdict"], "parse_rule": p["parse_rule"],
                    "parser_version": PARSER_VERSION})
    return out


# ------------------------------------------------------------------ closed book

def plan_closed_book(claims, model_keys, repeats, seed):
    plan = [{"kind": "closed_book", "claim": c, "condition": "closed_book", "model_key": m,
             "run_index": r, "live_search": False, "pack": None}
            for r in range(repeats) for c in claims for m in model_keys]
    random.Random(seed + 2).shuffle(plan)
    for i, p in enumerate(plan):
        p["call_order"] = i
    return plan


def summarise_closed_book(rows: list[dict], claims: dict) -> tuple[list[dict], list[dict]]:
    rows = _latest(rows)
    per = defaultdict(list)
    for r in rows:
        per[(r["claim_id"], r["model"])].append(r)

    flags = []
    for (cid, m), rs in sorted(per.items()):
        gold = claims[cid]["label"]
        verdicts = [r["parsed_verdict"] for r in sorted(rs, key=lambda x: x["run_index"])]
        knew = None
        if gold in COMMITTED:
            knew = all(v == gold for v in verdicts)
        flags.append({"claim_id": cid, "model": m, "set": claims[cid]["set"], "gold": gold,
                      "n_repeats": len(verdicts), "verdicts": "|".join(verdicts),
                      "knew_unaided": knew, **provenance()})

    agg = []
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], claims[r["claim_id"]]["set"])].append(r)
    for (m, s), rs in sorted(by.items()):
        golds = [claims[r["claim_id"]]["label"] for r in rs]
        scored = [r for r in rs if claims[r["claim_id"]]["label"] in COMMITTED]
        committed = [r for r in scored if r["parsed_verdict"] in COMMITTED]
        correct = sum(r["parsed_verdict"] == claims[r["claim_id"]]["label"] for r in scored)
        c_correct = sum(r["parsed_verdict"] == claims[r["claim_id"]]["label"] for r in committed)
        majority = max(golds.count(g) for g in set(golds)) / len(golds)
        agg.append({
            "model": m, "set": s, "n_calls": len(rs), "n_scored": len(scored),
            "commit_rate": round(len(committed) / len(scored), 3) if scored else None,
            "accuracy": round(correct / len(scored), 3) if scored else None,
            "committed_accuracy": round(c_correct / len(committed), 3) if committed else None,
            "p_committed_vs_coinflip": round(binom_sf(c_correct, len(committed), 0.5), 4) if committed else None,
            "majority_class_rate": round(majority, 3),
            "unparseable_rate": round(sum(r["parsed_verdict"] == "unparseable" for r in rs) / len(rs), 3),
            **provenance(),
        })
    return flags, agg


# ------------------------------------------------------------------ leak check

def leak_claims(claims: list[dict], seed: int) -> list[dict]:
    lc = config()["probe"]["leak_check"]
    pool = [c for c in claims
            if c["set"] == lc["claim_set"] and c["label"] in COMMITTED
            and (not lc["require_not_surface_detectable"] or c.get("surface_detectable") is False)]
    rng = random.Random(seed + 3)
    return sorted(rng.sample(pool, min(lc["n_claims"], len(pool))), key=lambda c: c["id"])


def judge_leak(rows: list[dict], claims: dict) -> list[dict]:
    alpha = config()["probe"]["leak_check"]["alpha"]
    out = []
    by = defaultdict(list)
    for r in _latest(rows):
        by[r["model"]].append(r)
    for m, rs in sorted(by.items()):
        tools = [r for r in rs if r["tool_calls"]]
        urls = [r for r in rs if r.get("cited_urls")]
        committed = [r for r in rs if r["parsed_verdict"] in COMMITTED]
        correct = sum(r["parsed_verdict"] == claims[r["claim_id"]]["label"] for r in committed)
        p = binom_sf(correct, len(committed), 0.5) if committed else 1.0
        # Smallest number correct (of n committed) that would reach alpha.
        n = len(committed)
        k_needed = next((k for k in range(n + 1) if binom_sf(k, n, 0.5) < alpha), None)
        reasons = []
        if tools:
            reasons.append(f"tool/grounding fields in {len(tools)} responses")
        if p < alpha:
            reasons.append(f"{correct}/{n} committed answers correct, p={p:.3g} vs coin flip")
        out.append({"model": m, "n": len(rs), "committed": n, "correct": correct, "p": p,
                    "responses_with_urls": len(urls), "tool_rows": len(tools),
                    "result": "FAIL" if reasons else "PASS", "reasons": reasons,
                    "detectable": (f"needs >= {k_needed}/{n} correct to fail" if k_needed is not None
                                   else f"cannot fail on accuracy with {n} committed answers")})
    return out


def print_leak(verdicts: list[dict]) -> None:
    print("\n=== SEARCH-LEAK CHECK ===")
    for v in verdicts:
        print(f"  {v['model']:<8} {v['result']}  committed {v['committed']}/{v['n']}, "
              f"correct {v['correct']}, p={v['p']:.3g}; {v['detectable']}; "
              f"{v['responses_with_urls']} responses contain URLs")
        for why in v["reasons"]:
            print(f"           -> {why}")
    print("  PASS means no evidence of retrieval was found. With this few claims it is a\n"
          "  weak check: it catches a model that reliably knows post-cutoff facts, not an\n"
          "  occasional leak. Tool fields are checked on every main-run row as well.")


# ------------------------------------------------------------------ CLI

def _write_csv(p: Path, rows: list[dict]) -> None:
    if not rows:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_meta(p)


def summarise() -> None:
    claims = {c["id"]: c for c in load_claims()}
    rows = read_jsonl(path("responses") / CLOSED)
    if rows:
        flags, agg = summarise_closed_book(rows, claims)
        _write_csv(path("results") / "knew_unaided.csv", flags)
        _write_csv(path("results") / "closed_book_summary.csv", agg)
        print("\n=== CLOSED-BOOK ===")
        print(f"  {'model':<8}{'set':<5}{'n':>5}{'commit':>8}{'acc':>7}{'acc|commit':>12}"
              f"{'p(coin)':>9}{'majority':>10}{'unparse':>9}")
        for a in agg:
            print(f"  {a['model']:<8}{a['set']:<5}{a['n_calls']:>5}{a['commit_rate']!s:>8}"
                  f"{a['accuracy']!s:>7}{a['committed_accuracy']!s:>12}"
                  f"{a['p_committed_vs_coinflip']!s:>9}{a['majority_class_rate']:>10}"
                  f"{a['unparseable_rate']:>9}")
        known = sum(1 for f in flags if f["knew_unaided"])
        print(f"  knew_unaided=True for {known}/{len(flags)} claim-model pairs "
              f"-> {path('results') / 'knew_unaided.csv'}")
    leak = read_jsonl(path("responses") / LEAK)
    if leak:
        print_leak(judge_leak(leak, claims))


def main() -> None:
    cfg = config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["closed-book", "leak-check", "summarise"])
    ap.add_argument("--models", default=",".join(cfg["study_models"]))
    ap.add_argument("--only", default="")
    ap.add_argument("--concurrency", type=int, default=cfg["run"]["concurrency"])
    args = ap.parse_args()
    if args.cmd == "summarise":
        summarise()
        return
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    only = {x for x in args.only.split(",") if x}
    claims = [c for c in load_claims() if not only or c["id"] in only]
    seed = cfg["seed"]
    if args.cmd == "closed-book":
        plan = plan_closed_book(claims, model_keys, cfg["probe"]["closed_book_repeats"], seed)
        out = path("responses") / CLOSED
    else:
        chosen = leak_claims(claims, seed)
        if not chosen:
            lc = cfg["probe"]["leak_check"]
            raise SystemExit(
                f"no eligible leak-check claims: need set {lc['claim_set']!r} (probe.leak_check."
                f"claim_set), gold supported/refuted"
                + (", and surface_detectable=false. CSV claims without a surface_detectable column "
                   "are 'unknown'; add the column or set require_not_surface_detectable: false"
                   if lc["require_not_surface_detectable"] else ""))
        print(f"leak check on {[c['id'] for c in chosen]}")
        plan = [{"kind": "leak_check", "claim": c, "condition": "free", "model_key": m,
                 "run_index": 0, "live_search": False, "pack": None, "call_order": i}
                for i, (c, m) in enumerate((c, m) for c in chosen for m in model_keys)]
        out = path("responses") / LEAK
    print(f"{len(plan)} calls -> {out}")
    print(asyncio.run(execute(plan, out, concurrency=args.concurrency)))
    summarise()


if __name__ == "__main__":
    main()
