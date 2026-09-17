"""Run the benchmark and report it.

    python -m eval.run     --corpus eval/dataset/corpus.jsonl --runs 3
    python -m eval.analyze --results eval/results/runs.jsonl

Two reporting choices worth understanding:

  * MACRO F1, not accuracy. With four unbalanced verdict classes, plain
    accuracy flatters a system that always guesses the majority class.
  * PER CATEGORY, always. An overall figure around 70% typically hides 95% on
    direct claims and 40% on hedged ones — and the breakdown is the finding,
    not the headline number.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .conditions import CONDITIONS


# --------------------------------------------------------------------- run

async def main_run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="eval/results/runs.jsonl")
    ap.add_argument("--runs", type=int, default=3,
                    help="Repeats per claim. Providers are not fully "
                         "deterministic even at temperature 0, so variance is "
                         "reported rather than assumed away.")
    ap.add_argument("--conditions", default="full,open_web_only,no_page_context,parametric")
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    names = [c.strip() for c in args.conditions.split(",") if c.strip() in CONDITIONS]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as fh:
        for run_i in range(args.runs):
            for rec in records:
                for name in names:
                    try:
                        res = await CONDITIONS[name](rec)
                    except Exception as exc:
                        res = {"condition": name, "claim_verdict": "error", "error": str(exc)}
                    fh.write(json.dumps({
                        "run": run_i, "id": rec["id"],
                        "gold": rec.get("label"),
                        "category": rec.get("category_gold") or rec["triage"]["category"],
                        **res,
                    }, default=str) + "\n")
            print(f"run {run_i + 1}/{args.runs} complete")

    print(f"\nResults written to {out}")


# ----------------------------------------------------------------- analysis

def macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = {g for g, _ in pairs} | {p for _, p in pairs}
    scores = []
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def analyze(path: Path) -> None:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    by_cond = defaultdict(list)
    for r in rows:
        if r.get("gold"):
            by_cond[r["condition"]].append(r)

    print("\n=== Macro F1 by condition and claim category ===\n")
    cats = sorted({r["category"] for rs in by_cond.values() for r in rs})
    header = f"{'condition':<18}" + "".join(f"{c[:12]:>14}" for c in cats) + f"{'ALL':>14}"
    print(header)
    print("-" * len(header))

    for cond, rs in sorted(by_cond.items()):
        line = f"{cond:<18}"
        for cat in cats:
            pairs = [(r["gold"], r["claim_verdict"]) for r in rs if r["category"] == cat]
            line += f"{macro_f1(pairs):>14.3f}" if pairs else f"{'—':>14}"
        line += f"{macro_f1([(r['gold'], r['claim_verdict']) for r in rs]):>14.3f}"
        print(line)

    # Risk-coverage. A system that is 95% accurate on the 60% it chooses to
    # answer may be far more useful than one that is 78% accurate on everything,
    # because the first never confidently lies to you.
    print("\n=== Risk / coverage ===\n")
    print(f"{'condition':<18}{'coverage':>12}{'accuracy@covered':>20}")
    for cond, rs in sorted(by_cond.items()):
        answered = [r for r in rs if r["claim_verdict"] != "unresolved"]
        cov = len(answered) / len(rs) if rs else 0.0
        acc = (sum(1 for r in answered if r["claim_verdict"] == r["gold"]) / len(answered)
               if answered else 0.0)
        print(f"{cond:<18}{cov:>12.2%}{acc:>20.2%}")

    # Stability across repeats. Rarely reported, costs only API calls, and if a
    # fixed scaffold proves more stable than an ad-hoc chat prompt that is a
    # genuine argument for purpose-built tooling.
    print("\n=== Verdict stability across runs ===\n")
    for cond, rs in sorted(by_cond.items()):
        groups = defaultdict(list)
        for r in rs:
            groups[r["id"]].append(r["claim_verdict"])
        stable = sum(1 for v in groups.values() if len(set(v)) == 1)
        print(f"{cond:<18}{stable}/{len(groups)} claims gave an identical verdict every run")

    # How often does retrieval overturn the model's prior, and who is right?
    prior = {r["id"]: r["claim_verdict"] for r in by_cond.get("parametric", [])}
    full = {r["id"]: (r["claim_verdict"], r["gold"]) for r in by_cond.get("full", [])}
    shared = set(prior) & set(full)
    if shared:
        disagree = [i for i in shared if prior[i] != full[i][0]]
        evidence_right = sum(1 for i in disagree if full[i][0] == full[i][1])
        print("\n=== Prior vs evidence ===\n")
        print(f"disagreed on {len(disagree)}/{len(shared)} claims "
              f"({len(disagree) / len(shared):.1%})")
        if disagree:
            print(f"when they disagreed, the evidence-grounded verdict was correct "
                  f"{evidence_right}/{len(disagree)} times")

    n_ev = [r["n_evidence"] for r in by_cond.get("full", []) if "n_evidence" in r]
    if n_ev:
        print(f"\nmean evidence items per claim (full): {statistics.mean(n_ev):.1f}")


def main_analyze() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval/results/runs.jsonl")
    analyze(Path(ap.parse_args().results))


if __name__ == "__main__":
    import sys

    if "analyze" in sys.argv[0]:
        main_analyze()
    else:
        asyncio.run(main_run())
