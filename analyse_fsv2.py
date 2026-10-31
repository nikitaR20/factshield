"""FactShield v2 — scoring and analysis of the frozen runs.

usage:
    python analyse_fsv2.py --dev DEV.jsonl --test TEST.jsonl --new NEW.jsonl
                           --gold GOLD.(json|csv) [--setb SETB.csv] [--out DIR]

GOLD: either a JSON {claim_id: label} or the pack_audit.csv written by
`python -m study.packs audit` (its gold_label column already includes
label_overrides). On the laptop, `--gold study/results/pack_audit.csv`.
SETB: the Set B claims CSV, for the surface_detectable subgroup (optional).

Scoring rules (fixed before scoring, 2026-10-29):
- conflicting_evidence is scored as not_enough_evidence (decided 2026-10-25).
- committed = supported or refuted. confidently wrong (CW) = committed and != gold.
- missed = answered not_enough_evidence when gold is supported/refuted.
- API error rows are ignored; the last successful row per (claim, condition,
  model) is used. Every claim must have exactly one successful row per
  condition or the script stops.
- Groups: dev = the 23 claims used in design (dev run 2 file); test = 51;
  new = 26 (B-0030..B-0055); unseen = test + new (77); all = 100.
- Paired tests: exact two-sided McNemar (binomial on discordant claims).
- Intervals: Wilson 95% for proportions; Clopper-Pearson 95% upper bound for
  a zero count.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

NEE = "not_enough_evidence"
LABELS = ("supported", "refuted", NEE)
MODELS = ("openai", "gemini", "groq")


def norm(v):
    if v is None:
        return None
    return NEE if v == "conflicting_evidence" else v


# ----------------------------------------------------------------- loading
def load_rows(path, group):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("error"):
            continue
        cond = r["condition"]
        key = (r["claim_id"], cond if cond != "fs2_single" else "single:" + r["model"].split(":")[1])
        out[key] = r  # last successful row wins
    for k in out:
        out[k]["_group"] = group
    return out


def load_gold(path):
    p = Path(path)
    if p.suffix == ".json":
        return {k: norm(v) for k, v in json.load(open(p, encoding="utf-8")).items()}
    csv.field_size_limit(10 ** 8)
    g = {}
    for x in csv.DictReader(open(p, encoding="utf-8-sig")):
        g[x["claim_id"]] = norm(x["gold_label"])
    return g


def load_surface(path):
    if not path:
        return {}
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return {f"claims_Dataset_B-{i:04d}": (r.get("surface_detectable") or "").strip().upper()
            for i, r in enumerate(rows, 1)}


# ----------------------------------------------------------------- stats
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cp_upper_zero(n, alpha=0.05):
    """Two-sided 95% Clopper-Pearson upper bound when 0 of n."""
    return 1 - (alpha / 2) ** (1 / n) if n else float("nan")


def mcnemar_exact(b, c):
    """Two-sided exact p-value; b, c = discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def macro_f1(pairs):
    f = []
    for lab in LABELS:
        tp = sum(p == lab and g == lab for p, g in pairs)
        fp = sum(p == lab and g != lab for p, g in pairs)
        fn = sum(p != lab and g == lab for p, g in pairs)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f) / len(f)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--setb")
    ap.add_argument("--out", default="fsv2_analysis")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = {}
    for path, grp in ((a.dev, "dev"), (a.test, "test"), (a.new, "new")):
        part = load_rows(path, grp)
        clash = set(k[0] for k in part) & set(k[0] for k in rows)
        if clash:
            raise SystemExit(f"claim in two files: {sorted(clash)[:5]}")
        rows.update(part)
    gold = load_gold(a.gold)
    surface = load_surface(a.setb)

    claims = sorted({k[0] for k in rows})
    group = {c: rows[(c, "fs2")]["_group"] for c in claims}
    conds = ["fs2", "fs2_no_check", "fs2_majority_r1"] + [f"single:{m}" for m in MODELS]
    for c in claims:
        for cond in conds:
            if (c, cond) not in rows:
                raise SystemExit(f"missing successful row: {c} {cond}")
        if c not in gold:
            raise SystemExit(f"no gold label for {c}")

    pred = {cond: {c: norm(rows[(c, cond)]["parsed_verdict"]) for c in claims} for cond in conds}
    pred["always_nee"] = {c: NEE for c in claims}
    conds_all = conds + ["always_nee"]
    names = {"fs2": "FactShield", "fs2_no_check": "FactShield without citation check",
             "fs2_majority_r1": "Majority 2-of-3 (round 1) + check",
             "single:openai": "OpenAI alone", "single:gemini": "Gemini alone",
             "single:groq": "Groq alone", "always_nee": "Always 'not enough evidence'"}

    sets = {
        "all (100)": claims,
        "unseen: test + new (77)": [c for c in claims if group[c] != "dev"],
        "test (51)": [c for c in claims if group[c] == "test"],
        "new (26)": [c for c in claims if group[c] == "new"],
        "dev / design set (23)": [c for c in claims if group[c] == "dev"],
        "Dataset A (45)": [c for c in claims if "_A-" in c],
        "Set B (55)": [c for c in claims if "_B-" in c],
        "unseen Dataset A": [c for c in claims if "_A-" in c and group[c] != "dev"],
        "unseen Set B": [c for c in claims if "_B-" in c and group[c] != "dev"],
    }
    if surface:
        sets["Set B surface_detectable=TRUE"] = [c for c in claims if surface.get(c) == "TRUE"]
        sets["Set B surface_detectable=FALSE"] = [c for c in claims if surface.get(c) == "FALSE"]

    def stats(cond, cs):
        p = pred[cond]
        n = len(cs)
        com = [c for c in cs if p[c] in ("supported", "refuted")]
        cw = [c for c in com if p[c] != gold[c]]
        cc = [c for c in com if p[c] == gold[c]]
        correct = [c for c in cs if p[c] == gold[c]]
        firm_gold = [c for c in cs if gold[c] != NEE]
        missed = [c for c in firm_gold if p[c] == NEE]
        return {"n": n, "committed": len(com), "committed_correct": len(cc), "cw": len(cw),
                "cw_ids": cw, "correct": len(correct), "missed": len(missed),
                "firm_gold": len(firm_gold), "macro_f1": macro_f1([(p[c], gold[c]) for c in cs])}

    lines = ["# FactShield v2 — results (scored 2026-10-29)", "",
             "Frozen design fsv2-0.2 / prompt v2 / MiniCheck-Flan-T5-L. "
             "conflicting_evidence scored as not enough evidence. "
             "CW = confidently wrong (a supported/refuted verdict that does not match the label). "
             "Coverage = share of claims given a supported/refuted verdict. "
             "Accuracy on answered = correct among supported/refuted verdicts. "
             "Brackets: 95% intervals (Wilson; for 0 errors the Clopper-Pearson upper bound).", ""]
    table_rows = []
    for sname, cs in sets.items():
        if not cs:
            continue
        gl = Counter(gold[c] for c in cs)
        lines += [f"## {sname}",
                  f"Gold: supported {gl['supported']}, refuted {gl['refuted']}, NEE {gl[NEE]}", "",
                  "| Condition | Correct | Answered (coverage) | Accuracy on answered | CW (rate of all) | Missed firm | Macro-F1 |",
                  "|---|---|---|---|---|---|---|"]
        for cond in conds_all:
            s = stats(cond, cs)
            n = s["n"]
            cov = s["committed"] / n
            acc = s["committed_correct"] / s["committed"] if s["committed"] else float("nan")
            if s["committed"]:
                lo, hi = wilson(s["committed_correct"], s["committed"])
                acc_s = f"{acc:.0%} [{lo:.0%}–{hi:.0%}]"
            else:
                acc_s = "–"
            if s["cw"]:
                lo, hi = wilson(s["cw"], n)
                cw_s = f"{s['cw']} ({s['cw']/n:.1%} [{lo:.1%}–{hi:.1%}])"
            else:
                cw_s = f"0 (0% [0–{cp_upper_zero(n):.1%}])"
            lines.append(f"| {names[cond]} | {s['correct']}/{n} ({s['correct']/n:.0%}) | "
                         f"{s['committed']} ({cov:.0%}) | {acc_s} | {cw_s} | "
                         f"{s['missed']}/{s['firm_gold']} | {s['macro_f1']:.2f} |")
            table_rows.append({"set": sname, "condition": names[cond], **{k: v for k, v in s.items() if k != "cw_ids"},
                               "cw_ids": " ".join(x.replace("claims_Dataset_", "") for x in s["cw_ids"])})
        lines.append("")

    # paired comparisons on the main sets
    lines += ["## Paired comparisons: FactShield vs each alternative (exact McNemar, two-sided)", "",
              "b = claims where only the alternative is wrong in that way; c = only FactShield is.", ""]
    for sname in ("all (100)", "unseen: test + new (77)"):
        cs = sets[sname]
        lines += [f"### {sname}", "",
                  "| Alternative | CW: alt only / FS only | p | Correct: FS only / alt only | p |",
                  "|---|---|---|---|---|"]
        for cond in conds_all[1:]:
            fs, al = pred["fs2"], pred[cond]
            cw = lambda p, c: p[c] in ("supported", "refuted") and p[c] != gold[c]
            b = sum(cw(al, c) and not cw(fs, c) for c in cs)
            cc = sum(cw(fs, c) and not cw(al, c) for c in cs)
            x = sum(fs[c] == gold[c] and al[c] != gold[c] for c in cs)
            y = sum(al[c] == gold[c] and fs[c] != gold[c] for c in cs)
            lines.append(f"| {names[cond]} | {b} / {cc} | {fmt_p(mcnemar_exact(b, cc))} | {x} / {y} | {fmt_p(mcnemar_exact(x, y))} |")
        lines.append("")

    # FactShield errors listed
    lines += ["## Every FactShield confidently-wrong verdict and every missed firm label (all 100)", ""]
    for c in claims:
        p = pred["fs2"][c]
        if p != gold[c] and (p != NEE or gold[c] != NEE):
            kind = "CONFIDENTLY WRONG" if p != NEE else "missed"
            singles = ", ".join(f"{m}={pred['single:' + m][c]}" for m in MODELS)
            lines.append(f"- {c.replace('claims_Dataset_', '')} [{group[c]}] {kind}: FactShield {p}, gold {gold[c]}; "
                         f"path {rows[(c, 'fs2')]['factshield_v2']['path']}; singles {singles}. "
                         f"Claim: {rows[(c, 'fs2')]['factshield_v2']['claim'][:140]}")
    lines.append("")

    # paths
    lines += ["## How FactShield reached its answer (all 100)", "",
              "| Path / reason | Claims | FactShield answer correct | Answered |", "|---|---|---|---|"]
    pr = defaultdict(list)
    for c in claims:
        r = rows[(c, "fs2")]["factshield_v2"]
        pr[f"{r['path']} / {r.get('reason') or '-'}"].append(c)
    for k, cs in sorted(pr.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {k} | {len(cs)} | {sum(pred['fs2'][c] == gold[c] for c in cs)} | "
                     f"{sum(pred['fs2'][c] != NEE for c in cs)} |")
    lines.append("")

    # citation check behaviour
    n_checked = n_down = 0
    probs_ok, probs_bad = [], []
    for c in claims:
        r = rows[(c, "fs2")]["factshield_v2"]
        if r.get("panel_label") in ("supported", "refuted"):
            n_checked += 1
            if r["final"] != r["panel_label"]:
                n_down += 1
        for ch in r.get("checks") or []:
            (probs_ok if ch.get("confirmed") else probs_bad).append(ch.get("prob"))
    diff = [c for c in claims if pred["fs2"][c] != pred["fs2_no_check"][c]]
    lines += ["## Citation check (MiniCheck)", "",
              f"- Panel agreed on supported/refuted for {n_checked} claims; the check downgraded {n_down} to NEE.",
              f"- Claims where FactShield and 'without citation check' differ: {len(diff)} "
              f"({', '.join(x.replace('claims_Dataset_', '') for x in diff) or 'none'}).",
              f"- Statements checked: {len(probs_ok) + len(probs_bad)}; confirmed {len(probs_ok)}, not confirmed {len(probs_bad)}.", ""]

    # error overlap (single models, all and unseen)
    lines += ["## Overlap of the three models' errors (single, round 1)", ""]
    for sname in ("all (100)", "unseen: test + new (77)"):
        cs = sets[sname]
        wrong = {m: [pred["single:" + m][c] != gold[c] for c in cs] for m in MODELS}

        def phi(x, y):
            n11 = sum(i and j for i, j in zip(x, y)); n00 = sum((not i) and (not j) for i, j in zip(x, y))
            n10 = sum(i and not j for i, j in zip(x, y)); n01 = sum((not i) and j for i, j in zip(x, y))
            d = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
            return (n11 * n00 - n10 * n01) / d if d else float("nan")

        ph = {f"{x}+{y}": phi(wrong[x], wrong[y]) for x, y in itertools.combinations(MODELS, 2)}
        rho = sum(ph.values()) / 3
        eff = 3 / (1 + 2 * rho)
        rnd = random.Random(0)
        boots = []
        for _ in range(4000):
            idx = [rnd.randrange(len(cs)) for _ in cs]
            w = {m: [wrong[m][i] for i in idx] for m in MODELS}
            ps = [phi(w[x], w[y]) for x, y in itertools.combinations(MODELS, 2)]
            if not any(math.isnan(v) for v in ps):
                boots.append(3 / (1 + 2 * sum(ps) / 3))
        boots.sort()
        all3 = sum(all(wrong[m][i] for m in MODELS) for i in range(len(cs)))
        all3_same = sum(all(wrong[m][i] for m in MODELS) and len({pred['single:' + m][cs[i]] for m in MODELS}) == 1
                        for i in range(len(cs)))
        exp = len(cs) * math.prod(sum(wrong[m]) / len(cs) for m in MODELS)
        lines += [f"### {sname}",
                  f"- Wrong (any): " + ", ".join(f"{m} {sum(wrong[m])}" for m in MODELS),
                  f"- Pairwise phi: " + ", ".join(f"{k} {v:.2f}" for k, v in ph.items()),
                  f"- All three wrong: {all3} (independence would give {exp:.1f}); all three the same wrong answer: {all3_same}",
                  f"- Effective independent models (Kish): {eff:.2f} of 3; bootstrap 95% "
                  f"{boots[int(.025 * len(boots))]:.2f}–{boots[int(.975 * len(boots))]:.2f}", ""]

    # cost
    lines += ["## Cost per claim (FactShield, all 100)", ""]
    calls = [rows[(c, "fs2")]["factshield_v2"]["n_calls"] for c in claims]
    secs = [rows[(c, "fs2")]["seconds"] for c in claims]
    tok = defaultdict(int)
    for c in claims:
        for s in rows[(c, "fs2")]["factshield_v2"]["samples"]:
            t = s.get("tokens") or {}
            tok[s["model_key"]] += t.get("total") or ((t.get("input") or 0) + (t.get("output") or 0))
    lines += [f"- Model calls per claim: mean {sum(calls)/len(calls):.1f} (min {min(calls)}, max {max(calls)}); a single model uses 1.",
              f"- Wall time per claim (s, includes rate-limit waits): median {sorted(secs)[len(secs)//2]:.0f}, mean {sum(secs)/len(secs):.0f}.",
              "- Tokens (all 100 claims, verdict calls): " + ", ".join(f"{k} {v:,}" for k, v in tok.items()), ""]

    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    with open(out / "summary_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table_rows[0].keys()))
        w.writeheader(); w.writerows(table_rows)
    with open(out / "per_claim.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["claim_id", "group", "set", "gold", *conds, "path", "reason", "surface_detectable", "claim"])
        for c in claims:
            r = rows[(c, "fs2")]["factshield_v2"]
            w.writerow([c, group[c], "A" if "_A-" in c else "B", gold[c], *[pred[k][c] for k in conds],
                        r["path"], r.get("reason") or "", surface.get(c, ""), r["claim"]])
    print((out / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
