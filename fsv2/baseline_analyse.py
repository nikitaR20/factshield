"""Plain-model baseline vs FactShield v2 — analysis fixed in fsv2/BASELINE_PLAN.md.

Offline; no model calls. Reads:
  * the saved FactShield v2 runs (fs2 and fs2_single rows),
  * the plain-model rows written by `python -m study.run` (study/responses/main.jsonl),
  * Claude's v2 single answers (fsv2_r1_claude.jsonl, from fsv2.fourth_model),
  * optional v2 repeats (fsv2_r1_<model>_rep1.jsonl) for the run-to-run noise measure.

    python -m fsv2.baseline_analyse --results study/results/factshield_v2p2_dev.jsonl \
        study/results/factshield_v2p2_test.jsonl study/results/factshield_v2p2_new.jsonl

Plain answers are re-parsed from response_text with the CURRENT study parser, so the
parser on the machine that runs this is the one applied (and recorded in the summary).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from . import fourth_model as fm  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = ("openai", "claude", "gemini")
CONDITIONS = ("format", "free", "citation")
UNP = "unparseable"


def _norm(v):
    v = fm.norm(v)
    return v if v in fm.LABELS else UNP


def load_plain(path, models=MODELS) -> tuple[dict, dict, str]:
    """{(model, condition, run): {claim: verdict}}, parse-rule counts, parser version.
    Last successful row per key wins (same rule as study.score)."""
    from study import prompts
    latest = {}
    for r in fm.read_jsonl(path):
        if r.get("kind") != "main" or r.get("live_search") or r.get("error"):
            continue
        if r.get("model") not in models:
            continue
        latest[(r["model"], r["condition"], r["run_index"], r["claim_id"])] = r
    out, rules = defaultdict(dict), Counter()
    for (m, cond, run, cid), r in latest.items():
        p = prompts.parse(r.get("response_text"), cond)
        out[(m, cond, run)][cid] = _norm(p["parsed_verdict"])
        rules[(m, cond, p["parse_rule"])] += 1
    return out, rules, getattr(prompts, "PARSER_VERSION", "unknown")


def flips(a: dict, b: dict, ids) -> dict:
    both = [c for c in ids if c in a and c in b]
    changed = [c for c in both if a[c] != b[c]]
    firm = [c for c in changed if a[c] in fm.COMMITTED or b[c] in fm.COMMITTED]
    return {"pairs": len(both), "changed": len(changed),
            "changed_share": round(len(changed) / len(both), 3) if both else None,
            "changed_involving_firm": len(firm)}


def analyse(a) -> dict:
    saved = fm.load_saved(a.results)
    gold, gold_src = fm.load_gold(a.gold)
    ids_all = sorted(saved["final"])
    plain, rules, parser_version = load_plain(a.plain)
    claude_v2, _ = fm.load_fourth(a.claude_v2, "claude")
    v2 = {"openai": {c: saved["single"].get((c, "openai")) for c in ids_all},
          "gemini": {c: saved["single"].get((c, "gemini")) for c in ids_all},
          "claude": claude_v2}
    fs = {c: saved["final"][c] for c in ids_all}

    from .run_packs import split_of
    new_ids, excl = fm.read_ids(a.new_ids), fm.read_ids(a.exclude)
    subsets = {"all": ids_all,
               "unseen (test + new)": [c for c in ids_all if c in new_ids or split_of(c) == "test"],
               f"without {len(excl & set(ids_all))} audited-label claims":
                   [c for c in ids_all if c not in excl]}

    def p(*x):
        print(*x)

    report = {"gold_source": gold_src, "parser_version": parser_version, "subsets": {},
              "primary": {}, "secondary": {}, "noise": {}, "parse_rules": {}, "missing": {}}
    p(f"gold: {gold_src}\nparser: {parser_version}")

    # completeness: every condition/model/run-0 must cover every claim
    for m in MODELS:
        for cond in CONDITIONS:
            have = plain.get((m, cond, 0), {})
            miss = [c for c in ids_all if c not in have]
            report["missing"][f"{m}/{cond}/r0"] = miss
            if miss:
                p(f"WARNING {m} {cond} run 0: {len(miss)} claims missing")
        miss_v2 = [c for c in ids_all if v2[m].get(c) is None]
        if miss_v2:
            p(f"WARNING v2 single {m}: {len(miss_v2)} claims missing")
        report["missing"][f"{m}/v2"] = miss_v2

    def usable(pred: dict, ids):
        return [c for c in ids if pred.get(c) is not None]

    for sname, ids in subsets.items():
        p(f"\n=== {sname}: {len(ids)} claims; gold "
          + ", ".join(f"{l} {sum(gold[c] == l for c in ids)}" for l in fm.LABELS) + " ===")
        p(f"{'condition':<34}{'n':>4}{'correct':>8}{'answered':>9}{'CW':>4}{'unparse':>8}  acc|ans")
        rows = {}
        conds = {"FactShield v2 as run": fs}
        for m in MODELS:
            conds[f"v2 single {m}"] = v2[m]
            for cond in CONDITIONS:
                conds[f"plain {cond} {m}"] = plain.get((m, cond, 0), {})
        for name, pred in conds.items():
            u = usable(pred, ids)
            if not u:
                continue
            s = fm.score({c: _norm(pred[c]) for c in u}, gold, u)
            rows[name] = s
            acc = f"{s['accuracy_on_answered']:.0%}" if s["accuracy_on_answered"] is not None else "-"
            p(f"{name:<34}{s['n']:>4}{s['correct']:>8}{s['answered']:>9}{s['confidently_wrong']:>4}"
              f"{s['unparseable']:>8}  {acc}")
        report["subsets"][sname] = rows

        def test(label, A, B, what):
            u = [c for c in ids if A.get(c) is not None and B.get(c) is not None]
            r = fm.paired({c: _norm(A[c]) for c in u}, {c: _norm(B[c]) for c in u}, gold, u, what)
            return {"test": label, "n": len(u), **r}

        # PRIMARY: FactShield vs plain format r0, CW, per model (Holm over 3)
        prim = [test(f"CW FactShield vs plain format {m}", fs, plain.get((m, "format", 0), {}),
                     "confidently_wrong") for m in MODELS]
        for t, q in zip(prim, fm.holm([t["p_exact"] for t in prim])):
            t["p_holm"] = q
        prim_correct = [test(f"correct FactShield vs plain format {m}", fs,
                             plain.get((m, "format", 0), {}), "correct") for m in MODELS]
        for t, q in zip(prim_correct, fm.holm([t["p_exact"] for t in prim_correct])):
            t["p_holm"] = q
        p("PRIMARY (A_only = FactShield has the event and the plain model does not; exact McNemar, Holm over 3):")
        for t in prim + prim_correct:
            p(f"  {t['test']:<44} n {t['n']:>3}  A_only {t['a_only']:>2}  B_only {t['b_only']:>2}"
              f"  p {t['p_exact']:.3f}  Holm {t['p_holm']:.3f}")
        report["primary"][sname] = prim + prim_correct

        # SECONDARY 1: v2 single vs plain format (prompt + evidence layout together)
        sec1 = []
        for m in MODELS:
            for what in ("confidently_wrong", "correct"):
                sec1.append(test(f"{what} v2 single vs plain format {m}", v2[m],
                                 plain.get((m, "format", 0), {}), what))
        for t, q in zip(sec1, fm.holm([t["p_exact"] for t in sec1])):
            t["p_holm"] = q
        # SECONDARY 2: plain free / citation vs plain format
        sec2 = []
        for m in MODELS:
            for cond in ("free", "citation"):
                for what in ("confidently_wrong", "correct"):
                    sec2.append(test(f"{what} plain {cond} vs plain format {m}",
                                     plain.get((m, cond, 0), {}), plain.get((m, "format", 0), {}), what))
        for t, q in zip(sec2, fm.holm([t["p_exact"] for t in sec2])):
            t["p_holm"] = q
        p("secondary (exploratory; Holm within each family):")
        for t in sec1 + sec2:
            p(f"  {t['test']:<52} n {t['n']:>3}  A_only {t['a_only']:>2}  B_only {t['b_only']:>2}"
              f"  p {t['p_exact']:.3f}  Holm {t['p_holm']:.3f}")
        report["secondary"][sname] = {"v2_vs_plain": sec1, "plain_conditions": sec2}

    # NOISE
    p("\n=== run-to-run change (all claims) ===")
    noise = {}
    for m in MODELS:
        for cond in CONDITIONS:
            r0, r1 = plain.get((m, cond, 0), {}), plain.get((m, cond, 1), {})
            if r0 and r1:
                noise[f"plain {cond} {m}"] = flips(r0, r1, ids_all)
    for spec in a.v2_repeat or []:
        m, path = spec.split("=", 1)
        rep, _ = fm.load_fourth(path, m)
        orig = v2.get(m, {})
        noise[f"v2 single {m}"] = flips({c: _norm(v) for c, v in orig.items() if v is not None},
                                        {c: _norm(v) for c, v in rep.items()}, ids_all)
    for k, v in noise.items():
        p(f"  {k:<26} pairs {v['pairs']:>3}  changed {v['changed']:>3} ({v['changed_share']})"
          f"  involving a firm verdict {v['changed_involving_firm']}")
    report["noise"] = noise

    p("\n=== parse rules (plain rows, all runs) ===")
    for (m, cond, rule), n in sorted(rules.items()):
        p(f"  {m:<7}{cond:<9}{rule:<32}{n}")
    report["parse_rules"] = {f"{m}/{c}/{r}": n for (m, c, r), n in rules.items()}

    if a.summary:
        Path(a.summary).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        p(f"\nsummary JSON: {a.summary}")
    return report


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
    ap.add_argument("--results", nargs="+", required=True, help="saved FactShield v2 jsonl files")
    ap.add_argument("--plain", default="study/responses/main.jsonl")
    ap.add_argument("--claude-v2", default="study/results/fsv2_r1_claude.jsonl")
    ap.add_argument("--v2-repeat", nargs="*",
                    help="model=path pairs, e.g. openai=study/results/fsv2_r1_openai_rep1.jsonl")
    ap.add_argument("--gold", help="JSON claim_id -> label (default: CSVs + label_overrides)")
    ap.add_argument("--new-ids", default=str(HERE / "test_ids_new_setB.txt"))
    ap.add_argument("--exclude", default=str(HERE / "claude_influenced_ids.txt"))
    ap.add_argument("--summary", default="study/results/baseline_summary.json")
    ap.add_argument("--report", default="baseline_report.txt", help="UTF-8 copy of the printed report")
    a = ap.parse_args(argv)
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
