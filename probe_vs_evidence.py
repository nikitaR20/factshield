"""Closed-book probe vs with-evidence answers (exploratory, 2026-10-29).

usage (from the factshield folder, so the current parser is used):
    python probe_vs_evidence.py --probe study/responses/closed_book.jsonl
        --dev study/results/factshield_v2p2_dev.jsonl --test study/results/factshield_v2p2_test.jsonl
        --new study/results/factshield_v2p2_new.jsonl --gold study/results/pack_audit.csv

Closed-book answers are re-parsed from the saved text with study.prompts.parse
(parser-v4). With-evidence answers = round-1 single-model rows (fs2_single).
"""
import argparse
import csv
import json

from study.prompts import PARSER_VERSION, parse

N = "not_enough_evidence"
C = ("supported", "refuted")
M = ("openai", "gemini", "groq")


def norm(v):
    return N if v == "conflicting_evidence" else v


def main():
    ap = argparse.ArgumentParser()
    for k in ("probe", "dev", "test", "new", "gold"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    csv.field_size_limit(10 ** 8)
    gold = {x["claim_id"]: norm(x["gold_label"]) for x in csv.DictReader(open(a.gold, encoding="utf-8-sig"))}
    best = {}
    for line in open(a.probe, encoding="utf-8"):
        r = json.loads(line)
        if r.get("error") is None:
            best[(r["claim_id"], r["model"], r["run_index"])] = parse(r.get("response_text"), "closed_book")["parsed_verdict"]
    ev = {}
    for p in (a.dev, a.test, a.new):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["condition"] == "fs2_single" and not r.get("error"):
                ev[(r["claim_id"], r["model"].split(":")[1])] = norm(r["parsed_verdict"])

    def cb(c, m):
        return [best[(c, m, i)] for i in (0, 1) if (c, m, i) in best]

    missing = sum(1 for c in gold for m in M for i in (0, 1) if (c, m, i) not in best)
    print(f"# Closed-book probe vs with-evidence answers ({PARSER_VERSION}; {missing} of 600 probe calls missing)\n")
    print("## Without evidence, on claims labelled supported/refuted\n")
    print("| Model | Set | Firm claims | Committed / calls | Correct | Correct when committed | Right on both repeats |")
    print("|---|---|---|---|---|---|---|")
    for m in M:
        for s in ("A", "B"):
            firm = [c for c in gold if f"_{s}-" in c and gold[c] in C]
            calls = [(c, v) for c in firm for v in cb(c, m)]
            com = [(c, v) for c, v in calls if v in C]
            cor = sum(v == gold[c] for c, v in com)
            knew = sum(1 for c in firm if len(cb(c, m)) == 2 and all(v == gold[c] for v in cb(c, m)))
            print(f"| {m} | {s} | {len(firm)} | {len(com)}/{len(calls)} | {cor} | "
                  f"{(cor / len(com)) if com else float('nan'):.0%} | {knew}/{len(firm)} |")
    print("\n## The 8 shared errors: no-evidence answers (repeat 1 | 2) -> with-evidence answer\n")
    for c in ["A-0023", "A-0039", "A-0040", "B-0007", "B-0018", "B-0048", "A-0005", "A-0044"]:
        cid = "claims_Dataset_" + c
        parts = "; ".join(f"{m}: {'|'.join(cb(cid, m))} -> {ev[(cid, m)]}" for m in M)
        print(f"- {c} (gold {gold[cid]}): {parts}")
    print("\n## Did evidence change firm no-evidence answers? (same firm answer on both repeats)\n")
    for m in M:
        base = [c for c in gold if len(cb(c, m)) == 2 and len(set(cb(c, m))) == 1 and cb(c, m)[0] in C]
        changed = [c for c in base if ev[(c, m)] != cb(c, m)[0]]
        better = sum(ev[(c, m)] == gold[c] and cb(c, m)[0] != gold[c] for c in changed)
        worse = sum(ev[(c, m)] != gold[c] and cb(c, m)[0] == gold[c] for c in changed)
        print(f"- {m}: {len(base)} firm without evidence; kept {len(base) - len(changed)}; changed {len(changed)} "
              f"(to right {better}, to wrong {worse})")
    print("\n## Firm answers on claims labelled NEE that equal the model's consistent no-evidence answer\n")
    for m in M:
        com = [c for c in gold if gold[c] == N and ev[(c, m)] in C]
        same = [c for c in com if len(cb(c, m)) == 2 and len(set(cb(c, m))) == 1 and cb(c, m)[0] == ev[(c, m)]]
        print(f"- {m}: {len(com)} firm answers on NEE-labelled claims; {len(same)} match its no-evidence answer "
              f"{[x.replace('claims_Dataset_', '') for x in same]}")


if __name__ == "__main__":
    main()
