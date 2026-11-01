"""Error overlap between the three single models on the dev run.

usage: python -m fsv2.overlap RESULTS_JSONL [GOLD_JSON]

Without GOLD_JSON the gold labels are read the same way the study reads them:
the claim CSVs in study/claims (or STUDY_CLAIMS_DIR) with
claims.csv.label_overrides from study/config.yaml applied. GOLD_JSON, if
given, maps claim_id -> gold label instead.

"Single" = each model's round-1 verdict on the same evidence (fs2_single rows).
conflicting_evidence is scored as not_enough_evidence. Written 2026-10-26 for
the dev run; run it on the test results once they are scored.
"""
import itertools
import json
import math
import sys

NEE = "not_enough_evidence"


def norm(v):
    if v is None:
        return None
    return NEE if v == "conflicting_evidence" else v


rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]


def load_gold():
    if len(sys.argv) > 2:
        return json.load(open(sys.argv[2], encoding="utf-8")), sys.argv[2]
    from study.claimsets import claims_folder, load_csv_claims
    from study.common import _apply_label_overrides, config
    folder = claims_folder()
    claims, _ = load_csv_claims(folder)
    _apply_label_overrides(claims)
    n_ov = len(config()["claims"]["csv"].get("label_overrides") or {})
    return {c["id"]: c["label"] for c in claims}, f"{folder} + {n_ov} label overrides"


_g, gold_src = load_gold()
gold = {k: norm(v) for k, v in _g.items()}
models = ["openai", "gemini", "groq"]

single = {}   # (claim, model) -> verdict
panel = {}
for r in rows:
    if r["condition"] == "fs2_single":
        single[(r["claim_id"], r["model"].split(":")[1])] = norm(r["parsed_verdict"])
    if r["condition"] == "fs2" and not r.get("error"):
        panel[r["claim_id"]] = norm(r["factshield_v2"]["final"])
ids = sorted({c for c, _ in single})
no_gold = [c for c in ids if c not in gold]
if no_gold:
    sys.exit(f"no gold label for {no_gold}")
n = len(ids)
print(f"gold labels from: {gold_src}")
print(f"claims: {n}; gold labels: " + ", ".join(
    f"{l} {sum(gold[i] == l for i in ids)}" for l in ("supported", "refuted", NEE)))
missing = [(c, m) for c in ids for m in models if single.get((c, m)) is None]
print("missing/unparsed single verdicts:", missing or "none")


def wrong(c, m):
    return single[(c, m)] != gold[c]


def conf_wrong(c, m):
    return single[(c, m)] in ("supported", "refuted") and single[(c, m)] != gold[c]


print("\n== Per model (round-1 verdict, same evidence) ==")
for m in models:
    w = sum(wrong(c, m) for c in ids)
    cw = sum(conf_wrong(c, m) for c in ids)
    commit = sum(single[(c, m)] in ("supported", "refuted") for c in ids)
    print(f"{m:7s} correct {n - w}/{n}  wrong {w}  confidently wrong {cw}  committed {commit}")
pw = sum(panel[c] in ("supported", "refuted") and panel[c] != gold[c] for c in ids)
pc = sum(panel[c] in ("supported", "refuted") for c in ids)
print(f"panel   correct {sum(panel[c] == gold[c] for c in ids)}/{n}  confidently wrong {pw}  committed {pc}")


def phi(a, b):
    n11 = sum(x and y for x, y in zip(a, b)); n00 = sum((not x) and (not y) for x, y in zip(a, b))
    n10 = sum(x and not y for x, y in zip(a, b)); n01 = sum((not x) and y for x, y in zip(a, b))
    d = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / d if d else float("nan")


print("\n== Pairs: any wrong answer ==")
phis = []
for a, b in itertools.combinations(models, 2):
    wa = [wrong(c, a) for c in ids]; wb = [wrong(c, b) for c in ids]
    both = sum(x and y for x, y in zip(wa, wb))
    same = sum(wrong(c, a) and wrong(c, b) and single[(c, a)] == single[(c, b)] for c in ids)
    exp = sum(wa) * sum(wb) / n
    p = phi(wa, wb); phis.append(p)
    print(f"{a}+{b}: both wrong {both} (independence would give {exp:.1f}); "
          f"same wrong answer {same}; phi {p:.2f}")

print("\n== Pairs: confident errors (S/R that is wrong) ==")
for a, b in itertools.combinations(models, 2):
    both = sum(conf_wrong(c, a) and conf_wrong(c, b) for c in ids)
    same = sum(conf_wrong(c, a) and conf_wrong(c, b) and single[(c, a)] == single[(c, b)] for c in ids)
    print(f"{a}+{b}: both confidently wrong {both}; same wrong label {same}")

all3 = [c for c in ids if all(wrong(c, m) for m in models)]
all3_same = [c for c in all3 if len({single[(c, m)] for m in models}) == 1]
rates = [sum(wrong(c, m) for c in ids) / n for m in models]
print("\n== All three ==")
print(f"all three wrong: {len(all3)} (independence would give {n * math.prod(rates):.1f})")
print(f"all three the SAME wrong answer (the panel's blind spot): {len(all3_same)}")
for c in all3_same:
    print(f"   {c}: all said {single[(c, models[0])]}, gold {gold[c]}, panel final {panel[c]}")
for c in all3:
    if c not in all3_same:
        print(f"   {c}: wrong but different answers {[single[(c, m)] for m in models]}, gold {gold[c]}")

k = len(models)
rho = sum(phis) / len(phis)
print(f"\nmean pairwise phi (error correlation): {rho:.2f}")
print(f"effective number of independent models (Kish, k/(1+(k-1)*rho)): {k / (1 + (k - 1) * rho):.2f} of {k}")

print("\n== Panel confident errors ==")
for c in ids:
    if panel[c] in ("supported", "refuted") and panel[c] != gold[c]:
        print(f"   {c}: panel {panel[c]}, gold {gold[c]}, singles {[single[(c, m)] for m in models]}")
print("\n== Per claim ==")
print("claim".ljust(24), "gold".ljust(20), *(m.ljust(20) for m in models), "panel")
for c in ids:
    print(c.replace("claims_Dataset_", "").ljust(24), gold[c].ljust(20),
          *((single[(c, m)] or "-").ljust(20) for m in models), panel[c])

# bootstrap the uncertainty of the effective number (claims resampled with replacement)
import random
random.seed(0)
vals = []
for _ in range(5000):
    s = [random.choice(ids) for _ in ids]
    ps = []
    for a, b in itertools.combinations(models, 2):
        ps.append(phi([wrong(c, a) for c in s], [wrong(c, b) for c in s]))
    if any(math.isnan(p) for p in ps):
        continue
    r = sum(ps) / len(ps)
    vals.append(k / (1 + (k - 1) * r))
vals.sort()
print(f"\nbootstrap 95% interval for effective number: {vals[int(.025*len(vals))]:.2f} to {vals[int(.975*len(vals))]:.2f} ({len(vals)} resamples)")
