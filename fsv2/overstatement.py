"""Overstatement analysis, as fixed in fsv2/OVERSTATEMENT_PLAN.md. Offline; no model calls.

Overstatement (OS) = a firm verdict (supported/refuted) on a claim whose gold label is
not_enough_evidence. Three questions: which kinds occur (coding sheet), under what
evidence conditions (pack features vs how often systems overstate), and whether output
constraints reduce them (paired prompt contrasts), with the cost in understatement.

    python -m fsv2.overstatement \
        --results study/results/factshield_v2p2_dev.jsonl study/results/factshield_v2p2_test.jsonl \
                  study/results/factshield_v2p2_new.jsonl \
        --plain study/responses/main.jsonl --claude-v2 study/results/fsv2_r1_claude.jsonl \
        --packs study/packs

Writes overstatement_report.txt (UTF-8), study/results/overstatement_summary.json,
study/results/overstatement_features.csv and study/results/overstatement_coding.csv.
Plain answers are re-parsed from response_text with the current study parser.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from . import evidence as ev  # noqa: E402
from . import fourth_model as fm  # noqa: E402

HERE = Path(__file__).resolve().parent
NEE, SUP, REF = "not_enough_evidence", "supported", "refuted"
FIRM = (SUP, REF)
UNP = "unparseable"
PLAIN_MODELS = ("openai", "claude")
PROMPTS = ("format", "free", "citation")
RUNS = (0, 1)
CUE = re.compile(r"\b(said|says|according to|reported|reportedly|alleged|allegedly|claimed|claims|"
                 r"announced|told|stated)\b", re.I)


# ----------------------------------------------------------------- loading

def _v(x):
    x = fm.norm(x)
    return x if x in (SUP, REF, NEE) else UNP


def load_plain(path, models=("openai", "claude", "gemini")):
    """{(model, prompt, run): {claim: verdict}}, {(model, prompt, run): {claim: text}}, parser.
    Last successful row per key wins (same rule as study.score)."""
    from study import prompts
    latest = {}
    for r in fm.read_jsonl(path):
        if r.get("kind") != "main" or r.get("live_search") or r.get("error"):
            continue
        if r.get("model") not in models:
            continue
        latest[(r["model"], r["condition"], r["run_index"], r["claim_id"])] = r
    verdicts, texts = defaultdict(dict), defaultdict(dict)
    for (m, cond, run, cid), r in latest.items():
        p = prompts.parse(r.get("response_text"), cond)
        verdicts[(m, cond, run)][cid] = _v(p["parsed_verdict"])
        texts[(m, cond, run)][cid] = r.get("response_text") or ""
    return verdicts, texts, getattr(prompts, "PARSER_VERSION", "unknown")


def load_filters(paths, version=fm.FROZEN_VERSION):
    """FactShield as run, unanimity without the citation check, majority 2/3 (round 1)."""
    out = {"FactShield (unanimity + check)": {}, "unanimity, no check": {}, "majority 2/3": {}}
    for p in paths:
        for r in fm.read_jsonl(p):
            if r.get("error") or r.get("condition") != "fs2":
                continue
            mv = str(r.get("model_version") or "")
            if not mv.startswith(version + "/") or "STUB" in mv:
                continue
            fv = r.get("factshield_v2") or {}
            ab = fv.get("ablations") or {}
            cid = r["claim_id"]
            out["FactShield (unanimity + check)"][cid] = _v(fv.get("final"))
            out["unanimity, no check"][cid] = _v(ab.get("fs2_no_check"))
            out["majority 2/3"][cid] = _v(ab.get("fs2_majority_r1"))
    return out


def load_codes(path) -> dict:
    """Coding sheet -> {claim: {code, secondary, self_source, checked}}. A filled
    FINAL_code column (after Nikita's check) wins over the draft primary_code."""
    out = {}
    if not path or not Path(path).exists():
        return out
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            final = (r.get("FINAL_code") or "").strip().upper()
            draft = (r.get("primary_code") or r.get("primary_code (A/E/S/M/I/N/L/O)") or "").strip().upper()
            code = final or draft
            if not code:
                continue
            out[r["claim_id"]] = {"code": code, "secondary": (r.get("secondary_code") or "").strip().upper(),
                                  "self_source": (r.get("self_source") or r.get("self_source (y/n)") or "").strip().lower(),
                                  "checked": bool(final), "draft": draft,
                                  "nikita_check": (r.get("NIKITA_check (agree/disagree)") or r.get("NIKITA_check") or "").strip().lower(),
                                  "evidence_quote": r.get("evidence_quote") or r.get("evidence_quote (pid: text)") or "",
                                  "draft_note": r.get("draft_note") or ""}
    return out


def load_pack(packs_dir, cid):
    f = Path(packs_dir) / f"{cid}.json"
    return json.load(open(f, encoding="utf-8")) if f.exists() else None


# ----------------------------------------------------------------- features (fixed in the plan)

def features(pack: dict) -> dict:
    from study import passages as P
    claim = pack.get("claim") or ""
    ps = pack.get("passages") or []
    items = ev.prepare(pack)
    n = len(ps)
    copies = sum(1 for it in items if it.same_text_as or it.also_published_by or it.same_org_as)
    cues = sum(1 for p in ps if CUE.search(p.get("text") or ""))
    ratings = [it.rating for it in items if it.rating is not None]
    dated = sum(1 for p in ps if p.get("published_date"))
    return {
        "n_passages": n,
        "n_orgs": len({ev.organisation(ev.host(p.get("source_domain") or "")) for p in ps}),
        "copy_share": round(copies / n, 3) if n else 0.0,
        "attrib_share": round(cues / n, 3) if n else 0.0,
        "max_overlap": round(max((P.overlap(claim, p.get("text") or "") for p in ps), default=0.0), 3),
        "mean_rating": round(sum(ratings) / len(ratings), 3) if ratings else None,
        "dated_share": round(dated / n, 3) if n else 0.0,
    }


FEATURES = ("n_passages", "n_orgs", "copy_share", "attrib_share", "max_overlap", "mean_rating", "dated_share")


# ----------------------------------------------------------------- statistics

def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = sum((x - ma) ** 2 for x in a) ** 0.5
    sb = sum((y - mb) ** 2 for y in b) ** 0.5
    if sa == 0 or sb == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def spearman_perm(x, y, reps=10000, seed=0):
    rx, ry = ranks(x), ranks(y)
    obs = pearson(rx, ry)
    rng = random.Random(seed)
    ry2 = list(ry)
    hits = 0
    for _ in range(reps):
        rng.shuffle(ry2)
        if abs(pearson(rx, ry2)) >= abs(obs) - 1e-12:
            hits += 1
    return round(obs, 3), (hits + 1) / (reps + 1)


def mean_diff_perm(vals_a, vals_b, reps=10000, seed=0):
    rng = random.Random(seed)
    obs = sum(vals_a) / len(vals_a) - sum(vals_b) / len(vals_b)
    pool = list(vals_a) + list(vals_b)
    na = len(vals_a)
    hits = 0
    for _ in range(reps):
        rng.shuffle(pool)
        d = sum(pool[:na]) / na - sum(pool[na:]) / (len(pool) - na)
        if abs(d) >= abs(obs) - 1e-12:
            hits += 1
    return round(obs, 3), (hits + 1) / (reps + 1)


# ----------------------------------------------------------------- counting

def os_counts(pred, gold, ids):
    nee = [c for c in ids if gold[c] == NEE and c in pred]
    firm_gold = [c for c in ids if gold[c] in FIRM and c in pred]
    os_s = [c for c in nee if pred[c] == SUP]
    os_r = [c for c in nee if pred[c] == REF]
    us = [c for c in firm_gold if pred[c] == NEE]
    wd = [c for c in firm_gold if pred[c] in FIRM and pred[c] != gold[c]]
    unp = [c for c in ids if c in pred and pred[c] == UNP]
    correct = sum(1 for c in ids if c in pred and pred[c] == gold[c])
    return {"n": sum(1 for c in ids if c in pred), "n_nee": len(nee), "os": len(os_s) + len(os_r),
            "os_supported": len(os_s), "os_refuted": len(os_r),
            "os_rate": round((len(os_s) + len(os_r)) / len(nee), 3) if nee else None,
            "us": len(us), "n_firm_gold": len(firm_gold), "wd": len(wd), "correct": correct,
            "unparseable": len(unp), "os_ids": sorted(os_s + os_r)}


def os_flag(pred, gold, c):
    return c in pred and gold[c] == NEE and pred[c] in FIRM


def contrast(name, A, B, gold, ids):
    """Exact McNemar: OS on NEE claims (A_only = A overstates, B does not), plus the costs."""
    nee = [c for c in ids if gold[c] == NEE and c in A and c in B]
    a_only = sum(1 for c in nee if os_flag(A, gold, c) and not os_flag(B, gold, c))
    b_only = sum(1 for c in nee if os_flag(B, gold, c) and not os_flag(A, gold, c))
    firm = [c for c in ids if gold[c] in FIRM and c in A and c in B]
    us_a = sum(1 for c in firm if A[c] == NEE)
    us_b = sum(1 for c in firm if B[c] == NEE)
    both = [c for c in ids if c in A and c in B]
    return {"test": name, "n_nee": len(nee), "os_A": sum(os_flag(A, gold, c) for c in nee),
            "os_B": sum(os_flag(B, gold, c) for c in nee), "A_only": a_only, "B_only": b_only,
            "p_exact": fm.mcnemar_exact(a_only, b_only),
            "us_A": us_a, "us_B": us_b,
            "correct_A": sum(1 for c in both if A[c] == gold[c]),
            "correct_B": sum(1 for c in both if B[c] == gold[c])}


def tertiles(vals):
    s = sorted(v for v in vals if v is not None)
    if not s:
        return None, None
    return s[len(s) // 3], s[(2 * len(s)) // 3]


# ----------------------------------------------------------------- main analysis

def analyse(a) -> dict:
    gold, gold_src = fm.load_gold(a.gold)
    saved = fm.load_saved(a.results)
    ids = sorted(saved["final"])
    filters = load_filters(a.results)
    plain, texts, parser = load_plain(a.plain)
    claude_v2, claude_reason = fm.load_fourth(a.claude_v2, "claude")
    v2 = {m: {c: _v(saved["single"].get((c, m))) for c in ids if saved["single"].get((c, m))}
          for m in ("openai", "gemini", "groq")}
    v2["claude"] = {c: _v(v) for c, v in claude_v2.items() if v}
    excl = fm.read_ids(a.exclude)

    conds = {}
    for m in PLAIN_MODELS:
        for pr in PROMPTS:
            for run in RUNS:
                conds[f"plain {pr} {m} r{run}"] = plain.get((m, pr, run), {})
    for m in ("openai", "gemini", "groq", "claude"):
        conds[f"v2 single {m}"] = v2[m]
    conds.update(filters)
    gem_plain = {f"plain {pr} gemini r{run}": plain.get(("gemini", pr, run), {})
                 for pr in PROMPTS for run in RUNS}
    pooled = [f"plain {pr} {m} r{run}" for m in PLAIN_MODELS for pr in PROMPTS for run in RUNS] + \
             ["v2 single openai", "v2 single groq", "v2 single claude"]

    lines = []

    def p(*x):
        s = " ".join(str(i) for i in x)
        lines.append(s)
        print(s)

    rep = {"gold_source": gold_src, "parser_version": parser, "n_claims": len(ids),
           "gold_counts": dict(Counter(gold[c] for c in ids))}
    p(f"gold: {gold_src}  ({dict(Counter(gold[c] for c in ids))})\nparser: {parser}\nclaims: {len(ids)}")

    missing = {k: len([c for c in ids if c not in v]) for k, v in conds.items()}
    if any(missing.values()):
        p("WARNING missing answers:", {k: v for k, v in missing.items() if v})
    rep["missing"] = missing

    # ---- per-condition counts
    for sname, sub in (("all", ids), (f"without {len(excl & set(ids))} audited-label claims",
                                      [c for c in ids if c not in excl])):
        p(f"\n=== Overstatement by condition: {sname} ({sum(gold[c]==NEE for c in sub)} NEE-gold claims) ===")
        p(f"{'condition':<34}{'OS':>4}{'OS-S':>6}{'OS-R':>6}{'OS rate':>9}{'US':>5}{'WD':>4}{'correct':>9}{'unp':>5}")
        rows = {}
        for k, pred in list(conds.items()) + list(gem_plain.items()):
            if not pred:
                continue
            s = os_counts(pred, gold, sub)
            rows[k] = s
            tag = "  (partial)" if k.startswith("plain") and "gemini" in k else ""
            p(f"{k:<34}{s['os']:>4}{s['os_supported']:>6}{s['os_refuted']:>6}{str(s['os_rate']):>9}"
              f"{s['us']:>5}{s['wd']:>4}{s['correct']:>9}{s['unparseable']:>5}  n={s['n']}{tag}")
        rep.setdefault("by_condition", {})[sname] = rows

    # ---- run-to-run change in OS (plain)
    p("\n=== Run-to-run change in overstatement (plain, NEE claims) ===")
    flips = {}
    for m in PLAIN_MODELS:
        for pr in PROMPTS:
            A, B = conds[f"plain {pr} {m} r0"], conds[f"plain {pr} {m} r1"]
            nee = [c for c in ids if gold[c] == NEE and c in A and c in B]
            ch = sum(1 for c in nee if os_flag(A, gold, c) != os_flag(B, gold, c))
            flips[f"{pr} {m}"] = {"nee": len(nee), "os_r0": sum(os_flag(A, gold, c) for c in nee),
                                  "os_r1": sum(os_flag(B, gold, c) for c in nee), "changed": ch}
            p(f"  {pr:<9}{m:<7} OS r0 {flips[f'{pr} {m}']['os_r0']:>3}  r1 {flips[f'{pr} {m}']['os_r1']:>3}"
              f"  claims whose OS status changed {ch}")
    rep["os_run_change"] = flips

    # ---- Q3 output constraints
    p("\n=== Q3 Output constraints [test] run 0, OS on NEE claims, exact McNemar, Holm over 4 ===")
    p("(A_only = A overstates where B does not; costs: US = NEE on firm-gold claims, correct of 100)")
    prim = []
    for m in PLAIN_MODELS:
        prim.append(contrast(f"citation vs free {m}", conds[f"plain citation {m} r0"],
                             conds[f"plain free {m} r0"], gold, ids))
    for m in PLAIN_MODELS:
        prim.append(contrast(f"format vs free {m}", conds[f"plain format {m} r0"],
                             conds[f"plain free {m} r0"], gold, ids))
    for t, q in zip(prim, fm.holm([t["p_exact"] for t in prim])):
        t["p_holm"] = q

    def show(t):
        hol = f"  Holm {t['p_holm']:.3f}" if "p_holm" in t else ""
        p(f"  {t['test']:<40} OS {t['os_A']:>2} vs {t['os_B']:>2}  (A_only {t['A_only']}, B_only {t['B_only']})"
          f"  p {t['p_exact']:.3f}{hol}  | US {t['us_A']} vs {t['us_B']}  correct {t['correct_A']} vs {t['correct_B']}")
    for t in prim:
        show(t)
    p("replication on run 1 [desc]:")
    repl = []
    for m in PLAIN_MODELS:
        repl.append(contrast(f"citation vs free {m} (r1)", conds[f"plain citation {m} r1"],
                             conds[f"plain free {m} r1"], gold, ids))
    for m in PLAIN_MODELS:
        repl.append(contrast(f"format vs free {m} (r1)", conds[f"plain format {m} r1"],
                             conds[f"plain free {m} r1"], gold, ids))
    for t in repl:
        show(t)
    p("secondary [desc] (uncorrected p):")
    sec = [contrast("v2 single vs plain format openai", conds["v2 single openai"],
                    conds["plain format openai r0"], gold, ids),
           contrast("v2 single vs plain format claude", conds["v2 single claude"],
                    conds["plain format claude r0"], gold, ids),
           contrast("majority 2/3 vs v2 single openai", conds["majority 2/3"],
                    conds["v2 single openai"], gold, ids),
           contrast("unanimity (no check) vs v2 single openai", conds["unanimity, no check"],
                    conds["v2 single openai"], gold, ids),
           contrast("unanimity + check vs unanimity, no check", conds["FactShield (unanimity + check)"],
                    conds["unanimity, no check"], gold, ids),
           contrast("FactShield vs plain format openai", conds["FactShield (unanimity + check)"],
                    conds["plain format openai r0"], gold, ids),
           contrast("FactShield vs plain format claude", conds["FactShield (unanimity + check)"],
                    conds["plain format claude r0"], gold, ids)]
    for t in sec:
        show(t)
    rep["q3"] = {"primary": prim, "replication_run1": repl, "secondary": sec}

    # ---- Q2 evidence conditions
    p("\n=== Q2 Evidence conditions ===")
    feats = {}
    for c in ids:
        pk = load_pack(a.packs, c)
        if pk is None:
            continue
        f = features(pk)
        f["set"] = "A" if "_A-" in c else "B"
        f["claim"] = pk.get("claim")
        feats[c] = f
    nee_ids = [c for c in ids if gold[c] == NEE and c in feats]
    if len(feats) < len(ids):
        p(f"WARNING packs found for {len(feats)} of {len(ids)} claims")
    for c in ids:
        if c not in feats:
            continue
        fl = [k for k in pooled if os_flag(conds[k], gold, c)]
        den = sum(1 for k in pooled if c in conds[k])
        feats[c]["os_n"] = len(fl)
        feats[c]["os_den"] = den
        feats[c]["os_share"] = round(len(fl) / den, 3) if den and gold[c] == NEE else None
        feats[c]["os_supported_n"] = sum(1 for k in pooled if c in conds[k] and gold[c] == NEE
                                         and conds[k][c] == SUP)
        feats[c]["os_refuted_n"] = sum(1 for k in pooled if c in conds[k] and gold[c] == NEE
                                       and conds[k][c] == REF)
    y = [feats[c]["os_share"] for c in nee_ids]
    p(f"NEE claims with packs: {len(nee_ids)}; os_share over {len(pooled)} pooled system-runs; "
      f"mean {sum(y)/len(y):.3f}; claims with os_share 0: {sum(v == 0 for v in y)}; >= 0.5: {sum(v >= 0.5 for v in y)}")
    tests = []
    for fe in FEATURES:
        pairs = [(feats[c][fe], feats[c]["os_share"]) for c in nee_ids if feats[c][fe] is not None]
        rho, pv = spearman_perm([u for u, _ in pairs], [v for _, v in pairs])
        tests.append({"feature": fe, "n": len(pairs), "rho": rho, "p_perm": round(pv, 4)})
    for t, q in zip(tests, fm.holm([t["p_perm"] for t in tests])):
        t["p_holm"] = round(q, 4)
    p("[test] Spearman rho(feature, os_share), permutation p (10,000), Holm over 7:")
    for t in tests:
        p(f"  {t['feature']:<14} n {t['n']:>3}  rho {t['rho']:>6}  p {t['p_perm']:.4f}  Holm {t['p_holm']:.4f}")
    a_vals = [feats[c]["os_share"] for c in nee_ids if feats[c]["set"] == "A"]
    b_vals = [feats[c]["os_share"] for c in nee_ids if feats[c]["set"] == "B"]
    d, pv = mean_diff_perm(a_vals, b_vals)
    p(f"set A vs B (outside Holm): mean os_share A {sum(a_vals)/len(a_vals):.3f} (n {len(a_vals)}) vs "
      f"B {sum(b_vals)/len(b_vals):.3f} (n {len(b_vals)}); diff {d}, permutation p {pv:.4f}")
    p("[desc] tertiles (NEE claims): mean os_share, and OS-S / OS-R instances")
    tert = {}
    for fe in ("copy_share", "attrib_share", "max_overlap"):
        lo, hi = tertiles([feats[c][fe] for c in nee_ids])
        groups = {"low": [], "mid": [], "high": []}
        for c in nee_ids:
            v = feats[c][fe]
            g = "low" if v < lo else ("high" if v >= hi else "mid")
            groups[g].append(c)
        tert[fe] = {}
        out = []
        for g, cs in groups.items():
            if not cs:
                continue
            ms = sum(feats[c]["os_share"] for c in cs) / len(cs)
            s_n = sum(feats[c]["os_supported_n"] for c in cs)
            r_n = sum(feats[c]["os_refuted_n"] for c in cs)
            tert[fe][g] = {"n": len(cs), "mean_os_share": round(ms, 3), "os_s": s_n, "os_r": r_n}
            out.append(f"{g} n {len(cs)} mean {ms:.2f} S/R {s_n}/{r_n}")
        p(f"  {fe:<13} cuts {lo}/{hi}: " + " | ".join(out))
    rep["q2"] = {"pooled": pooled, "tests": tests, "set_A_vs_B": {"diff": d, "p": pv},
                 "tertiles": tert}

    # ---- Q1 kinds: the claims to code
    p("\n=== Q1 Claims to code (NEE claims overstated by at least one non-Gemini-plain system) ===")
    q1_conds = [k for k in conds if conds[k]]
    to_code = []
    for c in nee_ids:
        who = {k: conds[k][c] for k in q1_conds if os_flag(conds[k], gold, c)}
        if who:
            to_code.append((c, who))
    to_code.sort(key=lambda x: -len(x[1]))
    p(f"{len(to_code)} distinct claims; by number of system-runs that overstated:")
    for c, who in to_code:
        dirs = Counter(who.values())
        p(f"  {c:<24} {len(who):>2} system-runs  S {dirs.get(SUP, 0):>2}  R {dirs.get(REF, 0):>2}"
          f"  pooled os_share {feats[c]['os_share']}  {'(audited label)' if c in excl else ''}")
    rep["q1_claims"] = [{"claim_id": c, "n_systems": len(w), "verdicts": w} for c, w in to_code]

    # ---- Q1 continued: summaries of the codes (draft until checked), and the L sensitivity
    codes = load_codes(a.codes) if a.codes else {}
    if codes:
        status = "CHECKED" if all(v["checked"] for v in codes.values()) else "DRAFT (not yet checked by Nikita)"
        p(f"\n=== Q1 Kinds: codes from {a.codes}: {status} ===")
        missing_codes = [c for c, _ in to_code if c not in codes]
        if missing_codes:
            p(f"WARNING {len(missing_codes)} claims to code have no code: {missing_codes}")
        coded = [(c, who) for c, who in to_code if c in codes]
        chk = [c for c, _ in coded if codes[c]["checked"]]
        if chk:
            same = [c for c in chk if codes[c]["draft"] and codes[c]["draft"] == codes[c]["code"]]
            changed = [c for c in chk if codes[c]["draft"] and codes[c]["draft"] != codes[c]["code"]]
            said = Counter(codes[c]["nikita_check"] or "blank" for c in chk)
            p(f"check: {len(chk)} of {len(coded)} claims have FINAL_code; draft primary code kept on {len(same)}, "
              f"changed on {len(changed)} ({len(same)/len(chk):.0%} agreement on the primary code); "
              f"NIKITA_check column: {dict(said)}")
            for c in changed:
                p(f"  changed {c}: {codes[c]['draft']} -> {codes[c]['code']}")
            rep["q1_check"] = {"checked": len(chk), "kept": len(same), "changed": {c: [codes[c]["draft"], codes[c]["code"]] for c in changed},
                               "nikita_check": dict(said)}
        by_claim = Counter(codes[c]["code"] for c, _ in coded)
        p("primary code by claim: " + ", ".join(f"{k} {v}" for k, v in sorted(by_claim.items(), key=lambda x: -x[1])))
        inst = Counter()
        inst_dir = Counter()
        for c, who in coded:
            for k, v in who.items():
                inst[codes[c]["code"]] += 1
                inst_dir[(codes[c]["code"], v)] += 1
        tot = sum(inst.values())
        p(f"by system-run instance ({tot} OS instances across all non-Gemini-plain conditions):")
        for k, n in sorted(inst.items(), key=lambda x: -x[1]):
            p(f"  {k}: {n} ({n/tot:.0%})  as supported {inst_dir[(k, SUP)]}, as refuted {inst_dir[(k, REF)]}")
        fams = {"plain reason-first (format)": lambda k: k.startswith("plain format"),
                "plain verdict-first (free+citation)": lambda k: k.startswith("plain free") or k.startswith("plain citation"),
                "FactShield-prompt singles": lambda k: k.startswith("v2 single"),
                "panel filters": lambda k: k in filters}
        p("instances by family x model x code [desc]:")
        fam_tab = {}
        for fam, f in fams.items():
            for m in ("openai", "claude", "gemini", "groq", "panel"):
                ks = [k for k in q1_conds if f(k) and ((m == "panel" and k in filters) or (m != "panel" and m in k))]
                if not ks:
                    continue
                cnt = Counter()
                for c, who in coded:
                    for k in ks:
                        if k in who:
                            cnt[codes[c]["code"]] += 1
                if cnt:
                    fam_tab[f"{fam} | {m}"] = dict(cnt)
                    p(f"  {fam:<38} {m:<7} runs {len(ks)}: " +
                      ", ".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda x: -x[1])))
        ss = Counter(codes[c]["self_source"] for c, _ in coded)
        p(f"self-source (only support is the subject's own statement): {dict(ss)}")
        l_ids = {c for c, v in codes.items() if v["code"] == "L"}
        p(f"L-coded (label arguable) claims: {len(l_ids)} of {len(coded)} "
          f"({'more than a third: label difficulty dominates' if len(l_ids) > len(coded) / 3 else 'not more than a third'})")
        sens_ids = [c for c in ids if c not in l_ids]
        p("sensitivity without L-coded claims: primary Q3 contrasts")
        sens = []
        for t in prim:
            name = t["test"]
            kind, _, _, m = name.split()
            A = conds[f"plain {kind} {m} r0"]
            B = conds[f"plain free {m} r0"]
            sens.append(contrast(name + " (no L)", A, B, gold, sens_ids))
        for t in sens:
            show(t)
        sens_nee = [c for c in nee_ids if c not in l_ids]
        rho, pv = spearman_perm([feats[c]["max_overlap"] for c in sens_nee], [feats[c]["os_share"] for c in sens_nee])
        p(f"sensitivity without L-coded claims: rho(max_overlap, os_share) {rho}, p {pv:.4f} (n {len(sens_nee)})")
        excl_nee = [c for c in nee_ids if c not in excl]
        rho2, pv2 = spearman_perm([feats[c]["max_overlap"] for c in excl_nee], [feats[c]["os_share"] for c in excl_nee])
        p(f"sensitivity without audited-label claims: rho(max_overlap, os_share) {rho2}, p {pv2:.4f} (n {len(excl_nee)})")
        by_code_overlap = defaultdict(list)
        for c, _ in coded:
            by_code_overlap[codes[c]["code"]].append(feats[c]["max_overlap"])
        uncoded = [feats[c]["max_overlap"] for c in nee_ids if c not in dict(coded)]
        p("[desc] mean max_overlap by code: " + ", ".join(
            f"{k} {sum(v)/len(v):.2f} (n {len(v)})" for k, v in sorted(by_code_overlap.items())) +
          (f", never overstated {sum(uncoded)/len(uncoded):.2f} (n {len(uncoded)})" if uncoded else ""))
        rep["q1_codes"] = {"status": status, "by_claim": dict(by_claim), "by_instance": dict(inst),
                           "by_instance_direction": {f"{k}|{d}": n for (k, d), n in inst_dir.items()},
                           "by_family": fam_tab, "self_source": dict(ss), "L_claims": sorted(l_ids),
                           "sensitivity_no_L": {"q3": sens, "rho_max_overlap": rho, "p": pv},
                           "sensitivity_no_audited": {"rho_max_overlap": rho2, "p": pv2}}

    # ---- files
    if a.features_csv:
        with open(a.features_csv, "w", newline="", encoding="utf-8") as fh:
            cols = ["claim_id", "set", "gold", *FEATURES, "os_n", "os_den", "os_share",
                    "os_supported_n", "os_refuted_n", "audited_label", "claim"]
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for c in ids:
                if c in feats:
                    w.writerow({"claim_id": c, "gold": gold[c], "audited_label": c in excl,
                                **{k: feats[c].get(k) for k in cols if k not in ("claim_id", "gold", "audited_label")}})
    if a.coding_csv and a.codes and Path(a.coding_csv).resolve() == Path(a.codes).resolve():
        p(f"coding sheet not rewritten: {a.coding_csv} is also the --codes input (your checks are kept)")
    elif a.coding_csv:
        with open(a.coding_csv, "w", newline="", encoding="utf-8") as fh:
            cols = ["claim_id", "claim", "gold", "n_systems_overstating", "directions", "systems",
                    "self_source", "primary_code", "secondary_code", "evidence_quote", "draft_note",
                    "NIKITA_check (agree/disagree)", "FINAL_code", "NIKITA_note",
                    "factshield_reasons", "plain_excerpt", "passages"]
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for c, who in to_code:
                pk = load_pack(a.packs, c) or {}
                reasons = " || ".join(f"{m}: {saved['reasons'].get((c, m))}" for m in ("openai", "gemini", "groq")
                                      if saved["reasons"].get((c, m)))
                if claude_reason.get(c):
                    reasons += f" || claude: {claude_reason[c]}"
                ex = []
                for k in who:
                    if k.startswith("plain"):
                        _, pr, m, run = k.split()
                        t = texts.get((m, pr, int(run[1])), {}).get(c, "")
                        ex.append(f"[{k}] " + re.sub(r"\s+", " ", t)[:500])
                    if len(ex) >= 2:
                        break
                pas = " || ".join(f"{q.get('pid')} {q.get('source_domain')}: "
                                  + re.sub(r"\s+", " ", q.get("text") or "")[:350]
                                  for q in pk.get("passages", []))
                cd = codes.get(c, {})
                w.writerow({"claim_id": c, "claim": pk.get("claim"), "gold": gold[c],
                            "self_source": cd.get("self_source", ""), "primary_code": cd.get("code", ""),
                            "secondary_code": cd.get("secondary", ""), "evidence_quote": cd.get("evidence_quote", ""),
                            "draft_note": cd.get("draft_note", ""),
                            "n_systems_overstating": len(who),
                            "directions": dict(Counter(who.values())), "systems": "; ".join(who),
                            "factshield_reasons": reasons, "plain_excerpt": " ## ".join(ex), "passages": pas})
    if a.summary:
        Path(a.summary).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        p(f"\nsummary JSON: {a.summary}")
    if a.report:
        Path(a.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rep


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", nargs="+", required=True, help="saved FactShield v2 jsonl files (dev, test, new)")
    ap.add_argument("--plain", default="study/responses/main.jsonl")
    ap.add_argument("--claude-v2", default="study/results/fsv2_r1_claude.jsonl")
    ap.add_argument("--packs", default="study/packs")
    ap.add_argument("--gold", help="JSON claim_id -> label (default: CSVs + label_overrides)")
    ap.add_argument("--exclude", default=str(HERE / "claude_influenced_ids.txt"))
    ap.add_argument("--codes", default=str(HERE / "overstatement_codes_draft.csv"),
                    help="coding sheet; a filled FINAL_code column overrides the draft code")
    ap.add_argument("--summary", default="study/results/overstatement_summary.json")
    ap.add_argument("--features-csv", default="study/results/overstatement_features.csv")
    ap.add_argument("--coding-csv", default="study/results/overstatement_coding.csv")
    ap.add_argument("--report", default="overstatement_report.txt")
    a = ap.parse_args(argv)
    analyse(a)
    print(f"report saved: {a.report}")


if __name__ == "__main__":
    main()
