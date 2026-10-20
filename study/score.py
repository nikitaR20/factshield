"""Automatic measures over the response rows.

    python -m study.score                      # responses/main.jsonl
    python -m study.score --responses study/responses/live_search.jsonl --tag live

Writes results/scored[_tag].csv (one row per call), summary[_tag].csv and
change[_tag].csv, each with a .meta.json sidecar.

Denominators (stated once, applied everywhere):
  * API errors are excluded from every measure and reported as a count. A
    resumed run keeps the last successful row per call.
  * UNPARSEABLE responses stay in every denominator. They are neither committed
    nor abstentions, and in macro-F1 they are a wrong prediction for the gold
    label. Dropping them would reward a condition for producing responses the
    parser cannot read.
  * Confidently wrong = committed verdict (supported/refuted) that is wrong.
    Identical in all three conditions; the Format condition's confidence field
    is NOT used as a filter. Its rate is over all scored calls.

Citations:
  * `cited_in_pack` / `cited_fabricated` are about passage ids (P1..). With a
    frozen pack, an id that is not in the pack is unambiguously fabricated.
  * URLs are reported separately as `urls_outside_pack`. A URL that is not in
    the pack may still be a real page the model remembers, so it is "outside
    the pack", which is checkable, rather than "fabricated", which is not.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from urllib.parse import urlparse

from .common import load_claims, path, provenance, read_jsonl, write_meta
from .packs import PackError, load_pack

VERDICTS = ("supported", "refuted", "not_enough_evidence")
COMMITTED = ("supported", "refuted")


def canon_url(u: str) -> str:
    p = urlparse(u.strip())
    return f"{p.netloc.lower().removeprefix('www.')}{p.path.rstrip('/')}"


def latest_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Last successful row per call key; count keys that never succeeded."""
    ok, failed = {}, set()
    for r in rows:
        k = (r["claim_id"], r["condition"], r["model"], r["run_index"], r["live_search"])
        if r.get("error") is None:
            ok[k] = r
            failed.discard(k)
        elif k not in ok:
            failed.add(k)
    return list(ok.values()), len(failed)


def _knew_unaided() -> dict:
    p = path("results") / "knew_unaided.csv"
    if not p.exists():
        return {}
    with p.open() as fh:
        return {(r["claim_id"], r["model"]): r["knew_unaided"] for r in csv.DictReader(fh)}


def score_rows(rows: list[dict], claims: dict, packs: dict) -> list[dict]:
    knew = _knew_unaided()
    out = []
    for r in rows:
        c = claims[r["claim_id"]]
        pk = packs.get(r["claim_id"])
        gold, pred = c["label"], r["parsed_verdict"]
        pids = {p["pid"] for p in pk["passages"]} if pk else set()
        purls = {canon_url(p["url"]) for p in pk["passages"]} if pk else set()
        ids = r.get("cited_ids") or []
        urls = r.get("cited_urls") or []
        retr = c.get("retrievable")
        committed = pred in COMMITTED
        out.append({
            "claim_id": c["id"], "claim_set": c["set"], "domain": c["domain"],
            "source_id": c["source_id"], "variant": c.get("variant"),
            "surface_detectable": c.get("surface_detectable"),
            "retrievable": int(all(retr.values())) if isinstance(retr, dict) and retr else None,
            "knew_unaided": knew.get((c["id"], r["model"])),
            "model": r["model"], "model_version": r["model_version"],
            "condition": r["condition"], "run_index": r["run_index"],
            "live_search": r["live_search"], "gold": gold, "pred": pred,
            "confidence": r.get("confidence"),
            "format_ok": r.get("format_ok"),
            "committed": int(committed),
            "abstained": int(pred == "not_enough_evidence"),
            "unparseable": int(pred == "unparseable"),
            "correct": int(pred == gold),
            "confidently_wrong": int(committed and pred != gold),
            "n_cited_ids": len(ids),
            "n_cited_in_pack": sum(i in pids for i in ids),
            "n_cited_fabricated": sum(i not in pids for i in ids),
            "n_urls": len(urls),
            "n_urls_outside_pack": sum(canon_url(u) not in purls for u in urls),
            "tool_calls_observed": int(bool(r.get("tool_calls"))),
            "pack_hash": r.get("pack_hash"),
            **provenance(),
        })
    return out


def macro_f1(gold: list[str], pred: list[str]) -> float | None:
    """Over the three gold labels. An unparseable prediction is a miss for its
    gold label and a false positive for nothing."""
    labels = [l for l in VERDICTS if l in gold or l in pred]
    if not labels:
        return None
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def _ratio(a, b):
    return round(a / b, 4) if b else None


def summarise(scored: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for s in scored:
        groups[tuple(s[k] for k in group_keys)].append(s)
    out = []
    for key, ss in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        n = len(ss)
        committed = [s for s in ss if s["committed"]]
        abstained = [s for s in ss if s["abstained"]]
        nei_gold = [s for s in ss if s["gold"] == "not_enough_evidence"]
        cited = sum(s["n_cited_ids"] for s in ss)
        out.append({
            **dict(zip(group_keys, key)), "n": n,
            "unparseable_rate": _ratio(sum(s["unparseable"] for s in ss), n),
            "coverage": _ratio(len(committed), n),
            "selective_accuracy": _ratio(sum(s["correct"] for s in committed), len(committed)),
            "macro_f1": round(macro_f1([s["gold"] for s in ss], [s["pred"] for s in ss]) or 0, 4),
            "confidently_wrong_rate": _ratio(sum(s["confidently_wrong"] for s in ss), n),
            "abstention_precision": _ratio(sum(s["gold"] == "not_enough_evidence" for s in abstained),
                                           len(abstained)),
            "abstention_recall": _ratio(sum(s["abstained"] for s in nei_gold), len(nei_gold)),
            "rows_citing_any_id": sum(s["n_cited_ids"] > 0 for s in ss),
            "citations": cited,
            "citation_in_pack_rate": _ratio(sum(s["n_cited_in_pack"] for s in ss), cited),
            "citations_fabricated": sum(s["n_cited_fabricated"] for s in ss),
            "urls_outside_pack": sum(s["n_urls_outside_pack"] for s in ss),
            "rows_with_tool_calls": sum(s["tool_calls_observed"] for s in ss),
            **provenance(),
        })
    return out


def change_rates(scored: list[dict], conditions: list[str]) -> list[dict]:
    """Verdict change across conditions (same claim, model, run) and run-to-run
    change (same claim, model, condition). Unparseable counts as a verdict
    value: a response that flips from readable to unreadable did change."""
    by_cond = defaultdict(dict)
    by_run = defaultdict(dict)
    for s in scored:
        by_cond[(s["claim_id"], s["model"], s["run_index"])][s["condition"]] = s["pred"]
        by_run[(s["claim_id"], s["model"], s["condition"])][s["run_index"]] = s["pred"]

    out = []
    for m in sorted({s["model"] for s in scored}):
        full = [v for (c, mm, r), v in by_cond.items() if mm == m and len(v) == len(conditions)]
        row = {"model": m, "measure": "verdict_change_any", "pair": "all",
               "n": len(full), "rate": _ratio(sum(len(set(v.values())) > 1 for v in full), len(full))}
        out.append(row)
        for a, b in combinations(conditions, 2):
            pairs = [v for (c, mm, r), v in by_cond.items() if mm == m and a in v and b in v]
            out.append({"model": m, "measure": "verdict_change_pair", "pair": f"{a}|{b}",
                        "n": len(pairs), "rate": _ratio(sum(v[a] != v[b] for v in pairs), len(pairs))})
        for cond in conditions:
            reps = [v for (c, mm, cc), v in by_run.items() if mm == m and cc == cond and len(v) > 1]
            out.append({"model": m, "measure": "run_to_run_change", "pair": cond,
                        "n": len(reps), "rate": _ratio(sum(len(set(v.values())) > 1 for v in reps), len(reps))})
    for o in out:
        o.update(provenance())
    return out


def write_csv(p: Path, rows: list[dict], **meta) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        if rows:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    write_meta(p, n_rows=len(rows), **meta)


def run_scoring(responses: Path, tag: str = "") -> dict:
    claims = {c["id"]: c for c in load_claims()}
    raw = read_jsonl(responses)
    rows, n_failed = latest_rows(raw)
    rows = [r for r in rows if r.get("kind", "main") == "main"]
    packs = {}
    for cid in {r["claim_id"] for r in rows}:
        try:
            packs[cid] = load_pack(cid)
        except PackError as exc:
            raise SystemExit(f"cannot score {cid}: {exc}")
    scored = score_rows(rows, claims, packs)
    conds = sorted({s["condition"] for s in scored})
    sfx = f"_{tag}" if tag else ""
    res = path("results")
    summary = summarise(scored, ("model", "condition")) + summarise(scored, ("model", "condition", "claim_set"))
    changes = change_rates(scored, conds)
    write_csv(res / f"scored{sfx}.csv", scored, source=str(responses), api_errors_unresolved=n_failed)
    write_csv(res / f"summary{sfx}.csv", summary, source=str(responses))
    write_csv(res / f"change{sfx}.csv", changes, source=str(responses))
    return {"scored": scored, "summary": summary, "changes": changes, "api_errors": n_failed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    src = Path(args.responses) if args.responses else path("responses") / "main.jsonl"
    out = run_scoring(src, args.tag)
    print(f"scored {len(out['scored'])} calls; {out['api_errors']} calls never succeeded (excluded)\n")
    cols = ["n", "unparseable_rate", "coverage", "selective_accuracy", "macro_f1",
            "confidently_wrong_rate", "abstention_precision", "abstention_recall",
            "citation_in_pack_rate", "citations_fabricated", "rows_with_tool_calls"]
    print(f"{'model':<8}{'condition':<10}" + "".join(f"{c[:11]:>12}" for c in cols))
    for s in out["summary"]:
        if "claim_set" in s:
            continue
        print(f"{s['model']:<8}{s['condition']:<10}" + "".join(f"{s[c]!s:>12}" for c in cols))
    print()
    for c in out["changes"]:
        print(f"  {c['model']:<8}{c['measure']:<22}{c['pair']:<18} n={c['n']:<5} rate={c['rate']}")


if __name__ == "__main__":
    main()
