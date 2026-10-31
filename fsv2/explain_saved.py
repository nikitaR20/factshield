"""Stage 6 (checked explanations) on SAVED FactShield results — no new verdicts.

The study runs (run_packs.py) were made without --explain, so the explanation
step was never measured. This script writes and checks an explanation for
each saved claim, exactly as check_claim(..., explain=True) would after the
verdict: same prompt (explain.<prompt_version>), same explainer model
(config `explainer`), same evidence preparation, same MiniCheck rule. The
verdict, reason, samples and cited passages are read from the saved row, so
verdicts cannot change. Each pack is loaded through study.packs.load_pack
(hash-verified) and must match the pack_hash recorded in the row.

    python -m fsv2.explain_saved --results study/results/factshield_v2p2_dev.jsonl \
        study/results/factshield_v2p2_test.jsonl study/results/factshield_v2p2_new.jsonl \
        --out study/results/factshield_v2p2_explanations.jsonl
    # default: only claims FactShield answered supported/refuted; --all for every claim
    python -m fsv2.explain_saved --summary study/results/factshield_v2p2_explanations.jsonl

Resumable: claims already in --out with no error are skipped. One explainer
call per claim (Groq in the current config).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import httpx

from study.packs import load_pack

from . import checker as chk
from . import evidence as ev
from .decide import explain_verdict
from .settings import cfg


def load_saved(paths: list[str]) -> dict:
    rows = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("condition") == "fs2" and not r.get("error"):
                rows[r["claim_id"]] = r  # last successful row wins
    return rows


def summarise(path: str) -> None:
    done = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        done[r["claim_id"]] = r
    rs = [r for r in done.values() if not r["explanation"].get("error")]
    errs = [r["claim_id"] for r in done.values() if r["explanation"].get("error")]
    n_kept = sum(len(r["explanation"]["sentences"]) for r in rs)
    kinds = Counter(s["kind"] for r in rs for s in r["explanation"]["sentences"])
    dropped = [d for r in rs for d in r["explanation"]["dropped"]]
    why = Counter("no valid citation" if d["why"] == "no valid citation" else "checker rejected" for d in dropped)
    q = [r for r in rs if r["explanation"]["quotes_only"]]
    print(f"claims explained: {len(rs)} (errors: {len(errs)} {errs})")
    print(f"sentences written (kept + dropped): {n_kept + len(dropped)}")
    print(f"  kept: {n_kept} (fact {kinds['fact']}, summary {kinds['summary']}) — counted only where the "
          f"explanation survived; quotes-only claims show none")
    print(f"  dropped: {len(dropped)} — {dict(why)}")
    print(f"claims shown as quotes only (fewer than "
          f"{cfg()['explanation']['min_surviving_share']:.0%} of factual sentences survived, or no output): "
          f"{len(q)} of {len(rs)}")
    by_v = Counter((r["final"], r["explanation"]["quotes_only"]) for r in rs)
    print("by verdict (verdict, quotes_only): " + ", ".join(f"{k}: {v}" for k, v in sorted(by_v.items())))


async def run(a) -> None:
    rows = load_saved(a.results)
    todo = [r for r in rows.values()
            if a.all or r["factshield_v2"]["final"] in ("supported", "refuted")]
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if not r["explanation"].get("error"):
                done.add(r["claim_id"])
    todo = [r for r in sorted(todo, key=lambda x: x["claim_id"]) if r["claim_id"] not in done]
    print(f"{len(todo)} claims to explain ({len(done)} already done) with explainer "
          f"{cfg()['explainer']}, prompt {cfg()['prompt_version']}")
    checker = chk.load(stub=a.stub_checker)
    async with httpx.AsyncClient() as client:
        with out.open("a", encoding="utf-8") as fh:
            for r in todo:
                cid = r["claim_id"]
                pack = load_pack(cid)
                if pack["pack_hash"] != r["pack_hash"]:
                    raise SystemExit(f"{cid}: pack changed since the run ({pack['pack_hash'][:10]} != "
                                     f"{r['pack_hash'][:10]}); refusing to explain against different evidence")
                v = r["factshield_v2"]
                items = ev.prepare(pack)
                t0 = time.perf_counter()
                try:
                    exp = await explain_verdict(pack["claim"], v["final"], v.get("reason"), items,
                                                v["samples"], v["cited_pids"], checker, a.stub_models,
                                                client, {})
                except Exception as exc:  # recorded, not hidden
                    exp = {"error": f"{type(exc).__name__}: {exc}", "sentences": [], "dropped": [],
                           "quotes_only": True}
                row = {"claim_id": cid, "final": v["final"], "pack_hash": r["pack_hash"],
                       "model_version_run": r["model_version"], "seconds": round(time.perf_counter() - t0, 2),
                       "checker": checker.name, "stub_models": a.stub_models, "explanation": exp}
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                fh.flush()
                e = exp
                status = ("ERROR " + str(e["error"])[:100]) if e.get("error") else \
                    (f"kept {len(e['sentences'])}, dropped {len(e['dropped'])}"
                     + (" -> QUOTES ONLY" if e["quotes_only"] else ""))
                print(f"{cid:<26}{v['final']:<22}{status}")
    summarise(str(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", nargs="+")
    ap.add_argument("--out", default="study/results/factshield_v2p2_explanations.jsonl")
    ap.add_argument("--all", action="store_true", help="explain every claim, not only supported/refuted")
    ap.add_argument("--summary", help="only print the summary of an existing output file")
    ap.add_argument("--stub-models", action="store_true")
    ap.add_argument("--stub-checker", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summarise(a.summary)
        return
    if not a.results:
        raise SystemExit("--results is required")
    if a.stub_checker and not a.stub_models:
        raise SystemExit("real explainer with the stub checker would not describe FactShield; refusing")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
