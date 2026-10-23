"""Run FactShield on the study's frozen evidence packs.

The comparison the thesis needs is FactShield versus a plain model reading THE
SAME evidence. If each system searched for itself, a difference could be a
search difference. So this reads the packs the study models were given
(study/packs/*.json, fact-check content already removed) and the study's gold
labels (study.common.load_claims, label overrides applied), and writes rows in
the study's response format, so `python -m study.score` scores FactShield and
the models with the same code.

Held constant, deliberately:
  * evidence: the pack passages, exactly as the models saw them
  * the claim: the pack's claim text as ONE sub-claim. FactShield's triage
    (splitting, pronoun resolution) is skipped, because the models got the
    raw claim too.
  * no fast path: the packs contain no fact-check ratings, so there is nothing
    for it to use. It stays switched off so that remains true by construction.

Conditions:
  fs_computed   grading + evidence state + score + abstention rules. No model call.
  fs_explained  as above, plus the explanation model's one-directional
                downgrade (one call per non-abstained claim, see models.yaml).
  fs_no_abstain ablation: abstention rules off; a verdict is forced from the
                evidence lean. Shows what the abstention rules buy, at what cost.

Verdict mapping onto the study's three labels:
  supported -> supported, refuted -> refuted, unresolved -> not_enough_evidence.
  partly_supported has no study label. It is kept as itself and scored as a
  COMMITTED answer that is wrong (study/score.py), which is the harsh choice
  for FactShield, made before seeing results.

    python -m eval.from_packs --out study/results/factshield.jsonl
    python -m study.score --responses study/results/factshield.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.config import PIPELINE_VERSION  # noqa: E402
from app.pipeline import decide, grading, scoring  # noqa: E402
from app.pipeline import evidence_state as ev  # noqa: E402
from app.providers import nli  # noqa: E402
from app.schemas import RawDocument, SubClaim  # noqa: E402

TO_STUDY = {"supported": "supported", "refuted": "refuted",
            "unresolved": "not_enough_evidence", "partly_supported": "partly_supported"}
CONDITIONS = ("fs_computed", "fs_explained", "fs_no_abstain")


def pack_docs(pack: dict) -> list[RawDocument]:
    docs = []
    for p in pack.get("passages", []):
        docs.append(RawDocument(
            url=p["url"], title=p.get("title") or "", text=p.get("text") or "",
            published_date=(p.get("published_date") or None) and str(p["published_date"])[:10],
            source_domain=p.get("source_domain") or "",
            # Packs only contain search channels; anything else would be a
            # fact-check, which the packs exclude.
            channel=p.get("channel") if p.get("channel") in ("authority", "open_web", "literature")
            else "open_web",
            key_findings=p.get("key_findings"),
        ))
    return docs


def forced_lean(items, category: str, n_domains: int) -> str:
    """For the no-abstention ablation: whichever side carries more weight."""
    sup = sum(scoring.item_weight(i, category, n_domains) for i in items if i.stance == "supports")
    ref = sum(scoring.item_weight(i, category, n_domains) for i in items if i.stance == "refutes")
    if sup > ref:
        return "supported"
    if ref > sup:
        return "refuted"
    return "unresolved"


async def run_one(claim: dict, pack: dict, condition: str) -> dict:
    category = pack.get("category") or "general"
    sub = SubClaim(id=1, text=pack["claim"])
    items = await grading.grade(pack_docs(pack), sub.text, 1, category, None)
    state = ev.compute(items, category, None, None)
    d = await decide.decide(sub, items, state, category,
                            explain=(condition == "fs_explained"), use_fast_path=False)
    verdict = d.verdict.claim_verdict
    forced = False
    if condition == "fs_no_abstain" and verdict == "unresolved":
        verdict, forced = forced_lean(items, category, state.n_independent_domains), True
    return {
        "kind": "main", "claim_id": claim["id"], "condition": condition,
        "model": "factshield", "model_version": f"{PIPELINE_VERSION}/nli:{nli.backend_name()}",
        "run_index": 0, "live_search": False, "pack_hash": pack.get("pack_hash"),
        "parsed_verdict": TO_STUDY[verdict], "response_text": None,
        "confidence": None, "cited_ids": [], "cited_urls": [], "tool_calls": [],
        "error": None,
        "factshield": {
            "claim_verdict": d.verdict.claim_verdict, "forced": forced,
            "evidence_standing": d.verdict.evidence_standing,
            "support_score": d.verdict.support_score, "score_basis": d.verdict.score_basis,
            "abstain_kind": d.abstain_kind, "abstention_reason": d.verdict.abstention_reason,
            "explained": d.explained, "downgraded": d.downgraded,
            "explanation": d.verdict.explanation,
            "n_items": len(items),
            "stances": [i.stance for i in items],
            "items": [{"url": str(i.url), "tier": i.tier, "role": i.role, "stance": i.stance,
                       "conf": i.stance_confidence, "quote": i.quote} for i in items],
        },
    }


async def main_async(args) -> None:
    from study.common import load_claims
    from study.packs import PackError, load_pack

    if nli.backend_name() != "transformers" and not args.allow_stub:
        raise SystemExit("NLI model not loaded (lexical stub). Results would not describe "
                         "FactShield. Install transformers/torch, or pass --allow-stub for a dry run.")
    claims = load_claims()
    if args.ids:
        want = set(args.ids.split(","))
        claims = [c for c in claims if c["id"] in want]
    conds = [c for c in args.conditions.split(",") if c in CONDITIONS]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("a", encoding="utf-8") as fh:
        for c in claims:
            try:
                pack = load_pack(c["id"])
            except PackError as exc:
                print(f"skip {c['id']}: {exc}")
                continue
            for cond in conds:
                t0 = time.perf_counter()
                try:
                    row = await run_one(c, pack, cond)
                except Exception as exc:  # recorded, not hidden
                    row = {"kind": "main", "claim_id": c["id"], "condition": cond,
                           "model": "factshield", "model_version": PIPELINE_VERSION,
                           "run_index": 0, "live_search": False, "parsed_verdict": None,
                           "error": f"{type(exc).__name__}: {exc}"}
                row["seconds"] = round(time.perf_counter() - t0, 2)
                fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
                fh.flush()
                n += 1
                print(f"{c['id']:<28}{cond:<15}{row.get('parsed_verdict')!s:<22}"
                      f"gold={c['label']}{'  ERROR ' + row['error'] if row.get('error') else ''}")
    print(f"\n{n} rows appended to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="study/results/factshield.jsonl")
    ap.add_argument("--conditions", default="fs_computed,fs_no_abstain",
                    help="fs_explained makes one model call per non-abstained claim")
    ap.add_argument("--ids", help="comma-separated claim ids (default: all)")
    ap.add_argument("--allow-stub", action="store_true",
                    help="run with the keyword stub; for testing the plumbing only")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
