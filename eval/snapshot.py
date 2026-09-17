"""Snapshot the retrieval corpus.

Retrieve ONCE per claim, store the documents, then run every condition against
the stored set. Two reasons, and the second matters more than cost:

  1. Search credits are consumed once, not once per condition per run.
  2. Ablations must differ only in the PIPELINE, never in what the web happened
     to return that afternoon. Without a snapshot, re-running the benchmark
     next month gives different numbers for reasons that have nothing to do
     with your system, and nobody — including you — can reproduce it.

Usage:
    python -m eval.snapshot --claims eval/dataset/claims.jsonl \
                            --out eval/dataset/corpus.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.pipeline import stages  # noqa: E402
from app.retrieval import retrieve  # noqa: E402
from app.schemas import CaptureContext  # noqa: E402


async def snapshot_one(record: dict) -> dict:
    ctx = CaptureContext(
        selection=record["claim"],
        surrounding_text=record.get("context", ""),
        page_published=record.get("page_published"),
        page_domain=record.get("page_domain"),
    )
    tri, _ = await stages.triage(ctx)

    docs_by_claim = {}
    for sub in tri.sub_claims:
        docs = await retrieve(sub.text, tri.category, tri.jurisdiction)
        docs_by_claim[sub.id] = [d.model_dump(mode="json") for d in docs]

    return {
        "id": record["id"],
        "claim": record["claim"],
        "label": record.get("label"),
        "category_gold": record.get("category"),
        "context": record.get("context", ""),
        "page_published": record.get("page_published"),
        "triage": tri.model_dump(mode="json"),
        "documents": docs_by_claim,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.claims).read_text().splitlines() if l.strip()]
    if args.limit:
        records = records[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()}
        print(f"resuming — {len(done)} already snapshotted")

    with out.open("a") as fh:
        for i, rec in enumerate(records, 1):
            if rec["id"] in done:
                continue
            try:
                snap = await snapshot_one(rec)
            except Exception as exc:  # one bad claim must not lose the run
                print(f"  [{i}/{len(records)}] {rec['id']} FAILED: {exc}")
                continue
            fh.write(json.dumps(snap, default=str) + "\n")
            fh.flush()
            n = sum(len(v) for v in snap["documents"].values())
            print(f"  [{i}/{len(records)}] {rec['id']}: {n} docs")
            await asyncio.sleep(0.4)  # be polite to the free APIs

    print(f"\nSnapshot written to {out}")


if __name__ == "__main__":
    asyncio.run(main())
