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
import os
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
        docs = await retrieve(sub.text, tri.category, tri.jurisdiction,
                              concepts=sub.search_concepts or None)
        docs_by_claim[sub.id] = [d.model_dump(mode="json") for d in docs]

    return {
        "id": record["id"],
        "claim": record["claim"],
        "label": record.get("label"),
        "category_gold": record.get("category"),
        "context": record.get("context", ""),
        "page_published": record.get("page_published"),
        "label_date": record.get("label_date"),
        # Identifiers a good search SHOULD find: PMIDs, DOIs, or URL
        # substrings. Lets the analysis report whether a verdict was right
        # for the right reason.
        "must_retrieve": record.get("must_retrieve", []),
        "triage": tri.model_dump(mode="json"),
        "documents": docs_by_claim,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay", type=float, default=2.0,
                    help="Seconds between claims. Raise it if you see timeouts "
                         "or 429s; snapshotting is a one-off cost.")
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(
        args.claims).read_text().splitlines() if l.strip()]
    if args.limit:
        records = records[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resume from whatever is already valid. A run interrupted mid-write can
    # leave a truncated or doubled-up line; crashing on it would discard every
    # claim already fetched, and re-fetching costs search credits.
    done: set[str] = set()
    if out.exists():
        salvaged, corrupt = [], 0
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                corrupt += 1
                continue
            salvaged.append(line)
            done.add(record["id"])
        if corrupt:
            # Rewrite without the damaged lines so the append below starts clean.
            out.write_text("\n".join(salvaged) + "\n", encoding="utf-8")
            print(f"dropped {corrupt} corrupt line(s) from {out.name}")
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
            # One complete line, flushed and synced, so an interruption leaves
            # whole records rather than a fragment the next run cannot parse.
            fh.write(json.dumps(snap, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            n = sum(len(v) for v in snap["documents"].values())
            print(f"  [{i}/{len(records)}] {rec['id']}: {n} docs")
            # Free tiers rate-limit on sustained bursts. Snapshotting is a
            # one-off cost, so pacing it is cheaper than re-running failures.
            await asyncio.sleep(args.delay)

    print(f"\nSnapshot written to {out}")


if __name__ == "__main__":
    asyncio.run(main())
