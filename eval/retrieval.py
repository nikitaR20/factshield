"""Retrieval recall — did the search find the evidence that settles the claim?

For each claim, list in `must_retrieve` the few sources a competent search
should surface: PMIDs, DOIs, or distinctive URL substrings. This reports, per
claim, what fraction were actually retrieved.

Why this matters: the Stanford preprint's central finding is that fact-checking
fails on ACCESS to the right evidence, not on reasoning over it. With retrieval
now the bottleneck, this is the metric that measures it directly — and it lets
you separate a verdict that was right for the right reason from one that was
right by accident.

    python -m eval.retrieval --corpus eval/dataset/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _matches(identifier: str, url: str) -> bool:
    ident, url = identifier.strip().lower(), url.lower()
    if ident.isdigit():                       # bare PMID
        return f"pubmed.ncbi.nlm.nih.gov/{ident}" in url
    if ident.startswith("10."):               # DOI
        return ident in url
    return ident in url                       # URL substring


def recall_for(record: dict) -> tuple[int, int, list[str]]:
    wanted = record.get("must_retrieve") or []
    urls = [d.get("url", "") for docs in record.get(
        "documents", {}).values() for d in docs]
    missed = [w for w in wanted if not any(_matches(w, u) for u in urls)]
    return len(wanted) - len(missed), len(wanted), missed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="eval/dataset/corpus.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(
        args.corpus).read_text().splitlines() if l.strip()]
    scored = [r for r in rows if r.get("must_retrieve")]
    if not scored:
        print("No claims have `must_retrieve` set. Add PMIDs, DOIs or URL "
              "substrings for the sources a good search should find.")
        return

    found_total = wanted_total = 0
    print(f"{'claim':<14}{'recall':>10}   missed")
    for r in scored:
        found, wanted, missed = recall_for(r)
        found_total += found
        wanted_total += wanted
        print(f"{r['id']:<14}{found:>4}/{wanted:<5}   {', '.join(missed) or '—'}")
    print(f"\noverall retrieval recall: {found_total}/{wanted_total} "
          f"({found_total / wanted_total:.0%})")
    print(f"claims annotated: {len(scored)}/{len(rows)}")


if __name__ == "__main__":
    main()
