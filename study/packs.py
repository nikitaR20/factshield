"""Evidence packs: retrieve once per claim, freeze, never touch again.

    python -m study.packs build                 # build every missing pack
    python -m study.packs build --only b015,b016
    python -m study.packs build --rebuild b015  # only after deleting the file
    python -m study.packs verify                # re-hash every pack

Each pack is study/packs/<claim_id>.json. Every build appends a row to
study/packs/manifest.jsonl with the content hash, so a pack that changes after
the fact is detected by `verify` and refused by `load`.

Retrieval reuses FactShield's `app.retrieval.retrieve`. Two caveats that the
pack records rather than hides:

  * `retrieve` needs a category. The product gets it from an LLM triage step,
    which the study must not call, so the category comes from the claim's
    `domain` through a fixed table in config.yaml.
  * Since main@451df3f, `retrieve` also takes `concepts` (synonym groups
    ANDed into the PubMed query). The product gets them from triage; the study
    takes them from an OPTIONAL hand-written `search_concepts` field on the
    claim. Without it, PubMed falls back to keyword extraction. What was used
    is recorded in the pack.
  * PubMed documents carry `key_findings` (results + conclusions). Those sit
    at the END of an abstract, and passages are cut to max_chars_per_passage,
    so the conclusion can be cut off. The pack records the findings and
    whether they survived the cut, and warns when they did not. What the model
    sees is not changed silently: that is a stimulus decision.
  * `retrieve` swallows every channel failure and returns fewer documents. A
    missing TAVILY_API_KEY therefore looks exactly like "the web had nothing".
    The pack lists which channels were attempted and which keys were present,
    and an empty pack is flagged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import stat
from collections import Counter
from pathlib import Path
from typing import Awaitable, Callable

from app.config import env, pack_for

from .common import (DurableAppender, config, load_claims, now_iso, path,
                     provenance, read_jsonl, sha256_json)

Retriever = Callable[..., Awaitable[list]]   # (claim, category, jurisdiction, concepts=None)


class PackError(RuntimeError):
    pass


def manifest_path() -> Path:
    return path("packs") / "manifest.jsonl"


def pack_path(claim_id: str) -> Path:
    return path("packs") / f"{claim_id}.json"


def category_for(domain: str) -> str:
    table = config()["packs"]["domain_to_category"]
    return table.get((domain or "").lower(), table["default"])


def content_hash(pack: dict) -> str:
    """Hash of what the model sees plus the claim it belongs to. Build-time
    metadata (timestamps, commit) is excluded so the hash identifies content."""
    return sha256_json({"claim_id": pack["claim_id"], "claim": pack["claim"],
                        "passages": pack["passages"]})


def _attempted_channels(category: str) -> dict:
    pk = pack_for(category)
    adapters = list(pk.get("adapters", ["factcheck"]))
    return {
        "adapters": adapters,
        "domain_search": True,
        "open_web": bool(pk.get("use_open_web", True)),
        "keys_present": {
            "TAVILY_API_KEY": bool(env("TAVILY_API_KEY")),
            "FACTCHECK_API_KEY": bool(env("FACTCHECK_API_KEY")),
        },
    }


def _flags(claim: dict, passages: list[dict], attempted: dict) -> list[str]:
    flags = []
    if not passages:
        flags.append("empty_pack")
    retr = claim.get("retrievable")
    if isinstance(retr, dict) and retr and not any(retr.values()):
        flags.append("unretrievable_by_all_providers")
    if not attempted["keys_present"]["TAVILY_API_KEY"]:
        flags.append("tavily_key_missing")
    return flags


async def _default_retriever(claim: str, category: str, jurisdiction: str | None,
                             concepts: list[list[str]] | None = None) -> list:
    from app.retrieval import retrieve
    return await retrieve(claim, category, jurisdiction, concepts=concepts)


async def build_pack(claim: dict, retriever: Retriever | None = None) -> dict:
    cfg = config()["packs"]
    category = category_for(claim.get("domain", ""))
    attempted = _attempted_channels(category)
    concepts = claim.get("search_concepts") or None
    docs = await (retriever or _default_retriever)(claim["claim"], category, None,
                                                    concepts=concepts)

    passages = []
    for i, d in enumerate(docs[: cfg["max_passages"]], 1):
        d = d.model_dump(mode="json") if hasattr(d, "model_dump") else dict(d)
        text = d.get("text") or ""
        kept = text[: cfg["max_chars_per_passage"]]
        findings = d.get("key_findings")
        passages.append({
            "pid": f"P{i}",
            "url": str(d.get("url")),
            "title": d.get("title"),
            "published_date": d.get("published_date"),
            "source_domain": d.get("source_domain"),
            "channel": d.get("channel"),
            "text": kept,
            "text_truncated": len(text) > cfg["max_chars_per_passage"],
            "text_chars_original": len(text),
            # Not shown to the model; recorded so truncation loss is measurable.
            "key_findings": findings,
            "key_findings_in_text": (findings.strip() in kept) if findings else None,
        })

    pack = {
        "claim_id": claim["id"],
        "claim": claim["claim"],
        "category": category,
        "built_at": now_iso(),
        **provenance(),
        "retrieval": {
            **attempted,
            "n_docs_returned": len(docs),
            "n_docs_kept": len(passages),
            "docs_per_channel": dict(Counter(p["channel"] for p in passages)),
            "search_concepts": concepts,
        },
        # Per-provider retrievability of the claim's SOURCE, as recorded in the
        # claim set. The pack cannot measure this itself: it would need each
        # provider's live search.
        "source_retrievable_by_provider": claim.get("retrievable"),
        "passages": passages,
    }
    pack["flags"] = _flags(claim, passages, attempted)
    # Warnings are reported but do not exclude the claim (flags do).
    lost = [p["pid"] for p in passages if p["key_findings_in_text"] is False]
    pack["warnings"] = [f"key_findings_truncated:{','.join(lost)}"] if lost else []
    pack["pack_hash"] = content_hash(pack)
    return pack


def _manifest_rows(claim_id: str) -> list[dict]:
    return [r for r in read_jsonl(manifest_path()) if r["claim_id"] == claim_id]


def write_pack(pack: dict, *, rebuild: bool = False) -> None:
    p = pack_path(pack["claim_id"])
    previous = _manifest_rows(pack["claim_id"])
    if p.exists():
        raise PackError(f"{p} exists. Packs are immutable. To rebuild, delete it "
                        f"explicitly and run with --rebuild {pack['claim_id']}.")
    if previous and not rebuild:
        raise PackError(f"{pack['claim_id']} was built before (hash {previous[-1]['pack_hash'][:12]}) "
                        f"and its file is gone. Rebuilding changes the evidence; pass "
                        f"--rebuild {pack['claim_id']} to do it on purpose.")
    p.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: two concurrent builders cannot both write the file.
    with p.open("x", encoding="utf-8") as fh:
        json.dump(pack, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    with DurableAppender(manifest_path()) as m:
        m.write({"claim_id": pack["claim_id"], "pack_hash": pack["pack_hash"],
                 "n_passages": len(pack["passages"]), "flags": pack["flags"],
                 "built_at": pack["built_at"], "rebuild": rebuild,
                 "previous_hash": previous[-1]["pack_hash"] if previous else None,
                 **provenance()})


def load_pack(claim_id: str) -> dict:
    """Load a pack and prove it is the one the manifest recorded."""
    p = pack_path(claim_id)
    if not p.exists():
        raise PackError(f"no pack for {claim_id}; run `python -m study.packs build`")
    pack = json.loads(p.read_text(encoding="utf-8"))
    actual = content_hash(pack)
    rows = _manifest_rows(claim_id)
    if not rows:
        raise PackError(f"{claim_id}: pack has no manifest row")
    if actual != pack["pack_hash"] or actual != rows[-1]["pack_hash"]:
        raise PackError(f"{claim_id}: pack content hash {actual[:12]} does not match "
                        f"the manifest ({rows[-1]['pack_hash'][:12]}). It was modified after freezing.")
    return pack


def verify() -> int:
    bad = 0
    latest = {}
    for r in read_jsonl(manifest_path()):
        latest[r["claim_id"]] = r
    for cid in sorted(latest):
        try:
            load_pack(cid)
            print(f"  ok      {cid}")
        except PackError as exc:
            bad += 1
            print(f"  FAILED  {exc}")
    print(f"{len(latest) - bad}/{len(latest)} packs verified")
    return bad


async def build_all(only: set[str] | None, rebuild: str | None,
                    retriever: Retriever | None = None) -> None:
    claims = load_claims()
    if rebuild:
        only = {rebuild}
    todo = [c for c in claims if (not only or c["id"] in only)]
    flagged = []
    for c in todo:
        if pack_path(c["id"]).exists():
            if rebuild:
                raise PackError(f"--rebuild {rebuild}: delete {pack_path(c['id'])} first.")
            continue
        pack = await build_pack(c, retriever)
        write_pack(pack, rebuild=bool(rebuild))
        print(f"  {c['id']}: {len(pack['passages'])} passages "
              f"{pack['retrieval']['docs_per_channel']} flags={pack['flags']}"
              + (f" warnings={pack['warnings']}" if pack["warnings"] else ""))
        if pack["flags"]:
            flagged.append((c["id"], pack["flags"]))
    if flagged:
        print(f"\n{len(flagged)} FLAGGED claims (excluded from runs unless --include-flagged):")
        for cid, fl in flagged:
            print(f"  {cid}: {', '.join(fl)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--only", default="", help="comma-separated claim ids")
    b.add_argument("--rebuild", default=None, metavar="CLAIM_ID")
    sub.add_parser("verify")
    args = ap.parse_args()
    if args.cmd == "verify":
        raise SystemExit(1 if verify() else 0)
    only = {x for x in args.only.split(",") if x} or None
    try:
        asyncio.run(build_all(only, args.rebuild))
    except PackError as exc:
        raise SystemExit(f"refused: {exc}")


if __name__ == "__main__":
    main()
