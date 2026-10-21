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


_LAST_FANOUT: dict = {}


def _canon(url: str) -> str:
    from urllib.parse import urlparse
    u = urlparse(str(url))
    return f"{u.netloc.lower().removeprefix('www.')}{u.path.rstrip('/')}"


async def _default_retriever(claim: str, category: str, jurisdiction: str | None,
                             concepts: list[list[str]] | None = None) -> list:
    """Retrieve through app.retrieval.

    fetch_per_channel null: `retrieve()` unchanged, as the product calls it.
    Otherwise: the same channels `retrieve()` uses (the category's adapters,
    domain_search, open_web if the category enables it), each asked for
    `fetch_per_channel` results. A channel listed in exclude_channels is not
    called at all. Deny-listed domains and duplicate URLs are removed exactly
    as retrieve() does. What each channel returned or raised is recorded,
    because retrieve() swallows failures and an empty pack needs a reason.
    """
    import asyncio

    from app import retrieval as R

    cfg = config()["packs"]
    n = cfg.get("fetch_per_channel")
    _LAST_FANOUT.clear()
    if not n:
        _LAST_FANOUT.update({"mode": "app.retrieval.retrieve"})
        return await R.retrieve(claim, category, jurisdiction, concepts=concepts)

    skip = set(cfg.get("exclude_channels") or [])
    pk = pack_for(category, jurisdiction)
    calls: dict[str, object] = {}
    for name in pk.get("adapters", ["factcheck"]):
        fn = R.ADAPTERS.get(name)
        # The factcheck adapter IS the prior_check channel.
        if fn is None or (name == "factcheck" and "prior_check" in skip):
            continue
        calls[name] = fn(claim, limit=n, category=category, jurisdiction=jurisdiction,
                         concepts=concepts)
    if "authority" not in skip:
        calls["domain_search"] = R.domain_search(claim, n, category=category,
                                                 jurisdiction=jurisdiction)
    if pk.get("use_open_web", True) and "open_web" not in skip:
        calls["open_web"] = R.open_web(claim, n)

    results = await asyncio.gather(*calls.values(), return_exceptions=True)
    docs, seen, per = [], set(), {}
    for name, res in zip(calls, results):
        if isinstance(res, Exception):
            per[name] = f"error: {type(res).__name__}: {res}"[:200]
            continue
        per[name] = len(res)
        for d in res:
            if R.DENY_DOMAINS and any(
                    d.source_domain.lower().removeprefix("www.") == x
                    or d.source_domain.lower().endswith("." + x) for x in R.DENY_DOMAINS):
                continue
            key = _canon(d.url)
            if key in seen:
                continue
            seen.add(key)
            docs.append(d)
    _LAST_FANOUT.update({"mode": "study_fanout", "fetch_per_channel": n,
                         "channels_called": list(calls), "returned_per_channel": per})
    return docs


async def build_pack(claim: dict, retriever: Retriever | None = None) -> dict:
    cfg = config()["packs"]
    category = category_for(claim.get("domain", ""))
    attempted = _attempted_channels(category)
    concepts = claim.get("search_concepts") or None
    docs = await (retriever or _default_retriever)(claim["claim"], category, None,
                                                    concepts=concepts)

    passages = []
    docs = [d.model_dump(mode="json") if hasattr(d, "model_dump") else dict(d) for d in docs]
    n_returned = len(docs)
    drop_ch = set(cfg.get("exclude_channels") or [])
    drop_dom = [x.lower().removeprefix("www.") for x in cfg.get("exclude_domains") or []]

    def _dropped(d: dict) -> bool:
        dom = (d.get("source_domain") or "").lower().removeprefix("www.")
        return d.get("channel") in drop_ch or any(dom == x or dom.endswith("." + x) for x in drop_dom)

    removed = [d for d in docs if _dropped(d)]
    docs = [d for d in docs if not _dropped(d)]
    for i, d in enumerate(docs[: cfg["max_passages"]], 1):
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
            "n_docs_returned": n_returned,
            "excluded_by_config": [{"url": str(d.get("url")), "channel": d.get("channel"),
                                    "source_domain": d.get("source_domain")} for d in removed],
            "n_docs_kept": len(passages),
            "docs_per_channel": dict(Counter(p["channel"] for p in passages)),
            "search_concepts": concepts,
            "fanout": dict(_LAST_FANOUT) if retriever is None else {"mode": "injected"},
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
    verdicts = [p["pid"] for p in passages if p["channel"] == "prior_check"]
    if verdicts:
        # The pack holds a published fact-check rating. If the claim came from
        # a fact-check table, this is likely its own answer key.
        pack["warnings"].append(f"contains_factcheck_verdict:{','.join(verdicts)}")
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


def summary() -> int:
    """Per-label pack sizes, channels, top domains, flags and warnings."""
    from collections import defaultdict

    claims = load_claims()
    by_label = defaultdict(Counter)
    channels, domains, flags, warns = Counter(), Counter(), Counter(), Counter()
    empty_ids = []
    for c in claims:
        try:
            pk = load_pack(c["id"])
        except PackError:
            by_label[c["label"]]["missing"] += 1
            continue
        n = len(pk["passages"])
        by_label[c["label"]]["empty" if n == 0 else "one" if n == 1 else "2+"] += 1
        if n == 0:
            empty_ids.append(c["id"])
        for ps in pk["passages"]:
            channels[ps["channel"]] += 1
            domains[(ps["source_domain"] or "").removeprefix("www.")] += 1
        flags.update(pk["flags"])
        warns.update(w.split(":")[0] for w in pk.get("warnings", []))
    print(f"{len(claims)} claims")
    print(f"\n{'gold label':<22}{'claims':>7}{'2+':>6}{'one':>6}{'empty':>7}{'missing':>9}")
    for lab in ("supported", "refuted", "not_enough_evidence"):
        v = by_label.get(lab, Counter())
        print(f"{lab:<22}{sum(v.values()):>7}{v['2+']:>6}{v['one']:>6}{v['empty']:>7}{v['missing']:>9}")
    print(f"\npassages per channel: {dict(channels)}")
    print("top domains: " + ", ".join(f"{d} {k}" for d, k in domains.most_common(12)))
    if flags:
        print(f"flags (claim excluded from runs): {dict(flags)}")
    if warns:
        print(f"warnings (kept): {dict(warns)}")
    if empty_ids:
        print(f"empty packs: {', '.join(empty_ids)}")
    missing = sum(v["missing"] for v in by_label.values())
    if missing:
        print(f"\n{missing} packs missing: run `python -m study.packs build` to finish.")
    return 0


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
    flagged, failed = [], []
    for c in todo:
        if pack_path(c["id"]).exists():
            if rebuild:
                raise PackError(f"--rebuild {rebuild}: delete {pack_path(c['id'])} first.")
            continue
        try:
            pack = await build_pack(c, retriever)
        except Exception as exc:  # one bad claim must not stop the build
            failed.append((c["id"], f"{type(exc).__name__}: {exc}"))
            print(f"  {c['id']}: FAILED {type(exc).__name__}: {str(exc)[:200]} (no pack written; "
                  f"re-run to retry)")
            continue
        write_pack(pack, rebuild=bool(rebuild))
        print(f"  {c['id']}: {len(pack['passages'])} passages "
              f"{pack['retrieval']['docs_per_channel']} flags={pack['flags']}"
              + (f" warnings={pack['warnings']}" if pack["warnings"] else ""))
        if pack["flags"]:
            flagged.append((c["id"], pack["flags"]))
    with_verdict = [c["id"] for c in todo if pack_path(c["id"]).exists()
                    and any(w.startswith("contains_factcheck_verdict")
                            for w in json.loads(pack_path(c["id"]).read_text(encoding="utf-8")).get("warnings", []))]
    if with_verdict:
        print(f"\n{len(with_verdict)} packs contain a published fact-check rating "
              f"(warning, not excluded). If these claims come from a fact-check table, the "
              f"pack may hold their answer. See packs.exclude_channels in config.yaml.")
    if failed:
        print(f"\n{len(failed)} claims FAILED and have no pack yet. Re-run the same command to retry:")
        for cid, why in failed:
            print(f"  {cid}: {why[:200]}")
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
    sub.add_parser("summary", help="pack sizes per gold label, channels, domains, flags")
    args = ap.parse_args()
    if args.cmd == "verify":
        raise SystemExit(1 if verify() else 0)
    if args.cmd == "summary":
        raise SystemExit(summary())
    only = {x for x in args.only.split(",") if x} or None
    try:
        asyncio.run(build_all(only, args.rebuild))
    except PackError as exc:
        raise SystemExit(f"refused: {exc}")


if __name__ == "__main__":
    main()
