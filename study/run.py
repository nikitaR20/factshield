"""Main study runner: every claim x condition x model, shuffled, resumable.

    python -m study.run                         # all claims, conditions, models
    python -m study.run --repeat 3              # run-to-run noise baseline
    python -m study.run --models stub --only t001,t002
    python -m study.run --live-search           # labelled live-search subset

Design points:
  * The whole plan (all repeats included) is shuffled once with the configured
    seed. Executing conditions or repeats in blocks would confound them with
    time, during which models and indexes drift.
  * Each row is appended and fsynced as it arrives. Re-running the same command
    resumes: keys with a successful row are skipped, keys whose last attempt
    errored are tried again. Error rows stay in the file for the audit trail.
  * The complete raw provider response is stored, not just the verdict.
  * Live-search rows go to a different file and carry live_search=true, so they
    cannot be pooled with closed-book rows by accident.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

import httpx

from . import models, prompts
from .common import (DurableAppender, config, load_claims, now_iso, path,
                     provenance, read_jsonl)
from .packs import PackError, load_pack

KEY_FIELDS = ("kind", "claim_id", "condition", "model", "run_index", "live_search")


def row_key(r: dict) -> tuple:
    return tuple(r[k] for k in KEY_FIELDS)


def make_row(*, kind: str, claim: dict, condition: str, model_key: str, run_index: int,
             live_search: bool, pack: dict | None, result: models.CallResult,
             prompt_meta: dict, call_order: int) -> dict:
    parsed = prompts.parse(result.text, condition)
    return {
        # --- required by the protocol
        "claim_id": claim["id"],
        "condition": condition,
        "model": model_key,
        "model_version": result.model_version,
        "run_index": run_index,
        "live_search": live_search,
        "pack_hash": pack["pack_hash"] if pack else None,
        "timestamp": result.timestamp,
        "raw_response": result.raw,
        "parsed_verdict": parsed["parsed_verdict"],
        "confidence": parsed["confidence"],
        "cited_ids": parsed["cited_ids"],
        "tool_calls": result.tool_calls,
        "latency_ms": result.latency_ms,
        "error": result.error,
        # --- provenance and audit
        "kind": kind,
        "claim_set": claim.get("set"),
        "call_order": call_order,
        "provider": result.provider,
        "model_requested": result.model_requested,
        "temperature_sent": result.temperature_sent,
        "params_sent": result.params_sent,
        "tools_attached": result.tools_attached,
        "finish_reason": result.finish_reason,
        "tokens": result.tokens,
        "attempts": result.attempts,
        "http_status": result.http_status,
        "latency_total_ms": result.latency_total_ms,
        "response_text": result.text,
        "reasoning_text": result.reasoning_text,
        "parse_rule": parsed["parse_rule"],
        "format_ok": parsed["format_ok"],
        "cited_urls": parsed["cited_urls"],
        **prompt_meta,
        **provenance(),
    }


async def execute(plan: list[dict], out: Path, *, concurrency: int) -> dict:
    """Run plan items in order with bounded concurrency. Each item has: kind,
    claim, condition, model_key, run_index, live_search, pack, call_order."""
    done = {row_key(r) for r in read_jsonl(out) if r.get("error") is None}
    todo = [p for p in plan if (p["kind"], p["claim"]["id"], p["condition"], p["model_key"],
                                p["run_index"], p["live_search"]) not in done]
    stats = {"planned": len(plan), "already_done": len(plan) - len(todo), "ok": 0, "errors": 0}
    if done:
        print(f"resuming: {stats['already_done']} of {len(plan)} calls already have a successful row")

    queue: asyncio.Queue = asyncio.Queue()
    for item in todo:
        queue.put_nowait(item)

    async with httpx.AsyncClient() as client:
        with DurableAppender(out) as sink:
            async def worker():
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    system, user, meta = prompts.build(item["condition"], item["claim"]["claim"],
                                                       item["pack"])
                    res = await models.complete(item["model_key"], system, user,
                                                live_search=item["live_search"], client=client)
                    row = make_row(kind=item["kind"], claim=item["claim"],
                                   condition=item["condition"], model_key=item["model_key"],
                                   run_index=item["run_index"], live_search=item["live_search"],
                                   pack=item["pack"], result=res, prompt_meta=meta,
                                   call_order=item["call_order"])
                    sink.write(row)
                    n = stats["ok"] + stats["errors"] + 1
                    if res.error:
                        stats["errors"] += 1
                        print(f"  [{n}/{len(todo)}] {item['claim']['id']} {item['condition']:<9}"
                              f" {item['model_key']:<7} r{item['run_index']}  ERROR {res.error[:120]}")
                    else:
                        stats["ok"] += 1
                        tools = "TOOLS OBSERVED" if res.tool_calls else "no tools"
                        print(f"  [{n}/{len(todo)}] {item['claim']['id']} {item['condition']:<9}"
                              f" {item['model_key']:<7} r{item['run_index']}  {row['parsed_verdict']:<20}"
                              f" {res.model_version}  {tools}")

            await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
    return stats


def select_claims(only: set[str] | None, include_flagged: bool) -> tuple[list[dict], dict, list]:
    claims = [c for c in load_claims() if not only or c["id"] in only]
    if only and len(claims) != len(only):
        missing = only - {c["id"] for c in claims}
        raise SystemExit(f"unknown claim ids: {sorted(missing)}")
    kept, packs, excluded = [], {}, []
    for c in claims:
        try:
            pk = load_pack(c["id"])
        except PackError as exc:
            excluded.append((c["id"], str(exc)))
            continue
        if pk["flags"] and not include_flagged:
            excluded.append((c["id"], f"flagged: {', '.join(pk['flags'])}"))
            continue
        kept.append(c)
        packs[c["id"]] = pk
    return kept, packs, excluded


def build_plan(claims: list[dict], packs: dict, model_keys: list[str], repeat: int,
               live: bool, seed: int) -> list[dict]:
    plan = []
    for r in range(repeat):
        for c in claims:
            for cond in config()["conditions"]:
                for m in model_keys:
                    plan.append({"kind": "main", "claim": c, "condition": cond, "model_key": m,
                                 "run_index": r, "live_search": live, "pack": packs[c["id"]]})
    random.Random(seed).shuffle(plan)
    for i, p in enumerate(plan):
        p["call_order"] = i
    return plan


def main() -> None:
    cfg = config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(cfg["study_models"]))
    ap.add_argument("--only", default="", help="comma-separated claim ids")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--live-search", action="store_true",
                    help="LABELLED MODE: attach each provider's native search, on a small "
                         "seeded subset of claims, written to a separate file")
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--concurrency", type=int, default=cfg["run"]["concurrency"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in model_keys:
        models.model_cfg(m)  # fail on an unknown key before any call
    if "stub" in model_keys:
        print("NOTE: the `stub` model is an offline stand-in. Its rows are not model output.")

    only = {x for x in args.only.split(",") if x} or None
    claims, packs, excluded = select_claims(only, args.include_flagged)
    for cid, why in excluded:
        print(f"  excluded {cid}: {why}")
    if not claims:
        raise SystemExit("no claims to run")

    seed = cfg["seed"]
    if args.live_search:
        n = cfg["run"]["live_search"]["n_claims"]
        claims = sorted(random.Random(seed + 1).sample(claims, min(n, len(claims))),
                        key=lambda c: c["id"])
        print("=" * 72)
        print(f"LIVE SEARCH MODE: provider search tools ATTACHED for {len(claims)} claims.")
        print("These rows are NOT part of the closed-book study and are written separately.")
        print("=" * 72)

    out = Path(args.out) if args.out else path("responses") / (
        "live_search.jsonl" if args.live_search else "main.jsonl")
    plan = build_plan(claims, packs, model_keys, args.repeat, args.live_search, seed)
    print(f"{len(claims)} claims x {len(cfg['conditions'])} conditions x {len(model_keys)} models"
          f" x {args.repeat} repeats = {len(plan)} calls -> {out}")

    with DurableAppender(path("responses") / "run_log.jsonl") as log:
        log.write({"event": "start", "at": now_iso(), "argv": sys.argv[1:], "out": str(out),
                   "n_calls": len(plan), "models": {m: cfg["models"][m] for m in model_keys},
                   "excluded": excluded, **provenance()})
        stats = asyncio.run(execute(plan, out, concurrency=args.concurrency))
        log.write({"event": "end", "at": now_iso(), **stats, **provenance()})
    print(f"\ndone: {stats}")
    if stats["errors"]:
        print("Some calls failed. Re-run the same command to retry only those.")


if __name__ == "__main__":
    main()
