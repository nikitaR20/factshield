"""Run FactShield v2 on the study's frozen packs (the thesis comparison).

Same evidence as the plain models, same gold labels, same dev/test split as
eval/from_packs.py, and rows in the study's response format, so

    python -m study.score --responses study/results/factshield_v2.jsonl --tag fs2

scores FactShield v2 with the same code as the models.

Rows written per claim (all from ONE set of model calls):
  condition fs2              model factshield-v2   the design as decided
  condition fs2_no_check     model factshield-v2   ablation: no citation check
  condition fs2_majority_r1  model factshield-v2   ablation: 2-of-3 in round 1, with check
  condition fs2_single       model fs2:<key>       ablation: one model, v2 prompt, no rule

Typical use, on the laptop with keys in .env:
    python -m fsv2.run_packs --split dev                # look at dev only
    python -m fsv2.run_packs --split test               # ONCE, after the design is frozen
Offline plumbing test (meaningless verdicts):
    python -m fsv2.run_packs --split dev --stub-models --stub-checker --out /tmp/x.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import httpx  # noqa: E402

from study.common import load_claims  # noqa: E402
from study.packs import PackError, load_pack  # noqa: E402

from . import checker as chk  # noqa: E402
from .decide import TO_STUDY, check_claim  # noqa: E402
from .settings import cfg  # noqa: E402


def split_of(claim_id: str) -> str:
    """Identical to eval/from_packs.py: fixed before any result was seen."""
    return "dev" if int(hashlib.sha256(claim_id.encode()).hexdigest(), 16) % 3 == 0 else "test"


def rows_for(claim: dict, pack: dict, res: dict, version: str, seconds: float) -> list[dict]:
    base = {"kind": "main", "claim_id": claim["id"], "split": split_of(claim["id"]),
            "run_index": 0, "live_search": False, "pack_hash": pack.get("pack_hash"),
            "response_text": None, "confidence": None, "cited_urls": [], "seconds": seconds}
    if res.get("error"):
        return [{**base, "condition": "fs2", "model": "factshield-v2", "model_version": version,
                 "parsed_verdict": None, "cited_ids": [], "tool_calls": [],
                 "error": res["error"], "factshield_v2": res}]
    tools = [t for s in res["samples"] for t in (s.get("tool_calls") or [])]
    rows = [{**base, "condition": "fs2", "model": "factshield-v2", "model_version": version,
             "parsed_verdict": TO_STUDY[res["final"]], "cited_ids": res["cited_pids"],
             "tool_calls": tools, "error": None, "factshield_v2": res}]
    for cond in ("fs2_no_check", "fs2_majority_r1"):
        rows.append({**base, "condition": cond, "model": "factshield-v2", "model_version": version,
                     "parsed_verdict": TO_STUDY[res["ablations"][cond]], "cited_ids": [],
                     "tool_calls": tools, "error": None})
    for s in res["samples"][: len(cfg()["panel"])]:  # round 1 only
        rows.append({**base, "condition": "fs2_single", "model": f"fs2:{s['model_key']}",
                     "model_version": f"{version}/{s['model_version']}",
                     "parsed_verdict": TO_STUDY[s["verdict"]], "cited_ids": [],
                     "tool_calls": s.get("tool_calls") or [], "error": None})
    return rows


async def main_async(a) -> None:
    if not a.stub_models and a.stub_checker and not a.allow_stub_checker:
        raise SystemExit("Real models with the stub checker would produce results that do not "
                         "describe FactShield v2. Install MiniCheck, or pass --allow-stub-checker.")
    checker = chk.load(stub=a.stub_checker)
    version = (f"{cfg()['version']}/{checker.name}"
               + ("/STUB-MODELS" if a.stub_models else ""))
    claims = load_claims()
    if a.ids:
        want = set(a.ids.split(","))
        claims = [c for c in claims if c["id"] in want]
    if a.split != "all":
        claims = [c for c in claims if split_of(c["id"]) == a.split]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    async with httpx.AsyncClient() as client:
        with out.open("a", encoding="utf-8") as fh:
            for c in claims:
                try:
                    pack = load_pack(c["id"])
                except PackError as exc:
                    print(f"skip {c['id']}: {exc}")
                    continue
                t0 = time.perf_counter()
                try:
                    res = await check_claim(pack["claim"], pack, checker=checker,
                                            stub=a.stub_models, client=client, explain=a.explain)
                except Exception as exc:  # recorded, not hidden
                    res = {"error": f"{type(exc).__name__}: {exc}"}
                secs = round(time.perf_counter() - t0, 2)
                for row in rows_for(c, pack, res, version, secs):
                    fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
                fh.flush()
                n += 1
                # Test verdicts are never printed: reading them would steer changes.
                if res.get("error"):
                    shown = "ERROR " + res["error"][:120]
                elif split_of(c["id"]) == "dev":
                    shown = (f"{res['final']:<22}gold={c['label']:<22}"
                             f"{res['path']}, {res.get('reason') or ''}")
                else:
                    shown = "(test: hidden)"
                print(f"{c['id']:<26}{secs:>7.1f}s  {shown}")
    print(f"\n{n} claims written to {out} ({version})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="study/results/factshield_v2.jsonl")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--ids", help="comma-separated claim ids")
    ap.add_argument("--explain", action="store_true",
                    help="also write the checked explanation (one extra call per claim)")
    ap.add_argument("--stub-models", action="store_true", help="offline stand-in models")
    ap.add_argument("--stub-checker", action="store_true", help="word-overlap stand-in for MiniCheck")
    ap.add_argument("--allow-stub-checker", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
