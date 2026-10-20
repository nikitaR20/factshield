from __future__ import annotations

import asyncio

from study import packs, probe, run
from study.common import config, path, read_jsonl

REQUIRED = ["claim_id", "condition", "model", "model_version", "run_index", "live_search",
            "pack_hash", "timestamp", "raw_response", "parsed_verdict", "confidence",
            "cited_ids", "tool_calls", "latency_ms", "error"]


def _setup(fake_retriever):
    asyncio.run(packs.build_all(None, None, fake_retriever))
    claims, pks, excluded = run.select_claims(None, include_flagged=False)
    return claims, pks, excluded


def test_flagged_claims_excluded(study_paths, fake_retriever):
    claims, _, excluded = _setup(fake_retriever)
    assert {c["id"] for c in claims} == {"t001", "t002"}
    assert excluded[0][0] == "t003" and "unretrievable" in excluded[0][1]


def test_rows_have_required_fields_and_order_is_shuffled(study_paths, fake_retriever):
    claims, pks, _ = _setup(fake_retriever)
    plan = run.build_plan(claims, pks, ["stub"], repeat=3, live=False, seed=config()["seed"])
    assert len(plan) == 2 * 3 * 3
    seq = [p["condition"] for p in plan]
    assert seq != sorted(seq)
    assert [p["run_index"] for p in plan] != sorted(p["run_index"] for p in plan)
    out = path("responses") / "main.jsonl"
    stats = asyncio.run(run.execute(plan, out, concurrency=3))
    assert stats["ok"] == 18
    rows = read_jsonl(out)
    for r in rows:
        for f in REQUIRED:
            assert f in r, f
        assert r["tool_calls"] == [] and r["live_search"] is False
        assert r["pack_hash"] == pks[r["claim_id"]]["pack_hash"]
        assert r["git_commit"] and r["seed"] == config()["seed"]


def test_resume_skips_done_and_retries_errors(study_paths, fake_retriever, monkeypatch):
    claims, pks, _ = _setup(fake_retriever)
    plan = run.build_plan(claims, pks, ["stub"], repeat=1, live=False, seed=1)
    out = path("responses") / "main.jsonl"

    real = run.models.complete
    calls = {"n": 0}

    async def flaky(*a, **k):
        calls["n"] += 1
        res = await real(*a, **k)
        if calls["n"] == 2:
            res.error = "simulated failure"
        return res

    monkeypatch.setattr(run.models, "complete", flaky)
    s1 = asyncio.run(run.execute(plan, out, concurrency=1))
    assert s1["errors"] == 1
    s2 = asyncio.run(run.execute(plan, out, concurrency=1))
    assert s2["already_done"] == 5 and s2["ok"] == 1
    rows = read_jsonl(out)
    assert len(rows) == 7     # the error row is kept


def test_closed_book_and_leak_summaries(study_paths, fake_retriever):
    from study.common import load_claims
    claims = load_claims()
    plan = probe.plan_closed_book(claims, ["stub"], 2, seed=1)
    out = path("responses") / probe.CLOSED
    asyncio.run(run.execute(plan, out, concurrency=2))
    flags, agg = probe.summarise_closed_book(read_jsonl(out), {c["id"]: c for c in claims})
    assert len(flags) == 3
    assert next(f for f in flags if f["claim_id"] == "t003")["knew_unaided"] is None  # NEI gold
    assert {a["set"] for a in agg} == {"B"}

    chosen = probe.leak_claims(claims, seed=1)
    assert {c["id"] for c in chosen} <= {"t001", "t002"}


def test_leak_judgement():
    claims = {f"c{i}": {"label": "refuted"} for i in range(8)}
    rows = [{"claim_id": f"c{i}", "model": "m", "run_index": 0, "error": None,
             "parsed_verdict": "refuted", "tool_calls": [], "cited_urls": []} for i in range(8)]
    assert probe.judge_leak(rows, claims)[0]["result"] == "FAIL"
    rows = [dict(r, parsed_verdict="not_enough_evidence") for r in rows]
    assert probe.judge_leak(rows, claims)[0]["result"] == "PASS"
    rows[0]["tool_calls"] = [{"type": "groundingMetadata"}]
    assert probe.judge_leak(rows, claims)[0]["result"] == "FAIL"
