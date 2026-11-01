"""FactShield on frozen study packs: plumbing and the scoring contract.

Runs with the keyword NLI stub, so the verdicts themselves mean nothing here;
what is tested is that pack passages reach FactShield's grading, that no fact-
check path can fire, and that the rows score with the study's own code.
"""

from __future__ import annotations

import asyncio

from eval import from_packs as F
from study import packs, score
from study.common import config, load_claims, path, read_jsonl
from study.tests.conftest import FIXTURES, fake_docs


def _setup(tmp_path, monkeypatch):
    paths = config()["paths"]
    claims_dir = tmp_path / "claims"
    claims_dir.mkdir()
    (claims_dir / "fixture.jsonl").write_text((FIXTURES / "claims.jsonl").read_text())
    for k, v in (("claims", claims_dir), ("packs", tmp_path / "packs"),
                 ("responses", tmp_path / "responses"), ("results", tmp_path / "results")):
        monkeypatch.setitem(paths, k, str(v))
    monkeypatch.setenv("TAVILY_API_KEY", "fixture")
    monkeypatch.setitem(config()["packs"], "processing",
                        {**(config()["packs"].get("processing") or {}), "enabled": False,
                         "warn_below_passages": 0, "warn_best_overlap_below": 0})

    async def retriever(claim, category, jurisdiction, concepts=None):
        return fake_docs(claim)
    asyncio.run(packs.build_all(None, None, retriever))


def test_rows_score_with_study_code(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out = path("responses") / "factshield.jsonl"

    class Args:
        pass
    a = Args()
    a.out, a.conditions, a.ids, a.allow_stub, a.split, a.nli_model = str(out), "fs_computed,fs_no_abstain", None, True, "all", None
    asyncio.run(F.main_async(a))

    rows = read_jsonl(out)
    n_claims = len(load_claims())
    assert len(rows) == 2 * n_claims and all(r["error"] is None for r in rows)
    assert all(r["parsed_verdict"] in ("supported", "refuted", "not_enough_evidence",
                                       "partly_supported") for r in rows)
    # no fact-check rating can be inherited from a pack
    assert all(not any(i["tier"] == "prior_check" for i in r["factshield"]["items"]) for r in rows)
    # the ablation never abstains where the evidence leans
    for r in rows:
        if r["condition"] == "fs_no_abstain" and r["factshield"]["forced"]:
            assert r["parsed_verdict"] in ("supported", "refuted", "not_enough_evidence")

    res = score.run_scoring(out)
    assert len(res["scored"]) == len(rows)


def test_partly_supported_is_committed_and_wrong():
    claims = {"a": {"id": "a", "set": "A", "domain": "x", "source_id": "s", "label": "supported"}}
    row = {"claim_id": "a", "condition": "fs_computed", "model": "factshield", "model_version": "v",
           "run_index": 0, "live_search": False, "parsed_verdict": "partly_supported",
           "response_text": None, "cited_ids": [], "cited_urls": [], "tool_calls": [],
           "pack_hash": "h", "error": None}
    s = score.score_rows([row], claims, {"a": {"passages": []}})[0]
    assert s["committed"] == 1 and s["correct"] == 0 and s["confidently_wrong"] == 1


def test_split_is_fixed_and_roughly_a_third():
    ids = [f"claims - Dataset A-{i:04d}" for i in range(1, 46)]
    dev = [i for i in ids if F.split_of(i) == "dev"]
    assert F.split_of(ids[0]) == F.split_of(ids[0])
    assert 8 <= len(dev) <= 22
