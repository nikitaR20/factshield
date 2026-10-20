from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from study import analyse, packs, run, sample_citations, score
from study.common import path, read_jsonl


def _row(cid, cond, pred, run_index=0, confidence=None, cited=(), urls=(), model="m"):
    return {"claim_id": cid, "condition": cond, "model": model, "model_version": "v",
            "run_index": run_index, "live_search": False, "parsed_verdict": pred,
            "confidence": confidence, "cited_ids": list(cited), "cited_urls": list(urls),
            "tool_calls": [], "pack_hash": "h", "error": None}


CLAIMS = {
    "a": {"id": "a", "set": "A", "domain": "x", "source_id": "s1", "label": "supported"},
    "b": {"id": "b", "set": "A", "domain": "x", "source_id": "s1", "label": "refuted"},
    "c": {"id": "c", "set": "B", "domain": "x", "source_id": "s2", "label": "not_enough_evidence"},
}
PACKS = {k: {"passages": [{"pid": "P1", "url": "https://www.e.example/p/"}]} for k in CLAIMS}


def test_confidently_wrong_ignores_format_confidence():
    rows = [_row("b", "format", "supported", confidence="low"),
            _row("b", "free", "supported")]
    s = score.score_rows(rows, CLAIMS, PACKS)
    assert [x["confidently_wrong"] for x in s] == [1, 1]


def test_unparseable_stays_in_denominators():
    rows = [_row("a", "free", "supported"), _row("b", "free", "unparseable"),
            _row("c", "free", "not_enough_evidence")]
    s = score.score_rows(rows, CLAIMS, PACKS)
    summ = score.summarise(s, ("model", "condition"))[0]
    assert summ["n"] == 3 and summ["coverage"] == round(1 / 3, 4)
    assert summ["unparseable_rate"] == round(1 / 3, 4)
    assert summ["abstention_precision"] == 1.0 and summ["abstention_recall"] == 1.0
    # supported F1=1, refuted F1=0 (missed), NEI F1=1 -> 2/3
    assert summ["macro_f1"] == round(2 / 3, 4)


def test_citation_in_pack_vs_fabricated_and_urls():
    rows = [_row("a", "citation", "supported", cited=["P1", "P7"],
                 urls=["http://e.example/p", "https://other.example/q"])]
    s = score.score_rows(rows, CLAIMS, PACKS)[0]
    assert (s["n_cited_in_pack"], s["n_cited_fabricated"]) == (1, 1)
    assert s["n_urls_outside_pack"] == 1   # www/scheme/trailing slash normalised
    assert s["source_in_pack"] is None     # fixture claim has no source_url


def test_source_in_pack():
    claims = {"a": dict(CLAIMS["a"], source_url="https://e.example/p")}
    s = score.score_rows([_row("a", "free", "supported")], claims, PACKS)[0]
    assert s["source_in_pack"] == 1


def test_change_rates():
    rows = [_row("a", "free", "supported", 0), _row("a", "format", "refuted", 0),
            _row("a", "citation", "supported", 0),
            _row("a", "free", "supported", 1), _row("a", "format", "refuted", 1),
            _row("a", "citation", "refuted", 1)]
    s = score.score_rows(rows, CLAIMS, PACKS)
    ch = {(c["measure"], c["pair"]): c["rate"] for c in score.change_rates(s, ["free", "format", "citation"])}
    assert ch[("verdict_change_any", "all")] == 1.0
    assert ch[("verdict_change_pair", "free|format")] == 1.0
    assert ch[("run_to_run_change", "citation")] == 1.0
    assert ch[("run_to_run_change", "free")] == 0.0


def test_latest_rows_prefers_success():
    rows = [dict(_row("a", "free", "supported"), error="boom"), _row("a", "free", "refuted"),
            dict(_row("b", "free", "supported"), error="boom")]
    kept, failed = score.latest_rows(rows)
    assert [r["parsed_verdict"] for r in kept] == ["refuted"] and failed == 1


def test_stratified_sample_reports_shortfall():
    units = ([{"condition": "citation", "model": "m", "claim_id": f"c{i}", "run_index": 0,
               "sentence_index": 0, "cited": "P1"} for i in range(40)]
             + [{"condition": "free", "model": "m", "claim_id": "c0", "run_index": 0,
                 "sentence_index": 0, "cited": "P1"}])
    drawn, report = sample_citations.stratified(units, 30, seed=1)
    rep = {r["condition"]: r for r in report}
    assert rep["citation"]["drawn"] == 10          # capped at its share, not refilled
    assert rep["free"]["drawn"] == 1 and rep["free"]["shortfall"] == 9
    assert rep["format"]["available"] == 0


def test_end_to_end_scoring_and_worksheet(study_paths, fake_retriever):
    asyncio.run(packs.build_all(None, None, fake_retriever))
    claims, pks, _ = run.select_claims(None, include_flagged=False)
    plan = run.build_plan(claims, pks, ["stub"], repeat=2, live=False, seed=7)
    out = path("responses") / "main.jsonl"
    asyncio.run(run.execute(plan, out, concurrency=2))
    res = score.run_scoring(out)
    assert len(res["scored"]) == 12
    rows, _ = score.latest_rows(read_jsonl(out))
    units = sample_citations.collect_units(rows, {c["id"]: c for c in claims})
    assert units and all("passage_text" in u for u in units)
    assert any(u["in_pack"] == 0 for u in units) or all(u["in_pack"] == 1 for u in units)


@pytest.mark.slow
def test_mixed_model_recovers_planted_effect():
    df = analyse.synthetic(20261019)
    df = df[df["run_index"] == 0]
    m = analyse.fit_mixed(df, "correct", "laplace").set_index("term")
    cit = m.loc["condition[citation]"]
    # Whether one dataset DETECTS the effect is a power question (see the
    # --reps coverage study). What must hold for one dataset: right sign,
    # truth inside the interval.
    assert cit["mean"] < 0
    assert cit["lo"] <= analyse.TRUTH["condition[citation]"] <= cit["hi"]
    fmt = m.loc["condition[format]"]
    assert fmt["lo"] <= 0 <= fmt["hi"]                     # null not "found"


def test_real_data_analysis_is_locked(monkeypatch):
    monkeypatch.setattr("sys.argv", ["analyse"])
    with pytest.raises(SystemExit, match="locked"):
        analyse.main()
