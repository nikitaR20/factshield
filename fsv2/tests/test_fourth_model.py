"""Tests for the Anthropic adapter and fsv2/fourth_model.py. No network."""

import asyncio
import json

import httpx
import pytest

import fsv2  # noqa: F401  (sets sys.path)
from fsv2 import fourth_model as fm
from study import models as sm

NEE = "not_enough_evidence"


# ------------------------------------------------------------ adapter

def _cfg(**over):
    c = {"provider": "anthropic", "model": "claude-haiku-5-5", "temperature": None,
         "thinking": "disabled", "effort": "low", "max_output_tokens": 4000, "timeout_s": 5}
    c.update(over)
    return c


def _call(cfg, handler, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    r = sm.CallResult(provider="anthropic", model_requested=cfg["model"])
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def go():
        try:
            await sm._anthropic(cfg, "SYS", "USER", False, client, r)
        finally:
            await client.aclose()
    asyncio.run(go())
    return r


def test_anthropic_request_and_parse(monkeypatch):
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["headers"] = dict(req.headers)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "model": "claude-haiku-5-5", "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": ""},
                        {"type": "text", "text": '{"verdict": "supported"'},
                        {"type": "text", "text": "}"}],
            "usage": {"input_tokens": 1200, "output_tokens": 80}})

    r = _call(_cfg(), handler, monkeypatch)
    b = seen["body"]
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "test-key"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert b["system"] == "SYS" and b["messages"] == [{"role": "user", "content": "USER"}]
    assert b["thinking"] == {"type": "disabled"}
    assert b["output_config"] == {"effort": "low"}
    assert "temperature" not in b            # Haiku 5.5 rejects non-default values
    assert b["max_tokens"] == 4000
    assert r.text == '{"verdict": "supported"}'
    assert r.reasoning_text is None          # empty thinking block ignored
    assert r.model_version == "claude-haiku-5-5" and r.finish_reason == "end_turn"
    assert r.tokens == {"input": 1200, "output": 80, "reasoning": None, "total": 1280}
    assert "messages" not in r.params_sent and "system" not in r.params_sent


def test_anthropic_optional_fields_omitted(monkeypatch):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "x"}], "usage": {}})

    _call(_cfg(thinking=None, effort=None, temperature=0.3), handler, monkeypatch)
    assert "thinking" not in seen["body"] and "output_config" not in seen["body"]
    assert seen["body"]["temperature"] == 0.3


def test_anthropic_400_is_fatal_not_retried(monkeypatch):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "temperature not supported"}})

    with pytest.raises(sm._Fatal):
        _call(_cfg(temperature=0), handler, monkeypatch)
    assert len(calls) == 1


def test_anthropic_registered_and_missing_key(monkeypatch):
    assert sm._PROVIDERS["anthropic"] is sm._anthropic
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = sm.CallResult(provider="anthropic", model_requested="m")

    async def go():
        async with httpx.AsyncClient() as c:
            await sm._anthropic(_cfg(), "s", "u", False, c, r)
    with pytest.raises(sm._Fatal, match="ANTHROPIC_API_KEY"):
        asyncio.run(go())


def test_claude_not_in_study_models():
    from study.common import config
    assert "claude" in config()["models"]
    assert "claude" not in config()["study_models"]      # the study runs are unchanged
    assert config()["models"]["claude"]["temperature"] is None


# ------------------------------------------------------------ rules and stats

def test_rules():
    assert fm.unanimous(["supported"] * 4) == "supported"
    assert fm.unanimous(["supported", "supported", "unparseable"]) is None
    assert fm.at_least(["refuted", "refuted", "refuted", NEE], 3) == "refuted"
    assert fm.at_least(["refuted", "refuted", NEE, NEE], 3) is None
    assert fm.rule_majority(("a", "b", "c", "d"))({"a": "supported", "b": "supported",
                                                   "c": NEE, "d": NEE}) == NEE
    assert fm.pattern(["s", "s", "s", "s"]) == "4"
    assert fm.pattern(["s", "r", "s", "n"]) == "2-1-1"
    assert fm.norm("conflicting_evidence") == NEE


def test_mcnemar_and_holm():
    assert fm.mcnemar_exact(0, 6) == pytest.approx(0.03125)
    assert fm.mcnemar_exact(3, 0) == pytest.approx(0.25)
    assert fm.mcnemar_exact(0, 0) == 1.0
    assert fm.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_score_counts():
    gold = {"a": "supported", "b": "refuted", "c": NEE, "d": NEE}
    pred = {"a": "supported", "b": "supported", "c": NEE, "d": "unparseable"}
    s = fm.score(pred, gold, gold)
    assert (s["correct"], s["answered"], s["confidently_wrong"]) == (2, 2, 1)
    assert s["abstention_accuracy"] == 1.0 and s["unparseable"] == 1


# ------------------------------------------------------------ end to end, offline

PSHA = {"system_sha": "s1", "verdict_sha": "v1"}


def _fs_rows(cid, final, singles, pack_hash="h1"):
    rows = [{"claim_id": cid, "condition": "fs2", "error": None, "pack_hash": pack_hash,
             "model_version": "fsv2-0.2/minicheck:flan-t5-large",
             "factshield_v2": {"final": final, "prompt_sha": PSHA, "samples": [
                 {"model_key": m, "reason": f"{m} reason"} for m in fm.BASE3]}}]
    rows += [{"claim_id": cid, "condition": "fs2_single", "model": f"fs2:{m}",
              "model_version": f"fsv2-0.2/minicheck:flan-t5-large/{m}-v",
              "parsed_verdict": v, "error": None} for m, v in zip(fm.BASE3, singles)]
    return rows


def test_run_and_analyse_offline(tmp_path, monkeypatch, capsys):
    from fsv2 import llm
    import study.packs

    # claim X: shared error (all three say supported, gold NEE); Claude says NEE
    # claim Y: all right; claim Z: models disagree
    res = tmp_path / "fs.jsonl"
    rows = (_fs_rows("claims_Dataset_A-9001", "supported", ["supported"] * 3)
            + _fs_rows("claims_Dataset_A-9002", "refuted", ["refuted"] * 3)
            + _fs_rows("claims_Dataset_A-9003", NEE, ["supported", NEE, "refuted"]))
    res.write_text("\n".join(json.dumps(r) for r in rows))
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"claims_Dataset_A-9001": NEE, "claims_Dataset_A-9002": "refuted",
                                "claims_Dataset_A-9003": "supported"}))

    def fake_pack(cid):
        return {"claim_id": cid, "claim": f"claim {cid}", "pack_hash": "h1",
                "passages": [{"pid": "P1", "url": "https://bbc.com/x", "title": "t",
                              "source_domain": "bbc.com", "published_date": None,
                              "text": "Some passage text about the matter at hand."}]}
    monkeypatch.setattr(study.packs, "load_pack", fake_pack)
    monkeypatch.setattr(llm, "panel_cfg", lambda key, **k: {"provider": "anthropic", "model": "m"})
    answers = {"claims_Dataset_A-9001": NEE, "claims_Dataset_A-9002": "refuted",
               "claims_Dataset_A-9003": "supported"}
    calls = []

    async def fake_call(cfg, system, user, client):
        cid = next(c for c in answers if f"claim {c}" in user)
        calls.append(cid)
        r = sm.CallResult(provider="anthropic", model_requested="m", model_version="m")
        r.text = json.dumps({"verdict": answers[cid], "passages": [], "key_statements": [],
                             "reason": "r"})
        r.tokens = {"input": 10, "output": 5}
        return r
    monkeypatch.setattr(llm, "call", fake_call)
    from fsv2 import decide
    real_build = decide.build_prompt
    monkeypatch.setattr(decide, "build_prompt",
                        lambda claim, items: (real_build(claim, items)[0],
                                              real_build(claim, items)[1], PSHA))

    out = tmp_path / "claude.jsonl"
    fm.main(["run", "--results", str(res), "--out", str(out), "--limit", "2"])
    fm.main(["run", "--results", str(res), "--out", str(out)])      # resumes: only the third
    assert len(calls) == 3 and len(set(calls)) == 3
    printed = capsys.readouterr().out
    assert "supported" not in printed and "refuted" not in printed  # verdicts never shown

    ids_new = tmp_path / "new.txt"; ids_new.write_text("")
    excl = tmp_path / "ex.txt"; excl.write_text("claims_Dataset_A-9001\n")
    rep = fm.analyse(fm.argparse.Namespace(
        results=[str(res)], fourth=str(out), model_key="claude", gold=str(gold),
        new_ids=str(ids_new), exclude=str(excl), worksheet=str(tmp_path / "w.csv"),
        summary=str(tmp_path / "s.json")))
    allc = rep["subsets"]["all"]
    assert allc["U3 openai+gemini+groq (simulated)"]["confidently_wrong"] == 1
    assert allc["U4 all four agree"]["confidently_wrong"] == 0
    assert allc["U4 all four agree"]["correct"] == 2
    assert allc["M4 at least 3 of 4 agree"]["confidently_wrong"] == 1
    assert rep["subsets"]["without 1 audited-label claims"]["U3 openai+gemini+groq (simulated)"]["n"] == 2
    assert rep["shared_errors"][0]["claude"] == NEE
    assert len(rep["combinations"]) == 4 + 6 + 4 * 2 + 2
    assert len((tmp_path / "w.csv").read_text().strip().splitlines()) == 1 + 2   # header + 2


def test_run_refuses_changed_pack(tmp_path, monkeypatch):
    from fsv2 import llm
    import study.packs
    res = tmp_path / "fs.jsonl"
    res.write_text("\n".join(json.dumps(r) for r in _fs_rows("claims_Dataset_A-9001", NEE,
                                                             [NEE] * 3, pack_hash="old")))
    monkeypatch.setattr(study.packs, "load_pack", lambda cid: {"claim": "c", "pack_hash": "new",
                                                               "passages": []})

    async def boom(*a, **k):
        raise AssertionError("must not call the model")
    monkeypatch.setattr(llm, "call", boom)
    out = tmp_path / "c.jsonl"
    fm.main(["run", "--results", str(res), "--out", str(out)])
    row = json.loads(out.read_text())
    assert "pack hash" in row["error"]


def test_old_version_rows_ignored(tmp_path, capsys):
    """A file mixing dev run 1 (fsv2-0.1) with v2 rows: only fsv2-0.2 is read,
    whatever the file order."""
    new = _fs_rows("claims_Dataset_A-9001", "supported", ["supported"] * 3)
    old = _fs_rows("claims_Dataset_A-9001", "refuted", ["refuted"] * 3)
    for r in old:
        r["model_version"] = "fsv2-0.1/minicheck:flan-t5-large"
    stub = _fs_rows("claims_Dataset_A-9001", NEE, [NEE] * 3)
    for r in stub:
        r["model_version"] = "fsv2-0.2/stub/STUB-MODELS"
    f = tmp_path / "mixed.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in new + old + stub))
    saved = fm.load_saved([str(f)])
    assert saved["final"]["claims_Dataset_A-9001"] == "supported"
    assert saved["single"][("claims_Dataset_A-9001", "groq")] == "supported"
    assert "skipped 8 rows" in capsys.readouterr().out


def test_run_refuses_changed_prompt(tmp_path, monkeypatch):
    from fsv2 import decide, llm
    import study.packs
    res = tmp_path / "fs.jsonl"
    res.write_text("\n".join(json.dumps(r) for r in _fs_rows("claims_Dataset_A-9001", NEE, [NEE] * 3)))
    monkeypatch.setattr(study.packs, "load_pack", lambda cid: {
        "claim": "c", "pack_hash": "h1", "passages": [
            {"pid": "P1", "url": "https://bbc.com/x", "title": "t", "source_domain": "bbc.com",
             "published_date": None, "text": "Some passage text about the matter at hand."}]})
    monkeypatch.setattr(decide, "build_prompt",
                        lambda claim, items: ("s", "u", {"system_sha": "s1", "verdict_sha": "CHANGED"}))

    async def boom(*a, **k):
        raise AssertionError("must not call the model")
    monkeypatch.setattr(llm, "call", boom)
    out = tmp_path / "c.jsonl"
    fm.main(["run", "--results", str(res), "--out", str(out)])
    assert "prompt differs" in json.loads(out.read_text())["error"]
