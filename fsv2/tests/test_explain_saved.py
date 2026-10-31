"""explain_saved: Stage 6 on saved results, offline (stub models, stub checker)."""

import asyncio
import json
from types import SimpleNamespace

import pytest

import fsv2  # noqa: F401
from fsv2 import checker as chk
from fsv2 import decide, explain_saved, llm
from fsv2.run_packs import rows_for



def _models():
    return {"models": {k: {"provider": k, "model": f"m-{k}", "temperature": 0,
                           "max_output_tokens": 100, "timeout_s": 1}
                       for k in ("openai", "gemini", "groq")}}


BASIC = {"claim_id": "c1", "claim": "The tower was completed in 1999.", "pack_hash": "h",
         "passages": [{"pid": f"P{i+1}", "url": f"https://{d}/a{i}", "title": "t",
                       "source_domain": d, "published_date": None, "text": t}
                      for i, (d, t) in enumerate([
                          ("bbc.com", "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."),
                          ("travelblog.example", "The tower lights up every evening and draws many visitors."),
                          ("reuters.com", "Engineers finished the iron tower in 1889, records show.")])]}


@pytest.fixture(autouse=True)
def fixed_models(monkeypatch):
    monkeypatch.setattr(llm, "study_config", lambda: _models())


def _saved_file(tmp_path):
    res = asyncio.run(decide.check_claim(BASIC["claim"], BASIC, checker=chk.StubChecker(),
                                         stub=True, client=None))
    rows = rows_for({"id": "c1"}, BASIC, res, "test-version", 0.1)
    p = tmp_path / "saved.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p, res


def _args(saved, out, **kw):
    return SimpleNamespace(results=[str(saved)], out=str(out), all=True, stub_models=True,
                           stub_checker=True, **kw)


def test_explains_saved_verdict_without_changing_it(tmp_path, monkeypatch, capsys):
    saved, res = _saved_file(tmp_path)
    monkeypatch.setattr(explain_saved, "load_pack", lambda cid: BASIC)
    out = tmp_path / "expl.jsonl"
    asyncio.run(explain_saved.run(_args(saved, out)))
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert r["claim_id"] == "c1" and r["final"] == res["final"]
    e = r["explanation"]
    assert set(e) >= {"sentences", "dropped", "quotes_only"}
    # resumable: a second run skips the finished claim
    asyncio.run(explain_saved.run(_args(saved, out)))
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1
    assert "claims explained: 1" in capsys.readouterr().out


def test_refuses_when_pack_changed(tmp_path, monkeypatch):
    saved, _ = _saved_file(tmp_path)
    monkeypatch.setattr(explain_saved, "load_pack", lambda cid: {**BASIC, "pack_hash": "different"})
    with pytest.raises(SystemExit):
        asyncio.run(explain_saved.run(_args(saved, tmp_path / "x.jsonl")))
