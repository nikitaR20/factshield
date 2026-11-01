"""Tests for the browser-demo pipeline (fsv2/live.py, fsv2/api.py). Offline."""

import asyncio
import time

import pytest

import fsv2  # noqa: F401
from fsv2 import api, live, llm
from fsv2.checker import StubChecker


@pytest.fixture(autouse=True)
def fixed_models(monkeypatch):
    models = {k: {"provider": k, "model": f"m-{k}", "temperature": 0, "max_output_tokens": 100,
                  "timeout_s": 1} for k in ("openai", "gemini", "groq")}
    monkeypatch.setattr(llm, "study_config", lambda: {"models": models})


def run(coro):
    return asyncio.run(coro)


def test_word_cap_is_enforced_not_truncated():
    with pytest.raises(live.TooLong) as e:
        run(live.extract({"selection": "word " * 801}, stub=True, client=None))
    assert e.value.n == 801 and e.value.cap == 800


def test_extract_returns_standalone_claims_and_skipped():
    sel = "The council voted to close the old bridge on Tuesday. Repairs will take two years, officials said."
    out = run(live.extract({"selection": sel}, stub=True, client=None))
    assert out["error"] is None and out["claims"] and out["skipped"]
    c = out["claims"][0]
    assert c["checking_as"] and len(c["queries"]) == 2 and isinstance(c["resolved"], bool)


class NoSearch(live.Search):
    async def factchecks(self, q, client):
        return [], None

    async def web(self, q, inc, client):
        return [], "web search HTTP 432 Tavily plan credits used up"


def test_no_sources_means_no_model_calls_and_error_is_shown():
    steps = []
    out = run(live.check_claim("Some claim about a bridge", ["bridge"], checker=StubChecker(),
                               search=NoSearch(), stub=True, client=None, progress=steps.append))
    assert out["label"] == "not_enough_evidence" and out["n_model_calls"] == 0
    assert "couldn't find sources" in out["reason_text"]
    assert any("432" in e for e in out["search_errors"])


def test_unrelated_factcheck_is_related_not_evidence():
    out = run(live.check_claim("The council voted to close the bridge", ["council bridge vote"],
                               checker=StubChecker(), search=live.StubSearch(), stub=True,
                               client=None, progress=lambda m: None))
    related = [r["reviewed_claim"] for r in out["related_factchecks"]]
    assert related == ["A different claim"]
    fc_sources = [s for s in out["sources"] if s["source_type"] == "fact-check"]
    assert len(fc_sources) == 1 and "council voted" in fc_sources[0]["text"]


def test_web_passages_drop_social_offtopic_and_duplicates():
    claim = "The council voted to close the old bridge for repairs"
    docs = [
        {"url": "https://x.com/a", "source_domain": "x.com", "text": claim * 3, "title": ""},
        {"url": "https://news.example/1", "source_domain": "news.example", "title": "",
         "text": "The council voted on Tuesday to close the old bridge for repairs lasting two years. "
                 "Engineers had warned about cracks."},
        {"url": "https://news.example/2", "source_domain": "news.example", "title": "",
         "text": "The council voted on Tuesday to close the old bridge for repairs lasting two years. "
                 "Engineers had warned about cracks."},
        {"url": "https://other.example/3", "source_domain": "other.example", "title": "",
         "text": "A recipe for lemon cake with butter, sugar, flour and eggs baked for forty minutes."},
    ]
    kept, dropped = live.web_passages(claim, docs, set())
    assert [k["url"] for k in kept] == ["https://news.example/1"]
    reasons = " ".join(d["because"] for d in dropped)
    assert "social" in reasons and "near-duplicate" in reasons and "off-topic" in reasons


def test_api_flow_and_limits(monkeypatch):
    monkeypatch.setattr(api, "STUB", True)
    from fastapi.testclient import TestClient
    with TestClient(api.app) as c:
        assert c.get("/v2/health").json()["stub"] is True
        assert c.post("/v2/extract", json={"selection": "w " * 900}).status_code == 413
        too_many = [{"checking_as": "a claim"}] * 4
        assert c.post("/v2/check", json={"claims": too_many}).status_code == 400
        jid = c.post("/v2/check", json={"claims": [{"checking_as": "The bridge closed in 2026",
                                                    "queries": ["bridge closed", "bridge 2026"]}]}
                     ).json()["job_id"]
        for _ in range(200):
            j = c.get(f"/v2/jobs/{jid}").json()
            if j["status"] == "done":
                break
            time.sleep(0.02)
        slot = j["claims"][0]
        assert slot["status"] == "done" and slot["steps"][0]["text"].startswith("Searching")
        r = slot["result"]
        for k in ("label", "label_text", "reason_text", "key_quotes", "explanation", "sources",
                  "related_factchecks", "checked_on", "n_sources"):
            assert k in r
        assert c.get("/v2/jobs/nope").status_code == 404


def test_friendly_errors():
    assert "overloaded" in api._friendly("HTTP 503: high demand")
    assert "daily allowance" in api._friendly("on tokens per day (TPD): Limit 200000")
    assert "API key" in api._friendly("OPENAI_API_KEY not set")
