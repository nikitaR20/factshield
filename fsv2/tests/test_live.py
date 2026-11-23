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


# ---- demo fixes 2026-11-09 (FT / GLP-1 check) ---------------------------

from fsv2 import evidence as ev  # noqa: E402
from fsv2.settings import prompt  # noqa: E402


def test_live_extraction_uses_v3_and_thesis_prompts_are_unchanged():
    v3, _ = prompt("extract", live._live()["extract_prompt_version"])
    assert "Keep who said it" in v3 and "Never turn \"associated with\" into \"causes\"" in v3
    # Thesis prompts: same files and hashes as the frozen runs.
    assert prompt("verdict")[1].startswith("148d0c5b")
    assert prompt("explain")[1].startswith("147a7538")
    assert "Keep who said it" not in prompt("extract")[0]


def test_live_source_types_for_the_glp1_sources():
    t = ev.live_source_type
    assert t("www.aao.org") == "medical or professional body"
    assert t("aoa.org") == "medical or professional body"
    assert t("nm.org") == "hospital or health system"
    assert t("goodrx.com") == "health information website"
    assert t("pmc.ncbi.nlm.nih.gov") == "research paper or journal"
    assert t("reuters.com") == "news outlet"            # falls back to the thesis rule
    assert t("example.org") == "other website"


def test_thesis_packs_render_exactly_as_before():
    # No source_type / rating_applies keys -> thesis labels and ratings.
    items = ev.prepare({"passages": [{"pid": "P1", "url": "https://goodrx.com/a", "title": "t",
                                      "source_domain": "goodrx.com", "text": "x " * 20}]})
    assert items[0].source_type == "news outlet" and items[0].rating_applies
    assert "news-quality rating 0." in ev.render(items)


def test_live_pack_labels_and_hides_news_rating_where_it_does_not_apply():
    pack = live.as_pack("c", [], [
        {"url": "https://pmc.ncbi.nlm.nih.gov/articles/1", "source_domain": "pmc.ncbi.nlm.nih.gov",
         "text": "x", "channel": "web"},
        {"url": "https://www.reuters.com/a", "source_domain": "reuters.com", "text": "x", "channel": "web"}])
    items = ev.prepare(pack)
    assert items[0].source_type == "research paper or journal" and not items[0].rating_applies
    assert items[1].source_type == "news outlet" and items[1].rating_applies
    text = ev.render(items)
    assert "not applicable (not a news site)" in text and "news-quality rating 1.00" in text


def test_quotes_are_cut_at_a_word_and_prefer_real_sentences():
    long = "Warning signs include " + "sudden vision loss in one eye, dark spots, " * 15
    q = live._best_sentence(long + ". NAION can cause sudden, painless vision loss.", "NAION vision loss")
    assert q == "NAION can cause sudden, painless vision loss."
    cut = live._trim("word " * 100)
    assert cut.endswith(" …") and len(cut) <= live.QUOTE_CHARS + 2 and "wor …" not in cut


def test_attributed_claims_are_flagged_and_wording_is_not_independent():
    assert live.is_attributed("The FDA said NAION can cause vision loss.")
    assert not live.is_attributed("NAION can cause sudden, painless vision loss.")
    txt = live._reason_text({"final": "supported"}, 8, 4)
    assert "independent" not in txt and "4 organisations" in txt


def test_attribution_detection_ignores_ambiguous_nouns():
    assert not live.is_attributed("The United States has 50 states.")
    assert not live.is_attributed("Insurance claims rose 5% in 2025.")
    assert not live.is_attributed("Reported cases of measles rose in Texas.")
    assert live.is_attributed("Reuters reported that the deal collapsed.")
    assert live.is_attributed("According to WHO, cases fell.")
