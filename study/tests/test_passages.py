"""Text rules for passages, tested on the cases found in the Dataset A audit.

Snippets are short excerpts of retrieved pages, quoted to reproduce the exact
failure each rule exists for.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from study import packs, passages as P
from study.common import config, load_claims


@pytest.fixture(autouse=True)
def rules_on(monkeypatch):
    monkeypatch.setitem(config()["packs"], "processing",
                        {**config()["packs"]["processing"], "enabled": True})


# ------------------------------------------------- fact-check attributions

@pytest.mark.parametrize("text", [
    # 0005: content-farm page reporting the fact-checkers' finding on this claim
    "These fact-checkers compare the alleged death date with public records. In this case, their "
    "analysis showed that the image used was from a 2014 interview, not a recent hospital scene.",
    # 0008: Salon
    "They were right. It wasn't a scam. As verified by Snopes, texting RBG to 50409 is part of an "
    "automated service run by Resistbot.",
    # 0031: iqmetrics
    "Snopes examined the claim and could not find a year, a test name, or an administering psychologist.",
    # 0031: riotiq
    "No biographer has produced documentation. According to Snopes, Monroe's extensive personal "
    "archives contain no test results.",
    # 0034: Yahoo
    "But a closer look at the intruders might change your assumptions. Snopes reports that the bugs "
    "resemble pantry moth larvae.",
    # 0003: a TV station's fact-check segment
    "With the passing of Pope Francis, we're looking back at one of our favorite 'Fact or Fictions'. "
    "But it was fiction.",
    "PolitiFact rated the statement Mostly False.",
    "The claim was checked by the fact-checkers at Lead Stories.",
])
def test_factcheck_attribution_detected(text):
    assert P.factcheck_mention(text)


@pytest.mark.parametrize("text", [
    # 0003 VOA: the outlet's own reporting, with evidence, not a borrowed verdict
    "But it turned out to be a fake. The image was taken from a video chat the pontiff did in "
    "September of 2014.",
    # 0017 CNN: "hoax" inside a quote
    'Woolery said he "never said Covid was a hoax, but it was the only way" his critics could twist it.',
    # 0023: the spreading site's own byline
    "Hillary Campaign Caught Overcharging Poor Donors. Fact checked by The People's Voice Community.",
    # a general remark, no finding on this claim
    "You might notice that sites like Snopes and FactCheck.org routinely debunk celebrity death hoaxes.",
])
def test_ordinary_text_not_flagged(text):
    assert P.factcheck_mention(text) is None


# ------------------------------------------------------------- cleaning

AP_MENU = """TOP STORIES
       Philadelphia Eagles quarterback Jalen Hurts joins list of athletes diagnosed with blood clots
Newsletters

AP Top 25 Poll Alerts Get email alerts for every college football Top 25 Poll release.

See All Newsletters
Entertainment
SECTIONS MoviesFashionTelevisionCelebrity InterviewsMusicBooksAP Breakthrough Entertainers [...] Newsletters
   Quizzes
   Games
Fact Check
TOP STORIES
       FACT FOCUS: Census Bureau report does not prove Trump's claim about noncitizen voting in 2020
Enable accessibility
Report AdImage 1"""


def test_menu_page_cleaned_and_dropped():
    cleaned, removed = P.clean(AP_MENU)
    assert "Fact Check" not in cleaned and "SECTIONS" not in cleaned and removed >= 8
    kept, dropped = P.process(
        "A Home Depot manager in Spokane Valley, WA, rudely refused to provide a veterans discount.",
        [{"url": "https://apnews.com/x", "channel": "authority", "source_domain": "apnews.com",
          "text": AP_MENU}], 1500)
    assert not kept and dropped[0]["because"].startswith(("off-topic", "too little"))


def test_short_real_snippet_is_kept():
    """0033: search returned only a snippet, but it is the evidence."""
    claim = ("President Trump raised his middle finger to make an obscene gesture at Italian Prime "
             "Minister Paolo Gentiloni during a meeting at the 2017 G7 summit in Italy.")
    snippet = ("At the 2017 G7 Summit in Italy, President Donald Trump appeared to give Italian Prime "
               "Minister Paolo Gentiloni the finger. The incident occurred on the")
    kept, dropped = P.process(claim, [{"url": "https://ranker.com/x", "channel": "open_web",
                                       "source_domain": "ranker.com", "text": snippet}], 1500)
    assert kept and not dropped


def test_paraphrase_survives_off_topic_rule():
    """0015: at a stricter threshold this deciding sentence was dropped."""
    claim = "Evidence shows rare night clouds are linked to human-caused climate change."
    text = ("It is thought that the increase in brightness and frequency of noctilucent clouds over "
            "past decades may be linked to global warming.")
    kept, _ = P.process(claim, [{"url": "https://bbc.co.uk/x", "channel": "authority",
                                 "source_domain": "bbc.co.uk", "text": text}], 1500)
    assert kept


def test_selection_keeps_relevant_sentences_in_order():
    claim = "Steven Hildreth Jr. was let go with a verbal warning after a broken headlight stop."
    filler = " ".join(f"Unrelated sentence number {i} about the weather and sport." for i in range(60))
    text = (filler + " Hildreth said he was stopped because one of his headlights was out. "
            + filler + " In the end, Hildreth was let go with a verbal warning. " + filler)
    out = P.select(claim, text, 500)
    assert len(out) <= 500
    assert out.index("headlights was out") < out.index("verbal warning")
    assert "…" in out


def test_near_duplicates_dropped():
    claim = "Dallas Police Chief David Brown lost a son, brother, and work partner to gun violence."
    body = ("Mr Brown's son, David O'Neal Brown Jr, first shot dead a private citizen, Jeremy McMillan. "
            "He then shot police officer Craig Shaw. Dallas Police Chief David Brown lost his brother "
            "and his former partner to gun violence as well.")
    docs = [{"url": "https://www.bbc.com/news/world-us-canada-36753964", "channel": "open_web",
             "source_domain": "bbc.com", "text": body},
            {"url": "https://www.bbc.co.uk/news/world-us-canada-36753964", "channel": "authority",
             "source_domain": "bbc.co.uk", "text": body + " Share this page."}]
    kept, dropped = P.process(claim, docs, 1500)
    assert len(kept) == 1 and dropped[0]["because"].startswith("near-duplicate of https://www.bbc.com")


# ----------------------------------------------- in the pack builder

def test_pack_records_every_removal_and_raw_cache(study_paths, monkeypatch):
    monkeypatch.setitem(config()["packs"], "processing",
                        {**config()["packs"]["processing"], "enabled": True})
    calls = {"n": 0}

    async def docs(claim, *a, **k):
        calls["n"] += 1
        return [
            {"url": "https://apnews.com/menu", "title": "AP", "channel": "authority",
             "source_domain": "apnews.com", "text": AP_MENU},
            {"url": "https://yahoo.com/a", "title": "Y", "channel": "open_web",
             "source_domain": "yahoo.com",
             "text": "Exampleton's sales tax story. Snopes reports that the rise to 7 percent was approved."},
            {"url": "https://news.example/tax", "title": "N", "channel": "open_web",
             "source_domain": "news.example",
             "text": "Exampleton's council voted to raise the sales tax to 7 percent in September 2026."},
        ]

    asyncio.run(packs.build_all({"t001"}, None, docs))
    p = packs.load_pack("t001")
    assert [x["url"] for x in p["passages"]] == ["https://news.example/tax"]
    why = {e["url"]: e["because"] for e in p["retrieval"]["excluded_by_config"]}
    assert why["https://yahoo.com/a"].startswith("attributes a finding to a fact-checker")
    assert why["https://apnews.com/menu"].startswith(("off-topic", "too little"))
    assert p["retrieval"]["processing"]["version"] == P.VERSION
    raw = json.loads(packs.raw_path("t001").read_text(encoding="utf-8"))
    assert len(raw["docs"]) == 3                                   # everything, before any rule

    # Reprocess from the cache with a changed rule: no new search, old packs kept aside.
    monkeypatch.setitem(config()["packs"]["processing"], "drop_factcheck_mentions", False)
    for cid in ("t002", "t003"):
        asyncio.run(packs.build_all({cid}, None, docs))
    n_before = calls["n"]
    archived = asyncio.run(packs.reprocess(docs))
    assert calls["n"] == n_before                                  # no search
    assert archived and (archived / "t001.json").exists()          # old version kept
    p2 = packs.load_pack("t001")
    assert "https://yahoo.com/a" in [x["url"] for x in p2["passages"]]
    assert p2["retrieval"]["fanout"].get("from_raw_cache") is True


def test_reprocess_refuses_without_raw(study_paths, fake_retriever):
    with pytest.raises(packs.PackError, match="no saved raw results"):
        asyncio.run(packs.reprocess())


def test_weak_pack_warnings(study_paths, monkeypatch):
    monkeypatch.setitem(config()["packs"], "processing",
                        {**config()["packs"]["processing"], "enabled": True,
                         "warn_below_passages": 3, "warn_best_overlap_below": 0.5})

    async def one(claim, *a, **k):
        return [{"url": "https://news.example/a", "title": "N", "channel": "open_web",
                 "source_domain": "news.example",
                 "text": "Exampleton's council discussed the city's tax plans at length last autumn."}]

    asyncio.run(packs.build_all({"t001"}, None, one))
    w = packs.load_pack("t001")["warnings"]
    assert "few_passages:1" in w and any(x.startswith("no_strong_match") for x in w)


# ----------------------------------------------------- label overrides

def test_label_override_recorded(study_paths, monkeypatch):
    monkeypatch.setitem(config()["claims"]["csv"], "label_overrides",
                        {"t002": {"label": "not_enough_evidence", "reason": "pack does not refute it"}})
    c = next(x for x in load_claims() if x["id"] == "t002")
    assert c["label"] == "not_enough_evidence" and c["label_original"] == "refuted"
    assert c["label_override_reason"] == "pack does not refute it"


def test_label_override_needs_reason(study_paths, monkeypatch):
    monkeypatch.setitem(config()["claims"]["csv"], "label_overrides",
                        {"t002": {"label": "not_enough_evidence"}})
    with pytest.raises(ValueError, match="reason"):
        load_claims()


def test_most_read_list_on_one_line_removed():
    """0008: AP's 'Most read' list arrives as one long line."""
    line = ("1. Alberta voters overwhelmingly reject separation process in a victory for Carney 4 MIN READ "
            "669 2. Trump and Vance announce $25M whistleblower program to encourage public to report "
            "alleged fraud 2 MIN READ 1172 3. From France to Albania, social unrest is setting Europe's "
            "streets ablaze 5 MIN READ 758")
    cleaned, removed = P.clean(line + "\nTrump urged the Senate to consider his pick without delay.")
    assert "MIN READ" not in cleaned and removed == 1


def test_most_read_items_on_separate_lines_removed():
    """0008 again: in the raw page each 'Most read' item is its own line."""
    page = ("1. Alberta voters overwhelmingly reject separation process in a victory for Carney 4 MIN READ 669\n"
            "2. Trump and Vance announce $25M whistleblower program to encourage public to report "
            "alleged fraud 2 MIN READ 1172\n"
            "Trump urged the Senate to consider his pick for the high court without delay.")
    cleaned, removed = P.clean(page)
    assert "MIN READ" not in cleaned and removed == 2 and "without delay" in cleaned
