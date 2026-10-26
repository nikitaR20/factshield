"""Tests for FactShield v2's evidence preparation and commit rule.

Model replies are scripted, so each test pins one branch of the rule.
No network, no MiniCheck: the checker is replaced by a scripted one.
"""

import asyncio
import hashlib
import json

import pytest

import fsv2  # noqa: F401  (sets sys.path)
from fsv2 import decide, evidence as ev, llm
from fsv2.run_packs import rows_for, split_of
from study.models import CallResult


# ------------------------------------------------------------ fixtures

def pack(*passages):
    return {"claim_id": "c1", "claim": "The tower was completed in 1999.", "pack_hash": "h",
            "passages": [{"pid": f"P{i+1}", "url": f"https://{d}/a{i}", "title": "t",
                          "source_domain": d, "published_date": None, "text": t}
                         for i, (d, t) in enumerate(passages)]}


BASIC = pack(("bbc.com", "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."),
             ("travelblog.example", "The tower lights up every evening and draws many visitors."),
             ("reuters.com", "Engineers finished the iron tower in 1889, records show."))


def reply(verdict, statements=(), marks=None):
    marks = marks or {"P1": "yes", "P2": "no", "P3": "yes"}
    return json.dumps({
        "passages": [{"id": k, "same_subject": v, "stance": "neither"} for k, v in marks.items()],
        "verdict": verdict,
        "key_statements": [{"text": t, "passages": list(p)} for t, p in statements],
        "reason": "x"})


class Script:
    """Replace llm.call: answers come from a per-model queue."""
    def __init__(self, answers):
        self.answers = {k: list(v) for k, v in answers.items()}
        self.calls = []

    async def __call__(self, cfg, system, user, client):
        key = cfg["model"].removeprefix("stub-")
        self.calls.append((key, cfg.get("temperature")))
        r = CallResult(provider="script", model_requested=cfg["model"])
        r.text = self.answers[key].pop(0)
        r.model_version = "scripted"
        return r


class Checker:
    name = "scripted"
    def __init__(self, prob):
        self.prob, self.seen = prob, []
    def score(self, docs, statements):
        self.seen += list(zip(docs, statements))
        return [self.prob] * len(docs)


def run(monkeypatch, answers, prob=0.9, explain=False):
    s = Script(answers)
    monkeypatch.setattr(llm, "call", s)
    chk = Checker(prob)
    res = asyncio.run(decide.check_claim(BASIC["claim"], BASIC, checker=chk, stub=True,
                                         client=None, explain=explain))
    return res, s, chk


REFUTE = reply("refuted", [("The Eiffel Tower was completed in 1889.", ["P1"])])
NEE = reply("not_enough_evidence")


# ------------------------------------------------------------ evidence

def test_organisation_groups_bbc_domains():
    assert ev.organisation("bbc.com") == ev.organisation("news.bbc.co.uk") == "bbc"
    assert ev.registrable("news.bbc.co.uk") == "bbc.co.uk"
    assert ev.organisation("abc.net.au") != ev.organisation("abcnews.com")


def test_rating_lookup_and_unrated():
    assert ev.rating("www.reuters.com") == pytest.approx(1.0)
    assert ev.rating("en.wikipedia.org") is not None      # suffix match
    assert ev.rating("travelblog.example") is None
    assert ev.source_type("en.wikipedia.org").startswith("encyclopedia")
    assert ev.source_type("travelblog.example") == "other website"


def test_near_copy_shown_once_and_named():
    text = "The council voted on Tuesday to close the old bridge for repairs lasting two years. " * 3
    items = ev.prepare(pack(("apnews.com", text), ("6abc.com", text + " Extra line here.")))
    assert items[1].same_text_as == "P1" and items[0].also_published_by == ["6abc.com"]
    out = ev.render(items)
    assert out.count("The council voted") == 3 and "Same text as [P1]" in out


def test_same_org_and_wire_labels():
    items = ev.prepare(pack(("bbc.com", "Story one about a flood (AP) in the valley today."),
                            ("news.bbc.co.uk", "A different story about the same valley flood.")))
    assert items[0].same_org_as == ["P2"] and items[1].same_org_as == ["P1"]
    assert items[0].credits_wire == ["AP"]
    assert "unrated" not in ev.render(items).split("[P1]")[1].split("URL")[0]


def test_text_for_check_resolves_copies():
    text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. " * 3
    items = ev.prepare(pack(("a.com", text), ("b.com", text)))
    assert ev.text_for_check(items, ["P2"]) == items[0].text


# ------------------------------------------------------------ parsing

def test_extract_json_handles_fences_and_prose():
    assert llm.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.extract_json('Sure! {"a": {"b": "}"}} done') == {"a": {"b": "}"}}
    assert llm.extract_json("no json here") is None


def test_parse_flags_fabricated_ids_and_bad_verdicts():
    p = llm.parse_verdict(reply("refuted", [("x", ["P1", "P99"])]), {"P1", "P2", "P3"})
    assert p["statements"][0]["passages"] == ["P1"]
    assert p["statements"][0]["fabricated_ids"] == ["P99"]
    assert llm.parse_verdict('{"verdict": "true"}', set())["verdict"] == "unparseable"


# ------------------------------------------------------------ commit rule

def test_unanimous_refuted_with_confirmed_statement(monkeypatch):
    res, s, chk = run(monkeypatch, {"openai": [REFUTE], "gemini": [REFUTE], "groq": [REFUTE]})
    assert res["final"] == "refuted" and res["path"] == "unanimous_round1"
    assert res["cited_pids"] == ["P1"] and len(s.calls) == 3
    assert all(t == 0 for _, t in s.calls) or all(t is None for _, t in s.calls)


def test_checker_failure_downgrades(monkeypatch):
    res, _, _ = run(monkeypatch, {"openai": [REFUTE], "gemini": [REFUTE], "groq": [REFUTE]}, prob=0.2)
    assert res["final"] == "not_enough_evidence" and res["reason"] == "citation_check_failed"
    assert res["ablations"]["fs2_no_check"] == "refuted"


def test_statement_citing_off_subject_passage_is_not_eligible(monkeypatch):
    off = reply("refuted", [("The tower lights up every evening.", ["P2"])])  # P2 marked 'no'
    res, _, chk = run(monkeypatch, {"openai": [off], "gemini": [off], "groq": [off]})
    assert res["final"] == "not_enough_evidence" and chk.seen == []


def test_disagreement_draws_extra_samples_at_extra_temperature(monkeypatch):
    res, s, _ = run(monkeypatch, {"openai": [REFUTE, REFUTE, NEE],
                                  "gemini": [NEE, REFUTE, REFUTE],
                                  "groq": [REFUTE, NEE, REFUTE]})
    assert len(s.calls) == 9
    assert sorted({t for _, t in s.calls[3:]}) == [0.7]
    assert res["final"] == "refuted" and res["path"] == "unanimous_after_extra"


def test_persistent_disagreement_is_not_enough_evidence(monkeypatch):
    res, _, _ = run(monkeypatch, {"openai": [REFUTE, REFUTE, REFUTE],
                                  "gemini": [NEE, NEE, NEE],
                                  "groq": [REFUTE, REFUTE, REFUTE]})
    assert res["final"] == "not_enough_evidence" and res["reason"] == "models_disagreed"
    assert res["ablations"]["fs2_majority_r1"] == "refuted"     # 2 of 3, check passes


def test_unparseable_never_counts_as_agreement(monkeypatch):
    res, _, _ = run(monkeypatch, {"openai": ["nonsense"] * 3, "gemini": ["nonsense"] * 3,
                                  "groq": ["nonsense"] * 3})
    assert res["final"] == "not_enough_evidence" and res["panel_label"] is None


def test_conflicting_evidence_needs_no_check(monkeypatch):
    c = reply("conflicting_evidence")
    res, _, chk = run(monkeypatch, {"openai": [c], "gemini": [c], "groq": [c]})
    assert res["final"] == "conflicting_evidence" and chk.seen == []
    assert decide.TO_STUDY["conflicting_evidence"] == "not_enough_evidence"


def test_api_error_is_recorded_not_guessed(monkeypatch):
    async def failing(cfg, system, user, client):
        r = CallResult(provider="x", model_requested=cfg["model"])
        r.error = "HTTP 429"
        return r
    monkeypatch.setattr(llm, "call", failing)
    res = asyncio.run(decide.check_claim(BASIC["claim"], BASIC, checker=Checker(0.9), stub=True,
                                         client=None))
    assert res["error"] and "429" in res["error"]
    rows = rows_for({"id": "claims_Dataset_A-0003", "label": "refuted"}, BASIC, res, "v", 1.0)
    assert len(rows) == 1 and rows[0]["error"] and rows[0]["parsed_verdict"] is None


def test_models_without_temperature_never_get_one(monkeypatch):
    import study.models as sm
    real = sm.config()
    fake = {**real, "models": {**real["models"],
                               "openai": {**real["models"]["openai"], "temperature": None}}}
    monkeypatch.setattr(llm, "study_config", lambda: fake)
    assert llm.panel_cfg("openai", temperature=0.7)["temperature"] is None
    assert llm.panel_cfg("gemini", temperature=0.7)["temperature"] == 0.7


# ------------------------------------------------------------ explanation

def test_explanation_keeps_checked_and_summary_drops_uncited(monkeypatch):
    expl = json.dumps({"sentences": ["The tower was completed in 1889 [P1].",
                                     "It is very famous.",
                                     "Summary: no passage mentions 1999."]})
    res, _, _ = run(monkeypatch, {"openai": [REFUTE], "gemini": [REFUTE], "groq": [REFUTE, expl]},
                    explain=True)
    e = res["explanation"]
    assert [k["kind"] for k in e["sentences"]] == ["fact", "summary"]
    assert e["dropped"][0]["why"] == "no valid citation" and not e["quotes_only"]


# ------------------------------------------------------------ split & rows

def test_split_matches_from_packs_and_counts():
    ref = lambda cid: "dev" if int(hashlib.sha256(cid.encode()).hexdigest(), 16) % 3 == 0 else "test"
    ids = [f"claims_Dataset_A-{i:04d}" for i in range(1, 46)]
    assert all(split_of(i) == ref(i) for i in ids)
    assert sum(split_of(i) == "dev" for i in ids) == 17


def test_rows_have_study_fields(monkeypatch):
    res, _, _ = run(monkeypatch, {"openai": [REFUTE], "gemini": [REFUTE], "groq": [REFUTE]})
    rows = rows_for({"id": "claims_Dataset_A-0003", "label": "refuted"}, BASIC, res, "v", 1.0)
    assert [r["condition"] for r in rows] == ["fs2", "fs2_no_check", "fs2_majority_r1",
                                              "fs2_single", "fs2_single", "fs2_single"]
    for r in rows:
        for k in ("claim_id", "condition", "model", "model_version", "run_index",
                  "live_search", "parsed_verdict", "error", "pack_hash"):
            assert k in r
