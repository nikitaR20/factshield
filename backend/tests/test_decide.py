"""The decision function shared by the API and the benchmark.

These tests pin the two things that had drifted between the copies, and the
validation crash on downgraded verdicts.
"""

import asyncio
import sys
from datetime import date
from pathlib import Path

from app.pipeline import decide as D
from app.pipeline import evidence_state as ev
from app.pipeline import stages
from app.schemas import EvidenceItem, SubClaim


def item(**kw) -> EvidenceItem:
    base = dict(
        id=1, sub_claim_id=1, url="https://example.com/a", title="t",
        quote="q", published_date=date(2024, 1, 1), source_domain="example.com",
        channel="open_web", tier="reporting", stance="supports", role="assesses",
        stance_confidence=0.9,
    )
    base.update(kw)
    return EvidenceItem(**base)


CLAIM = SubClaim(id=1, text="The claim.")


def run(coro):
    return asyncio.run(coro)


def test_fast_path_runs_before_abstention():
    """A matching fact-check resolves the claim even when every other source
    only reports it — the report-only rule would otherwise abstain."""
    items = [
        item(id=1, tier="prior_check", channel="prior_check", stance="refutes",
             stance_confidence=0.95, claim_match=0.9, url="https://snopes.com/x",
             source_domain="snopes.com"),
    ]
    state = ev.compute(items, "general")
    d = run(D.decide(CLAIM, items, state, "general", explain=False))
    assert d.fast_path and d.verdict.claim_verdict == "refuted"


def test_no_evidence_abstains_without_model_call(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("explanation model must not be called")
    monkeypatch.setattr(stages, "explain", boom)
    state = ev.compute([], "general")
    d = run(D.decide(CLAIM, [], state, "general"))
    assert d.verdict.abstained and d.abstain_kind == "nothing_found"


def _strong_support():
    return [item(id=i, url=f"https://s{i}.example/a", source_domain=f"s{i}.example",
                 tier="authority") for i in range(1, 5)]


def test_downgraded_verdict_is_valid(monkeypatch):
    """apply_pattern_downgrade used to put the pattern NAME into score_basis,
    a dict of floats, so every downgraded verdict failed validation."""
    async def fake_explain(claim, items, state, verdict, standing, max_items=6):
        return "subset_stated_as_general", "Sources support a narrower claim [1].", [1], "v"
    monkeypatch.setattr(stages, "explain", fake_explain)
    monkeypatch.setattr(stages.nli, "entails", lambda p, h, threshold=None: True)
    items = _strong_support()
    state = ev.compute(items, "general")
    d = run(D.decide(CLAIM, items, state, "general"))
    assert d.downgraded and d.verdict.claim_verdict == "partly_supported"
    assert d.verdict.explanation_pattern == "subset_stated_as_general"
    assert all(isinstance(v, float) for v in d.verdict.score_basis.values())


def test_explain_false_makes_no_model_call(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("explain=False must not call the model")
    monkeypatch.setattr(stages, "explain", boom)
    items = _strong_support()
    d = run(D.decide(CLAIM, items, ev.compute(items, "general"), "general", explain=False))
    assert d.verdict.claim_verdict == "supported" and not d.explained


def test_benchmark_uses_the_same_decision(monkeypatch):
    """eval/conditions.py must return what decide() returns for the same input."""
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from eval import conditions

    async def fake_grade(docs, claim, sub_id, category, jurisdiction=None, start_id=1):
        return _strong_support()
    monkeypatch.setattr(conditions.grading, "grade", fake_grade)
    record = {"claim": "The claim.", "documents": {"1": []},
              "triage": {"category": "general", "jurisdiction": None,
                         "sub_claims": [{"id": 1, "text": "The claim."}]}}
    out = run(conditions.run_retrieval_condition(record, "full"))
    items = _strong_support()
    d = run(D.decide(CLAIM, items, ev.compute(items, "general"), "general", explain=False))
    assert out["claim_verdict"] == d.verdict.claim_verdict
    assert out["support_score"] == d.verdict.support_score
