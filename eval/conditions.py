"""Benchmark conditions.

Each condition is the full pipeline with one thing removed, so a difference in
results is attributable to that one thing. That is what makes this a controlled
instrument rather than a scoreboard.

    parametric      no retrieval at all — the offline-LLM baseline
    open_web_only   general search only, no source control, no fact-check channel
    no_page_context bare selection, no surrounding paragraphs — the context ablation
    full            everything

IMPORTANT: `parametric` lives here and ONLY here. Running a no-retrieval pass
inside the production pipeline and feeding its answer to the verdict step would
anchor the output and destroy the claim that verdicts are evidence-grounded.
Run it separately, log it, and compare — which yields the offline-LLM baseline
for free AND a reportable disagreement rate between model prior and evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.pipeline import evidence_state as ev  # noqa: E402
from app.pipeline import decide, grading  # noqa: E402
from app.providers.llm import LLMError, complete_json  # noqa: E402
from app.schemas import RawDocument, SubClaim  # noqa: E402

PARAMETRIC_PROMPT = """You are judging a factual claim using only your own knowledge.
You have no retrieval and no sources. Answer with your best judgement.

Reply with JSON only:
{"verdict": "supported" | "refuted" | "partly_supported" | "unresolved"}"""


async def run_parametric(record: dict) -> dict:
    """No retrieval. The model's prior, logged as a measured variable.

    The disagreement rate between this and the evidence-grounded verdict is a
    result in its own right: how often does retrieval overturn the model's
    memory, and which is right when they differ?
    """
    claim = record["triage"]["sub_claims"][0]["text"] if record["triage"]["sub_claims"] else record["claim"]
    try:
        data = await complete_json("decide", PARAMETRIC_PROMPT, json.dumps({"claim": claim}))
        verdict = data.get("verdict", "unresolved")
    except LLMError:
        verdict = "unresolved"
    return {
        "condition": "parametric",
        "claim_verdict": verdict,
        "evidence_standing": None,
        "support_score": None,
        "n_evidence": 0,
    }


def _filter_docs(docs: list[dict], condition: str) -> list[RawDocument]:
    items = [RawDocument(**d) for d in docs]
    if condition == "open_web_only":
        return [d for d in items if d.channel == "open_web"]
    return items


async def run_retrieval_condition(record: dict, condition: str, explain: bool = False) -> dict:
    """full | open_web_only | no_page_context — all run against the snapshot.

    `explain=False` (the default) makes no model call: the verdict is the
    computed one, before the explanation model's one-directional downgrade.
    `full_explained` runs that step too, at the cost of one call per claim.
    """
    tri = record["triage"]
    category = "general" if condition == "open_web_only" else tri["category"]
    jurisdiction = tri.get("jurisdiction")

    sub_claims = [SubClaim(**c) for c in tri["sub_claims"]]
    if condition == "no_page_context":
        # The selection as highlighted, unresolved. This is what a copy-paste
        # into a general assistant receives.
        sub_claims = [SubClaim(id=1, text=record["claim"])]

    results = []
    next_id = 1
    for sub in sub_claims:
        raw = record["documents"].get(
            str(sub.id)) or record["documents"].get(sub.id) or []
        docs = _filter_docs(raw, condition)
        items = await grading.grade(docs, sub.text, sub.id, category, jurisdiction, start_id=next_id)
        next_id += len(items)

        state = ev.compute(items, category, jurisdiction,
                           record.get("page_published"))
        # The API's own decision function. This copy used to check abstention
        # before the fast path and skip the explanation downgrade, so it
        # measured a slightly different system from the one users get.
        d = await decide.decide(sub, items, state, category, explain=explain)
        v = d.verdict
        results.append({
            "condition": condition, "sub_claim_id": sub.id,
            "claim_verdict": v.claim_verdict,
            "evidence_standing": v.evidence_standing,
            "support_score": v.support_score, "score_basis": v.score_basis,
            "abstained": v.abstained, "abstain_kind": d.abstain_kind,
            "fast_path": d.fast_path, "explained": d.explained,
            "downgraded": d.downgraded, "n_evidence": len(items),
        })

    return results[0] if len(results) == 1 else {
        "condition": condition, "sub_results": results,
        "claim_verdict": _combine([r["claim_verdict"] for r in results]),
        "evidence_standing": results[0]["evidence_standing"] if results else None,
        "support_score": None,
        "n_evidence": sum(r["n_evidence"] for r in results),
    }


def _combine(verdicts: list[str]) -> str:
    """A decomposed claim resolves to `partly_supported` when its parts differ."""
    uniq = set(verdicts)
    if len(uniq) == 1:
        return verdicts[0]
    if "supported" in uniq and "refuted" in uniq:
        return "partly_supported"
    if "unresolved" in uniq and len(uniq) > 1:
        return "partly_supported"
    return "unresolved"


CONDITIONS = {
    "parametric": run_parametric,
    "full": lambda r: run_retrieval_condition(r, "full"),
    "full_explained": lambda r: run_retrieval_condition(r, "full", explain=True),
    "open_web_only": lambda r: run_retrieval_condition(r, "open_web_only"),
    "no_page_context": lambda r: run_retrieval_condition(r, "no_page_context"),
}
