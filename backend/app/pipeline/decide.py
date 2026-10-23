"""One sub-claim from graded evidence to a verdict.

This is the decision logic the API serves AND the logic the benchmark
measures. It used to exist twice — once in `main.check` and once in
`eval/conditions.py` — and the copies had drifted: the benchmark checked the
abstention rules before the fact-check fast path (the API had been fixed to do
the opposite), passed no evidence state to the fast path, and never ran the
explanation downgrade. A thesis result measured on the benchmark copy would not
have described the system users see. One function, two callers, no drift.

`explain=False` stops before the only model call. The verdict is then the
computed one, without the one-directional downgrade the explanation model may
apply; the result records which mode produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import EvidenceItem, EvidenceState, SubClaim, SubClaimVerdict
from . import evidence_state as ev
from . import scoring, stages

TOO_THIN_REASON = (
    "Sources were found, but too few took a clear position to support a verdict."
)


@dataclass
class Decision:
    verdict: SubClaimVerdict
    fast_path: bool = False
    abstain_kind: str | None = None     # why it abstained, as a countable code
    explained: bool = False             # did the explanation model run
    downgraded: bool = False
    grounding_total: int = 0
    grounding_dropped: int = 0


async def decide(
    claim: SubClaim,
    items: list[EvidenceItem],
    state: EvidenceState,
    category: str,
    *,
    explain: bool = True,
    use_fast_path: bool = True,
    max_items: int = 6,
) -> Decision:
    # A fact-checker who reviewed THIS claim outranks every abstention rule:
    # they are a human expert who already resolved it. Checking the rules
    # first meant a claim backed by a matching Snopes verdict could still be
    # abstained on.
    if use_fast_path:
        fast = stages.fast_path_verdict(claim, items, state)
        if fast is not None:
            return Decision(verdict=fast, fast_path=True)

    # Short circuit BEFORE any model call. No evidence, no verdict request —
    # the model cannot hallucinate on a call that is not made.
    reason = ev.should_abstain(state, items)
    if reason:
        return Decision(
            verdict=SubClaimVerdict(
                sub_claim_id=claim.id, sub_claim_text=claim.text,
                claim_verdict="unresolved",
                # Derived from WHY it abstained, not hardcoded.
                evidence_standing=ev.standing_for_abstention(state, items),
                support_score=None, abstained=True, abstention_reason=reason,
                explanation=stages.describe_evidence(items, state),
            ),
            abstain_kind=ev.abstention_kind(state, items),
        )

    score, basis = scoring.compute_score(
        items, category, state.n_independent_domains,
        attribution_extracted=claim.attribution_extracted,
    )
    verdict, standing = scoring.derive_verdict(
        score, state, items, attribution_extracted=claim.attribution_extracted
    )

    # Too little evidence to score IS an abstention, and stopping here also
    # saves the model call.
    if score is None:
        return Decision(
            verdict=SubClaimVerdict(
                sub_claim_id=claim.id, sub_claim_text=claim.text,
                claim_verdict="unresolved",
                evidence_standing=ev.standing_for_abstention(state, items),
                support_score=None, score_basis=basis, abstained=True,
                abstention_reason=TOO_THIN_REASON,
                explanation=stages.describe_evidence(items, state),
                cited_evidence_ids=[i.id for i in items
                                    if i.stance in ("supports", "refutes")][:3],
            ),
            abstain_kind=ev.abstention_kind(state, items, scored=False),
        )

    if not explain:
        return Decision(
            verdict=SubClaimVerdict(
                sub_claim_id=claim.id, sub_claim_text=claim.text,
                claim_verdict=verdict, evidence_standing=standing,
                support_score=score, score_basis=basis,
                explanation=stages.describe_evidence(items, state),
                cited_evidence_ids=[i.id for i in items[:3]],
            ),
        )

    pattern, explanation, cited, _ = await stages.explain(
        claim, items, state, verdict, standing, max_items=max_items
    )
    # The model picks a pattern that describes the evidence. Reject any
    # pattern the evidence plainly contradicts.
    pattern = scoring.validate_pattern(pattern, items)

    # One-directional: the model may weaken the computed verdict, never
    # strengthen it. See scoring.apply_pattern_downgrade.
    before = verdict
    verdict, standing = scoring.apply_pattern_downgrade(verdict, standing, pattern)
    downgraded = verdict != before
    if downgraded:
        # `score_basis` is dict[str, float]; the pattern NAME is already
        # carried by `explanation_pattern`. Putting the string in here (as the
        # API used to) fails validation, so every downgraded verdict was a 500.
        basis = {**basis, "downgraded_from": 1.0}
    explanation, cited, total, dropped = stages.ground(explanation, items)

    # Grounding can legitimately delete everything the model wrote. Show a
    # computed description rather than an empty box.
    if not explanation.strip():
        explanation = stages.describe_evidence(
            items, state, downgrade_pattern=pattern if downgraded else None
        )
        cited = [i.id for i in items[:3]]

    return Decision(
        verdict=SubClaimVerdict(
            sub_claim_id=claim.id, sub_claim_text=claim.text,
            claim_verdict=verdict, evidence_standing=standing,
            support_score=score, score_basis=basis,
            explanation_pattern=pattern or None,
            explanation=explanation, cited_evidence_ids=cited,
        ),
        explained=True, downgraded=downgraded,
        grounding_total=total, grounding_dropped=dropped,
    )
