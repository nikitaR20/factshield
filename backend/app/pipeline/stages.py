"""Triage, verdict and grounding.

Cost control lives here. Exactly one potentially-paid call per request, and it
is skipped whenever a published fact-check already resolves the claim — which
is common, because viral claims are precisely what fact-checkers cover.
"""

from __future__ import annotations

import json
import re

from ..config import flag, prompt
from ..providers import nli
from ..providers.llm import LLMError, complete_json
from ..schemas import (
    CaptureContext,
    EvidenceItem,
    EvidenceState,
    SubClaim,
    SubClaimVerdict,
    TriageOutput,
)

_CITE = re.compile(r"\[(\d+)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ------------------------------------------------------------------ triage

async def triage(ctx: CaptureContext) -> tuple[TriageOutput, str]:
    system, version = prompt("triage")
    user = json.dumps(
        {
            "selection": ctx.selection,
            "surrounding_text": ctx.surrounding_text[:4000],
            "page_title": ctx.page_title,
            "page_published": str(ctx.page_published) if ctx.page_published else None,
            "page_domain": ctx.page_domain,
        },
        ensure_ascii=False,
    )
    try:
        data = await complete_json("triage", system, user)
    except LLMError:
        # FACTSHIELD_STRICT=true surfaces the real cause instead of quietly
        # degrading. Use it while setting up; turn it off for a study, where a
        # dead provider should not take the whole request down.
        if flag("FACTSHIELD_STRICT"):
            raise
        # Degrade rather than fail: treat the selection as a single claim.
        return (
            TriageOutput(
                checkable=True,
                sub_claims=[SubClaim(id=1, text=ctx.selection.strip())],
                resolution_confident=False,
            ),
            version,
        )

    claims = [
        SubClaim(
            id=int(c.get("id", i + 1)),
            text=str(c.get("text", "")).strip(),
            attribution_extracted=bool(c.get("attribution_extracted", False)),
        )
        for i, c in enumerate(data.get("sub_claims", []))
        if str(c.get("text", "")).strip()
    ]
    return (
        TriageOutput(
            checkable=bool(data.get("checkable")) and bool(claims),
            reason=data.get("reason"),
            sub_claims=claims,
            category=data.get("category") or "general",
            jurisdiction=data.get("jurisdiction"),
            resolution_confident=bool(data.get("resolution_confident", True)),
        ),
        version,
    )


# ---------------------------------------------------------------- fast path

def fast_path_verdict(
    sub_claim: SubClaim,
    items: list[EvidenceItem],
    state: EvidenceState | None = None,
) -> SubClaimVerdict | None:
    """Skip the verdict model when a fact-checker already published a rating.

    Snopes has done the work; re-reasoning over it adds latency and cost and
    cannot improve on a human verdict. Roughly 1.5s instead of 8s, and zero
    paid tokens.
    """
    checks = [
        i for i in items
        if i.tier == "prior_check"
        and i.stance in ("supports", "refutes")
        # The fact-checker must have reviewed THIS claim, not a neighbouring
        # one. Without this gate, "Covid originated from China" inherits the
        # rating of "Wuhan Lab Leak Theory CONFIRMED".
        and i.claim_match >= 0.6
    ]
    if not checks:
        return None

    top = max(checks, key=lambda i: (i.claim_match, i.stance_confidence))
    if top.stance_confidence < 0.9:
        return None

    # Fact-checkers disagreeing on the same claim is not a fast path.
    if len({c.stance for c in checks}) > 1:
        return None

    supported = top.stance == "supports"

    # Axis 2 is computed, never assumed. Hardcoding "consistent" here meant the
    # shortcut that saves money also discarded the distinction the system
    # exists for: a confirmed PROMISE is not a confirmed OUTCOME.
    if sub_claim.attribution_extracted:
        standing = "overstated"
        caveat = (
            " This confirms the statement was made; it does not establish that"
            " the promised outcome will occur."
        )
    elif state is not None and state.shares_common_origin and state.sources_postdate_claim:
        standing, caveat = "amplified", ""
    elif state is not None and state.supporting_predate_refuting:
        standing, caveat = "outdated", ""
    else:
        standing, caveat = "consistent", ""

    return SubClaimVerdict(
        sub_claim_id=sub_claim.id,
        sub_claim_text=sub_claim.text,
        claim_verdict="supported" if supported else "refuted",
        evidence_standing=standing,
        support_score=95 if supported else 5,
        score_basis={"source": 1.0, "prior_check": 1.0},
        explanation_pattern="directly_confirmed" if supported else "directly_contradicted",
        explanation=(
            f"This claim has already been fact-checked by "
            f"{top.source_domain} [{top.id}], which rated it "
            f"{'accurate' if supported else 'false'}.{caveat}"
        ),
        cited_evidence_ids=[top.id],
    )


# ------------------------------------------------------------------ decide

async def explain(
    sub_claim: SubClaim,
    items: list[EvidenceItem],
    state: EvidenceState,
    claim_verdict: str,
    evidence_standing: str,
    max_items: int = 6,
) -> tuple[str, str, list[int], str]:
    """Generate the explanation only.

    The verdict and standing are already computed by `scoring`; the model does
    not decide them. Its job is to justify them from evidence — a strictly
    narrower task with far less room to invent.
    """
    system, version = prompt("decide")

    ranked = sorted(
        items,
        key=lambda i: (i.stance in ("supports", "refutes"),
                       i.stance_confidence),
        reverse=True,
    )[:max_items]

    user = json.dumps(
        {
            "claim": sub_claim.text,
            "claim_verdict": claim_verdict,
            "evidence_standing": evidence_standing,
            "evidence_state": state.model_dump(mode="json"),
            "evidence": [
                {
                    "id": i.id, "tier": i.tier, "stance": i.stance, "role": i.role,
                    "date": str(i.published_date) if i.published_date else None,
                    "source": i.source_domain, "quote": i.quote,
                }
                for i in ranked
            ],
        },
        ensure_ascii=False,
    )

    try:
        data = await complete_json("decide", system, user)
    except LLMError:
        if flag("FACTSHIELD_STRICT"):
            raise
        return "", "", [], version

    return (
        str(data.get("explanation_pattern") or ""),
        str(data.get("explanation") or ""),
        [int(x) for x in data.get("cited_evidence_ids", []) if str(x).isdigit()],
        version,
    )


# ---------------------------------------------------------------- grounding

_DOWNGRADE_REASON = {
    "quote_omits_context": "The sources show context the claim leaves out.",
    "subset_stated_as_general": "The sources support a narrower version of the claim than stated.",
    "timeframe_cherry_picked": "The sources support this only for a narrower period than implied.",
    "correlation_as_causation": "The sources show an association, not the causal link the claim asserts.",
    "study_not_human": "The supporting research was not conducted in humans.",
    "figure_outdated": "The supporting figure has since been superseded.",
    "source_is_reporting_not_assessing": "The sources report the claim rather than verify it.",
    "single_origin_amplification": "The supporting sources trace to a single origin.",
}


def describe_evidence(
    items: list[EvidenceItem],
    state: EvidenceState,
    downgrade_pattern: str | None = None,
) -> str:
    """A factual summary assembled from counts, used when the model produced
    nothing that survived grounding.

    Every clause here is arithmetic over the evidence table, so it cannot
    fabricate. An empty explanation box is worse than a plain one: the user is
    left with a verdict and no visible reason for it.
    """
    if not items:
        return "No sources were found for this claim."

    tiers: dict[str, int] = {}
    for i in items:
        tiers[i.tier] = tiers.get(i.tier, 0) + 1
    composition = ", ".join(
        f"{n} {t.replace('_', ' ')}" for t, n in sorted(tiers.items(), key=lambda x: -x[1])
    )

    supports = sum(1 for i in items if i.stance == "supports")
    refutes = sum(1 for i in items if i.stance == "refutes")
    unclear = sum(1 for i in items if i.stance in (
        "insufficient", "discusses"))

    parts = [
        f"{len(items)} sources across {state.n_independent_domains} "
        f"independent domains ({composition})."
    ]
    if supports or refutes:
        parts.append(f"{supports} support the claim, {refutes} contradict it.")
    if unclear:
        near = sum(
            1 for i in items
            if i.tier == "prior_check" and i.stance == "discusses" and i.claim_match < 0.8
        )
        if near:
            parts.append(
                f"{near} fact-checks address related but different claims and "
                "were not treated as verdicts on this one."
            )
        elif unclear == len(items):
            parts.append("None takes a clear position on this specific claim.")
    if not state.expected_tier_present:
        parts.append(
            "No source of the type this claim would require was found.")
    # Without this, a verdict of "partly supported" sat beside "4 support, 0
    # contradict" with nothing explaining the gap between them.
    if downgrade_pattern and downgrade_pattern in _DOWNGRADE_REASON:
        parts.append(_DOWNGRADE_REASON[downgrade_pattern])
    return " ".join(parts)


def ground(
    explanation: str, items: list[EvidenceItem]
) -> tuple[str, list[int], int, int]:
    """Drop any generated sentence that its cited evidence does not support.

    Telling a model to use only the supplied evidence does not enforce it. The
    instruction has no mechanism behind it — when evidence is thin the model
    fills gaps from training data, and the filled-in part reads exactly like
    the grounded part. This layer is what converts "we told it not to" into
    "we verified it didn't", and the drop count is a reportable metric.
    """
    if not explanation.strip():
        return "", [], 0, 0

    by_id = {i.id: i for i in items}
    kept: list[str] = []
    cited: set[int] = set()
    total = dropped = 0

    for sent in (s.strip() for s in _SENT_SPLIT.split(explanation) if s.strip()):
        total += 1
        ids = [int(m) for m in _CITE.findall(sent) if int(m) in by_id]
        if not ids:
            dropped += 1          # no citation -> unsupported by construction
            continue
        premise = " ".join(by_id[i].quote for i in ids)
        if nli.entails(premise, _CITE.sub("", sent).strip()):
            kept.append(sent)
            cited.update(ids)
        else:
            dropped += 1

    return " ".join(kept), sorted(cited), total, dropped
