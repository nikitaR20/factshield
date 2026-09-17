"""Score and verdict derivation — pure arithmetic, no model calls.

The support score is computed from retrieval metadata: tier quality, whether a
source assesses or merely repeats the claim, recency, and independence. It is
NOT a model's self-reported confidence, which is uncalibrated and cannot be
justified against a neighbouring value.

Two consequences follow, and both matter:

  * `score_basis` records every component, so the number is auditable. If a
    supervisor asks "why 72 and not 65", the answer is in the dict.
  * The score is None below a minimum evidence weight. "Unresolved" must never
    collapse into a misleading 50 — those are different states and a single
    number cannot express both.
"""

from __future__ import annotations

import math
from datetime import date

from ..config import tiers
from ..schemas import (
    ClaimVerdict,
    EvidenceItem,
    EvidenceStanding,
    EvidenceState,
)


def _cfg() -> dict:
    return tiers()


def recency_factor(published: date | None, category: str, today: date | None = None) -> float:
    """Exponential decay on a per-category half-life. Medical findings age
    slowly; breaking news ages in hours."""
    if published is None:
        return 0.7  # unknown date: discount, do not discard
    today = today or date.today()
    half_life = _cfg().get("recency_half_life_days", {}).get(category, 730)
    age_days = max(0, (today - published).days)
    return 0.5 ** (age_days / half_life)


def item_weight(item: EvidenceItem, category: str, n_domains: int, today: date | None = None) -> float:
    cfg = _cfg()
    if item.retracted and cfg.get("scoring", {}).get("exclude_retracted", True):
        return 0.0

    tier_w = cfg.get("tier_weights", {}).get(item.tier, 0.1)
    role_w = cfg.get("role_weights", {}).get(item.role, 0.2)
    rec_w = recency_factor(item.published_date, category, today)

    # Independence damping: ten documents from two domains should not outweigh
    # three documents from three domains.
    indep = min(1.0, math.sqrt(max(1, n_domains)) / 2.0)

    return tier_w * role_w * rec_w * indep * item.stance_confidence


def compute_score(
    items: list[EvidenceItem],
    category: str,
    n_domains: int,
    today: date | None = None,
) -> tuple[int | None, dict[str, float]]:
    """Return (score, basis). Score is None when evidence is too thin to
    support any number at all."""
    cfg = _cfg().get("scoring", {})
    min_weight = cfg.get("min_total_weight", 0.6)

    support = refute = 0.0
    for it in items:
        w = item_weight(it, category, n_domains, today)
        if it.stance == "supports":
            support += w
        elif it.stance == "refutes":
            refute += w

    total = support + refute
    basis = {
        "support_weight": round(support, 4),
        "refute_weight": round(refute, 4),
        "total_weight": round(total, 4),
        "min_required": min_weight,
        "n_independent_domains": float(n_domains),
    }

    if total < min_weight:
        basis["reason_no_score"] = 1.0
        return None, basis

    score = int(round(100.0 * support / total))
    basis["score"] = float(score)
    return score, basis


def derive_verdict(
    score: int | None,
    state: EvidenceState,
    items: list[EvidenceItem],
) -> tuple[ClaimVerdict, EvidenceStanding]:
    """Map computed evidence into the two axes.

    Axis 1 answers: are the propositions supported?
    Axis 2 answers: does the evidence base match what the claim implies?

    They are genuinely independent. "Some scientists believe X" can be
    supported on axis 1 and overstated on axis 2, and a single label cannot
    express that.
    """
    band = _cfg().get("scoring", {}).get("contested_band", [35, 65])
    low, high = band[0], band[1]

    # ---- axis 1
    if score is None:
        verdict: ClaimVerdict = "unresolved"
    elif score >= high:
        verdict = "supported"
    elif score <= low:
        verdict = "refuted"
    else:
        verdict = "partly_supported"

    # ---- axis 2, most specific first
    top_tiers = {"prior_check", "definitional", "peer_reviewed", "authority"}
    top_items = [i for i in items if i.tier in top_tiers]
    top_stances = {i.stance for i in top_items if i.stance in ("supports", "refutes")}

    if state.supporting_predate_refuting:
        standing: EvidenceStanding = "outdated"
    elif state.shares_common_origin and state.sources_postdate_claim:
        standing = "amplified"
    elif not state.expected_tier_present:
        # Nothing of the type this claim would require exists. For a recent
        # claim that is "not yet"; otherwise the claim overstates its backing.
        recent = state.claim_age_hours is not None and state.claim_age_hours < 72
        standing = "unestablished" if recent else "overstated"
    elif len(top_stances) > 1:
        standing = "contested"
    elif score is not None and low < score < high:
        standing = "contested"
    elif not top_items and items:
        standing = "overstated"
    else:
        standing = "consistent"

    return verdict, standing
