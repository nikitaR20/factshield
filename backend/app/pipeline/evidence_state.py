"""Evidence state: arithmetic over metadata, no model calls.

This module decides claim age, source ordering, tier coverage and origin
overlap. Routing any of it through a model would introduce fabrication into
the one layer that currently cannot fabricate — and these are precisely the
facts that produce the "too early to tell" and "amplified" explanations.

Every function here is pure. Write the tests for this file first.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
from urllib.parse import urlparse

from ..config import pack_for
from ..schemas import EvidenceItem, EvidenceState


def registrable_domain(url_or_domain: str) -> str:
    """Crude eTLD+1. Good enough to spot that bbc.co.uk and www.bbc.co.uk are
    one source, without pulling in a public-suffix dependency."""
    host = url_or_domain
    if "://" in host:
        host = urlparse(host).netloc
    host = host.lower().removeprefix("www.")
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    # Handle common two-part suffixes (co.uk, ac.uk, com.au, gov.uk ...)
    if parts[-2] in {"co", "ac", "gov", "org", "net", "com", "edu"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def independent_domains(items: list[EvidenceItem]) -> set[str]:
    return {registrable_domain(str(i.url)) for i in items}


def _as_date(value) -> date | None:
    """Dates arrive as `date` from Pydantic but as ISO strings from snapshot
    JSON. Accept both rather than making every caller remember which."""
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def claim_age_hours(page_published, now: datetime | None = None) -> float | None:
    published_date = _as_date(page_published)
    if published_date is None:
        return None
    now = now or datetime.now(timezone.utc)
    published = datetime.combine(
        published_date, datetime.min.time(), tzinfo=timezone.utc)
    return max(0.0, (now - published).total_seconds() / 3600.0)


def compute(
    items: list[EvidenceItem],
    category: str,
    jurisdiction: str | None = None,
    page_published: date | None = None,
    now: datetime | None = None,
) -> EvidenceState:
    if not items:
        return EvidenceState(n_documents=0, n_independent_domains=0, expected_tier_present=False)

    dates = [i.published_date for i in items if i.published_date]
    domains = independent_domains(items)
    published = _as_date(page_published)
    age = claim_age_hours(published, now)

    pack = pack_for(category, jurisdiction)
    expected = set(pack.get("expected_tiers", []))
    present = {i.tier for i in items}
    expected_present = bool(expected & present) if expected else True

    # A fact-checker who reviewed THIS claim is the best evidence available in
    # any category. It satisfies the tier requirement regardless of what the
    # pack lists.
    if any(i.tier == "prior_check" and i.claim_match >= 0.8 for i in items):
        expected_present = True

    # Every retrieved document merely repeats the claim; none evaluates it.
    # Coverage volume is not corroboration.
    #
    # But a primary source stating its own action IS evidence. The Federal
    # Reserve publishing its rate decision is not "repeating a claim", and
    # treating it that way abstained on a decision the Fed itself announced.
    primary = {"prior_check", "definitional", "peer_reviewed"}
    has_primary = any(
        i.tier in primary and i.stance in ("supports", "refutes") for i in items
    )
    all_report_only = (
        not has_primary and all(i.role == "reports" for i in items)
    )

    # Sources that all postdate the claim and cluster on one domain are an
    # amplification signature, not independent confirmation.
    postdate = False
    if published and dates:
        postdate = all(d >= published for d in dates)
    # Amplification means MANY documents from VERY few origins. Five distinct
    # domains over eighteen documents is normal retrieval, not a single story
    # being echoed, so the old ratio test fired far too readily.
    common_origin = len(items) >= 4 and len(domains) <= 2

    split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i in items:
        split[i.tier][i.stance] += 1

    # Supporting evidence systematically older than refuting evidence is the
    # signature of a fact that used to be true.
    sup = [i.published_date for i in items if i.stance ==
           "supports" and i.published_date]
    ref = [i.published_date for i in items if i.stance ==
           "refutes" and i.published_date]
    predate = bool(sup and ref and max(sup) < min(ref))

    return EvidenceState(
        n_documents=len(items),
        n_independent_domains=len(domains),
        claim_age_hours=age,
        earliest_source=min(dates) if dates else None,
        latest_source=max(dates) if dates else None,
        sources_postdate_claim=postdate,
        shares_common_origin=common_origin,
        expected_tier_present=expected_present,
        all_sources_report_only=all_report_only,
        tier_stance_split={k: dict(v) for k, v in split.items()},
        supporting_predate_refuting=predate,
    )


# ----------------------------------------------------------- abstention

#: Ordered — the first matching rule wins, so the most specific reason surfaces.
ABSTENTION_RULES = [
    (
        lambda st, items: st.n_documents == 0,
        "No sources were found for this claim in the vetted source list.",
    ),
    (
        lambda st, items: st.all_sources_report_only,
        "Every source found repeats the claim rather than assessing it. "
        "Coverage is not confirmation.",
    ),
    (
        lambda st, items: not st.expected_tier_present,
        "No source of the type this claim would require was found.",
    ),
    (
        lambda st, items: st.shares_common_origin and st.sources_postdate_claim,
        "All supporting sources postdate the claim and trace to a small number "
        "of origins, which indicates repetition rather than independent evidence.",
    ),
]


def standing_for_abstention(state: EvidenceState, items: list[EvidenceItem]) -> str:
    """Which axis-2 label describes an abstention.

    Previously hardcoded to "amplified" whenever any document existed, which
    was simply wrong for claims whose sources share no common origin.
    """
    if state.n_documents == 0:
        return "unestablished"
    if state.shares_common_origin and state.sources_postdate_claim:
        return "amplified"
    if state.all_sources_report_only:
        return "amplified"
    if not state.expected_tier_present:
        recent = state.claim_age_hours is not None and state.claim_age_hours < 72
        return "unestablished" if recent else "overstated"
    return "contested"


def should_abstain(state: EvidenceState, items: list[EvidenceItem]) -> str | None:
    """Return an abstention reason, or None to proceed."""
    for predicate, reason in ABSTENTION_RULES:
        if predicate(state, items):
            return reason
    return None
