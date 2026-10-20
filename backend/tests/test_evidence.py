"""Tests for the model-free layer.

`evidence_state` and `scoring` take no model calls, so they are exhaustively
testable with no mocks and no network. They also produce the explanations that
matter most — "too early to tell", "amplified", "outdated". Write these first.
"""

from datetime import date, datetime, timezone

import pytest

from app.providers import nli
from app.pipeline import evidence_state as ev
from app.pipeline import scoring
from app.schemas import EvidenceItem


def item(**kw) -> EvidenceItem:
    base = dict(
        id=1, sub_claim_id=1, url="https://example.com/a", title="t",
        quote="q", published_date=date(2024, 1, 1), source_domain="example.com",
        channel="open_web", tier="reporting", stance="supports", role="assesses",
        stance_confidence=0.9,
    )
    base.update(kw)
    return EvidenceItem(**base)


# ------------------------------------------------------------ domain parsing

@pytest.mark.parametrize("raw,expected", [
    ("https://www.bbc.co.uk/news/x", "bbc.co.uk"),
    ("bbc.co.uk", "bbc.co.uk"),
    ("https://pubmed.ncbi.nlm.nih.gov/123/", "nih.gov"),
    ("https://snopes.com/fact-check/x", "snopes.com"),
])
def test_registrable_domain(raw, expected):
    assert ev.registrable_domain(raw) == expected


def test_www_and_bare_domain_count_as_one_source():
    items = [
        item(id=1, url="https://www.reuters.com/a",
             source_domain="www.reuters.com"),
        item(id=2, url="https://reuters.com/b", source_domain="reuters.com"),
    ]
    assert len(ev.independent_domains(items)) == 1


# -------------------------------------------------------------- claim age

def test_claim_age_hours():
    now = datetime(2026, 1, 3, tzinfo=timezone.utc)
    assert ev.claim_age_hours(date(2026, 1, 1), now) == pytest.approx(48.0)
    assert ev.claim_age_hours(None, now) is None


# ------------------------------------------------------------ report-only

def test_all_report_only_is_detected():
    """Coverage volume is not corroboration. Fifty outlets repeating a claim
    is one claim, repeated fifty times."""
    items = [
        item(id=i, url=f"https://news{i}.com/x",
             source_domain=f"news{i}.com", role="reports")
        for i in range(1, 6)
    ]
    state = ev.compute(items, "general")
    assert state.all_sources_report_only is True
    assert ev.should_abstain(state, items) is not None


def test_one_assessing_source_stops_abstention():
    items = [
        item(id=1, url="https://news1.com/x",
             source_domain="news1.com", role="reports"),
        item(id=2, url="https://snopes.com/x", source_domain="snopes.com",
             role="assesses", tier="prior_check", stance="refutes"),
    ]
    state = ev.compute(items, "general")
    assert state.all_sources_report_only is False


def test_empty_evidence_abstains():
    state = ev.compute([], "medical")
    assert state.n_documents == 0
    assert ev.should_abstain(state, []) is not None


# --------------------------------------------------------------- outdated

def test_supporting_older_than_refuting_flags_outdated():
    items = [
        item(id=1, stance="supports", published_date=date(2015, 1, 1),
             url="https://a.com/x", source_domain="a.com"),
        item(id=2, stance="refutes", published_date=date(2024, 1, 1),
             url="https://b.com/x", source_domain="b.com"),
    ]
    state = ev.compute(items, "general")
    assert state.supporting_predate_refuting is True
    _, standing = scoring.derive_verdict(50, state, items)
    assert standing == "outdated"


# ---------------------------------------------------------- expected tiers

def test_missing_expected_tier_for_medical():
    """A medical claim with only blog sources has no evidence of the type it
    would require. That absence is itself informative."""
    items = [item(id=1, tier="low", url="https://blog.com/x",
                  source_domain="blog.com")]
    state = ev.compute(items, "medical")
    assert state.expected_tier_present is False
    assert ev.should_abstain(state, items) is not None


def test_present_expected_tier_for_medical():
    items = [item(id=1, tier="peer_reviewed", channel="literature",
                  url="https://pubmed.ncbi.nlm.nih.gov/1/", source_domain="pubmed.ncbi.nlm.nih.gov")]
    state = ev.compute(items, "medical")
    assert state.expected_tier_present is True


# ------------------------------------------------------------------ scoring

def test_no_score_when_evidence_too_thin():
    """`unresolved` must never collapse into a misleading 50."""
    items = [item(id=1, tier="low", role="asserts", stance_confidence=0.6)]
    score, basis = scoring.compute_score(items, "general", 1)
    assert score is None
    assert basis["total_weight"] < basis["min_required"]


def test_strong_refutation_scores_low():
    items = [
        item(id=1, tier="prior_check", role="assesses", stance="refutes",
             stance_confidence=0.95, published_date=date(2025, 6, 1),
             url="https://snopes.com/x", source_domain="snopes.com"),
        item(id=2, tier="peer_reviewed", role="assesses", stance="refutes",
             stance_confidence=0.9, published_date=date(2025, 1, 1),
             url="https://pubmed.ncbi.nlm.nih.gov/1/", source_domain="nih.gov"),
    ]
    score, _ = scoring.compute_score(
        items, "general", 2, today=date(2026, 1, 1))
    assert score is not None and score < 20
    state = ev.compute(items, "general")
    verdict, _ = scoring.derive_verdict(score, state, items)
    assert verdict == "refuted"


def test_reporting_role_is_heavily_discounted():
    """Five sources that merely report must not outweigh one that assesses."""
    reporters = [
        item(id=i, tier="reporting", role="reports", stance="supports",
             published_date=date(2025, 12, 1),
             url=f"https://n{i}.com/x", source_domain=f"n{i}.com")
        for i in range(1, 6)
    ]
    assessor = item(id=9, tier="prior_check", role="assesses", stance="refutes",
                    stance_confidence=0.95, published_date=date(2025, 12, 1),
                    url="https://snopes.com/x", source_domain="snopes.com")
    score, _ = scoring.compute_score(reporters + [assessor], "general", 6,
                                     today=date(2026, 1, 1))
    assert score is not None and score < 50


def test_score_basis_is_auditable():
    items = [item(id=1, tier="authority", role="assesses",
                  published_date=date(2025, 1, 1))]
    _, basis = scoring.compute_score(
        items, "general", 1, today=date(2026, 1, 1))
    for field in ("support_weight", "refute_weight", "total_weight", "min_required"):
        assert field in basis


def test_retracted_sources_carry_no_weight():
    items = [item(id=1, tier="peer_reviewed", role="assesses", retracted=True)]
    score, basis = scoring.compute_score(items, "medical", 1)
    assert score is None
    assert basis["total_weight"] == 0.0


# ------------------------------------------------------------ two axes

def test_contested_when_top_tiers_disagree():
    """The COVID-origins case: excellent sources that disagree is a different
    state from no evidence at all, and needs a different label."""
    items = [
        item(id=1, tier="authority", stance="supports", role="assesses",
             url="https://who.int/a", source_domain="who.int"),
        item(id=2, tier="peer_reviewed", stance="refutes", role="assesses",
             url="https://nature.com/b", source_domain="nature.com"),
    ]
    state = ev.compute(items, "science")
    _, standing = scoring.derive_verdict(50, state, items)
    assert standing == "contested"


def test_axes_are_independent():
    """Supported on axis 1, overstated on axis 2 — the case a single label
    cannot express. Several outlets assess the claim and agree, but no
    fact-check, literature or institutional source addresses it, so the
    evidence is thinner than the claim implies."""
    items = [
        item(id=i, tier="reporting", role="assesses", stance="supports",
             stance_confidence=0.9, published_date=date(2025, 1, 1),
             url=f"https://news{i}.com/x", source_domain=f"news{i}.com")
        for i in range(1, 5)
    ]
    state = ev.compute(items, "general")
    score, _ = scoring.compute_score(
        items, "general", 4, today=date(2025, 6, 1))
    verdict, standing = scoring.derive_verdict(score, state, items)
    assert verdict == "supported"
    assert standing == "overstated"


def test_low_tier_only_yields_no_score_at_all():
    """Four blogs is not evidence. The system must decline to put a number on
    it rather than reporting a confident-looking figure."""
    items = [
        item(id=i, tier="low", role="asserts", stance="supports",
             stance_confidence=0.9, published_date=date(2025, 1, 1),
             url=f"https://blog{i}.com/x", source_domain=f"blog{i}.com")
        for i in range(1, 5)
    ]
    state = ev.compute(items, "general")
    score, _ = scoring.compute_score(
        items, "general", 4, today=date(2025, 6, 1))
    verdict, _ = scoring.derive_verdict(score, state, items)
    assert score is None
    assert verdict == "unresolved"


# ------------------------------------------ fast path must match the claim

def test_fast_path_rejects_mismatched_factcheck():
    """Regression: "Covid originated from China" was returned as `refuted`
    because the fast path inherited a FactCheck.org rating for the DIFFERENT
    claim "Wuhan Lab Leak Theory CONFIRMED".

    ClaimReview search returns loosely related claims. A rating may only be
    inherited when the fact-checker reviewed the same claim."""
    from app.pipeline import stages
    from app.schemas import SubClaim

    mismatched = item(
        id=1, tier="prior_check", channel="prior_check", role="assesses",
        stance="refutes", stance_confidence=0.95, claim_match=0.15,
        url="https://factcheck.org/x", source_domain="factcheck.org",
    )
    claim = SubClaim(id=1, text="Covid originated from China")
    assert stages.fast_path_verdict(claim, [mismatched]) is None


def test_fast_path_accepts_matched_factcheck():
    from app.pipeline import stages
    from app.schemas import SubClaim

    matched = item(
        id=1, tier="prior_check", channel="prior_check", role="assesses",
        stance="refutes", stance_confidence=0.95, claim_match=0.93,
        url="https://snopes.com/x", source_domain="snopes.com",
    )
    v = stages.fast_path_verdict(
        SubClaim(id=1, text="A Fargo man was arrested."), [matched])
    assert v is not None and v.claim_verdict == "refuted"


def test_fast_path_declines_when_factcheckers_disagree():
    """Two fact-checks of the same claim reaching opposite ratings is not a
    shortcut — it is exactly the case that needs full evidence weighing."""
    from app.pipeline import stages
    from app.schemas import SubClaim

    items = [
        item(id=1, tier="prior_check", channel="prior_check", role="assesses",
             stance="refutes", stance_confidence=0.95, claim_match=0.9,
             url="https://a.org/x", source_domain="a.org"),
        item(id=2, tier="prior_check", channel="prior_check", role="assesses",
             stance="supports", stance_confidence=0.95, claim_match=0.9,
             url="https://b.org/x", source_domain="b.org"),
    ]
    assert stages.fast_path_verdict(SubClaim(id=1, text="X"), items) is None


def test_junk_boilerplate_never_becomes_a_quote():
    """Search extraction returns navigation furniture as body text. Left in,
    the user sees a cookie banner where evidence should be."""
    from app.pipeline.grading import sentences

    text = ("When autocomplete results are available use up and down arrows to review. "
            "The WHO joint mission found no evidence supporting that conclusion. "
            "Skip to main content and subscribe to our newsletter today please.")
    out = sentences(text)
    assert any("joint mission" in s for s in out)
    assert not any("autocomplete" in s.lower() for s in out)
    assert not any("subscribe to our" in s.lower() for s in out)


# --------------------------------------- fact-check ratings, not article prose

def test_nuanced_ratings_are_never_a_verdict():
    """Regression: "Distorts the Facts" was not in the rating vocabulary, so
    stance fell through to NLI over the article prose — which quotes the
    falsehood verbatim — and came back `supports`."""
    from app.pipeline.grading import _stance_from_rating

    for nuanced in ("Distorts the Facts", "Mixture", "Half True",
                    "Unproven", "Needs Context", "Partly false"):
        assert _stance_from_rating(nuanced) is None, nuanced


def test_clear_ratings_map_to_stance():
    from app.pipeline.grading import _stance_from_rating

    assert _stance_from_rating("False") == ("refutes", 0.95)
    assert _stance_from_rating("Pants on Fire") == ("refutes", 0.95)
    assert _stance_from_rating("No Evidence") == ("refutes", 0.95)
    assert _stance_from_rating("Accurate") == ("supports", 0.95)


@pytest.mark.asyncio
async def test_factcheck_quote_reads_as_a_rating_not_an_assertion():
    """The stored quote for a fact-check must never be the debunked claim
    presented as if the source asserted it."""
    from datetime import date as _date

    from app.pipeline import grading
    from app.schemas import RawDocument

    doc = RawDocument(
        url="https://www.factcheck.org/x", title="Baseless Claim",
        text="This virus, SARS-CoV-2, actually is not from nature.",
        published_date=_date(2020, 9, 1), source_domain="factcheck.org",
        channel="prior_check", prior_rating="No Evidence",
        prior_publisher="FactCheck.org",
        prior_claim_text="It is a man-made virus created in the lab.",
    )
    items = await grading.grade([doc], "Covid originated from China", 1, "medical")
    quote = items[0].quote
    assert quote.startswith("FactCheck.org rated the claim")
    assert "No Evidence" in quote
    # The debunked assertion must not stand alone as the evidence text.
    assert "actually is not from nature" not in quote


# ------------------------------------------- computed explanation fallback

def test_describe_evidence_never_returns_empty():
    """Regression: grounding deleted every generated sentence and the user was
    shown a verdict with a blank explanation. The fallback is assembled from
    counts, so it cannot fabricate."""
    from app.pipeline import evidence_state as _ev
    from app.pipeline.stages import describe_evidence

    items = [
        item(id=1, tier="prior_check", channel="prior_check", stance="discusses",
             claim_match=0.2, url="https://a.org/x", source_domain="a.org"),
        item(id=2, tier="authority", channel="authority", stance="supports",
             role="asserts", url="https://who.int/x", source_domain="who.int"),
    ]
    state = _ev.compute(items, "medical")
    text = describe_evidence(items, state)
    assert text.strip()
    assert "2 sources" in text
    assert "related but different claims" in text

    assert describe_evidence([], _ev.compute([], "medical")).strip()


@pytest.mark.skipif(
    nli.backend_name() != "transformers",
    reason="tests real NLI behaviour; the lexical stub cannot detect contradiction",
)
def test_grounding_rejects_contradicted_sentences():
    """The bar was lowered so valid paraphrases survive, but a sentence its own
    cited evidence contradicts must still be deleted."""
    from app.pipeline.stages import ground

    ev_items = [item(id=1, quote="There is no evidence this claim is true.")]
    kept, cited, total, dropped = ground(
        "The evidence confirms this claim is accurate [1].", ev_items
    )
    assert dropped == 1
    assert kept == ""


def test_grounding_drops_uncited_sentences():
    from app.pipeline.stages import ground

    ev_items = [item(id=1, quote="A study found no effect.")]
    kept, cited, total, dropped = ground(
        "This claim is unsupported by the literature.", ev_items
    )
    assert dropped == 1 and kept == ""


# ------------------------------- expected_tiers must match real tier names

def test_every_pack_expects_tiers_its_own_sources_produce():
    """Regression: the political pack listed Reuters/AP/BBC under `authority`
    but expected [definitional, independent_analysis, reporting]. Its own
    trusted sources could therefore never satisfy it, and a claim backed by
    Snopes, Reuters and AP was abstained on for "no source of the required
    type"."""
    from app.config import source_packs
    from app.schemas import Tier
    from typing import get_args

    valid = set(get_args(Tier))
    for name, pack in source_packs().items():
        expected = set(pack.get("expected_tiers", []))
        unknown = expected - valid
        assert not unknown, f"{name}: unknown tier names {unknown}"

        # If a pack names authority domains, `authority` must be acceptable.
        if pack.get("authority"):
            assert "authority" in expected, (
                f"{name} lists authority domains but does not expect the "
                "`authority` tier — its own sources could never satisfy it"
            )


def test_matching_factcheck_satisfies_tier_requirement():
    """A fact-checker who reviewed THIS claim is the best evidence available in
    any category, whatever the pack lists."""
    items = [item(id=1, tier="prior_check", channel="prior_check",
                  role="assesses", stance="supports", claim_match=0.95,
                  url="https://snopes.com/x", source_domain="snopes.com")]
    state = ev.compute(items, "medical")
    assert state.expected_tier_present is True
    assert ev.should_abstain(state, items) is None


def test_abstention_standing_is_derived_not_hardcoded():
    """"amplified" was returned for any non-empty abstention, including claims
    whose sources share no common origin."""
    assert ev.standing_for_abstention(
        ev.compute([], "medical"), []) == "unestablished"

    items = [item(id=i, tier="low", role="reports", stance="supports",
                  url=f"https://n{i}.com/x", source_domain=f"n{i}.com")
             for i in range(1, 6)]
    state = ev.compute(items, "medical")
    assert ev.standing_for_abstention(state, items) == "amplified"


# ------------------------------- promise vs outcome (the second axis's job)

def test_confirmed_promise_is_flagged_overstated():
    """Regression: "Trump will give $5,000 to every citizen" was verified as a
    PROMISE and reported with evidence_standing "consistent".

    The promise is real; the payment is not established. Axis 1 says the
    attribution holds, axis 2 must say the claim implies more than the evidence
    supports. This is the exact distinction the second axis exists for."""
    from app.pipeline import stages
    from app.schemas import SubClaim

    check = item(
        id=1, tier="prior_check", channel="prior_check", role="assesses",
        stance="supports", stance_confidence=0.95, claim_match=0.98,
        url="https://snopes.com/x", source_domain="snopes.com",
    )
    claim = SubClaim(
        id=1,
        text="Donald Trump promised $5,000 to every US adult.",
        attribution_extracted=True,
    )
    state = ev.compute([check], "political")
    v = stages.fast_path_verdict(claim, [check], state)

    assert v is not None
    assert v.claim_verdict == "supported"
    assert v.evidence_standing == "overstated"
    assert "does not establish" in v.explanation


def test_ordinary_confirmed_claim_stays_consistent():
    from app.pipeline import stages
    from app.schemas import SubClaim

    check = item(
        id=1, tier="prior_check", channel="prior_check", role="assesses",
        stance="supports", stance_confidence=0.95, claim_match=0.98,
        url="https://snopes.com/x", source_domain="snopes.com",
    )
    claim = SubClaim(id=1, text="The Eiffel Tower is in Paris.")
    state = ev.compute([check], "general")
    v = stages.fast_path_verdict(claim, [check], state)
    assert v.evidence_standing == "consistent"


def test_attribution_flag_also_applies_without_fast_path():
    state = ev.compute([], "political")
    _, standing = scoring.derive_verdict(
        90, state, [], attribution_extracted=True)
    assert standing == "overstated"


# ------------------------------------------------- role, denylist, URL dates

def test_role_is_judged_on_the_quote_not_the_document():
    """Regression: role was computed from the whole document. A long article
    contains "said" somewhere, so nearly everything was marked `reports`,
    which tripped the "everyone is just repeating this" abstention and killed
    valid claims."""
    from app.pipeline.grading import assign_role

    # The sentence used as evidence states the fact directly.
    assert assign_role(
        "The Federal Reserve raised interest rates by a quarter point on Wednesday."
    ) == "asserts"
    # A sentence that merely relays someone else's statement.
    assert assign_role(
        "According to officials, the policy said to be under review."
    ) == "reports"
    # A sentence that evaluates.
    assert assign_role(
        "Our analysis found no evidence supporting the claim.") == "assesses"


def test_primary_source_is_not_mere_repetition():
    """A central bank publishing its own rate decision is not "repeating a
    claim". Treating it that way abstained on a decision the Fed announced."""
    primary = item(id=1, tier="definitional", channel="authority",
                   role="asserts", stance="supports",
                   url="https://federalreserve.gov/x", source_domain="federalreserve.gov")
    press = [item(id=i, tier="authority", role="reports", stance="supports",
                  url=f"https://n{i}.com/x", source_domain=f"n{i}.com")
             for i in range(2, 5)]
    state = ev.compute([primary] + press, "financial")
    assert state.all_sources_report_only is False
    assert ev.should_abstain(state, [primary] + press) is None


def test_social_and_prediction_markets_are_never_evidence():
    """A Facebook video and a Polymarket page appeared as evidence for a
    Federal Reserve decision. Trading odds measure what bettors expect, not
    what happened."""
    from app.retrieval import _denied

    for bad in ("facebook.com", "www.youtube.com", "polymarket.com",
                "reddit.com", "x.com", "kalshi.com"):
        assert _denied(bad), bad
    for good in ("reuters.com", "federalreserve.gov", "who.int", "snopes.com"):
        assert not _denied(good), good


def test_dates_recovered_from_urls():
    """Institutional pages often expose no date to the search API, so a stale
    2026-04 FOMC minutes page looked as current as this week's statement — and
    contradicted the claim with no recency penalty."""
    from datetime import date as _date

    from app.retrieval import _date_from_url

    assert _date_from_url(
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260429.htm"
    ) == _date(2026, 4, 29)
    assert _date_from_url(
        "https://www.nytimes.com/live/2026/09/16/business/fed-meeting"
    ) == _date(2026, 9, 16)
    assert _date_from_url("https://www.who.int/news-room/questions") is None


# ------------------------------- the model may weaken, never strengthen

def test_mismatch_pattern_downgrades_a_supported_verdict():
    """Regression: "The Fed confirmed more hikes are coming" scored 97 and
    `supported`, on the strength of the Fed's note about THIS week's hike —
    topically close, semantically wrong.

    The explanation model diagnosed it correctly as `subset_stated_as_general`.
    That signal now weakens the verdict."""
    v, st = scoring.apply_pattern_downgrade(
        "supported", "consistent", "subset_stated_as_general"
    )
    assert v == "partly_supported"
    assert st == "overstated"


def test_ordinary_pattern_changes_nothing():
    v, st = scoring.apply_pattern_downgrade(
        "supported", "consistent", "directly_confirmed")
    assert (v, st) == ("supported", "consistent")
    v, st = scoring.apply_pattern_downgrade("refuted", "consistent", None)
    assert (v, st) == ("refuted", "consistent")


def test_downgrade_never_strengthens():
    """The model must not be able to manufacture support. Every mismatch
    pattern can only move a verdict downwards."""
    for pattern in scoring._MISMATCH_PATTERNS:
        v, _ = scoring.apply_pattern_downgrade(
            "unresolved", "contested", pattern)
        assert v == "unresolved"
        v, _ = scoring.apply_pattern_downgrade(
            "refuted", "consistent", pattern)
        assert v == "refuted"


# --------------------------------- dates are arithmetic, not entailment

def test_explicit_dates_stripped_before_stance_judging():
    """Regression: "raised interest rates during the week of September 17,
    2026" scored 0.066 entailment against CNN's near-identical "raised
    interest rates by a quarter of a percentage point".

    The assertion matched perfectly; the date phrasing did not. Sources say
    "Wednesday", "September 16", "effective September 17" — none of which an
    entailment model reconciles. The model judges meaning; the code checks
    dates."""
    from app.pipeline.grading import strip_explicit_date as strip

    assert strip("The Federal Reserve raised interest rates during the week of "
                 "September 17, 2026.") == "The Federal Reserve raised interest rates"
    assert strip(
        "Pluto was reclassified on August 24, 2006.") == "Pluto was reclassified"

    # Place names and figures must survive untouched.
    assert "Wuhan, China" in strip(
        "The COVID-19 outbreak was first detected in Wuhan, China.")
    assert "$5,000" in strip("Trump promised $5,000 to every US adult.")

    # A claim that is nothing but a date must not be emptied.
    assert strip("September 17, 2026").strip()


def test_headline_is_scored_as_evidence():
    """Regression: a Reuters article titled "Fed forecasts see latest hike
    followed by another before end year" was used only as premise context. The
    body sentence actually scored was about 2029 rates and came back `refutes`
    at 0.884, flipping the verdict."""
    from app.pipeline.grading import _candidate_sentences

    title = "Fed forecasts see latest hike followed by another before end year"
    body = ("Their new forecasts sees rates coming back down in 2028. "
            "The federal funds rate is projected to stand between 3.5% and 3.75% in 2029.")
    out = _candidate_sentences(
        title, body, "The Fed expects another rate hike this year", 3)
    assert out[0] == title


def test_junk_titles_are_not_scored():
    from app.pipeline.grading import _candidate_sentences

    out = _candidate_sentences("Skip to main content", "A real sentence about the claim here.",
                               "the claim", 2)
    assert not any("Skip to main" in s for s in out)


# ------------------------------- reporting IS the evidence for attributions

def test_reporting_counts_fully_for_attribution_claims():
    """Regression: "Trump stated the US will control Greenland's security" —
    BBC, the NYT and Reuters all reported him saying it, and it scored 0.316
    against a 0.35 threshold.

    The `reports` discount stops repetition counting as confirmation of a
    claim's CONTENT. But an attribution claim is about the utterance itself,
    and a credible outlet reporting that it was said is the primary evidence."""
    from datetime import date as _date

    reporters = [
        item(id=i, tier="authority", role="reports", stance="supports",
             stance_confidence=0.9, published_date=_date(2026, 9, 18),
             url=f"https://n{i}.com/x", source_domain=f"n{i}.com")
        for i in range(1, 4)
    ]
    today = _date(2026, 9, 20)

    content, _ = scoring.compute_score(reporters, "political", 3, today)
    attribution, _ = scoring.compute_score(
        reporters, "political", 3, today, attribution_extracted=True
    )
    assert content is None            # repetition does not confirm content
    assert attribution == 100          # but it does confirm the statement


def test_attribution_does_not_rescue_low_tier_reporting():
    """The fix lifts the ROLE weight, not the tier. Three blogs reporting a
    quote are still three blogs."""
    from datetime import date as _date

    blogs = [
        item(id=i, tier="low", role="reports", stance="supports",
             stance_confidence=0.9, published_date=_date(2026, 9, 18),
             url=f"https://b{i}.com/x", source_domain=f"b{i}.com")
        for i in range(1, 4)
    ]
    score, _ = scoring.compute_score(
        blogs, "political", 3, _date(2026, 9, 20), attribution_extracted=True
    )
    assert score is None


def test_downgrade_is_explained_in_fallback_text():
    """Regression: "partly supported" appeared beside "4 support the claim, 0
    contradict it" and a score of 100, with nothing explaining the gap. The
    model's reason had been deleted by grounding."""
    from app.pipeline.stages import describe_evidence

    items = [item(id=i, tier="authority", stance="supports",
                  url=f"https://n{i}.com/x", source_domain=f"n{i}.com")
             for i in range(1, 4)]
    state = ev.compute(items, "political")

    plain = describe_evidence(items, state)
    explained = describe_evidence(
        items, state, downgrade_pattern="quote_omits_context")
    assert "context" not in plain
    assert "context the claim leaves out" in explained


# ======================================================================
# Vitamin D regression: six bugs from one pilot claim
# ======================================================================

def test_bug1_tier_is_earned_by_domain_not_channel():
    """A supplement retailer arriving through the authority channel was tiered
    `authority`, cited as the sole support, and satisfied the expected-tier
    check that should have flagged the claim."""
    from app.pipeline.grading import assign_tier
    from app.schemas import RawDocument

    retailer = RawDocument(url="https://www.metagenics.ie/vitamin-d", title="t",
                           text="x", source_domain="metagenics.ie", channel="authority")
    assert assign_tier(retailer, "medical", None) == "low"

    who = RawDocument(url="https://www.who.int/x", title="t", text="x",
                      source_domain="who.int", channel="authority")
    assert assign_tier(who, "medical", None) == "authority"

    # Credible, but not a medical authority: journalism, not domain expertise.
    reuters = RawDocument(url="https://www.reuters.com/x", title="t", text="x",
                          source_domain="reuters.com", channel="authority")
    assert assign_tier(reuters, "medical", None) == "reporting"


def test_bug2a_hedged_evidence_does_not_support_categorical_claim():
    """"Supplements MAY HELP to prevent" was graded as supporting
    "supplements PREVENT"."""
    from app.pipeline.grading import hedge_mismatch

    claim = "Vitamin D supplements prevent respiratory infections"
    assert hedge_mismatch(
        "Vitamin D supplements may help to prevent infections.", claim)
    assert hedge_mismatch(
        "Some trials found a protective effect and others did not.", claim)
    assert not hedge_mismatch(
        "Vitamin D supplementation prevented infections.", claim)
    # A hedged claim matched by hedged evidence is not a mismatch.
    assert not hedge_mismatch(
        "Supplements may reduce risk.", "Supplements may reduce risk")


def test_bug2b_headlines_are_discounted():
    from app.pipeline.grading import HEADLINE_DISCOUNT

    assert 0 < HEADLINE_DISCOUNT < 1


@pytest.mark.asyncio
async def test_bug1b_out_of_pack_results_leave_the_authority_channel(monkeypatch):
    """The provider's domain filter is not strictly enforced. Out-of-pack
    results must not ride in on the authority channel."""
    import app.retrieval as R
    from app.schemas import RawDocument

    async def fake_tavily(claim, limit, include, channel):
        return [
            RawDocument(url="https://www.who.int/a", title="t", text="x",
                        source_domain="who.int", channel=channel),
            RawDocument(url="https://www.metagenics.ie/b", title="t", text="x",
                        source_domain="metagenics.ie", channel=channel),
        ]

    monkeypatch.setattr(R, "_tavily", fake_tavily)
    docs = await R.domain_search("claim", category="medical")
    by_domain = {d.source_domain: d.channel for d in docs}
    assert by_domain["who.int"] == "authority"
    assert by_domain["metagenics.ie"] == "open_web"


def test_bug3_pubmed_query_uses_keywords_not_sentence():
    """The raw claim returned one dose-comparison trial and missed every
    meta-analysis. Relation words are what's being tested, so requiring them
    filters out null results."""
    from app.retrieval import _pubmed_terms

    q = _pubmed_terms("Vitamin D supplements prevent respiratory infections.")
    assert "prevent" not in q.lower()
    assert "D" in q.split(), "the D in 'Vitamin D' must survive"
    assert "5G" in _pubmed_terms(
        "Some scientists believe 5G signals cause cancer.")
    assert "B12" in _pubmed_terms("Vitamin B12 deficiency causes memory loss.")


def test_bug4_thin_evidence_cannot_score_100():
    """Total weight 0.3678 — barely over the 0.35 floor — produced a score of
    100 because nothing refuted it. The ratio says how evidence leans, not how
    much there is."""
    from datetime import date as _date

    thin = [item(id=1, tier="authority", role="asserts", stance="supports",
                 stance_confidence=0.9, published_date=_date(2026, 1, 1),
                 url="https://a.com/x", source_domain="a.com"),
            item(id=2, tier="authority", role="asserts", stance="supports",
                 stance_confidence=0.9, published_date=_date(2026, 1, 1),
                 url="https://b.com/x", source_domain="b.com")]
    score, basis = scoring.compute_score(
        thin, "medical", 2, today=_date(2026, 2, 1))
    assert score is not None
    assert basis["raw_ratio"] == 100.0
    assert score < 80, "one-sided but thin evidence must not read as certainty"

    strong = [item(id=i, tier="peer_reviewed", role="assesses", stance="supports",
                   stance_confidence=0.95, published_date=_date(2026, 1, 1),
                   url=f"https://s{i}.org/x", source_domain=f"s{i}.org")
              for i in range(1, 5)]
    score, _ = scoring.compute_score(
        strong, "medical", 4, today=_date(2026, 2, 1))
    assert score == 100, "the same lean with strong evidence reaches full"


def test_bug5_overstated_fires_once_tiers_are_honest():
    """With the retailer correctly tiered `low`, a medical claim with no
    peer-reviewed or institutional support no longer passes the tier check."""
    items = [
        item(id=1, tier="low", role="asserts", stance="supports",
             url="https://metagenics.ie/x", source_domain="metagenics.ie"),
        item(id=2, tier="reporting", role="reports", stance="supports",
             url="https://news.harvard.edu/x", source_domain="news.harvard.edu"),
    ]
    state = ev.compute(items, "medical")
    assert state.expected_tier_present is False
    assert ev.should_abstain(state, items) is not None


def test_bug6_plural_claim_needs_plural_citations():
    """"Multiple authority sources" cited one source and passed grounding."""
    from app.pipeline.stages import ground

    ev_items = [item(id=1, quote="Supplementation reduced infection rates."),
                item(id=2, quote="Supplementation reduced infection rates in trials.",
                     url="https://b.com/x", source_domain="b.com")]
    _, _, _, dropped = ground(
        "Multiple authority sources confirm supplementation reduces infections [1].",
        ev_items)
    assert dropped == 1


# ======================================================================
# Round two: right answer, wrong reason
# ======================================================================

def test_retrieved_but_silent_top_tier_does_not_count():
    """Two peer-reviewed papers were retrieved, neither addressed the claim
    (both `insufficient`), and they still satisfied the expected-tier check.
    Retrieving something is not the same as finding evidence."""
    items = [
        item(id=1, tier="peer_reviewed", channel="literature", stance="insufficient",
             url="https://pubmed.ncbi.nlm.nih.gov/1/", source_domain="pubmed.ncbi.nlm.nih.gov"),
        item(id=2, tier="peer_reviewed", channel="literature", stance="insufficient",
             url="https://pubmed.ncbi.nlm.nih.gov/2/", source_domain="nih.gov"),
        item(id=3, tier="reporting", role="asserts", stance="supports",
             url="https://news.harvard.edu/x", source_domain="news.harvard.edu"),
    ]
    state = ev.compute(items, "medical")
    assert state.expected_tier_retrieved is True     # the search worked
    assert state.expected_tier_present is False      # nothing took a position
    reason = ev.should_abstain(state, items)
    # Must NOT blame the search when the search succeeded.
    assert reason and "limits of the search" not in reason
    assert "none took a clear position" in reason


def test_abstention_kind_separates_retrieval_failure_from_thin_evidence():
    """The examiner's question: how many good abstentions are failed searches?"""
    assert ev.abstention_kind(ev.compute([], "medical"), []) == "nothing_found"

    silent = [item(id=1, tier="peer_reviewed", stance="insufficient",
                   url="https://p.org/1", source_domain="p.org")]
    assert ev.abstention_kind(ev.compute(
        silent, "medical"), silent) == "retrieved_but_silent"
    assert "retrieved_but_silent" in ev.READING_LIMITED
    assert "retrieved_but_silent" not in ev.RETRIEVAL_LIMITED

    blogs = [item(id=1, tier="low", stance="supports",
                  url="https://b.com/1", source_domain="b.com")]
    kind = ev.abstention_kind(ev.compute(blogs, "medical"), blogs)
    assert kind == "expected_tier_not_retrieved"
    assert kind in ev.RETRIEVAL_LIMITED

    thin = [item(id=1, tier="peer_reviewed", stance="supports", role="assesses",
                 url="https://p.org/1", source_domain="p.org")]
    assert ev.abstention_kind(ev.compute(
        thin, "medical"), thin, scored=False) == "too_thin"
    assert "too_thin" not in ev.RETRIEVAL_LIMITED


def test_pattern_rejected_when_evidence_contradicts_it():
    """`expert_sources_disagree` was returned when nobody disagreed."""
    silent = [item(id=1, tier="peer_reviewed", stance="insufficient"),
              item(id=2, tier="reporting", stance="supports",
                   url="https://b.com/x", source_domain="b.com")]
    assert scoring.validate_pattern("expert_sources_disagree", silent) is None

    split = [item(id=1, tier="peer_reviewed", stance="supports"),
             item(id=2, tier="authority", stance="refutes",
                  url="https://b.com/x", source_domain="b.com")]
    assert scoring.validate_pattern(
        "expert_sources_disagree", split) == "expert_sources_disagree"

    assert scoring.validate_pattern("directly_contradicted", silent) is None
    assert scoring.validate_pattern(
        "figure_outdated", silent) == "figure_outdated"


def test_thin_evidence_is_not_labelled_contested_unless_it_disagrees():
    one_sided = [item(id=1, tier="peer_reviewed", stance="supports", role="assesses",
                      url="https://p.org/1", source_domain="p.org")]
    assert ev.standing_for_abstention(ev.compute(
        one_sided, "medical"), one_sided) == "unestablished"


def test_pubmed_month_dates_are_parsed():
    """PubMed's usual "2008 Mar" silently became None, starving the outdated
    detection of dates. PubMed always has a date."""
    from datetime import date as _date

    from app.retrieval import _parse_date

    assert _parse_date("2008 Mar") == _date(2008, 3, 1)
    assert _parse_date("2008 Mar 15") == _date(2008, 3, 15)
    assert _parse_date("2008 Mar-Apr") == _date(2008, 3, 1)
    assert _parse_date("2008") == _date(2008, 1, 1)
    assert _parse_date("2026-09-16T10:00:00Z") == _date(2026, 9, 16)
    assert _parse_date(None) is None


def test_partial_effect_is_not_prevention():
    """"Reduce risk by up to 12%" was graded `supports` at 0.887 for
    "prevent". The hedge check caught "may" but not a weaker effect stated
    without hedging."""
    from app.pipeline.grading import hedge_mismatch

    claim = "Vitamin D supplements prevent respiratory infections"
    assert hedge_mismatch(
        "Vitamin D reduced the risk of infection by 12%.", claim)
    assert hedge_mismatch(
        "A modest reduction in infection rates was observed.", claim)
    assert not hedge_mismatch(
        "Vitamin D prevented infections in all participants.", claim)
    # A non-categorical claim is not overstated by a partial effect.
    assert not hedge_mismatch("Vitamin D reduced the risk of infection by 12%.",
                              "Vitamin D reduces the risk of respiratory infection")


def test_concept_query_ors_synonyms_and_ands_concepts():
    from app.retrieval import _concept_query

    q = _concept_query([["vitamin D", "cholecalciferol"],
                        ["respiratory infection", "common cold"]])
    assert q == ('("vitamin D" OR cholecalciferol) AND '
                 '("respiratory infection" OR "common cold")')
    assert "prevent" not in q


def test_claim_match_is_null_outside_factchecks():
    """It defaulted to 1.0, so an unrelated 2008 article carried claim_match 1
    — a field uniformly 1 distinguishes nothing."""
    from app.schemas import EvidenceItem

    assert EvidenceItem.model_fields["claim_match"].default is None


def test_retrieval_recall_matching():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eval.retrieval import recall_for

    rec = {"must_retrieve": ["12345", "10.1136/bmj.i6583", "who.int/news/item/vitd"],
           "documents": {"1": [{"url": "https://pubmed.ncbi.nlm.nih.gov/12345/"},
                               {"url": "https://www.who.int/news/item/vitd-2025"}]}}
    found, wanted, missed = recall_for(rec)
    assert (found, wanted) == (2, 3)
    assert missed == ["10.1136/bmj.i6583"]


# ======================================================================
# Round three: the search worked, the reading didn't
# ======================================================================

def test_pubmed_entities_are_decoded():
    """"OR 1·40; 95% CI 0·70, 2·79" reached the model as
    "OR 1&#xb7;40; 95 % CI 0&#xb7;70" — the numbers carrying the finding were
    unreadable."""
    from app.retrieval import _clean

    assert _clean(
        "OR, 1&#xb7;40; 95 % CI: 0&#xb7;70, 2&#xb7;79") == "OR, 1.40; 95 % CI: 0.70, 2.79"
    assert _clean(
        "<i>in vitro</i> &amp; <b>in vivo</b>") == "in vitro & in vivo"


def test_structured_abstract_keeps_its_findings():
    """The parser discarded AbstractText attributes — exactly where PubMed puts
    Label="CONCLUSIONS"."""
    from app.retrieval import _parse_abstracts

    xml = ('<PubmedArticle><PMID Version="1">1</PMID>'
           '<AbstractText Label="BACKGROUND">In 2017, 18% of adults exceeded 1000 IU.</AbstractText>'
           '<AbstractText Label="RESULTS">No significant effect on incidence.</AbstractText>'
           '<AbstractText Label="CONCLUSIONS">Supplementation did not reduce infections.</AbstractText>'
           '</PubmedArticle>')
    full, findings = _parse_abstracts(xml)["1"]
    assert "18% of adults" in full
    assert "18% of adults" not in findings
    assert "did not reduce" in findings


def test_findings_are_scored_before_background():
    from app.pipeline.grading import _candidate_sentences

    out = _candidate_sentences(
        "A systematic review", "In 2017, 18% of adults exceeded 1000 IU daily. Background.",
        "Vitamin D prevents colds", 2,
        key_findings="Supplementation did not reduce the incidence of respiratory infection.")
    assert out[0].startswith("Supplementation did not reduce")


def test_null_result_refutes_a_categorical_claim():
    """A 2024 meta-analysis reporting "no significant effect" scored
    `insufficient` at 0.104. NLI reasonably reads a statistical null as
    neutral, but against "prevents" it is precisely the evidence that matters."""
    from app.pipeline.grading import (NULL_RESULT_CONFIDENCE,
                                      NULL_SUBGROUP_CONFIDENCE, null_result_against)

    claim = "Vitamin D supplements prevent colds and flu"
    assert null_result_against(
        "Supplementation did not reduce the incidence of respiratory infection.", claim
    ) == NULL_RESULT_CONFIDENCE
    # A subgroup null says less about the general claim.
    assert null_result_against(
        "There was no significant effect on ARTI incidence in ethnic minority groups.", claim
    ) == NULL_SUBGROUP_CONFIDENCE
    # A positive finding is not a null.
    assert null_result_against(
        "Supplementation reduced the risk of infection.", claim) is None
    # A non-categorical claim is left to the NLI model.
    assert null_result_against("did not reduce infection.",
                               "Vitamin D is associated with fewer colds") is None


@pytest.mark.asyncio
async def test_null_result_rule_applies_only_to_literature():
    """A news article saying a study "did not reduce" something is reporting,
    not the finding itself. The rule is confined to peer-reviewed abstracts."""
    from app.pipeline import grading
    from app.schemas import RawDocument

    lit = RawDocument(url="https://pubmed.ncbi.nlm.nih.gov/9/", title="A review",
                      text="Supplementation did not reduce respiratory infection incidence.",
                      source_domain="pubmed.ncbi.nlm.nih.gov", channel="literature",
                      key_findings="Supplementation did not reduce respiratory infection incidence.")
    news = RawDocument(url="https://news.example.com/x", title="A story",
                       text="Supplementation did not reduce respiratory infection incidence.",
                       source_domain="news.example.com", channel="open_web")
    items = await grading.grade([lit, news], "Vitamin D supplements prevent colds", 1, "medical")
    by_channel = {i.channel: i for i in items}
    assert by_channel["literature"].stance == "refutes"
    assert by_channel["literature"].stance_confidence == grading.NULL_RESULT_CONFIDENCE
    # The news item may still read as refuting via the NLI model, but never
    # through the null-result rule — so it must not carry the rule's constant.
    assert by_channel["open_web"].stance_confidence != grading.NULL_RESULT_CONFIDENCE


# ======================================================================
# Round four: preferring findings, and research as assessment
# ======================================================================

@pytest.mark.asyncio
async def test_findings_sentence_wins_over_background():
    """Listing findings first was not enough — selection picks the highest NLI
    score, so a background sentence with good word overlap still won. Item 1 of
    the vitamin D trace kept quoting hypercalcaemia statistics from a review
    whose conclusion was the point."""
    from app.pipeline import grading
    from app.schemas import RawDocument

    background = ("In 2017, a study uncovered increasing trends in vitamin D "
                  "supplementation, revealing that 18% of adults exceeded 1000 IU daily.")
    conclusion = "Vitamin D supplementation did not prevent respiratory infection."

    doc = RawDocument(
        url="https://pubmed.ncbi.nlm.nih.gov/1/", title="A systematic review of trials",
        text=f"{background} {conclusion}", source_domain="pubmed.ncbi.nlm.nih.gov",
        channel="literature", key_findings=conclusion,
    )
    items = await grading.grade([doc], "Vitamin D supplements prevent colds", 1, "medical")
    assert "18% of adults" not in items[0].quote


@pytest.mark.asyncio
async def test_peer_reviewed_research_counts_as_assessment():
    """Every paper scored `asserts` (0.4) because its findings sentence contains
    no phrase like "our analysis found" — so a meta-analysis directly on the
    question counted for less than a news article. Research evaluates a
    hypothesis; that is what the genre is."""
    from app.pipeline import grading
    from app.schemas import RawDocument

    paper = RawDocument(
        url="https://pubmed.ncbi.nlm.nih.gov/2/", title="A meta-analysis",
        text="There was no significant effect on infection incidence.",
        source_domain="pubmed.ncbi.nlm.nih.gov", channel="literature",
        key_findings="There was no significant effect on infection incidence.")
    items = await grading.grade([paper], "Vitamin D supplements prevent colds", 1, "medical")
    assert items[0].role == "assesses"


def test_corrupted_sentences_are_not_quotable():
    """"more susceptible13108-1/pdf) to getting colds" — link markup spliced
    into prose."""
    from app.pipeline.grading import sentences

    text = ("Low vitamin D may make you more susceptible13108-1/pdf) to getting colds. "
            "Vitamin D supports immune function in several understood ways.")
    kept = sentences(text)
    assert not any("13108-1/pdf" in s for s in kept)
    assert any("immune function" in s for s in kept)
    assert not sentences(
        "See [the study](https://example.com/paper.pdf) for details here.")


# ======================================================================
# Round five: reading a systematic review
# ======================================================================

def test_null_results_without_the_word_significant():
    """"11 studies revealed no Vitamin D impact on outcomes including
    respiratory infections" scored 0.066 — the pattern required the literal
    word "significant"."""
    from app.pipeline.grading import null_result_against

    claim = "Vitamin D supplements prevent colds and flu"
    for sentence in [
        "Conversely, 11 studies revealed no Vitamin D impact on outcomes "
        "including respiratory infections.",
        "The review found no protective effect.",
        "Supplementation showed no benefit over placebo.",
        "There was no significant difference between groups.",
    ]:
        assert null_result_against(sentence, claim), sentence


def test_safety_findings_never_refute_an_efficacy_claim():
    """"No adverse effects were reported" is a finding about harm. Reading it
    as evidence against efficacy would invert its meaning."""
    from app.pipeline.grading import null_result_against

    claim = "Vitamin D supplements prevent colds and flu"
    assert null_result_against(
        "No adverse effects were reported in any trial.", claim) is None
    assert null_result_against(
        "No serious side effects occurred.", claim) is None
    assert null_result_against(
        "No safety concerns were identified.", claim) is None


def test_sample_descriptions_are_not_findings():
    """PubMed files sample sizes under RESULTS, so the findings boost lifted
    "The analysis included 43 RCTs with 49320 participants" — a description of
    the study, not of what it found."""
    from app.pipeline.grading import is_study_description

    assert is_study_description(
        "The analysis included 43 RCTs with 49320 participants.")
    assert is_study_description(
        "We searched CINAHL, Medline, Embase and Cochrane for RCTs.")
    # A result that happens to count studies is still a result.
    assert not is_study_description(
        "11 studies revealed no Vitamin D impact on respiratory infections.")
    assert not is_study_description(
        "Supplementation reduced infection risk by 12%.")


def test_conclusions_are_ordered_before_results():
    """A conclusion states the verdict; results state the numbers."""
    from app.retrieval import _parse_abstracts

    xml = ('<PubmedArticle><PMID Version="1">1</PMID>'
           '<AbstractText Label="METHODS">We searched four databases.</AbstractText>'
           '<AbstractText Label="RESULTS">The analysis included 43 RCTs.</AbstractText>'
           '<AbstractText Label="CONCLUSIONS">Supplementation did not prevent infection.</AbstractText>'
           '</PubmedArticle>')
    _, findings = _parse_abstracts(xml)["1"]
    assert findings.index("did not prevent") < findings.index(
        "included 43 RCTs")
    assert "four databases" not in findings


def test_fallback_text_distinguishes_missing_from_silent():
    """"No source of the type this claim would require was found" was shown
    when sources HAD been found and simply took no position."""
    from app.pipeline.stages import describe_evidence

    silent = [item(id=1, tier="peer_reviewed", channel="literature", stance="insufficient",
                   url="https://pubmed.ncbi.nlm.nih.gov/1/",
                   source_domain="pubmed.ncbi.nlm.nih.gov")]
    text = describe_evidence(silent, ev.compute(silent, "medical"))
    assert "none took a clear position" in text
    assert "would require was found" not in text

    absent = [item(id=1, tier="low", stance="supports",
                   url="https://b.com/x", source_domain="b.com")]
    assert "would require was found" in describe_evidence(
        absent, ev.compute(absent, "medical"))
