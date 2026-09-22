"""Data contracts for the FactShield pipeline.

Every stage is a function over these types. Nothing downstream knows which
retrieval channel produced a document, which keeps adapters swappable.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

Category = Literal[
    "medical", "science", "political", "financial", "breaking_news", "general"
]

Tier = Literal[
    "prior_check",          # a published fact-check
    "definitional",         # the body that sets the rule
    "peer_reviewed",        # journal literature
    "authority",            # recognised institution, no API
    "independent_analysis", # CBO, IFS, academic commentary
    "reporting",            # news
    "low",                  # blogs, forums, content farms
]

Stance = Literal["supports", "refutes", "insufficient", "discusses"]
Role = Literal["assesses", "reports", "asserts"]
Channel = Literal["prior_check", "literature", "authority", "open_web"]

ClaimVerdict = Literal["supported", "refuted", "partly_supported", "unresolved"]
EvidenceStanding = Literal[
    "consistent",     # evidence quality matches what the claim implies
    "overstated",     # claim implies stronger backing than exists
    "outdated",       # was supported; evidence has moved
    "contested",      # comparable-standing sources disagree
    "unestablished",  # expected evidence type does not exist yet
    "amplified",      # many sources, one origin
]


# ---------------------------------------------------------------- capture

class CaptureContext(BaseModel):
    """What the extension sends. Context is always captured, never conditional:
    the page date is required for outdated-fact detection even when the
    selection is self-contained."""

    selection: str = Field(..., max_length=5000)
    surrounding_text: str = Field("", max_length=8000)
    page_title: str | None = None
    page_published: date | None = None
    page_domain: str | None = None
    page_lang: str | None = None


# ---------------------------------------------------------------- triage

class SubClaim(BaseModel):
    id: int
    text: str = Field(..., description="Standalone, pronouns resolved")
    # True when the original selection asserted a FUTURE OUTCOME and this
    # sub-claim is the underlying attribution. "Trump will give $5,000 to every
    # citizen" becomes "Trump promised $5,000 to every citizen": the promise is
    # verifiable, the payment is not. Confirming the promise must not be
    # presented as confirming the payment.
    attribution_extracted: bool = False


class TriageOutput(BaseModel):
    checkable: bool
    reason: str | None = None
    sub_claims: list[SubClaim] = []
    category: Category = "general"
    jurisdiction: str | None = None
    resolution_confident: bool = True

    @property
    def primary_claim(self) -> str | None:
        return self.sub_claims[0].text if self.sub_claims else None


# ---------------------------------------------------------------- retrieval

class RawDocument(BaseModel):
    """Uniform output of every adapter and search wrapper."""

    url: HttpUrl
    title: str
    text: str
    published_date: date | None = None
    source_domain: str
    channel: Channel
    structured_type: str | None = None   # e.g. "Systematic Review" from PubMed
    retracted: bool = False
    # Populated only by the prior_check channel
    prior_rating: str | None = None
    prior_publisher: str | None = None
    prior_claim_text: str | None = None   # the claim the fact-checker reviewed


# ---------------------------------------------------------------- grading

class EvidenceItem(BaseModel):
    id: int
    sub_claim_id: int
    url: HttpUrl
    title: str
    quote: str = Field(..., description="Verbatim sentence carrying the stance")
    published_date: date | None
    source_domain: str
    channel: Channel
    tier: Tier
    stance: Stance
    role: Role
    stance_confidence: float = Field(ge=0.0, le=1.0)
    # For prior_check items: how closely the fact-checker's reviewed claim
    # matches OURS. ClaimReview search returns loosely related claims, so a
    # rating cannot be trusted without checking it is about the same thing.
    claim_match: float = Field(default=1.0, ge=0.0, le=1.0)
    retracted: bool = False


class EvidenceState(BaseModel):
    """Computed by arithmetic over metadata. No model call. This is the one
    layer that structurally cannot fabricate, which is why the explanations
    that matter most are derived here."""

    n_documents: int
    n_independent_domains: int
    claim_age_hours: float | None = None
    earliest_source: date | None = None
    latest_source: date | None = None
    sources_postdate_claim: bool = False
    shares_common_origin: bool = False
    expected_tier_present: bool = True
    all_sources_report_only: bool = False
    tier_stance_split: dict[str, dict[str, int]] = {}
    supporting_predate_refuting: bool = False


# ---------------------------------------------------------------- verdict

class SubClaimVerdict(BaseModel):
    sub_claim_id: int
    sub_claim_text: str

    claim_verdict: ClaimVerdict
    evidence_standing: EvidenceStanding

    # Computed from retrieval metadata, never generated by a model.
    # None when evidence weight is below threshold — "unresolved" must not
    # collapse into a misleading 50.
    support_score: int | None = Field(
        default=None, ge=0, le=100,
        description="Evidence-weighted support. 100 = strongly supported, "
                    "0 = strongly refuted. None when evidence is insufficient.",
    )
    score_basis: dict[str, float] = Field(
        default_factory=dict,
        description="Components of support_score, so the number is auditable",
    )

    explanation_pattern: str | None = None
    explanation: str = ""
    cited_evidence_ids: list[int] = []

    abstained: bool = False
    abstention_reason: str | None = None


class CheckResponse(BaseModel):
    request_id: str
    checkable: bool
    not_checkable_reason: str | None = None

    resolved_claims: list[SubClaim] = []
    resolution_confident: bool = True
    category: Category = "general"

    verdicts: list[SubClaimVerdict] = []
    evidence: list[EvidenceItem] = []
    evidence_state: EvidenceState | None = None

    fast_path: bool = Field(
        default=False,
        description="True when a published fact-check resolved this directly, "
                    "skipping the verdict model entirely",
    )
    timings_ms: dict[str, float] = {}
    errors: list[str] = []


# ---------------------------------------------------------------- logging

class RequestLog(BaseModel):
    """This schema IS the evaluation dataset. Benchmark ablations are filters
    over this table, not a separate script that can drift from production."""

    request_id: str
    timestamp: datetime
    participant_id: str | None = None
    ui_condition: str | None = None
    extension_version: str | None = None
    pipeline_version: str
    prompt_versions: dict[str, str] = {}

    raw_selection_hash: str
    raw_selection: str | None = None      # study mode only
    resolved_claims: list[str] = []
    category: str
    resolution_confident: bool

    evidence: list[EvidenceItem] = []
    evidence_state: EvidenceState | None = None
    verdicts: list[SubClaimVerdict] = []

    fast_path: bool = False
    cache_hit: bool = False
    grounding_total: int = 0
    grounding_dropped: int = 0

    timings_ms: dict[str, float] = {}
    errors: list[str] = []