# FactShield — Architecture

Version 2.0 · Supersedes SDD v1.1

---

## 1. What this system is

FactShield is a browser extension and backend that verifies a highlighted claim against
controlled sources and explains its reasoning using only retrieved evidence.

It is built as a **controlled instrument**, not a competitor to general assistants. The
research questions require a system whose retrieval, presentation, and failure modes can
be held fixed or varied deliberately — something no commercial tool permits. Design
decisions below favour *controllability and traceability* over raw accuracy where the two
conflict.

Three properties follow from that, and they constrain everything else:

1. **The model never supplies facts.** It arranges and explains retrieved evidence. Every
   factual statement in the output traces to a specific sentence in a specific document.
2. **Abstention is a first-class outcome.** "Cannot determine" is a correct answer, not a
   failure path. The system must be good at recognising when evidence is insufficient.
3. **Every stage is logged in a schema that doubles as evaluation data.** The benchmark is
   a query over production logs, not a parallel script that may drift from what production
   does.

---

## 2. Output model

### No numeric score

The system outputs categorical verdicts only. A model-generated confidence number is not
calibrated, cannot be justified against a neighbouring value, and would act as an
uncontrolled trust signal in the interface. Where a strength indicator is useful, it is
**computed from retrieval metadata** (independent source count, tier, agreement, recency),
never asked of the model.

### Two axes

A single label cannot express the cases that matter most. *"Some scientists believe 5G
causes cancer"* is literally true and epistemically misleading; satirical framing is
factually accurate and deceptive in context. These are not edge cases — two of the five
target claim categories are defined by them.

**Axis 1 — `claim_verdict`**: are the constituent propositions supported by evidence?

| Value | Meaning |
|---|---|
| `supported` | Evidence supports the claim as stated |
| `refuted` | Evidence contradicts the claim |
| `partly_supported` | Decomposed sub-claims resolve differently |
| `unresolved` | Evidence insufficient to determine |

**Axis 2 — `evidence_standing`**: does the evidence base match what the claim implies?

| Value | Meaning | Detected by |
|---|---|---|
| `consistent` | Evidence quality matches the claim's implied backing | default |
| `overstated` | Claim implies stronger backing than exists | expected tier absent; only low-tier sources |
| `outdated` | Was supported; evidence has since moved | supporting sources predate contradicting ones |
| `contested` | Comparable-standing sources disagree | stance split within top tier |
| `unestablished` | No evidence of the expected type exists yet | expected tier empty, claim recent |
| `amplified` | Many sources, one origin | sources postdate claim and share a root |

Axis 2 is computed largely in code from document metadata, not inferred by the model. That
is deliberate: arithmetic over dates and tiers cannot hallucinate.

---

## 3. Pipeline

```
capture → triage → retrieve (parallel) → grade → assess → decide → ground → present
```

### 3.1 Capture (extension, no model)

Always collects, regardless of whether the selection appears self-contained:

- The selected text, verbatim
- The containing paragraph and the two preceding it
- Page headline, publication date, author, domain
- Page metadata (`<meta>`, JSON-LD) — more reliable than visible text for dates

The publication date is required even when the selection is unambiguous, because
outdated-fact detection depends on comparing it to source dates.

### 3.2 Triage (one small-model call)

Single structured call producing:

- `checkable`: is this a verifiable factual claim, or opinion / question / chit-chat?
- `resolved_claim`: the selection rewritten as a standalone sentence, with pronouns and
  deixis resolved against captured context
- `category`: routing key for source selection
- `jurisdiction`: for political and legal claims, which country's authorities apply
- `sub_claims`: decomposition, where the selection contains more than one proposition
- `resolution_confident`: false when antecedents are ambiguous

If `checkable` is false, return immediately. If `resolution_confident` is false, proceed but
surface the ambiguity to the user rather than guessing silently.

**The resolved claim is the cache key and the retrieval input. The raw selection is what the
user sees.** Caching on raw text collides across unrelated articles.

### 3.3 Retrieve (three channels, concurrent)

| Channel | Source | Answers |
|---|---|---|
| Prior checks | Google Fact Check Tools API (ClaimReview) | Has this been checked before? |
| Domain authority | Tavily, restricted to the category's source pack | What do the expected authorities say? |
| Open web | Tavily, unrestricted | Anything too recent for the above |

Run concurrently. Sequential execution triples latency for no benefit.

The prior-checks channel queries a live API, not a stored corpus — fact-checkers' ClaimReview
markup is indexed as they publish. No local claim database.

**Low-tier results are retained, not discarded.** Fifteen blogs tracing to one interview is a
finding (`amplified`), and discarding them makes that undetectable.

### 3.4 Grade each document (NLI model, batched)

Per retrieved document:

| Field | Source |
|---|---|
| `stance` | NLI model: supports / refutes / insufficient / discusses |
| `quote` | The exact sentence carrying that stance, extracted verbatim |
| `published_date` | Document metadata |
| `tier` | Domain lookup against the category's source pack |
| `role` | Does this document **assess** the claim, or merely **report** that someone made it? |

`role` matters for prominent claims. Hundreds of outlets reporting *that a figure was
stated* is not evidence the figure is correct. Without this field the aggregator counts
coverage volume as corroboration.

`quote` must be verifiable by string match against the source document. A paraphrase can
drift; a quote either appears or it does not.

Use an off-the-shelf NLI model fine-tuned on FEVER. No training required to start.

### 3.5 Assess evidence state (code, no model)

Computed, not inferred:

- Claim age, where determinable
- Earliest independent source, and whether it predates or postdates the claim
- Whether sources share a common origin
- Whether the category's expected top tier is represented at all
- Stance distribution within each tier
- Whether supporting sources systematically predate refuting ones

This block produces `evidence_standing` and drives abstention.

### 3.6 Decide

**Short-circuit before the model:** if no documents survive grading, return `unresolved` in
code. Do not call the model. It cannot hallucinate on a call that was never made.

Otherwise, one structured call receiving the **evidence table only** — never raw concatenated
snippets, never the model's own prior knowledge. The prompt states that the supplied evidence
is the sole permissible basis, and that insufficiency is a valid answer.

Explanation is selected from a fixed enum of patterns rather than written freely:

```
figure_outdated · study_not_human · quote_omits_context · correlation_as_causation
subset_stated_as_general · timeframe_cherry_picked · no_independent_confirmation
source_is_reporting_not_assessing · single_origin_amplification
```

Each pattern carries slots filled from evidence. Free-form prose is where invention happens;
a constrained pattern plus filled slots is far harder to fabricate.

### 3.7 Grounding enforcement

Four layers, each catching what the previous misses:

1. **Evidence IDs.** Documents numbered; every statement must cite one. A statement with no
   valid ID is unsupported by construction and is dropped in code.
2. **Quote verification.** Cited quotes are string-matched against the source document. No
   match, no use.
3. **Entailment check.** Each sentence of the generated explanation is run back through the
   NLI model against its cited evidence. Unentailed sentences are dropped, and one
   regeneration is attempted with the failure flagged.
4. **Faithfulness logging.** The rate of dropped statements is recorded per request. This is
   a reportable metric, separable from verdict accuracy, and almost never reported in the
   literature.

Layers 1–2 reduce fabrication. Layer 3 detects it. Layer 4 measures it.

### 3.8 Present

Order matters and is a study variable:

1. Resolved claim — *"Checking: [X] said [Y] would double in 2026"* — so a bad resolution is
   correctable rather than silent
2. Evidence summary — count and tier composition
3. Source cards — quote, publication, date, stance
4. Verdict on both axes
5. Explanation

Colour is reinforcement only, never the sole carrier: every verdict also has a text label and
a distinct icon shape. Red/green alone fails WCAG 1.4.1 Level A and is the most common form of
colour vision deficiency.

---

## 4. Data model

```python
from datetime import date
from typing import Literal
from pydantic import BaseModel, Field, HttpUrl

class CaptureContext(BaseModel):
    selection: str
    surrounding_text: str
    page_title: str | None
    page_published: date | None
    page_domain: str

class TriageOutput(BaseModel):
    checkable: bool
    resolved_claim: str | None
    sub_claims: list[str] = []
    category: Literal["medical", "science", "political", "financial",
                      "breaking_news", "general"]
    jurisdiction: str | None = None
    resolution_confident: bool = True

class EvidenceItem(BaseModel):
    id: int
    title: str
    url: HttpUrl
    quote: str = Field(description="Verbatim sentence carrying the stance")
    published_date: date | None
    channel: Literal["prior_check", "domain_authority", "open_web"]
    tier: Literal["definitional", "peer_reviewed", "authority",
                  "independent_analysis", "reporting", "low"]
    stance: Literal["supports", "refutes", "insufficient", "discusses"]
    role: Literal["assesses", "reports", "asserts"]

class EvidenceState(BaseModel):
    claim_age_hours: float | None
    earliest_independent_source: date | None
    sources_predate_claim: bool
    shares_common_origin: bool
    expected_tier_present: bool
    tier_stance_split: dict[str, dict[str, int]]

class Verdict(BaseModel):
    claim_verdict: Literal["supported", "refuted", "partly_supported", "unresolved"]
    evidence_standing: Literal["consistent", "overstated", "outdated",
                               "contested", "unestablished", "amplified"]
    explanation_pattern: str | None
    explanation: str
    cited_evidence_ids: list[int]
    abstained: bool
    abstention_reason: str | None
```

---

## 5. Source tiers

Tier hierarchy is **per category**. A systematic review tops the medical hierarchy; for
breaking news no peer-reviewed source can exist, and two independent wire services is the
ceiling.

| Category | Top tier | Second | Notes |
|---|---|---|---|
| Medical | Systematic reviews, Cochrane, WHO, NICE | Primary studies (PubMed) | Single study ≠ consensus |
| Science | Consensus bodies, IPCC, NASA | Peer-reviewed primary | Watch for superseded findings |
| Political — definitional | The issuing authority (USCIS, gov.uk, Federal Register) | — | Government defining its own rule is ground truth |
| Political — interpretive | Independent analysis (CBO, IFS, academic) | Official figures, as *one* voice | Government claims about its own outcomes are a party to the dispute |
| Financial | Regulatory filings, SEC EDGAR, central banks | Financial press | |
| Breaking news | Two independent wire services | Single wire | Peer review cannot exist yet |

**The political split is load-bearing.** Claims about *what a rule says* have an authoritative
source. Claims about *what a rule does* are contested empirical questions, and treating the
incumbent administration's figures as ground truth makes the tool inherit whoever is in office.
For interpretive claims, the honest output names the disagreement rather than resolving it.

Source packs live in versioned config, not code, so changes are auditable and the list can be
published in the thesis.

---

## 6. Abstention rules

The system returns `unresolved` when:

- No documents survive grading
- No document in the category's expected top tier addresses the claim
- All retrieved documents `report` rather than `assess` the claim
- Top-tier sources split on stance with no tiebreaker
- Every supporting source postdates the claim and shares a common origin
- Triage resolution was not confident and the ambiguity is material

Abstention is **explained**, not bare: *"No sources from the vetted medical list address this
claim; it appears in four blog posts, all citing one 2018 interview."* That is more useful to
a reader than a verdict, and it is fully derived from logged metadata.

---

## 7. Repository layout

```
factshield/
├── extension/
│   ├── manifest.json            # MV3
│   ├── background.js            # context menu, registered on install
│   ├── content/
│   │   ├── capture.js           # DOM walk, metadata extraction
│   │   └── popup/               # shadow DOM, two layout variants
│   └── config.js                # participant id, ui_condition
├── backend/
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py           # Pydantic contracts above
│   ├── pipeline/
│   │   ├── triage.py
│   │   ├── retrieval/
│   │   │   ├── prior_checks.py
│   │   │   ├── domain_authority.py
│   │   │   └── open_web.py
│   │   ├── grading.py           # NLI stance, quote extraction, role
│   │   ├── evidence_state.py    # pure functions, no model calls
│   │   ├── decide.py
│   │   └── grounding.py         # ID check, quote match, entailment
│   ├── config/
│   │   ├── source_packs.yaml    # versioned, per category and jurisdiction
│   │   ├── models.yaml          # provider and model per stage
│   │   └── tiers.yaml
│   └── cache.py                 # Redis, keyed on resolved_claim
├── eval/
│   ├── conditions/
│   │   ├── parametric.py        # no retrieval — the offline-LLM baseline
│   │   ├── open_web_only.py     # approximates a general search assistant
│   │   ├── no_page_context.py   # bare selection, for the context ablation
│   │   └── full.py
│   ├── dataset/                 # claims, labels, annotator agreement
│   └── analysis/                # queries over the production log schema
└── docs/
```

**The parametric baseline lives in `eval/`, never in `backend/pipeline/`.** Running a
no-retrieval pass and feeding its answer to the verdict step would anchor the output and
destroy the claim that verdicts are evidence-grounded. Run it separately, log it, and compare
— that yields the offline-LLM condition for free *and* a reportable disagreement rate between
model prior and evidence.

---

## 8. Model configuration

Models are config values, never hardcoded. Use a provider abstraction (LiteLLM or equivalent)
so a provider change is a config edit.

| Stage | Requirement | Candidate |
|---|---|---|
| Triage | Fast, reliable structured output | Groq (free tier) |
| Stance + quote | NLI, runs locally | FEVER-tuned NLI model |
| Verdict | Careful reasoning over conflicting evidence | Gemini / GPT-4o — measure and choose |
| Entailment check | Same as stance | Reuse local NLI model |

Which model to use at the verdict step is an **empirical question**, not a design decision.
Run the claim set through several and report the spread — a purpose-built pipeline on a
cheaper model versus a general tool on a better one is itself a result.

Free-tier terms often permit training on submitted data. Acceptable for benchmarking;
resolve before any session involving participants.

---

## 9. Performance

| Path | Target |
|---|---|
| Cache hit | < 100 ms |
| Cold, cached-adjacent | < 10 s |
| Cold, full retrieval | < 15 s |

The earlier 1.5 s target was not achievable. Advanced-depth search alone runs 2–5 s before any
model call; published RAG fact-checking extensions report 8–17 s end to end.

Mitigations: concurrent retrieval, batched grading, Redis on `resolved_claim` with a 48-hour
TTL, and **streaming the popup** — evidence renders as it lands, verdict resolves last.
Perceived latency drops sharply even when total time does not.

---

## 10. Instrumentation

Built in from the first commit, not retrofitted.

- `participant_id` and `ui_condition`, set at install, immutable
- Per-request log: raw selection, resolved claim, every `EvidenceItem` with full metadata,
  `EvidenceState`, stage timings, verdict, abstention reason, faithfulness drop count
- Client events: request issued, popup opened, sources expanded, source clicked, time to close

**The log schema is the evaluation schema.** Ablations become log filters. The circularity
analysis is already present because publication dates are recorded. This avoids the common
failure where a separate evaluation script diverges from production behaviour.

---

## 11. Build order

A working end-to-end path first, depth after. Each stage leaves something that runs.

1. **Skeleton** — context menu → capture → one search → one model → popup. Plus participant
   ID, condition flag, and the full log schema.
2. **Substance** — NLI grading with quotes and dates, tiered source packs, prior-checks
   channel, abstention rules, both popup layouts.
3. **Grounding** — evidence IDs, quote verification, entailment check, split verdict and
   explanation calls, decomposition.
4. **Freeze.** Pick the date in advance. After it, no pipeline changes — everything goes to
   evaluation.

Build a 20-claim regression set (four per category) as soon as the skeleton runs. Without it,
prompt changes are untestable and a regression on hedged claims goes unnoticed for weeks.

---

## 12. Known limits

State these rather than let an examiner find them.

- **Hedged claims remain hard.** Entailment is structurally the wrong frame: an article
  debunking a topic does not contradict a statement about what some people believe. Axis 2
  mitigates this; it does not solve it.
- **Peer review is not a guarantee.** Predatory journals, retractions, single-study results,
  superseded findings. Retraction checking is out of scope for v2.0.
- **Source control does not deliver neutrality.** Credibility and political balance are
  different properties — retrieved evidence may skew even when every source is individually
  credible. Measure the skew and report it rather than claiming its absence.
- **Domain tiering is uneven.** Clean for medicine and science, contested for politics,
  absent for breaking news.
- **Abstention rate is a design parameter.** A system that abstains too readily is useless;
  one that never abstains is dangerous. Report the risk–coverage curve, not a single accuracy
  figure.
