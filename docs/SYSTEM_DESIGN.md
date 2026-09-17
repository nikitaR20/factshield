# FactShield — System Design Document

Version 2.0 · Companion to `ARCHITECTURE.md`

`ARCHITECTURE.md` states what the system does and why. This document states how it is
built: module boundaries, interfaces, sequence, failure behaviour, configuration,
deployment, and test strategy.

---

## 1. Components

```
┌─ extension ──────────────────────────────────────────┐
│  background.js      context menu, request dispatch   │
│  capture.js         DOM walk, metadata extraction    │
│  popup/             shadow DOM, two layout variants  │
└──────────────────────────┬───────────────────────────┘
                           │ HTTPS / JSON
┌──────────────────────────┴───────────────────────────┐
│  api/          routing, validation, rate limiting    │
│  pipeline/     triage → retrieve → grade → decide    │
│  retrieval/    adapters + domain search              │
│  grounding/    citation, quote, entailment checks    │
│  cache/        Redis                                 │
│  logging/      structured per-request records        │
└──────────────────────────┬───────────────────────────┘
                           │
   ┌───────────────────────┼───────────────────────┐
   │                       │                       │
 model providers      free index APIs        search provider
 (triage, verdict)    (PubMed, CrossRef,     (domain-restricted
 local NLI            Fact Check Tools)       open web)
```

### 1.1 Extension

| Module | Responsibility | Must not |
|---|---|---|
| `background.js` | Register context menu on install; dispatch requests; hold no state | Assume it stays alive — MV3 service workers terminate after ~30 s idle |
| `capture.js` | Walk the DOM upward from selection; extract metadata | Send the whole page |
| `popup/` | Render inside shadow DOM; stream results as they arrive | Leak styles into the host page, or depend on host CSS |
| `config.js` | Hold `participant_id`, `ui_condition`, `extension_version` | Change condition after install |

Register the context menu in `chrome.runtime.onInstalled`, not at worker startup —
the worker restarts frequently and re-registering throws.

Popup renders in a closed shadow root. Host pages routinely define aggressive global CSS;
without isolation the popup breaks on a meaningful fraction of sites.

### 1.2 Backend modules

Each pipeline stage is a pure function over its input where possible, with I/O confined
to the retrieval and model-call layers. This is what makes the stages independently
testable and independently ablatable.

| Module | Input | Output | Model calls |
|---|---|---|---|
| `triage` | `CaptureContext` | `TriageOutput` | 1 (small) |
| `retrieval/*` | `TriageOutput` | `list[RawDocument]` | 0 |
| `grading` | `RawDocument`, claim | `EvidenceItem` | local NLI, batched |
| `evidence_state` | `list[EvidenceItem]` | `EvidenceState` | **0 — pure arithmetic** |
| `decide` | evidence table, state | `Verdict` | 1 (large) |
| `grounding` | `Verdict`, evidence | `Verdict` (filtered) | local NLI |

`evidence_state` takes no model call by design. Claim age, source ordering, tier coverage
and origin overlap are arithmetic over metadata; routing them through a model would
introduce fabrication into the one layer that currently cannot fabricate.

---

## 2. Retrieval layer

Two mechanisms, not one. The distinction is **index versus site**.

### 2.1 Index adapters

An adapter wraps a structured API that indexes many sources. Small in number.

| Adapter | Covers | Returns | Auth |
|---|---|---|---|
| `pubmed` | Medical literature (thousands of journals) | Title, abstract, publication type, date, DOI | None for modest use |
| `crossref` | Academic publishing broadly | Metadata, retraction notices | None |
| `factcheck` | Every publisher emitting ClaimReview | Claim, rating, publisher, date | API key, free |
| `federal_register` | US rulemaking | Document text, effective dates | None |
| `govuk` | UK government content | Page content, updated dates | None |

Adapters are preferred where they exist because they return **structured metadata** —
publication type, date, retraction status — which populates `tier` and `published_date`
directly rather than inferring them from a URL.

`pubmed` returning `Systematic Review` or `Randomized Controlled Trial` as a field is
strictly better than guessing tier from `pubmed.ncbi.nlm.nih.gov`.

### 2.2 Domain-restricted search

For authoritative sites with no API — Mayo Clinic, Cleveland Clinic, NHS, WHO guidance
pages, wire services.

**One search call per category, not one per site.** The whole authority pack for the
category is passed as a domain filter and the search provider ranks across it:

```python
search(query=resolved_claim,
       include_domains=source_packs[category][jurisdiction]["authority"],
       depth="advanced")
```

A cardiology claim and a dermatology claim search the same pack. Whichever site covers it
surfaces. No per-claim source selection is required, and no routing logic is needed beyond
the category already assigned at triage.

### 2.3 Channel to tier mapping

| Channel | Mechanism | Tiers it can produce |
|---|---|---|
| Prior checks | `factcheck` adapter | `prior_check` |
| Literature | `pubmed` / `crossref` adapter | `peer_reviewed` |
| Authority | Domain-restricted search | `authority`, `definitional` |
| Open web | Unrestricted search | `reporting`, `low` |

Open web runs only for `breaking_news` and `general` categories, and as a fallback when
the other channels return nothing. This is the only channel with meaningful per-query cost.

### 2.4 Adapter contract

Every adapter and search wrapper returns the same shape, so the grading stage is agnostic
to where a document came from:

```python
class RawDocument(BaseModel):
    url: HttpUrl
    title: str
    text: str
    published_date: date | None
    source_domain: str
    channel: Literal["prior_check", "literature", "authority", "open_web"]
    structured_type: str | None   # e.g. "Systematic Review" from PubMed
    retracted: bool = False
```

Adding a new adapter means implementing one function returning `list[RawDocument]`.
Nothing downstream changes.

---

## 3. Request sequence

```
extension                backend                     external
    │
    │ POST /v1/check ──────▶│
    │                       │ cache lookup (resolved claim unknown yet → skip)
    │                       │
    │                       │ triage ──────────────▶ small model
    │                       │◀─────────────────────
    │                       │
    │                       │ cache lookup on resolved_claim
    │                       │  hit → return, done
    │                       │
    │                       ├─ adapter calls ──────▶ PubMed / FactCheck / …
    │                       ├─ domain search ──────▶ search provider
    │                       └─ open web (cond.) ───▶ search provider
    │                       │◀───────────────────── (concurrent, gather)
    │                       │
    │◀── stream: evidence ──│ grading (local NLI, batched)
    │                       │
    │                       │ evidence_state (pure)
    │                       │
    │                       │ short-circuit if empty → abstain
    │                       │
    │                       │ decide ──────────────▶ large model
    │                       │◀─────────────────────
    │                       │
    │                       │ grounding checks (local NLI)
    │                       │
    │◀── stream: verdict ───│ cache write, log write
```

Two cache lookups: the first is a cheap miss (raw text is not the key), the second after
triage uses `resolved_claim` and is where hits actually occur. The triage call is the
price of a correct cache key, and it is the cheapest call in the pipeline.

**Streaming matters more than total latency.** Evidence cards render as grading completes;
the verdict arrives last. Perceived wait drops sharply even though wall-clock time does not.

---

## 4. Error handling

Every failure has a defined user-visible outcome. No spinner may hang.

| Failure | Detection | Behaviour | User sees |
|---|---|---|---|
| Selection too long | `len > 5000` | Reject at schema | "Select a shorter passage" |
| Not a checkable claim | `triage.checkable == false` | Return 200 with `not_checkable` | "This looks like an opinion, not a factual claim" |
| Ambiguous resolution | `resolution_confident == false` | Proceed, flag | Resolved claim shown with "is this what you meant?" |
| Adapter timeout | 8 s per channel | Drop that channel, continue | Fewer sources, noted in evidence summary |
| All channels empty | zero `RawDocument` | Abstain in code, **no model call** | "No sources found in the vetted list for this claim" |
| Model refusal | Provider safety response | Catch, return structured | "Unable to process this text" |
| Malformed model JSON | Parse failure | One repair retry, then abstain | Abstention with reason |
| Grounding drops all statements | Empty after filter | Abstain | Verdict without explanation, flagged |
| Provider quota exhausted | 429 from provider | Fail fast, log | "Service temporarily unavailable" |
| Backend unreachable | Extension fetch error | Retry once, then stop | "Cannot reach FactShield" |
| Non-English selection | Language detect at triage | Decline cleanly | "Only English is supported" |

The empty-retrieval short circuit is the important one. **The model is never asked to
evaluate a claim with no evidence attached.** It cannot hallucinate on a call that is not
made. This replaces the v1.1 design's fallback to parametric knowledge, which was the
single most dangerous behaviour in that document.

---

## 5. Caching

| Property | Value |
|---|---|
| Store | Redis |
| Key | `fs:v{pipeline_version}:{sha256(resolved_claim_normalised)}` |
| Normalisation | Lowercase, strip punctuation, collapse whitespace |
| TTL | 48 h |
| Eviction | `volatile-lru` |
| Dev cache | Separate keyspace, no TTL |

The `pipeline_version` in the key matters: changing a prompt or a source pack invalidates
cached results automatically rather than silently serving output from an older system.

The permanent dev keyspace means repeated benchmark runs during development cost nothing
after the first pass.

---

## 6. Configuration

Everything that could change without code changing lives in versioned YAML.

```
config/
├── models.yaml         provider + model + temperature per stage
├── source_packs.yaml   per category, per jurisdiction
├── tiers.yaml          tier hierarchy per category
└── prompts/
    ├── triage.v3.txt
    └── decide.v5.txt
```

Prompts are files, versioned, and the version is logged with every request. A prompt
changed mid-benchmark otherwise means half the results came from a different system with
no record of it.

`temperature: 0` everywhere. Providers are still not fully deterministic at zero, so the
benchmark runs each claim multiple times and reports variance.

Source packs are published in the thesis appendix. A source list that cannot be inspected
is not a controlled variable.

---

## 7. Privacy and security

### 7.1 Data handling

Selected text may contain anything the user highlights, including personal data.

| Mode | Raw selection | Resolved claim | Page URL |
|---|---|---|---|
| Normal | Hash only | Hash only | Domain only |
| Study mode | Stored, with consent | Stored | Stored |

Study mode is off by default and enabled per install. A privacy policy must exist before
the extension is distributed to anyone, including informally.

### 7.2 Gatekeeping

- `max_length=5000` on the request schema, enforced at validation
- Rate limit keyed on `participant_id` or token, **not IP** — university networks NAT
  many users behind one address, so IP limiting would have participants blocking each other
- All provider keys server-side in environment config, never in extension source
- Hard spending caps set in every provider dashboard, plus alerts

### 7.3 Content handling

- Quotes extracted for display are single sentences with attribution and a link
- PubMed abstracts only, never full texts
- `robots.txt` respected by any first-party fetching
- The popup must not become a way to read an article without visiting it

---

## 8. Deployment

| Environment | Backend | Redis | Notes |
|---|---|---|---|
| Local dev | `uvicorn --reload` | Docker container | Permanent dev cache |
| Study | Managed host (Render/Railway) | Managed Redis | Paid tier — free tiers sleep, causing 30 s+ cold starts |
| Extension | Unlisted or unpacked | — | Store listing not required for a study |

Every request logs `extension_version`. If the backend is updated while participants hold
an older extension, the data mixes two systems; the version field makes that detectable.

---

## 9. Testing

| Level | What | How |
|---|---|---|
| Unit | `evidence_state` functions | Pure functions, table-driven tests, no mocks needed |
| Unit | Adapters | Recorded fixtures, no live calls |
| Contract | Pydantic schemas | Validation on known-good and known-bad payloads |
| Integration | Full pipeline | Snapshotted corpus, no network |
| Regression | 20-claim set, 4 per category | Run on every prompt or config change |
| Manual | Extension on 10 diverse sites | Shadow DOM isolation, selection across elements |

The 20-claim regression set is built as soon as the skeleton runs, not at evaluation time.
Without it, a prompt change that degrades hedged-claim handling goes unnoticed for weeks.

`evidence_state` being model-free is what makes the most important logic exhaustively
testable. Write those tests first.

---

## 10. Logging

One structured record per request. This schema **is** the evaluation dataset.

```python
class RequestLog(BaseModel):
    request_id: str
    timestamp: datetime
    participant_id: str | None
    ui_condition: str | None
    extension_version: str
    pipeline_version: str
    prompt_versions: dict[str, str]

    raw_selection_hash: str
    raw_selection: str | None        # study mode only
    resolved_claim: str
    category: str
    resolution_confident: bool

    evidence: list[EvidenceItem]
    evidence_state: EvidenceState

    verdict: Verdict
    abstained: bool
    abstention_reason: str | None

    grounding_statements_total: int
    grounding_statements_dropped: int

    timings_ms: dict[str, float]
    errors: list[str]
```

Client events logged separately: request issued, popup opened, sources expanded, source
clicked, time to close.

Benchmark ablations become filters over this table. The circularity analysis is already
present because `published_date` is recorded per document. This avoids the common failure
where a separate evaluation script drifts from production behaviour.

---

## 11. Build sequence

| Phase | Deliverable | Done when |
|---|---|---|
| 1 | Skeleton — context menu → capture → one search → one model → popup. Participant ID, condition flag, full log schema | A claim goes in, a verdict comes out, and the log record is complete |
| 2 | NLI grading with quotes and dates; index adapters; tiered packs; abstention rules; both popup layouts | Regression set passes; abstention fires correctly on empty retrieval |
| 3 | Grounding — evidence IDs, quote verification, entailment filter, split verdict/explanation calls, decomposition | Faithfulness rate measurable and logged |
| 4 | **Freeze** | Date fixed in advance; no pipeline changes after it |

Phase 1 exists to surface integration problems — MV3 worker lifecycle, CORS, shadow DOM,
streaming — while they are cheap. A working ugly path in week three beats a polished
retrieval layer with no client in month three.

---

## 12. Open items

Resolve before Phase 2.

- **Which field marks this thesis** — CS, HCI, or communications. Determines which
  literature goes deep and which methods examiners expect.
- **Study mode consent flow** — needed before any distribution, even informal.
- **Provider terms on training data** — free tiers often permit it. Acceptable for
  benchmarking, not for participant sessions.
- **Prior work declaration** — the Android repo is reused code and most departments
  require it to be declared.
- **Success under null results** — write the one-page statement of what is claimed if the
  system performs at, above, or below baseline. If the below-baseline case leaves nothing,
  the framing needs changing now rather than in month ten.
