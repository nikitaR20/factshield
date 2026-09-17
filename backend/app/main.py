"""FactShield API.

Request flow, with the cost decisions made explicit:

  triage (free tier)
    -> cache lookup on the RESOLVED claim
    -> retrieval, all channels concurrent (adapters free; search metered)
    -> grading, local NLI, batched (free)
    -> evidence state + score, pure arithmetic (free)
    -> fast path: published fact-check resolves it -> NO verdict model call
    -> otherwise one verdict call, top 6 documents only
    -> grounding verification, local NLI (free)

The empty-retrieval short circuit is the important safety property: the model
is never asked to evaluate a claim with no evidence attached. It cannot
hallucinate on a call that is not made.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import cache
from .config import PIPELINE_VERSION, env, flag, models
from .pipeline import evidence_state as ev
from .pipeline import grading, scoring, stages
from .providers import discover, nli
from .retrieval import retrieve
from .schemas import (
    CaptureContext,
    CheckResponse,
    EvidenceItem,
    RequestLog,
    SubClaimVerdict,
)

app = FastAPI(title="FactShield", version=PIPELINE_VERSION)


@app.on_event("startup")
async def _warm() -> None:
    """Load the NLI weights before the first request, not during it."""
    await asyncio.to_thread(nli.warmup)
    print(f"[factshield] NLI backend ready: {nli.backend_name()}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*"],
    allow_origin_regex=r"chrome-extension://.*",
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Rate limit is keyed on participant, NOT IP. University networks put many
# users behind one address; IP limiting would have study participants blocking
# each other, and you would find out mid-session.
_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_LIMIT = int(env("RATE_LIMIT_PER_MIN", "20"))


def _rate_limit(participant: str) -> None:
    now = time.time()
    bucket = _BUCKETS[participant]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= _LIMIT:
        raise HTTPException(429, "Rate limit exceeded")
    bucket.append(now)


@app.get("/health")
async def health() -> dict:
    """Diagnostic. Says what is configured, what is missing, and what each
    missing piece actually blocks — so a half-configured install is obvious
    rather than quietly producing 'cannot determine' for everything."""
    checks = {
        "GROQ_API_KEY": ("triage", "claim splitting and pronoun resolution"),
        "GEMINI_API_KEY": ("explain", "written explanations"),
        "FACTCHECK_API_KEY": ("prior_check channel", "the fast path for viral claims"),
        "TAVILY_API_KEY": ("authority + open web channels", "most source retrieval"),
    }
    configured = {k: bool(env(k)) for k in checks}
    blocked = [
        f"{k} missing -> {stage} disabled ({effect})"
        for k, (stage, effect) in checks.items()
        if not configured[k]
    ]

    nli_backend = nli.backend_name()
    if nli_backend == "lexical-stub":
        blocked.append(
            "transformers/torch not installed -> stance classification is "
            "keyword matching, not a real NLI model "
            "(pip install transformers torch sentencepiece)"
        )

    return {
        "status": "ok",
        "ready": not blocked,
        "pipeline_version": PIPELINE_VERSION,
        "nli_backend": nli_backend,
        "strict_mode": flag("FACTSHIELD_STRICT"),
        "providers": configured,
        "blocked": blocked,
    }


@app.get("/v1/models")
async def list_models() -> dict:
    """What can this key actually call, and does the configured model work?

    Model names go stale and a stale name 404s, which looks exactly like a
    broken key. Discovery removes the guesswork.
    """
    cfg = models()
    return {
        "configured": {
            "triage": f"{cfg.get('triage', {}).get('provider')}/{cfg.get('triage', {}).get('model')}",
            "decide": f"{cfg.get('decide', {}).get('provider')}/{cfg.get('decide', {}).get('model')}",
        },
        "probe": await discover.probe(),
        "nli_benchmark": await asyncio.to_thread(discover.nli_benchmark),
        "groq": await discover.groq_models(),
        "gemini": await discover.gemini_models(),
    }


@app.post("/v1/event")
async def event(
    payload: dict,
    x_participant_id: str | None = Header(default=None),
    x_ui_condition: str | None = Header(default=None),
) -> dict:
    """Client-side behavioural events the backend log cannot see.

    Source-opening rate is the primary behavioural measure separating the two
    UI conditions — self-reported trust and actual reliance diverge constantly,
    so what people *do* matters more than what they say.
    """
    cache.write_log(
        {
            "kind": "client_event",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "participant_id": x_participant_id,
            "ui_condition": x_ui_condition,
            **payload,
        }
    )
    return {"ok": True}


@app.post("/v1/check", response_model=CheckResponse)
async def check(
    ctx: CaptureContext,
    x_participant_id: str | None = Header(default=None),
    x_ui_condition: str | None = Header(default=None),
    x_extension_version: str | None = Header(default=None),
) -> CheckResponse:
    participant = x_participant_id or "anonymous"
    _rate_limit(participant)

    request_id = f"fc_{uuid.uuid4().hex[:12]}"
    timings: dict[str, float] = {}
    errors: list[str] = []
    t0 = time.perf_counter()

    def mark(stage: str, since: float) -> float:
        now = time.perf_counter()
        timings[stage] = round((now - since) * 1000, 1)
        return now

    # ---------------------------------------------------------- triage
    t = time.perf_counter()
    tri, triage_version = await stages.triage(ctx)
    t = mark("triage", t)

    if not tri.checkable:
        return CheckResponse(
            request_id=request_id,
            checkable=False,
            not_checkable_reason=tri.reason or "This does not appear to be a verifiable factual claim.",
            category=tri.category,
            timings_ms=timings,
        )

    # ------------------------------------------------- cache (resolved claim)
    cache_key_text = " || ".join(c.text for c in tri.sub_claims)
    cached = await cache.get(cache_key_text)
    if cached:
        cached["request_id"] = request_id
        cached.setdefault("timings_ms", {})["cache"] = round(
            (time.perf_counter() - t0) * 1000, 1)
        return CheckResponse(**cached)

    # -------------------------------------------------------- retrieval
    retrieval_cfg = models().get("retrieval", {})
    results = await asyncio.gather(
        *[retrieve(c.text, tri.category, tri.jurisdiction)
          for c in tri.sub_claims],
        return_exceptions=True,
    )
    t = mark("retrieval", t)

    # ---------------------------------------------- grading (local, batched)
    all_items: list[EvidenceItem] = []
    per_claim: dict[int, list[EvidenceItem]] = {}
    next_id = 1
    for claim, docs in zip(tri.sub_claims, results):
        if isinstance(docs, Exception):
            errors.append(f"retrieval failed for sub-claim {claim.id}")
            per_claim[claim.id] = []
            continue
        items = await grading.grade(
            docs, claim.text, claim.id, tri.category, tri.jurisdiction, start_id=next_id
        )
        next_id += len(items)
        per_claim[claim.id] = items
        all_items.extend(items)
    t = mark("grading", t)

    # ------------------------------------------- state, score, verdict
    max_items = models().get("decide", {}).get("max_evidence_items", 6)
    verdicts: list[SubClaimVerdict] = []
    fast_path_used = False
    grounding_total = grounding_dropped = 0
    state_overall = None

    for claim in tri.sub_claims:
        items = per_claim.get(claim.id, [])
        state = ev.compute(items, tri.category,
                           tri.jurisdiction, ctx.page_published)
        state_overall = state_overall or state

        # A fact-checker who reviewed THIS claim outranks every abstention
        # rule: they are a human expert who already resolved it. Checking the
        # rules first meant a claim backed by a matching Snopes verdict could
        # still be abstained on.
        fast = stages.fast_path_verdict(claim, items)
        if fast is not None:
            fast_path_used = True
            verdicts.append(fast)
            continue

        # Otherwise short circuit BEFORE any model call. No evidence, no
        # verdict request — the model cannot hallucinate on a call not made.
        reason = ev.should_abstain(state, items)
        if reason:
            verdicts.append(
                SubClaimVerdict(
                    sub_claim_id=claim.id, sub_claim_text=claim.text,
                    claim_verdict="unresolved",
                    # Derived from WHY it abstained, not hardcoded. Reporting
                    # "amplified" for a claim with no common origin was simply
                    # inaccurate.
                    evidence_standing=ev.standing_for_abstention(state, items),
                    support_score=None, abstained=True, abstention_reason=reason,
                    explanation=stages.describe_evidence(items, state),
                )
            )
            continue

        score, basis = scoring.compute_score(
            items, tri.category, state.n_independent_domains)
        verdict, standing = scoring.derive_verdict(score, state, items)

        pattern, explanation, cited, _ = await stages.explain(
            claim, items, state, verdict, standing, max_items=max_items
        )
        explanation, cited, total, dropped = stages.ground(explanation, items)
        grounding_total += total
        grounding_dropped += dropped

        # Grounding can legitimately delete everything the model wrote. Show a
        # computed description rather than an empty box — it is assembled from
        # counts, so it cannot fabricate.
        if not explanation.strip():
            explanation = stages.describe_evidence(items, state)
            cited = [i.id for i in items[:3]]

        verdicts.append(
            SubClaimVerdict(
                sub_claim_id=claim.id, sub_claim_text=claim.text,
                claim_verdict=verdict, evidence_standing=standing,
                support_score=score, score_basis=basis,
                explanation_pattern=pattern or None,
                explanation=explanation, cited_evidence_ids=cited,
            )
        )
    t = mark("verdict", t)
    timings["total"] = round((time.perf_counter() - t0) * 1000, 1)

    response = CheckResponse(
        request_id=request_id,
        checkable=True,
        resolved_claims=tri.sub_claims,
        resolution_confident=tri.resolution_confident,
        category=tri.category,
        verdicts=verdicts,
        evidence=all_items,
        evidence_state=state_overall,
        fast_path=fast_path_used,
        timings_ms=timings,
        errors=errors,
    )

    await cache.put(cache_key_text, response.model_dump(mode="json"))

    cache.write_log(
        RequestLog(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            participant_id=x_participant_id,
            ui_condition=x_ui_condition,
            extension_version=x_extension_version,
            pipeline_version=PIPELINE_VERSION,
            prompt_versions={"triage": triage_version},
            raw_selection_hash=hashlib.sha256(
                ctx.selection.encode()).hexdigest()[:32],
            raw_selection=ctx.selection if flag(
                "FACTSHIELD_STUDY_MODE") else None,
            resolved_claims=[c.text for c in tri.sub_claims],
            category=tri.category,
            resolution_confident=tri.resolution_confident,
            evidence=all_items,
            evidence_state=state_overall,
            verdicts=verdicts,
            fast_path=fast_path_used,
            grounding_total=grounding_total,
            grounding_dropped=grounding_dropped,
            timings_ms=timings,
            errors=errors,
        )
    )

    return response
