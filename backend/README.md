# FactShield — Backend

Evidence-grounded claim verification. The model arranges retrieved evidence;
it never supplies facts of its own.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # all keys optional — see "Degraded mode"
uvicorn app.main:app --reload
pytest tests/ -q              # 20 tests, no network, no mocks
```

`GET /health` reports which providers are configured and whether the real NLI
model or the lexical stub is live.

## Cost

Exactly **one potentially-paid call per request**, and it is skipped whenever a
published fact-check already resolves the claim.

| Stage | Where it runs | Cost |
|---|---|---|
| Triage | Groq free tier | free |
| Fact-check channel | Google Fact Check Tools | free |
| Literature channel | PubMed / CrossRef | free |
| Authority + open web | Tavily | metered — the only real cost |
| Grading (10–15×/request) | local NLI | free |
| Verdict | Gemini free tier | free at project volume |
| Grounding | same local NLI | free |

Grading is the stage that would dominate cost if it were API calls, which is
why it runs locally and batched. Open web is disabled for medical and science,
where the free adapters already cover the expected tiers.

Set `FACTSHIELD_DEV_CACHE=true` during development: cached results never
expire, so repeated benchmark runs cost nothing after the first pass.

## Degraded mode

Every external dependency fails soft:

- No `GROQ_API_KEY` → triage treats the selection as a single claim, flags
  `resolution_confident=false`
- No search keys → channels return empty, and the request **abstains** rather
  than falling back to model memory
- No `transformers` → deterministic lexical NLI stub, so tests and the
  evaluation harness still run offline
- No `REDIS_URL` → in-process cache

The pipeline never guesses to fill a gap. That is the point.

## Two decisions worth understanding before reading the code

**The score is computed, not generated.** `scoring.compute_score` derives it
from tier weights, role weights, recency decay and independence damping. Every
component is recorded in `score_basis`, so "why 72 and not 65" has an answer. A
model's self-reported confidence would have none.

It returns `None` below a minimum evidence weight. "Unresolved" must never
collapse into a misleading 50 — those are different states.

**The verdict is decided before the model is called.** `scoring.derive_verdict`
sets both axes from the evidence table. The model's only job is writing the
explanation, and `stages.ground` then drops any sentence its cited evidence
does not entail. Telling a model to use only supplied evidence does not enforce
it; this does, and the drop count is logged as a reportable faithfulness metric.

## Layout

```
app/
  schemas.py              contracts every stage speaks
  config.py               YAML loading, versioned prompts
  main.py                 orchestration, rate limiting, logging
  providers/llm.py        provider-agnostic, fallback chain, JSON repair
  providers/nli.py        local entailment + offline stub
  retrieval/__init__.py   adapters (free) + domain search (metered)
  pipeline/
    evidence_state.py     pure arithmetic — no model calls
    scoring.py            pure arithmetic — no model calls
    grading.py            stance, quote, tier, role
    stages.py             triage, fast path, explain, ground
config/
  source_packs.yaml       per category; published in the thesis appendix
  tiers.yaml              weights feeding the score
  models.yaml             provider per stage
  prompts/*.vN.txt        versioned, logged per request
```

`evidence_state.py` and `scoring.py` take no model calls, so they are testable
with no mocks and no network — and they produce the explanations that matter
most ("too early to tell", "amplified", "outdated"). Write tests for those first.

## Not yet built

Extension client, evaluation harness (`eval/conditions/`), and the study
instrumentation beyond the headers already accepted. The parametric baseline
belongs in `eval/`, never in `app/pipeline/` — running a no-retrieval pass and
feeding its answer into the verdict step would anchor the output and destroy
the claim that verdicts are evidence-grounded.
