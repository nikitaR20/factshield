# FactShield

Evidence-grounded claim verification in the browser. Highlight a sentence,
right-click, get a verdict with its sources — or an honest "cannot determine".

**New here? Read `docs/WALKTHROUGH.md` first.** Plain-language explanation of
every file.

## Quick start

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload      # http://localhost:8000
pytest tests/ -q                   # 20 tests, no network needed
```

Extension: `chrome://extensions` → Developer mode → Load unpacked → `extension/`
(add three PNGs to `extension/icons/` first).

## Layout

```
backend/     FastAPI pipeline — see backend/README.md
extension/   Chrome MV3 client, two study layouts
eval/        snapshot corpus, four ablation conditions, analysis
docs/        ARCHITECTURE (what/why) · SYSTEM_DESIGN (how) · WALKTHROUGH (plain language)
```

## Three design rules

1. **The model never supplies facts.** It arranges retrieved evidence, and a
   grounding check deletes any sentence its cited source doesn't support.
2. **Abstention is a real answer.** Four rules stop a request before any model
   call when the evidence is insufficient.
3. **Every number is computed.** The score comes from tier weights, role
   weights, recency and independence — with the components logged, so "why 72"
   has an answer.

## Cost

One potentially-paid call per request, skipped whenever a published fact-check
resolves the claim. Everything else is free-tier or local. Tavily offers
students four months of 4,000 credits/month — claim it when you start the
benchmark, not while building.

## Benchmark

```bash
python -m eval.snapshot --claims eval/dataset/claims.jsonl --out eval/dataset/corpus.jsonl
python -m eval.run      --corpus eval/dataset/corpus.jsonl --runs 3
python -m eval.analyze  --results eval/results/runs.jsonl
```

Snapshot once, then run every condition offline. Ablations must differ only in
the pipeline, never in what the web returned that day.
