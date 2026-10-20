# study/ — the thesis experiment

Measures how three commercial models (OpenAI, Gemini, an open-weight model on
Groq) behave when asked to fact-check the same claim, against the same frozen
evidence, three different ways (`free`, `format`, `citation`).

It is **not** FactShield. It never imports FactShield's verdict logic, and
`study/tests/test_import_boundary.py` proves that on every test run by walking
the import graph from every file in `study/` into `backend/app`. Allowed:
`app.retrieval`, `app.providers.nli`, `app.config`, `app.schemas` (plus the
empty package `__init__` files they pull in). Forbidden: `app.pipeline.*`
verdict modules and `app.providers.llm`. `eval/` is the product's own benchmark
and stays separate.

## Order of work

```bash
pip install -r backend/requirements.txt -r study/requirements.txt
# 1. fill model ids in study/config.yaml (they are CHANGE_ME on purpose)
# 2. put claim sets in study/claims/*.jsonl (see study/claims/README.md)
python -m study.packs build              # retrieve once, freeze, hash
python -m study.packs verify
#    audit gold labels against the frozen packs before going further
python -m study.probe closed-book        # what each model knows unaided
python -m study.probe leak-check         # is anything retrieving server-side?
python -m study.run --repeat 3           # the study; resumable
python -m study.score
python -m study.sample_citations --nli   # worksheet for the coder + closed key
python -m study.analyse --synthetic --reps 100   # check the analysis first
python -m study.analyse --i-have-finished-the-pilot   # real data, after the pilot
python -m pytest study/tests -q
```

## Files

| File | What it does |
|---|---|
| `config.yaml` | Every tunable: model ids, seed, prompt version, paths, retry, pack size, probe and analysis settings |
| `common.py` | Config, provenance (seed + git commit on every output), fsynced JSONL |
| `models.py` | One `complete()` for all providers; search off and evidenced; served model version; backoff |
| `prompts.py` + `prompts/*.v1.txt` | Condition text as files, condition builder, conservative verdict parser |
| `packs.py` | Frozen evidence packs via `app.retrieval`, manifest of hashes |
| `probe.py` | Closed-book probe (`knew_unaided`) and search-leak check |
| `run.py` | Shuffled, resumable main runner; `--repeat`; labelled `--live-search` |
| `score.py` | Per-call scoring and summaries |
| `sample_citations.py` | Stratified, blinded citation worksheet |
| `analyse.py` | Mixed model, GEE, paired McNemar, format-vs-free CI, noise comparison |

## Decisions a reader of the thesis should know about

- **Parsing never guesses.** A `Verdict:` line or the first line decides; a
  hedged or negated label ("not supported", "partly supported") is
  `unparseable`, counted in every denominator.
- **Packs are the ground for gold labels.** `not_enough_evidence` means "not
  in this pack", so labels are audited against packs after they are built.
- **Retrieval category.** The product chooses a retrieval category with an LLM
  triage call. The study cannot call triage, so it maps each claim's `domain`
  to a category with a fixed table in `config.yaml`.
- **Search concepts and truncation.** Medical/science claims may carry
  hand-written `search_concepts` for PubMed. Each passage records PubMed's
  `key_findings` and whether they survived truncation to
  `max_chars_per_passage`; packs where they did not get a warning.
- **Mixed-model fit.** statsmodels' binomial mixed model is Bayesian. Its
  variational fit under-covered badly on synthetic data (69–76% for nominal
  95%, and a true null called non-zero 25–31% of the time); the Laplace fit
  covered at 96–99% with 1% false positives. Laplace is the default.
- **Live search** is a separate mode, separate output file, `live_search=true`
  on every row, never pooled with closed-book rows.
- **`STUDY_CONFIG`** points the scripts at another config for dry runs; any row
  produced that way records the config file it used.
