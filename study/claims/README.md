# Claim sets

## CSV files (fact-check tables)

Put `.csv` or `.tsv` files in this folder, or point `paths.claims` in
`study/config.yaml` (or the `STUDY_CLAIMS_DIR` environment variable) at any
folder on your computer. Expected columns:

    question  comment  claim  rate  what's true  what's false  what's unknown  origin  summary

- `claim` is the text the models see. Nothing else is shown to a model.
- `rate` becomes the gold label through `claims.csv.rating_to_label` in
  config. Ratings that are not clearly supported / refuted / not enough
  evidence are excluded and counted, never guessed.
- The other columns are kept under `factcheck` for auditing labels and for
  human coders.
- Ids are `<file name>-<row number>` unless there is an `id` column. Do not
  reorder or delete rows after building packs: the runner will refuse any
  claim whose pack was built for different text.

Check before building anything:

    python -m study.claimsets check

It prints every `rate` value found and what it became, claims kept per set
and label, and every excluded row's reason.

## JSON-lines files

Frozen inputs to the study. One JSON object per line, in `*.jsonl` files here.
These ARE version-controlled (`.gitignore` un-ignores `study/claims/*.jsonl`);
any change after the pilot is a protocol deviation.

Required fields: `id`, `claim`, `set`, `label`, `domain`, `source_id`.
`label` is one of `supported`, `refuted`, `not_enough_evidence`.

Optional `search_concepts`: synonym groups ANDed into the PubMed query, e.g.
`[["vitamin D", "cholecalciferol"], ["respiratory infection"]]`. The product's
triage writes these with an LLM; the study must not call triage, so write them
by hand (and publish them) for medical and science claims, or leave the field
out and PubMed falls back to keyword search. Decide one policy for all claims.

```json
{"id":"b015","claim":"...","set":"B","label":"refuted","domain":"economics",
 "source_url":"...","source_published":"2026-09-24","source_id":"b_src_05",
 "variant":"altered","alteration_type":"number","surface_detectable":true,
 "factcheck_rating":null,"retrievable":{"openai":true,"gemini":true,"groq":true},
 "notes":"..."}
```

The gold label must be correct **relative to the frozen evidence pack**, not
only relative to the world. A claim that is true but whose pack does not
contain the evidence has gold label `not_enough_evidence` for the in-context
conditions. Audit labels against packs after `python -m study.packs build`.

No claim sets are committed yet. Test fixtures live in `study/tests/fixtures/`
and are invented.
