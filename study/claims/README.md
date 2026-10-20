# Claim sets

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
