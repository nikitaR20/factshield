# FactShield v2 — thesis core

Stages 3–6 of the v2 design, run on the study's frozen packs. Full design and
the evidence behind each choice: `claude/factshield-v2-design.md` in the
Factshield project.

**Status (2026-10-25):** built and tested offline only. No real model and no
real MiniCheck has run yet, so nothing is known about its accuracy.

## What it does per claim
1. Prepares the pack passages: copies shown once, same-organisation and
   news-agency notes, source type, Lin et al. (2023) news-quality rating.
2. Asks the three study models (names/settings from `study/config.yaml`) for a
   verdict in JSON: per-passage relevance and stance, verdict, key statements.
3. Commits only if all three agree (extra samples at temperature 0.7 when they
   don't), and only if MiniCheck confirms a cited, on-subject passage
   supports a key statement. Otherwise: not enough evidence, with the reason.
4. Optional (`--explain`): one explanation, every factual sentence checked.

## Laptop steps (in your local repo, branch csv-claims)
1. Copy the `fsv2/` folder into the repo root. Your packs are already in
   `study/packs/`.
2. Install the checker (CPU is fine):
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
   python -c "import nltk; nltk.download('punkt_tab')"
   ```
3. Measure it: `python -m fsv2.speedtest`. The first run downloads the
   weights from Hugging Face (Flan-T5-Large size; I could not download them
   here to confirm the exact size).
4. Make sure `study/config.yaml` has the three real model names (not CHANGE_ME)
   and your keys are in the same `.env` the study already uses.
5. Plumbing check, no keys used, verdicts meaningless:
   `python -m fsv2.run_packs --split dev --stub-models --stub-checker --out fs2_stub.jsonl`
6. Dev run (17 claims): `python -m fsv2.run_packs --split dev --explain`
   then `python -m study.score --responses study/results/factshield_v2.jsonl --tag fs2`
7. Test run, **once, after the design is frozen**:
   `python -m fsv2.run_packs --split test`

Test verdicts are never printed. Every row records the checker and whether
stub models were used.

## Rows written per claim (one set of model calls)
| condition | model | meaning |
|---|---|---|
| fs2 | factshield-v2 | the design |
| fs2_no_check | factshield-v2 | ablation: unanimity without the MiniCheck step |
| fs2_majority_r1 | factshield-v2 | ablation: 2 of 3 in round 1, with the check |
| fs2_single | fs2:<model> | ablation: one model with the v2 prompt, no rule |

## Known limits
- Calls per claim: 3 when the panel agrees at once, 9 when it doesn't, +1 with `--explain`.
- Each call uses the study's `max_output_tokens`. A reasoning model can spend
  that budget on thinking and return no text; that shows up as an error row
  ("empty response text"), never as a guessed verdict. If it happens, raise the
  limit for the panel and record the change.
- Models the study runs with `temperature: null` keep no temperature for the
  extra samples too (their default sampling is already random).
- Refuted verdicts: MiniCheck confirms the counter-fact is in the passage;
  that it contradicts the claim is the panel's judgement.
- Domain ratings cover 230 of 306 pack passages; the rest show "unrated".
  The ratings repo has no licence file; its README invites reuse with citation.
- Same-organisation labels group by the domain's name label (bbc.com and
  bbc.co.uk → bbc); unrelated sites sharing a label would be grouped wrongly.

## Tests
`python -m pytest fsv2/tests -q` (19 tests, offline).
