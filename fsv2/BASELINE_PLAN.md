# Plain-model baseline — analysis plan, fixed 2026-10-31 BEFORE any baseline call

## Why
The thesis compares FactShield with models reading the same evidence. The "single model"
rows reported so far (fs2_single) were not plain models: they saw FactShield's evidence
layer (copies collapsed, Lin et al. ratings, same-organisation and news-agency notes) and the
v2 verdict prompt (passage-by-passage rubric, "copies count once", the refutation rule).
The planned plain-model study (`study/run.py`, prompts free / format / citation) was built
and tested but deferred on 2026-10-26 and never run. This run fills that gap.

## Status
- The plain conditions themselves were planned before any result (study/README).
- Decided now, AFTER the main results: which models (Claude added, Groq left out), the
  primary condition, and the repeat count. Reported as such.
- Groq is left out: about 600 calls on its free plan (8K tokens/min, 200K/day) would take
  about a week.

## What is run
- `python -m study.run --models openai,claude,gemini --include-flagged --repeat 2`
  -> 100 claims x 3 conditions x 3 models x 2 repeats = 1,800 calls, written to
  study/responses/main.jsonl.
- Same frozen packs (hash-checked by the runner), rendered plainly: title, domain, date,
  URL, text. No copy marks, no ratings, no organisation notes.
- Model settings exactly as in study/config.yaml (openai gpt-6-luna reasoning none, T 0;
  claude-haiku-5-5 thinking disabled, effort low, no temperature; gemini-3.5-flash-lite,
  thinking low, no temperature).
- v2 repeat (run-to-run noise of the v2 single-model answers):
  `python -m fsv2.fourth_model run --model-key <m> --out study/results/fsv2_r1_<m>_rep1.jsonl`
  for openai, gemini and claude (100 calls each, same v2 prompt and packs, prompt hash checked).

## Gold labels and scoring
- The same 100 labels as the main results (CSVs + 11 label_overrides; S32 / R17 / NEE51).
- Verdicts parsed with the study parser (never guesses). An unparseable answer is not
  correct and not a confident error; parse rates are reported per model and condition.
- Run 0 is the main answer; run 1 is used only for the noise measure.

## Primary comparison (3 tests, Holm-corrected together)
Confidently wrong (CW = supported/refuted that is not gold), FactShield v2 as run vs the
plain model in the `format` condition (run 0), for each of openai, claude, gemini; all 100
claims; two-sided exact McNemar. `format` is primary because it asks for reasoning before
the verdict and has a strict Verdict line, so it is the plain prompt closest to v2.
Correct answers are reported beside it with the same test.

## Secondary (exploratory; Holm within each family)
1. Per model: v2 single (FactShield's evidence layer + v2 prompt, no panel) vs plain
   `format` run 0, CW and correct. This changes the prompt AND the evidence layout together;
   it cannot say which of the two matters.
2. Per model: plain `free` and `citation` vs plain `format`.
3. Subsets: unseen 77, and without the 11 audited-label claims.

## Noise (descriptive)
- Plain: share of claims whose verdict differs between run 0 and run 1, per model and
  condition; number of firm (S/R) verdicts that change.
- v2 single: the same, original round-1 answer vs the repeat.

## Interpretation rules, fixed now
- A difference is called a finding only if its (Holm-adjusted) p < 0.05.
- A difference no larger than the measured run-to-run change for that model is described
  as within run-to-run noise.
- No new label changes. The comparison of the plain prompt with v2 is described as
  "prompt and evidence layout together".

## Power, stated now
FactShield had 6 CW. If a plain model makes 12 and all 6 FactShield errors are among them
(0 vs 6 discordant), p = 0.031; if only 3 are shared (3 vs 9), p = 0.15; 11 vs 6 with all
shared gives p = 0.06. Only a gap of about 6 or more confident errors, mostly one way, can
reach significance on 100 claims.
