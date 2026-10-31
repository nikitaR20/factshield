# Fourth panel model (Claude Haiku 5.5) — analysis plan, fixed 2026-10-31 BEFORE any Claude call

Status: EXPLORATORY. Decided after the FactShield v2 results (2026-10-29) were seen.
It does not replace or change the frozen design (fsv2-0.2, prompt v2) or its results.

## What is run
- One round-1 answer per claim from `claude-haiku-5-5` on the same 100 frozen packs, with the same
  system prompt, verdict prompt v2 and parser as the panel (`python -m fsv2.fourth_model run`).
- Settings follow the rule used for the other three (lowest reasoning the model allows):
  thinking disabled, effort low, max 4000 output tokens.
- No temperature is sent: Haiku 5.5 rejects any non-default value (HTTP 400). Its answers are
  therefore not deterministic, like Gemini's in this study. Not repeated (one answer per claim).
- Pack hashes are checked against the saved FactShield rows; a mismatch is an error row.
- Verdicts are not printed during the run.

## Conditions (all simulated from round-1 answers: no extra samples, no citation check)
- U3: openai + gemini + groq all agree (baseline; recomputed so it is like-for-like).
- U4: all four agree.
- M4: at least 3 of 4 agree.
- U3': openai + gemini + claude all agree (Claude replaces gpt-oss).
- Each single model. FactShield v2 as run is shown for reference only.

## Primary comparison (one test)
Confidently wrong (CW = supported/refuted that is not gold), U4 vs U3, all 100 claims,
two-sided exact McNemar.

Power, stated before the run: U3 has about 6 CW. U4 can only remove errors, never add them.
6 -> 0 gives p = 0.031; 6 -> 3 gives p = 0.25. So only a near-complete removal of the
shared errors can reach p < 0.05 on 100 claims. Any smaller change is reported as a
count, not as a finding.

## Secondary (Holm-corrected as one family of five)
CW: M4 vs U3; U3' vs U3; single claude vs single openai. Correct: U4 vs U3; M4 vs U3.
Also reported: correct verdicts U3 gave that U4 no longer gives (the coverage cost).

## Reported for every subset: all 100, unseen 77 (test + new), and without the 11 audited-label claims
correct, answered (coverage), CW, accuracy on answered with Wilson 95% interval, abstention
accuracy, per-class precision/recall, macro-F1.

## Descriptive only (no tests)
- Effective number of independent models (Kish, as in fsv2/overlap.py) for the 3 as run,
  openai+gemini+claude, and all four; bootstrap 95% interval.
- All-wrong counts against independence.
- Agreement split among the four (4 / 3-1 / 2-2 / 2-1-1 / 1-1-1-1) against gold.
- All 15 model combinations (unanimous; majority for 3+).
- What Claude answered on FactShield's 6 confident errors and the 3-model shared errors.
- Error-coding worksheet (claims with >= 2 of 4 wrong) for manual coding by Nikita.

## Interpretation rule, fixed now
"Adding Claude reduced shared errors" is claimed only if the primary test is significant.
Otherwise the result is described as counts, with the coverage cost beside it.
Any result that holds only when the audited-label claims are included is flagged.

## Known limits
- Same 100 claims as the main study; post hoc model choice.
- Claude helped audit the gold labels and write the prompts; disclosed, and handled only
  partly by the exclusion subset.
- Claude and Gemini answers are not deterministic; one answer per claim.
