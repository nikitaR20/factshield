# Overstatement analysis: plan (written 2026-11-01, before any count by evidence feature, type or prompt contrast on NEE claims)

## Status, stated honestly

This is a **re-analysis of answers already collected**. No new model calls.
It is **exploratory**:

- Aggregate confident-error counts per condition are already known.
- FactShield's 6 confident errors have a draft coding.
- Over-commit counts for the four FactShield-prompt singles on the 77 test and new claims were computed on 2026-11-01 as a feasibility check.
- The attribution and echo ideas came from these same claims.

Results can **describe** where overstatement happens in this data. They cannot confirm a cause. Every result is labelled `[desc]` (descriptive) or `[test]` (a test fixed below).

## Questions

1. **Kinds.** Which kinds of overstatement occur?
2. **Evidence conditions.** Under what evidence conditions do they occur?
3. **Output constraints.** Do output constraints reduce them, and at what cost?

## Definitions

- **Gold.** Claim CSVs plus 11 label overrides: S 32 / R 17 / NEE 51, the same gold as every other analysis. "conflicting_evidence" counts as NEE.
- **Overstatement (OS).** A firm verdict (supported or refuted) on a claim whose gold label is NEE. Counted per (claim, condition, run) and split by direction:
  - OS-S: said supported;
  - OS-R: said refuted.
- **Wrong direction (WD).** A firm verdict opposite to a firm gold label. Reported separately; it is not OS.
- **Understatement (US).** NEE on a claim whose gold label is supported or refuted. Reported alongside OS, because answering less often lowers OS by itself.
- **OS rate.** OS divided by the number of NEE-gold claims the condition answered (normally 51). Unparseable answers count as not firm, and their number is reported.
- **Sensitivity.** Repeat the main tables:
  - without the 11 audited-label claims (`claude_influenced_ids.txt`);
  - without claims coded L (label arguable) once the coding is checked.

## Conditions

| Family | Conditions | Runs |
|---|---|---|
| Plain prompts (study, same passages) | format = reason then verdict; free = verdict first; citation = verdict first + cite passages | OpenAI and Claude, runs 0 and 1 |
| FactShield prompt (v2), single models | openai, gemini, groq (round 1); claude (fourth-model run) | 1 |
| FactShield filters | majority 2/3 (round 1); unanimity without citation check; unanimity + citation check (FactShield as run) | 1 |

- **Gemini plain** is incomplete (free quota). It is shown descriptively, on the claims it covers, and is left out of every test and pooled count.
- **All three FactShield files** are used: dev, test and new. Only fsv2-0.2 rows are read.

## Q1. Kinds

- **Unit.** Each distinct claim with at least one OS by any non-Gemini-plain condition or run. The count of conditions and runs that overstated on it is recorded.
- **Codes.** Each claim gets one primary code and an optional secondary. These are the error-analysis scheme codes, plus N, which is added now, before reading:

| Code | Name | Meaning |
|---|---|---|
| A | Attribution | A passage reports that someone said or claimed X; the model treats X as established. |
| E | Echo | Several passages repeat one original report and are read as independent support. |
| S | Scope / detail | The gist matches, but the claim's specific detail (number, date, superlative, "systematically") is not established. |
| M | Misreading | Wrong entity, event, number or time read from a passage. |
| I | Inference | The verdict rests on an inference that goes beyond what the passages state. |
| N | Absence as refutation | No passage confirms the claim, so it is called refuted. |
| L | Label arguable | The pack arguably does settle the claim. |
| O | Other | Described in words. |

- **Self-source.** One extra column per claim, yes or no: is the only support the claim subject's own statement (a company or person reporting about itself)?
- **Roles.**
  - Claude drafts every row with passage quotes and the models' stated reasons.
  - Nikita checks every row against the pack.
  - Until checked, counts are reported as "draft".
  - Claude is the same model family as one tested model, so the draft is not an independent annotation.
- **Report.**
  - Counts of primary codes by claim and by system-instance, split OS-S and OS-R.
  - The share of OS that is L, reported separately from the rest.

## Q2. Evidence conditions

**Features**, computed mechanically from each frozen pack, with code fixed before outcomes are joined:

| Feature | What it measures |
|---|---|
| F1 `n_passages` | Number of passages |
| F2 `n_orgs` | Distinct organisations (fsv2.evidence.organisation) |
| F3 `copy_share` | Share of passages flagged by fsv2.evidence.prepare as same text as, also published by, or same organisation as another passage |
| F4 `attrib_share` | Share of passages containing a reporting cue (case-insensitive word match: said, says, according to, reported, reportedly, alleged, allegedly, claimed, claims, announced, told, stated) |
| F5 `max_overlap` | Highest claim–passage word overlap (study passages' overlap function, as in pack building) |
| F6 `mean_rating` | Mean Lin et al. rating over rated passages (None if none rated) |
| F7 `dated_share` | Share of passages with a publication date |
| F8 `set` | A or B |

**Outcome.** For each NEE claim, `os_share` = the fraction of the 15 non-Gemini-plain system-runs that overstated on it:

- plain: 3 prompts × 2 models × 2 runs = 12;
- FactShield-prompt singles for openai, groq and claude = 3.

Gemini's v2 single is excluded so that every claim has the same denominator; Gemini v2 is reported separately. Filters (majority, unanimity) are not part of the outcome, because they are functions of the singles.

**Test** `[test]`: Spearman correlation between each of F1–F7 and `os_share` on the 51 NEE claims.

- Two-sided permutation p, 10,000 permutations, seed 0.
- Holm correction across the 7 features.
- F8 is compared with a Mann-Whitney-style permutation test on mean `os_share`, outside the Holm family.

**Directional expectations**, written now: higher F4 → more OS; higher F3 → more OS; higher F5 → more OS (closer topical match invites a firm answer). Everything else is descriptive.

**Also reported** `[desc]`: tertile tables of `os_share`, and OS-S vs OS-R, by F3, F4 and F5.

**Caveats.**
- With 51 claims, only large associations can be detected.
- The data is observational: features co-vary with claim difficulty.
- No causal reading is allowed. A causal test would edit the evidence on purpose, on new claims (future work).

## Q3. Output constraints

**Primary family** `[test]`, run 0, exact McNemar on OS over the 51 NEE claims, Holm across 4 tests:

1. citation vs free, OpenAI: the citation requirement, with verdict order held fixed;
2. citation vs free, Claude;
3. format vs free, OpenAI: reason-first vs verdict-first, without a citation requirement;
4. format vs free, Claude.

**Replication** `[desc]`: the same 4 contrasts on run 1.

**Secondary** `[desc]`, with McNemar p but no correction claimed:
- FactShield-prompt single vs plain format, per model (openai, claude);
- majority vs unanimity vs best single;
- unanimity + check vs unanimity without check.

**Cost, reported for every contrast:**
- US on the 49 firm-gold claims;
- correct answers on all 100;
- WD.

A constraint "reduces overstatement" only if OS falls. Whether US rises is reported beside it, so caution is not mistaken for accuracy.

## Outputs

- `overstatement_report.txt` (UTF-8)
- `overstatement_summary.json`
- `overstatement_features.csv`: one row per claim with F1–F8, gold, `os_share`, and which systems overstated
- `overstatement_coding.csv`: one row per distinct OS claim, with:
  - the verdicts per system;
  - the round-1 reasons and plain-response excerpts;
  - top passages;
  - empty columns for codes, self-source and Nikita's check.

## What would change the conclusions

- Codes are draft until Nikita checks them.
- If more than a third of OS claims turn out to be L, the main finding is about label difficulty rather than model behaviour, and the report says so.
