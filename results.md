# FactShield v2 — results (scored 2026-10-29)

Frozen design fsv2-0.2 / prompt v2 / MiniCheck-Flan-T5-L. conflicting_evidence scored as not enough evidence. CW = confidently wrong (a supported/refuted verdict that does not match the label). Coverage = share of claims given a supported/refuted verdict. Accuracy on answered = correct among supported/refuted verdicts. Brackets: 95% intervals (Wilson; for 0 errors the Clopper-Pearson upper bound).

## all (100)
Gold: supported 32, refuted 17, NEE 51

| Condition | Correct | Answered (coverage) | Accuracy on answered | CW (rate of all) | Missed firm | Macro-F1 |
|---|---|---|---|---|---|---|
| FactShield | 83/100 (83%) | 43 (43%) | 86% [73%–93%] | 6 (6.0% [2.8%–12.5%]) | 11/49 | 0.78 |
| FactShield without citation check | 83/100 (83%) | 43 (43%) | 86% [73%–93%] | 6 (6.0% [2.8%–12.5%]) | 11/49 | 0.78 |
| Majority 2-of-3 (round 1) + check | 84/100 (84%) | 50 (50%) | 82% [69%–90%] | 9 (9.0% [4.8%–16.2%]) | 7/49 | 0.81 |
| OpenAI alone | 83/100 (83%) | 47 (47%) | 83% [70%–91%] | 8 (8.0% [4.1%–15.0%]) | 9/49 | 0.80 |
| Gemini alone | 82/100 (82%) | 58 (58%) | 76% [63%–85%] | 14 (14.0% [8.5%–22.1%]) | 4/49 | 0.81 |
| Groq alone | 73/100 (73%) | 60 (60%) | 67% [54%–77%] | 20 (20.0% [13.3%–28.9%]) | 7/49 | 0.69 |
| Always 'not enough evidence' | 51/100 (51%) | 0 (0%) | – | 0 (0% [0–3.6%]) | 49/49 | 0.23 |

## unseen: test + new (77)
Gold: supported 26, refuted 15, NEE 36

| Condition | Correct | Answered (coverage) | Accuracy on answered | CW (rate of all) | Missed firm | Macro-F1 |
|---|---|---|---|---|---|---|
| FactShield | 62/77 (81%) | 37 (48%) | 84% [69%–92%] | 6 (7.8% [3.6%–16.0%]) | 9/41 | 0.77 |
| FactShield without citation check | 62/77 (81%) | 37 (48%) | 84% [69%–92%] | 6 (7.8% [3.6%–16.0%]) | 9/41 | 0.77 |
| Majority 2-of-3 (round 1) + check | 65/77 (84%) | 40 (52%) | 85% [71%–93%] | 6 (7.8% [3.6%–16.0%]) | 6/41 | 0.83 |
| OpenAI alone | 63/77 (82%) | 40 (52%) | 82% [68%–91%] | 7 (9.1% [4.5%–17.6%]) | 7/41 | 0.80 |
| Gemini alone | 64/77 (83%) | 47 (61%) | 79% [65%–88%] | 10 (13.0% [7.2%–22.3%]) | 3/41 | 0.83 |
| Groq alone | 57/77 (74%) | 47 (61%) | 70% [56%–81%] | 14 (18.2% [11.2%–28.2%]) | 6/41 | 0.71 |
| Always 'not enough evidence' | 36/77 (47%) | 0 (0%) | – | 0 (0% [0–4.7%]) | 41/41 | 0.21 |

## test (51)
Gold: supported 15, refuted 10, NEE 26

| Condition | Correct | Answered (coverage) | Accuracy on answered | CW (rate of all) | Missed firm | Macro-F1 |
|---|---|---|---|---|---|---|
| FactShield | 41/51 (80%) | 24 (47%) | 79% [60%–91%] | 5 (9.8% [4.3%–21.0%]) | 5/25 | 0.77 |
| FactShield without citation check | 41/51 (80%) | 24 (47%) | 79% [60%–91%] | 5 (9.8% [4.3%–21.0%]) | 5/25 | 0.77 |
| Majority 2-of-3 (round 1) + check | 42/51 (82%) | 25 (49%) | 80% [61%–91%] | 5 (9.8% [4.3%–21.0%]) | 4/25 | 0.80 |
| OpenAI alone | 41/51 (80%) | 24 (47%) | 79% [60%–91%] | 5 (9.8% [4.3%–21.0%]) | 5/25 | 0.77 |
| Gemini alone | 42/51 (82%) | 29 (57%) | 76% [58%–88%] | 7 (13.7% [6.8%–25.7%]) | 2/25 | 0.81 |
| Groq alone | 37/51 (73%) | 31 (61%) | 65% [47%–79%] | 11 (21.6% [12.5%–34.6%]) | 3/25 | 0.70 |
| Always 'not enough evidence' | 26/51 (51%) | 0 (0%) | – | 0 (0% [0–7.0%]) | 25/25 | 0.23 |

## new (26)
Gold: supported 11, refuted 5, NEE 10

| Condition | Correct | Answered (coverage) | Accuracy on answered | CW (rate of all) | Missed firm | Macro-F1 |
|---|---|---|---|---|---|---|
| FactShield | 21/26 (81%) | 13 (50%) | 92% [67%–99%] | 1 (3.8% [0.7%–18.9%]) | 4/16 | 0.75 |
| FactShield without citation check | 21/26 (81%) | 13 (50%) | 92% [67%–99%] | 1 (3.8% [0.7%–18.9%]) | 4/16 | 0.75 |
| Majority 2-of-3 (round 1) + check | 23/26 (88%) | 15 (58%) | 93% [70%–99%] | 1 (3.8% [0.7%–18.9%]) | 2/16 | 0.89 |
| OpenAI alone | 22/26 (85%) | 16 (62%) | 88% [64%–97%] | 2 (7.7% [2.1%–24.1%]) | 2/16 | 0.85 |
| Gemini alone | 22/26 (85%) | 18 (69%) | 83% [61%–94%] | 3 (11.5% [4.0%–29.0%]) | 1/16 | 0.85 |
| Groq alone | 20/26 (77%) | 16 (62%) | 81% [57%–93%] | 3 (11.5% [4.0%–29.0%]) | 3/16 | 0.72 |
| Always 'not enough evidence' | 10/26 (38%) | 0 (0%) | – | 0 (0% [0–13.2%]) | 16/16 | 0.19 |

## dev / design set (23)
Gold: supported 6, refuted 2, NEE 15

| Condition | Correct | Answered (coverage) | Accuracy on answered | CW (rate of all) | Missed firm | Macro-F1 |
|---|---|---|---|---|---|---|
| FactShield | 21/23 (91%) | 6 (26%) | 100% [61%–100%] | 0 (0% [0–14.8%]) | 2/8 | 0.84 |
| FactShield without citation check | 21/23 (91%) | 6 (26%) | 100% [61%–100%] | 0 (0% [0–14.8%]) | 2/8 | 0.84 |
| Majority 2-of-3 (round 1) + check | 19/23 (83%) | 10 (43%) | 70% [40%–89%] | 3 (13.0% [4.5%–32.1%]) | 1/8 | 0.73 |
| OpenAI alone | 20/23 (87%) | 7 (30%) | 86% [49%–97%] | 1 (4.3% [0.8%–21.0%]) | 2/8 | 0.80 |
| Gemini alone | 18/23 (78%) | 11 (48%) | 64% [35%–85%] | 4 (17.4% [7.0%–37.1%]) | 1/8 | 0.69 |
| Groq alone | 16/23 (70%) | 13 (57%) | 54% [29%–77%] | 6 (26.1% [12.5%–46.5%]) | 1/8 | 0.62 |
| Always 'not enough evidence' | 15/23 (65%) | 0 (0%) | – | 0 (0% [0–14.8%]) | 8/8 | 0.26 |

## Dataset A (45)
Gold: supported 11, refuted 4, NEE 30

| Condition | Correct | Answered (coverage) | Accuracy on answered | CW (rate of all) | Missed firm | Macro-F1 |
|---|---|---|---|---|---|---|
| FactShield | 37/45 (82%) | 13 (29%) | 77% [50%–92%] | 3 (6.7% [2.3%–17.9%]) | 5/15 | 0.67 |
| FactShield without citation check | 37/45 (82%) | 13 (29%) | 77% [50%–92%] | 3 (6.7% [2.3%–17.9%]) | 5/15 | 0.67 |
| Majority 2-of-3 (round 1) + check | 37/45 (82%) | 17 (38%) | 71% [47%–87%] | 5 (11.1% [4.8%–23.5%]) | 3/15 | 0.73 |
| OpenAI alone | 37/45 (82%) | 15 (33%) | 73% [48%–89%] | 4 (8.9% [3.5%–20.7%]) | 4/15 | 0.74 |
| Gemini alone | 35/45 (78%) | 19 (42%) | 63% [41%–81%] | 7 (15.6% [7.7%–28.8%]) | 3/15 | 0.68 |
| Groq alone | 30/45 (67%) | 24 (53%) | 50% [31%–69%] | 12 (26.7% [16.0%–41.0%]) | 3/15 | 0.56 |
| Always 'not enough evidence' | 30/45 (67%) | 0 (0%) | – | 0 (0% [0–7.9%]) | 15/15 | 0.27 |

## Set B (55)
Gold: supported 21, refuted 13, NEE 21

| Condition | Correct | Answered (coverage) | Accuracy on answered | CW (rate of all) | Missed firm | Macro-F1 |
|---|---|---|---|---|---|---|
| FactShield | 46/55 (84%) | 30 (55%) | 90% [74%–97%] | 3 (5.5% [1.9%–14.9%]) | 6/34 | 0.81 |
| FactShield without citation check | 46/55 (84%) | 30 (55%) | 90% [74%–97%] | 3 (5.5% [1.9%–14.9%]) | 6/34 | 0.81 |
| Majority 2-of-3 (round 1) + check | 47/55 (85%) | 33 (60%) | 88% [73%–95%] | 4 (7.3% [2.9%–17.3%]) | 4/34 | 0.84 |
| OpenAI alone | 46/55 (84%) | 32 (58%) | 88% [72%–95%] | 4 (7.3% [2.9%–17.3%]) | 5/34 | 0.82 |
| Gemini alone | 47/55 (85%) | 39 (71%) | 82% [67%–91%] | 7 (12.7% [6.3%–24.0%]) | 1/34 | 0.86 |
| Groq alone | 43/55 (78%) | 36 (65%) | 78% [62%–88%] | 8 (14.5% [7.6%–26.2%]) | 4/34 | 0.76 |
| Always 'not enough evidence' | 21/55 (38%) | 0 (0%) | – | 0 (0% [0–6.5%]) | 34/34 | 0.18 |

## unseen Dataset A
Gold: supported 6, refuted 2, NEE 20

| Condition | Correct | Answered (coverage) | Accuracy on answered | CW (rate of all) | Missed firm | Macro-F1 |
|---|---|---|---|---|---|---|
| FactShield | 22/28 (79%) | 8 (29%) | 62% [31%–86%] | 3 (10.7% [3.7%–27.2%]) | 3/8 | 0.54 |
| FactShield without citation check | 22/28 (79%) | 8 (29%) | 62% [31%–86%] | 3 (10.7% [3.7%–27.2%]) | 3/8 | 0.54 |
| Majority 2-of-3 (round 1) + check | 23/28 (82%) | 9 (32%) | 67% [35%–88%] | 3 (10.7% [3.7%–27.2%]) | 2/8 | 0.71 |
| OpenAI alone | 23/28 (82%) | 9 (32%) | 67% [35%–88%] | 3 (10.7% [3.7%–27.2%]) | 2/8 | 0.71 |
| Gemini alone | 21/28 (75%) | 11 (39%) | 55% [28%–79%] | 5 (17.9% [7.9%–35.6%]) | 2/8 | 0.64 |
| Groq alone | 18/28 (64%) | 14 (50%) | 43% [21%–67%] | 8 (28.6% [15.3%–47.1%]) | 2/8 | 0.46 |
| Always 'not enough evidence' | 20/28 (71%) | 0 (0%) | – | 0 (0% [0–12.3%]) | 8/8 | 0.28 |

## unseen Set B
Gold: supported 20, refuted 13, NEE 16

| Condition | Correct | Answered (coverage) | Accuracy on answered | CW (rate of all) | Missed firm | Macro-F1 |
|---|---|---|---|---|---|---|
| FactShield | 40/49 (82%) | 29 (59%) | 90% [74%–96%] | 3 (6.1% [2.1%–16.5%]) | 6/33 | 0.79 |
| FactShield without citation check | 40/49 (82%) | 29 (59%) | 90% [74%–96%] | 3 (6.1% [2.1%–16.5%]) | 6/33 | 0.79 |
| Majority 2-of-3 (round 1) + check | 42/49 (86%) | 31 (63%) | 90% [75%–97%] | 3 (6.1% [2.1%–16.5%]) | 4/33 | 0.85 |
| OpenAI alone | 40/49 (82%) | 31 (63%) | 87% [71%–95%] | 4 (8.2% [3.2%–19.2%]) | 5/33 | 0.80 |
| Gemini alone | 43/49 (88%) | 36 (73%) | 86% [71%–94%] | 5 (10.2% [4.4%–21.8%]) | 1/33 | 0.88 |
| Groq alone | 39/49 (80%) | 33 (67%) | 82% [66%–91%] | 6 (12.2% [5.7%–24.2%]) | 4/33 | 0.77 |
| Always 'not enough evidence' | 16/49 (33%) | 0 (0%) | – | 0 (0% [0–7.3%]) | 33/33 | 0.16 |

## Set B surface_detectable=TRUE
Gold: supported 21, refuted 8, NEE 5

| Condition | Correct | Answered (coverage) | Accuracy on answered | CW (rate of all) | Missed firm | Macro-F1 |
|---|---|---|---|---|---|---|
| FactShield | 28/34 (82%) | 25 (74%) | 96% [80%–99%] | 1 (2.9% [0.5%–14.9%]) | 5/29 | 0.73 |
| FactShield without citation check | 28/34 (82%) | 25 (74%) | 96% [80%–99%] | 1 (2.9% [0.5%–14.9%]) | 5/29 | 0.73 |
| Majority 2-of-3 (round 1) + check | 29/34 (85%) | 26 (76%) | 96% [81%–99%] | 1 (2.9% [0.5%–14.9%]) | 4/29 | 0.78 |
| OpenAI alone | 29/34 (85%) | 26 (76%) | 96% [81%–99%] | 1 (2.9% [0.5%–14.9%]) | 4/29 | 0.78 |
| Gemini alone | 32/34 (94%) | 29 (85%) | 97% [83%–99%] | 1 (2.9% [0.5%–14.9%]) | 1/29 | 0.92 |
| Groq alone | 29/34 (85%) | 27 (79%) | 93% [77%–98%] | 2 (5.9% [1.6%–19.1%]) | 3/29 | 0.76 |
| Always 'not enough evidence' | 5/34 (15%) | 0 (0%) | – | 0 (0% [0–10.3%]) | 29/29 | 0.09 |

## Set B surface_detectable=FALSE
Gold: supported 0, refuted 4, NEE 16

| Condition | Correct | Answered (coverage) | Accuracy on answered | CW (rate of all) | Missed firm | Macro-F1 |
|---|---|---|---|---|---|---|
| FactShield | 18/20 (90%) | 4 (20%) | 75% [30%–95%] | 1 (5.0% [0.9%–23.6%]) | 1/4 | 0.60 |
| FactShield without citation check | 18/20 (90%) | 4 (20%) | 75% [30%–95%] | 1 (5.0% [0.9%–23.6%]) | 1/4 | 0.60 |
| Majority 2-of-3 (round 1) + check | 18/20 (90%) | 6 (30%) | 67% [30%–90%] | 2 (10.0% [2.8%–30.1%]) | 0/4 | 0.61 |
| OpenAI alone | 17/20 (85%) | 5 (25%) | 60% [23%–88%] | 2 (10.0% [2.8%–30.1%]) | 1/4 | 0.59 |
| Gemini alone | 15/20 (75%) | 9 (45%) | 44% [19%–73%] | 5 (25.0% [11.2%–46.9%]) | 0/4 | 0.54 |
| Groq alone | 14/20 (70%) | 8 (40%) | 38% [14%–69%] | 5 (25.0% [11.2%–46.9%]) | 1/4 | 0.51 |
| Always 'not enough evidence' | 16/20 (80%) | 0 (0%) | – | 0 (0% [0–16.8%]) | 4/4 | 0.30 |

## Paired comparisons: FactShield vs each alternative (exact McNemar, two-sided)

b = claims where only the alternative is wrong in that way; c = only FactShield is.

### all (100)

| Alternative | CW: alt only / FS only | p | Correct: FS only / alt only | p |
|---|---|---|---|---|
| FactShield without citation check | 0 / 0 | 1.000 | 0 / 0 | 1.000 |
| Majority 2-of-3 (round 1) + check | 3 / 0 | 0.250 | 3 / 4 | 1.000 |
| OpenAI alone | 2 / 0 | 0.500 | 3 / 3 | 1.000 |
| Gemini alone | 8 / 0 | 0.008 | 8 / 7 | 1.000 |
| Groq alone | 14 / 0 | <0.001 | 13 / 3 | 0.021 |
| Always 'not enough evidence' | 0 / 6 | 0.031 | 37 / 5 | <0.001 |

### unseen: test + new (77)

| Alternative | CW: alt only / FS only | p | Correct: FS only / alt only | p |
|---|---|---|---|---|
| FactShield without citation check | 0 / 0 | 1.000 | 0 / 0 | 1.000 |
| Majority 2-of-3 (round 1) + check | 0 / 0 | 1.000 | 0 / 3 | 0.250 |
| OpenAI alone | 1 / 0 | 1.000 | 2 / 3 | 1.000 |
| Gemini alone | 4 / 0 | 0.125 | 4 / 6 | 0.754 |
| Groq alone | 8 / 0 | 0.008 | 7 / 2 | 0.180 |
| Always 'not enough evidence' | 0 / 6 | 0.031 | 31 / 5 | <0.001 |

## Every FactShield confidently-wrong verdict and every missed firm label (all 100)

- A-0001 [test] missed: FactShield not_enough_evidence, gold refuted; path models_disagreed; singles openai=refuted, gemini=refuted, groq=not_enough_evidence. Claim: The name of Boxing Day comes from the need to rid the house of empty boxes the day after Christmas
- A-0005 [test] missed: FactShield not_enough_evidence, gold refuted; path unanimous_round1; singles openai=not_enough_evidence, gemini=not_enough_evidence, groq=not_enough_evidence. Claim: Clint Eastwood died of a heart attack while filming a commercial in August 2019.
- A-0018 [test] missed: FactShield not_enough_evidence, gold supported; path models_disagreed; singles openai=not_enough_evidence, gemini=not_enough_evidence, groq=supported. Claim: A man was burned to death attempting to rescue his dog from a boiling hot spring.
- A-0019 [dev] missed: FactShield not_enough_evidence, gold supported; path models_disagreed; singles openai=not_enough_evidence, gemini=supported, groq=supported. Claim: As of October 2017, would-be recipients of a repair grant from Dickinson, Texas, must promise not to boycott Israel.
- A-0023 [test] CONFIDENTLY WRONG: FactShield supported, gold not_enough_evidence; path unanimous_round1; singles openai=supported, gemini=supported, groq=supported. Claim: Hillary Clinton's campaign has been systemically overcharging small donors, including the elderly and poor people.
- A-0039 [test] CONFIDENTLY WRONG: FactShield refuted, gold not_enough_evidence; path unanimous_round1; singles openai=refuted, gemini=refuted, groq=refuted. Claim: Jelly sandals sold by Walmart contain unsafe amounts of lead.
- A-0040 [test] CONFIDENTLY WRONG: FactShield supported, gold not_enough_evidence; path unanimous_round1; singles openai=supported, gemini=supported, groq=supported. Claim: An Indian man who was bitten by a snake then bit his wife because he wanted them to 'die together.'
- A-0044 [dev] missed: FactShield not_enough_evidence, gold refuted; path models_disagreed; singles openai=not_enough_evidence, gemini=not_enough_evidence, groq=not_enough_evidence. Claim: A baby was thrown from Grenfell Tower while it was on fire and caught by an unidentified man.
- B-0007 [test] CONFIDENTLY WRONG: FactShield supported, gold refuted; path unanimous_round1; singles openai=supported, gemini=supported, groq=supported. Claim: US has encountered highest number of measles cases since 1993
- B-0014 [test] missed: FactShield not_enough_evidence, gold refuted; path models_disagreed; singles openai=not_enough_evidence, gemini=refuted, groq=not_enough_evidence. Claim: the Democratic candidate for US Senate in Michigan, Abdul El-Sayed called for Sharia law in the US during a 2022 speech
- B-0018 [test] CONFIDENTLY WRONG: FactShield supported, gold not_enough_evidence; path unanimous_round1; singles openai=supported, gemini=supported, groq=supported. Claim: US-based gangster Anil Pandit (tied to the Lawrence Bishnoi network) have claimed responsibility of attack on Tiger Shroff, citing links to 
- B-0029 [test] missed: FactShield not_enough_evidence, gold refuted; path models_disagreed; singles openai=not_enough_evidence, gemini=refuted, groq=supported. Claim: In 2025, newyork has officially banned construction of new data centers for 1 year
- B-0032 [new] missed: FactShield not_enough_evidence, gold refuted; path models_disagreed; singles openai=not_enough_evidence, gemini=refuted, groq=not_enough_evidence. Claim: The Dallas Cowboys defeated the Philadelphia Eagles 37:21 in an NFL game
- B-0035 [new] missed: FactShield not_enough_evidence, gold supported; path models_disagreed; singles openai=not_enough_evidence, gemini=not_enough_evidence, groq=supported. Claim: In UK, between January and August 2026, criminals stole £24.3 million through fake investment schemes where artificial intelligence (AI), in
- B-0048 [new] CONFIDENTLY WRONG: FactShield supported, gold not_enough_evidence; path unanimous_round1; singles openai=supported, gemini=supported, groq=supported. Claim: OpenAI's system proved the Exponential Time Hypothesis false. The hypothesis concerns the computational complexity of Boolean formula satisf
- B-0052 [new] missed: FactShield not_enough_evidence, gold refuted; path models_disagreed; singles openai=refuted, gemini=refuted, groq=not_enough_evidence. Claim: Trump declares US has victory acquiring Greenland
- B-0054 [new] missed: FactShield not_enough_evidence, gold refuted; path models_disagreed; singles openai=refuted, gemini=refuted, groq=not_enough_evidence. Claim: DHS has imposed a $90000 fee for initial OPT

## How FactShield reached its answer (all 100)

| Path / reason | Claims | FactShield answer correct | Answered |
|---|---|---|---|
| unanimous_round1 / - | 68 | 61 | 42 |
| models_disagreed / models_disagreed | 29 | 19 | 0 |
| unanimous_after_extra / - | 3 | 3 | 1 |

## Citation check (MiniCheck)

- Panel agreed on supported/refuted for 43 claims; the check downgraded 0 to NEE.
- Claims where FactShield and 'without citation check' differ: 0 (none).
- Statements checked: 242; confirmed 210, not confirmed 32.

## Overlap of the three models' errors (single, round 1)

### all (100)
- Wrong (any): openai 17, gemini 18, groq 27
- Pairwise phi: openai+gemini 0.48, openai+groq 0.38, gemini+groq 0.42
- All three wrong: 8 (independence would give 0.8); all three the same wrong answer: 8
- Effective independent models (Kish): 1.62 of 3; bootstrap 95% 1.37–1.98

### unseen: test + new (77)
- Wrong (any): openai 14, gemini 13, groq 20
- Pairwise phi: openai+gemini 0.60, openai+groq 0.49, gemini+groq 0.37
- All three wrong: 7 (independence would give 0.6); all three the same wrong answer: 7
- Effective independent models (Kish): 1.53 of 3; bootstrap 95% 1.29–1.89

## Cost per claim (FactShield, all 100)

- Model calls per claim: mean 4.9 (min 3, max 9); a single model uses 1.
- Wall time per claim (s, includes rate-limit waits): median 43, mean 46.
- Tokens (all 100 claims, verdict calls): openai 514,491, gemini 601,431, groq 549,234

## Sensitivity check: labels flagged as borderline BEFORE any run on them
Flagged in claude/setB-pack-audit-2026-10-26.md (B-0007, B-0014, B-0018) and claude/setB-new-claims-audit-2026-10-28.md (B-0047, B-0048, B-0052, B-0053). Excluding these 7 (labels themselves unchanged):

| Condition | All minus flagged (93): answered / CW / correct | Unseen minus flagged (70): answered / CW / correct |
|---|---|---|
| FactShield | 40 / 3 / 81 | 34 / 3 / 60 |
| Majority 2-of-3 + check | 46 / 6 / 81 | 36 / 3 / 62 |
| OpenAI alone | 42 / 4 / 81 | 35 / 3 / 61 |
| Gemini alone | 52 / 10 / 79 | 41 / 6 / 61 |
| Groq alone | 57 / 17 / 71 | 44 / 11 / 55 |

Same pattern as the main result: fewer confident errors than Gemini and Groq alone; no measurable difference from OpenAI alone or from majority vote on unseen claims.
