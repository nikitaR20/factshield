# Closed-book probe vs with-evidence answers (parser-v4; 14 of 600 probe calls missing)

## Without evidence, on claims labelled supported/refuted

| Model | Set | Firm claims | Committed / calls | Correct | Correct when committed | Right on both repeats |
|---|---|---|---|---|---|---|
| openai | A | 15 | 13/30 | 13 | 100% | 6/15 |
| openai | B | 34 | 18/68 | 13 | 72% | 6/34 |
| gemini | A | 15 | 24/29 | 20 | 83% | 10/15 |
| gemini | B | 34 | 42/63 | 24 | 57% | 10/34 |
| groq | A | 15 | 14/29 | 12 | 86% | 6/15 |
| groq | B | 34 | 42/68 | 20 | 48% | 10/34 |

## The 8 shared errors: no-evidence answers (repeat 1 | 2) -> with-evidence answer

- A-0023 (gold not_enough_evidence): openai: not_enough_evidence|not_enough_evidence -> supported; gemini: refuted|refuted -> supported; groq: not_enough_evidence|not_enough_evidence -> supported
- A-0039 (gold not_enough_evidence): openai: not_enough_evidence|not_enough_evidence -> refuted; gemini: not_enough_evidence|not_enough_evidence -> refuted; groq: not_enough_evidence|not_enough_evidence -> refuted
- A-0040 (gold not_enough_evidence): openai: not_enough_evidence|supported -> supported; gemini: supported|supported -> supported; groq: not_enough_evidence|not_enough_evidence -> supported
- B-0007 (gold refuted): openai: not_enough_evidence|supported -> supported; gemini: refuted -> supported; groq: supported|supported -> supported
- B-0018 (gold not_enough_evidence): openai: not_enough_evidence|not_enough_evidence -> supported; gemini: not_enough_evidence|refuted -> supported; groq: not_enough_evidence|not_enough_evidence -> supported
- B-0048 (gold not_enough_evidence): openai: refuted|refuted -> supported; gemini: refuted|refuted -> supported; groq: refuted|refuted -> supported
- A-0005 (gold refuted): openai: refuted|refuted -> not_enough_evidence; gemini: refuted|refuted -> not_enough_evidence; groq: refuted|refuted -> not_enough_evidence
- A-0044 (gold refuted): openai: not_enough_evidence|not_enough_evidence -> not_enough_evidence; gemini: supported|supported -> not_enough_evidence; groq: refuted|refuted -> not_enough_evidence

## Did evidence change firm no-evidence answers? (same firm answer on both repeats)

- openai: 22 firm without evidence; kept 11; changed 11 (to right 9, to wrong 1)
- gemini: 57 firm without evidence; kept 22; changed 35 (to right 29, to wrong 2)
- groq: 47 firm without evidence; kept 14; changed 33 (to right 23, to wrong 6)

## Firm answers on claims labelled NEE that equal the model's consistent no-evidence answer

- openai: 7 firm answers on NEE-labelled claims; 0 match its no-evidence answer []
- gemini: 13 firm answers on NEE-labelled claims; 4 match its no-evidence answer ['A-0026', 'A-0029', 'A-0035', 'A-0040']
- groq: 18 firm answers on NEE-labelled claims; 3 match its no-evidence answer ['A-0029', 'A-0036', 'A-0045']
