# Live search on the 55 Set B claims — plan, fixed 2026-10-31 BEFORE any live call

## Question
What does a person checking recent news with a search-enabled chatbot get? Each model
searches the web with its provider's own search tool; no evidence pack is given.

This is descriptive. It is NOT comparable like-for-like with the frozen-pack results:
each provider uses a different search engine (model and search engine are confounded),
results change from day to day, and the gold label is different (see below).

## Prior work (checked 2026-10-31; not a systematic review)
- DeVerna et al., ACL Findings 2026: provider web search raised accuracy on >6,000
  PolitiFact claims, and search-enabled GPT-4o cited the exact PolitiFact article
  46–59% of the time. Older, already-fact-checked claims.
- Walters et al., arXiv 2610.21731 (2026): search-augmented models often retrieve but do
  not use the evidence.
- Lenz Research 2026 (self-published): 5 frontier models with search disagreed on 63% of
  997 user-submitted claims; no ground truth.
- Wu Lu & Setiawan, arXiv 2610.04888 (2026): evidence published after the claim inflates
  accuracy; report per-class recall.
What looked less covered (not proven new): recent claims, some with one detail changed by
the annotator, where no published fact-check exists to find.

## Claims and gold
- The 55 Set B claims (B-0001..B-0055), claim text as in the claims CSV.
- WORLD labels, filled in by Nikita in fsv2/live_world_labels.csv BEFORE any live call, by
  these rules:
  - supported: reliable sources establish the claim, as worded, is true on the check date;
  - refuted: reliable sources establish it is false, including a changed detail that makes
    it false under its natural reading;
  - not_enough_evidence: not settled;
  - "X claimed / said Y": label whether X made that claim (as worded). A claim that asserts
    Y itself is labelled on Y, whoever announced it;
  - any source may be used, fact-checks included; one URL recorded per claim.
- ORIGIN per claim: real (as published) or modified (and what changed). Needed for the
  real-vs-modified breakdown.
- The pack-relative labels of the main study are NOT used here.

## Models and settings (lowest reasoning each provider allows WITH search)
- openai gpt-6-luna: Responses API, `web_search` tool, sources included,
  `max_tool_calls` 3, reasoning effort none, temperature 0. If the API refuses search at
  effort none or temperature 0 in the smoke test, the next setting it accepts is used and
  recorded, before the full run.
- claude claude-haiku-5-5: `web_search_20250305`, `max_uses` 3, effort low, thinking left
  at its default (adaptive). Not "disabled": Anthropic's docs warn that Haiku 5.5 with
  thinking off can skip a tool call it needs when JSON output is asked for. This differs
  from the frozen-pack runs and is reported.
- gemini gemini-3.5-flash-lite: `google_search` tool, thinking low, no temperature.
- Groq is left out (free-plan token limits).
- tool_choice is automatic everywhere; the prompt asks the model to search first. Whether
  it searched, how often, and what it found are recorded for every answer.
- One answer per claim per model (55 x 3 = 165 calls), all on the same day where possible.

## Prompt (fsv2/prompts/live.v1.txt)
Asks for a search, gives today's date, defines the three labels as in the study, and asks
for JSON: verdict, the source URLs relied on, one-sentence reason. No rule about
attribution or copies (that would be a new design, not the plain user scenario).

## Measures (all descriptive; no tests between providers)
Per model, against WORLD labels: correct, answered, confidently wrong, accuracy on
answered (Wilson 95%), per-class recall, unparseable; searched share, searches per claim;
share of answers whose sources include a fact-check site (study fact-check domain list +
URL patterns), and accuracy with vs without one; split by ORIGIN (real / modified).
Across models: all three agree / all three wrong the same way.
Beside it (descriptive): each model's frozen-pack v2 answer scored against the same world
labels, to show what changes when the model searches itself. Not a fair comparison:
different evidence and different date.

## Cost cap
Searches about $10 per 1,000 on OpenAI and Claude, plus result text as tokens; Gemini within
the free monthly allowance. Expected about $1–1.5 per paid provider; the smoke test's
printed token and search counts are multiplied out before the full run.

## Rules
- Verdicts are not printed during the run.
- World labels are not changed after the run. Disagreements found later are reported as
  such, never fixed silently.
