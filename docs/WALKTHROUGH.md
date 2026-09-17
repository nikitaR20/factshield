# FactShield — Code Walkthrough

Plain-language explanation of every file. Read this before the code.

---

## The one-paragraph version

You highlight a sentence. The extension grabs it plus the paragraphs around it.
The backend works out what claim is actually being made, searches three places
at once, sorts every result by how trustworthy it is and whether it agrees,
does some counting, and then either gives you a verdict or says it can't tell.
A model writes the explanation, but only from the sources — and a separate
check throws away anything it made up.

---

## The three rules everything follows

**1. The model never supplies facts.** It arranges evidence someone else
published. If a sentence in the explanation isn't in a source, it gets deleted
before you see it.

**2. Saying "I don't know" is a real answer.** Most of the hard work is
recognising when the evidence isn't good enough, rather than producing a verdict
anyway.

**3. Every number is computed, not guessed.** The score comes from arithmetic
over source quality and dates. Ask "why 72?" and there's an answer in the logs.

---

# Part 1 — The Backend

## `app/schemas.py` — the shapes

Every stage hands the next one a defined object. Think of these as forms with
required fields.

The important ones:

**`CaptureContext`** — what the browser sends. Selection, surrounding text,
page title, page date.

**`TriageOutput`** — what the first model call produces. Is it checkable, what
are the actual claims, what kind of claim, is it one claim or several.

**`RawDocument`** — a search result, before it's been judged. Every search
source produces this exact shape, which is why swapping a search provider
changes one function and nothing else.

**`EvidenceItem`** — a document *after* judging. Now it carries stance
(agrees/disagrees/unclear), the exact quote, tier (how trustworthy), role
(did it check the claim or just repeat it), and date.

**`SubClaimVerdict`** — the answer. Two axes plus the score.

**`RequestLog`** — one record per request. This is also your research dataset.
Your benchmark is a query over these, not a separate program that might behave
differently from the real thing.

---

## `app/config.py` — settings loader

Loads the YAML files. Nothing exciting, except one thing worth knowing:

**Prompts are versioned files.** `triage.v1.txt`, `decide.v1.txt`. The version
gets logged with every request. If you change a prompt halfway through your
benchmark, you'll be able to tell which results came from which version. Without
this, half your numbers come from a different system and you never find out.

---

## `config/source_packs.yaml` — where to look for each kind of claim

A medical claim goes to PubMed, WHO, Mayo Clinic, Cochrane. A UK policy claim
goes to gov.uk and legislation.gov.uk.

**All the sites for a category go into ONE search**, not one search per site.
The search engine ranks across the whole list, so a heart claim and a skin claim
both search the same medical sites and whichever one covers it comes up first.
No per-claim routing needed.

Two things in here worth understanding:

**`expected_tiers`** — what kind of source this claim *ought* to have. A medical
claim should have medical literature behind it. If none turns up, that absence
is informative, and the system says so instead of guessing.

**The political split.** Claims about what a rule *says* (visa fee, effective
date) have an authoritative answer — the government that wrote it. Claims about
what a rule *does* (did it create jobs) don't; the government is a party to that
argument. If you treat both the same, your tool quietly adopts whoever's in
office. So interpretive claims weight independent analysts above official
sources.

---

## `config/tiers.yaml` — how much each source counts for

**Tier weights** — a fact-check or a government document counts 1.0. A
systematic review 0.9. A news article 0.35. A blog 0.05.

**Role weights** — this is the one that saves you. A source that *checked* the
claim counts 1.0. A source that merely *reports* someone said it counts 0.15.

Why it matters: fifty news sites covering a claim is fifty repetitions of one
claim, not fifty pieces of evidence. Without role weighting, coverage volume
gets counted as agreement and your verdict flips.

**Recency half-life** — how fast evidence goes stale, per category. Medical
findings: years. Breaking news: days.

---

## `app/providers/llm.py` — talking to language models

One function, several providers. Groq, Gemini, and easy to add others.

Three things it handles:

- **Fallbacks.** If Groq is rate-limited, try Gemini. Configured in YAML.
- **JSON repair.** Models sometimes wrap JSON in code fences or add a sentence
  before it. This strips that. If it's still broken, it asks once more, then
  gives up.
- **Giving up properly.** When everything fails it raises an error, and the
  caller abstains. It never falls back to guessing.

---

## `app/providers/nli.py` — the local model

NLI means: given a document and a claim, does the document support it,
contradict it, or not say?

This runs 10–15 times per request. If those were API calls, that's where all
your money and all your waiting would go. So it runs on your own machine.

The model is `DeBERTa-v3-base-mnli-fever-anli` — about 370MB, trained on
fact-checking data specifically. Install `transformers` and `torch` and it
downloads automatically.

**If you don't install it**, a crude keyword-matching stub takes over. It's not
good, but it means the tests and the whole pipeline run offline with no setup.
`GET /health` tells you which one is live so you never accidentally ship the
stub.

---

## `app/retrieval/__init__.py` — finding sources

Three kinds of place to look, and the difference matters:

**Free structured APIs (adapters).** PubMed, CrossRef, Google Fact Check Tools.
These are *indexes* — PubMed alone covers thousands of journals. One adapter
covers a whole field.

They're better than search, not just cheaper: PubMed tells you a paper is a
"Systematic Review" as an actual field. That fills in your tier directly instead
of you guessing from the web address.

**Domain-restricted search (Tavily).** For trusted sites with no API — Mayo
Clinic, NHS, Reuters. One search, the whole site list passed as a filter.

**Open web search.** Only for breaking news and general claims. Turned off for
medical and science because the free adapters already cover what those need.

All three produce identical `RawDocument` objects, so everything downstream
doesn't know or care where a document came from.

**`retrieve()`** fires them all at the same time. One after another would take
three times as long for no benefit. If one fails, you get fewer sources — never
a dead request.

---

## `app/pipeline/grading.py` — judging each source

For every document, work out five things:

1. **Stance** — agrees, disagrees, or doesn't say. Local NLI.
2. **Quote** — the exact sentence that carries that stance. Not a summary. A
   quote can be checked against the original; a paraphrase can drift.
3. **Tier** — how trustworthy. Uses structured metadata where an adapter gives
   it, falls back to matching the domain against the source pack.
4. **Role** — did this source *check* the claim or just *repeat* it? Done with
   word patterns: "said", "according to", "reportedly" means reporting;
   "fact-check", "analysis found", "no evidence" means assessing.
5. **Date.**

One special case: if a fact-checker published a rating ("False", "Mostly
True"), that's used directly rather than re-derived from their prose. They
already did the work.

---

## `app/pipeline/evidence_state.py` — the counting

**Read this file first.** No model calls anywhere in it. Just arithmetic.

It works out:

- How old is the claim?
- How many genuinely independent sources? (`www.bbc.co.uk` and `bbc.co.uk` are
  one source, not two.)
- Do all the sources come after the claim and cluster on a few domains? That's
  a story spreading, not evidence.
- Is there any source of the expected type at all?
- Are the supporting sources all older than the disagreeing ones? That means
  something changed.

Because it's arithmetic, it **cannot make things up**. That's why your most
important explanations come from here — "this is four hours old and every source
traces to one post" is computed, not written by a model.

**`should_abstain()`** is at the bottom. Four rules, checked in order, first
match wins:

1. Nothing found
2. Every source just repeats the claim
3. No source of the required type
4. Everything postdates the claim and shares an origin

If any fires, the request stops **before any model call**. The model can't
hallucinate on a call you never made.

---

## `app/pipeline/scoring.py` — the number and the two answers

Also pure arithmetic, no models.

**`compute_score()`** — for each source:

```
weight = tier × role × recency × independence × confidence
```

Add up the supporting weights and the disagreeing weights. Score is the
supporting share, as 0–100.

Two things that matter:

- **`score_basis`** records every component. "Why 72 and not 65" has an answer.
- **Below a minimum total weight, it returns nothing at all.** Four blogs is not
  evidence, and reporting "50" would look like a finding when it's actually an
  absence. Unresolved and half-true must not look the same.

**`derive_verdict()`** produces both answers:

*Axis 1 — are the facts supported?* From the score. High = supported, low =
refuted, middle = partly, no score = unresolved.

*Axis 2 — is the evidence as good as the claim implies?* Checked most specific
first:

| Result | When |
|---|---|
| outdated | supporting sources older than disagreeing ones |
| amplified | many sources, few origins, all after the claim |
| unestablished | expected source type missing, claim is recent |
| overstated | expected source type missing, claim isn't recent |
| contested | good sources on both sides |
| consistent | none of the above |

This is why you need two axes. *"Some scientists believe 5G causes cancer"* is
**supported** on axis 1 (some do) and **overstated** on axis 2 (no research
behind it). One label can't say both, and either one alone is wrong.

---

## `app/pipeline/stages.py` — triage, shortcut, explain, check

**`triage()`** — one small model call. Is this checkable, what's the actual
claim with pronouns filled in, what category, split it if it's several claims.
If the model is unavailable, it treats the selection as one claim and flags
low confidence rather than failing.

**`fast_path_verdict()`** — **the big cost saving.** If a fact-checker already
published a rating, return it and skip the expensive model entirely. Snopes
did the work; reasoning over it again adds seconds and money and can't improve
on a human verdict.

This catches most viral misinformation, because viral claims are exactly what
fact-checkers cover. Roughly 1.5 seconds instead of 8, and zero paid tokens.

**`explain()`** — the only call that might cost money. Note what it's *not*
doing: the verdict is already decided by `scoring`. The model only writes the
explanation. A narrower job means much less room to invent.

Only the top 6 documents go in. Models get worse with long inputs, not better.

**`ground()`** — the honesty check. Take each sentence of the explanation:

- No `[3]` citation? Delete it.
- Has a citation? Check the quoted source actually supports that sentence
  using the local NLI model. Doesn't? Delete it.

This is the thing that makes "only use the evidence" real instead of a wish.
Telling a model not to invent doesn't stop it — when the evidence is thin it
fills the gap from memory, and the invented part reads exactly like the real
part. The number of deleted sentences is logged, and it's a research metric
nobody reports.

---

## `app/cache.py` — remembering, and the log

**Cache key = hash of the RESOLVED claim, plus the pipeline version.**

Both halves matter. Resolved, because "he said it would double" would collide
across completely unrelated articles. Pipeline version, because changing a
prompt should invalidate old results instead of silently serving answers from a
different system.

Set `FACTSHIELD_DEV_CACHE=true` while building — cached results never expire, so
re-running the same test claims costs nothing after the first time.

**`write_log()`** appends one JSON line per request with everything: the claim,
every source with its date and tier and stance, the timings, the verdict, how
many sentences the grounding check deleted.

This file is your research data. Wrapped so it can never break a request.

---

## `app/main.py` — putting it together

The order, and why:

1. **Triage.** Cheapest call, and you need the resolved claim before the cache
   key means anything.
2. **Cache check.** Hit: done, under 100ms.
3. **Retrieve.** All channels at once.
4. **Grade.** Local, batched, free.
5. **Count and abstain.** If the evidence is bad, stop here. **No model call.**
6. **Fast path.** Fact-check found? Done. **No model call.**
7. **Score, verdict, explain.** The one paid call.
8. **Ground.** Delete anything invented.
9. **Cache and log.**

One detail worth noting: **rate limiting is per participant, not per IP
address.** Universities put everyone behind one address — IP limiting would have
your study participants blocking each other, and you'd find out mid-session.

---

# Part 2 — The Extension

## `manifest.json`

Manifest V3. Minimal permissions — no reading your browsing history, no
background tracking. `contextMenus` for the right-click item, `storage` for the
participant ID.

## `background.js` — the service worker

Two things about MV3 that shape this file:

**It gets killed after about 30 seconds idle** and restarts on the next event.
So the context menu is registered in `onInstalled`, not at the top of the file —
re-registering an existing menu throws an error.

**It keeps nothing in memory.** The participant ID lives in `chrome.storage`,
which survives restarts.

It also **assigns the study condition once, at install**. The ID decides which
popup layout you get, deterministically, and it never changes. You can't unsee
a layout, so this has to be fixed per person.

The worker also proxies the API call, because a content script calling out
directly would trip the host page's security policy.

## `capture.js` — grabbing the context

This is the part a copy-paste can't do automatically.

When you highlight *"he said it would double by next year"*, the name is two
paragraphs up and "next year" only means something relative to the article's
date. This walks up the DOM, grabs the containing paragraph and the two before
it, and pulls the publication date from the page's metadata.

It reads JSON-LD first, then meta tags, then `<time>` elements — most reliable
first. Parsing visible text is a last resort because date formats vary by
language and site.

**It always captures context**, even when the sentence looks self-contained,
because the page date feeds outdated-fact detection regardless.

## `popup/popup.js` — the panel

Rendered inside a **closed shadow root**. Websites define aggressive global CSS;
without isolation the panel breaks on a lot of sites, and your styles would leak
into theirs.

**Two layouts, one backend.** Same data, same verdicts, different presentation:

- **evidence_first** — source cards at the top, verdict quietly below
- **verdict_first** — big coloured banner, sources hidden behind a click

Because the backend is identical, any difference in behaviour is caused by the
layout alone. That's the experiment.

**Colour is never the only signal.** Every verdict also has a text label and a
distinct symbol. Red-green alone fails the most basic accessibility standard,
and red-green colour blindness is the most common kind.

**The resolved claim is shown first, always.** If the pronoun resolution got it
wrong, you see that immediately instead of getting a confident verdict about a
claim you never made.

It also records what you do: opened the sources, clicked through to one, how
long the panel stayed open. What people *do* and what they *say* about trust
diverge constantly, so the behaviour is the real measure.

---

# Part 3 — The Evaluation Harness

## `eval/snapshot.py` — freeze the search results

Search once per claim, save the documents, then run every experiment against
the saved copy.

Two reasons, and the second is the important one:

1. You pay for search once instead of once per condition per run.
2. **Your experiments must differ only in the pipeline**, not in what the web
   happened to return that afternoon. Without this, re-running your benchmark
   next month gives different numbers for reasons that have nothing to do with
   your system — and nobody, including you, can reproduce it.

It resumes if interrupted, so a crash on claim 150 doesn't cost you the first
149.

## `eval/conditions.py` — the four experiments

Each one is the full system with exactly one thing removed:

| Condition | What's removed | What it tells you |
|---|---|---|
| `full` | nothing | your system |
| `open_web_only` | source control, fact-check channel | what curated sources are worth |
| `no_page_context` | surrounding paragraphs | what the browser integration is worth |
| `parametric` | all retrieval | what the model knew already |

**`parametric` lives here and only here.** Never in the live pipeline. If you
ran a no-retrieval pass in production and fed its answer into the verdict step,
the model's memory would anchor the result and you could no longer claim the
verdict is evidence-based.

Run separately, you get two things free: the offline-LLM baseline for your
comparison table, *and* a number for how often retrieval overturns the model's
memory — and who's right when they differ. That second one is a finding.

## `eval/run.py` — running and reporting

Runs every claim through every condition, several times.

Several times because **models aren't fully deterministic even at temperature
zero**. Verdict stability across runs is almost never reported in this
literature, it costs only API calls, and if your fixed pipeline proves steadier
than an ad-hoc chat prompt, that's a real argument for purpose-built tooling.

What it reports:

**Macro F1, not accuracy.** With four unbalanced verdict types, plain accuracy
flatters a system that always guesses the common one.

**Broken down by claim category, always.** An overall figure around 70% usually
hides 95% on easy claims and 40% on hedged ones. The breakdown is the finding.

**Risk-coverage.** How much did it answer, and how accurate was it on what it
answered? A system that's 95% right on the 60% it chooses to answer may be far
better than one that's 78% right on everything — because the first never
confidently lies to you.

**Prior vs evidence.** How often the model's memory disagreed with the
evidence, and which was correct.

---

# What to read, in order

1. `pipeline/evidence_state.py` — no models, pure counting, and where the best
   explanations come from
2. `pipeline/scoring.py` — the number and the two axes
3. `tests/test_evidence.py` — every design decision as an assertion
4. `pipeline/stages.py` — where the model calls actually are
5. `main.py` — the order everything happens in

Files 1 and 2 have no model calls and no mocks in their tests. Start there.

---

# What isn't built yet

- **Real streaming.** The panel shows a loading state then the full result. True
  progressive rendering needs server-sent events from the backend.
- **Retraction checking** beyond what CrossRef gives for free.
- **Study stimulus mode** — serving pre-written verdicts for seeded items.
- **Icons** — `extension/icons/` needs three PNGs before Chrome will load it.

# Things to verify before trusting the code

Every line that touches an external service is written from documentation, not
from a live response. Before building on it, hit each one once with a real key
and print the raw JSON:

- Does Tavily's `include_domains` return results when the domains are narrow,
  or come back empty?
- Does `content` contain enough text for quote extraction? (Set
  `search_depth: "advanced"` if you have the student credits — better extraction
  directly improves your quotes.)
- Is PubMed's `pubdate` in the format the parser expects?
- How often is `published_date` actually populated? Three of your six
  `evidence_standing` values depend on dates.

Expect one or two mismatches. Ten minutes of checking now saves a confusing
afternoon later.
