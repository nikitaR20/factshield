"""Turn raw documents into a structured evidence table.

Per document: stance, the verbatim sentence carrying that stance, tier, role,
and date. The verdict model never sees raw concatenated snippets — it sees
rows. That single change is what makes grounding enforceable rather than
merely requested.

All model work here is local NLI, batched. This stage runs 10-15x per request
and would dominate cost and latency if it were API calls.
"""

from __future__ import annotations

import re

from ..config import pack_for, tiers
from ..providers import nli
from ..schemas import EvidenceItem, RawDocument, Role, Stance, Tier

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Boilerplate that search extraction routinely returns as body text. Left in,
# it becomes the stored "quote" and the user sees navigation furniture where
# evidence should be.
_JUNK = re.compile(
    r"(autocomplete results are available|cookie|privacy policy|skip to (main )?content"
    r"|subscribe to our|sign in to continue|enable javascript|^title:|all rights reserved"
    r"|follow us on|share this (article|page))", re.I)

# Extraction splices link markup into prose: "more susceptible13108-1/pdf) to
# getting colds". Such a sentence is unusable as a displayed quote.
_BROKEN = re.compile(r"\]\(|https?://|\.pdf\)|\w\d{3,}-\d+/|\[\d+\]\(")

# A source that merely reports THAT someone made a claim is not evidence the
# claim is true. Hundreds of outlets covering a statement is coverage volume,
# not corroboration — and without this distinction the aggregator counts it
# as agreement.
_REPORTING = re.compile(
    r"\b(said|says|claimed|claims|announced|according to|told reporters|"
    r"stated|alleged|reportedly|posted|tweeted|wrote)\b", re.I)
_NEGATION = re.compile(
    r"\b(no|not|never|false|fake|hoax|debunk\w*|untrue|incorrect|misleading|baseless|refut\w*|disprov\w*)\b", re.I)
_ASSESSING = re.compile(
    r"\b(fact[- ]check\w*|we rated|our review|analysis (found|shows)|"
    r"study (found|shows)|evidence (shows|suggests|indicates)|"
    r"research (found|shows)|investigation found|no evidence|"
    r"rating:|verdict|debunk\w*|conclude[sd]?)\b", re.I)


def sentences(text: str, limit: int = 40) -> list[str]:
    parts = [
        s.strip()
        for s in _SENT.split(text or "")
        if len(s.strip()) > 25 and not _JUNK.search(s) and not _BROKEN.search(s)
    ]
    if parts:
        return parts[:limit]
    # Fallback for text too short or unpunctuated to split — but it must pass
    # the same filters, or it reintroduces exactly the junk they removed.
    whole = (text or "").strip()
    if whole and not _JUNK.search(whole) and not _BROKEN.search(whole):
        return [whole]
    return []


_CLAIM_DATE = re.compile(
    r"\s*(?:\b(?:in|on|during|as of|by)\b\s+)?"
    r"(?:the\s+)?(?:week\s+of\s+)?"
    r"(?:\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\b\s+\d{1,2}(?:st|nd|rd|th)?,?\s*(?:20\d{2})?"
    r"|\d{1,2}\s+\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b,?\s*(?:20\d{2})?"
    r"|\b20\d{2}-\d{2}-\d{2}\b)",
    re.I,
)


def strip_explicit_date(claim: str) -> str:
    """Remove an explicit calendar date from a claim before stance judging.

    A claim resolved to "raised interest rates during the week of September 17,
    2026" scored 0.066 entailment against CNN's near-identical "raised interest
    rates by a quarter of a percentage point" — the assertion matches perfectly
    and the date phrasing does not. Sources say "Wednesday", "September 16",
    "effective September 17"; the model cannot reconcile those.

    So the model judges the ASSERTION and the code checks the DATE, which is
    the same division of labour the rest of the pipeline uses: arithmetic for
    anything computable, the model only for meaning.
    """
    out = _CLAIM_DATE.sub(" ", claim)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,.")
    return out or claim


def _candidates(text: str, claim: str, top_k: int = 4) -> list[str]:
    """Pick the sentences worth running NLI on.

    Without this, grading creates a pair for every sentence in every document —
    700+ CPU inferences per request, which takes minutes. Lexical overlap is a
    cheap prefilter: a sentence sharing no content words with the claim is
    almost never the one carrying the stance, and scoring it costs nothing.
    """
    claim_terms = {w for w in re.findall(r"\w+", claim.lower()) if len(w) > 3}
    if not claim_terms:
        return sentences(text)[:top_k]

    scored: list[tuple[float, str]] = []
    for sent in sentences(text):
        terms = {w for w in re.findall(r"\w+", sent.lower()) if len(w) > 3}
        if not terms:
            continue
        overlap = len(claim_terms & terms) / len(claim_terms)
        # A negation or verdict cue makes a sentence worth checking even when
        # overlap is low — that is often exactly where a refutation lives.
        if _ASSESSING.search(sent) or _NEGATION.search(sent):
            overlap += 0.25
        # Search extraction truncates mid-sentence ("...according to a"). A
        # fragment can lose the qualifier that changes its meaning, so complete
        # sentences are preferred when one is available.
        if not re.search(r"[.!?\"\u201d')\]]\s*$", sent):
            overlap *= 0.8
        scored.append((overlap, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for score, s in scored[:top_k] if score > 0] or sentences(text)[:1]


def _candidate_sentences(doc_title: str, text: str, claim: str, top_k: int,
                         key_findings: str | None = None) -> list[str]:
    """Headline first, then body sentences.

    A Reuters article titled "Fed forecasts see latest hike followed by another
    before end year" is the clearest statement of the claim in the whole
    document. It was previously used only as premise context, while the
    sentence actually scored was about 2029 rates — and came back `refutes` at
    0.884, flipping the verdict.
    """
    out: list[str] = []
    # A structured abstract's RESULTS and CONCLUSIONS are where a systematic
    # review states its verdict. Scoring them first stops the selector
    # preferring a background sentence that merely mentions the intervention.
    if key_findings:
        out.extend(sentences(key_findings)[:top_k])
    title = (doc_title or "").strip()
    if len(title) > 20 and not _JUNK.search(title):
        out.append(title)
    out.extend(_candidates(text, claim, top_k))
    seen: set[str] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def assign_tier(doc: RawDocument, category: str, jurisdiction: str | None) -> Tier:
    """Structured metadata beats hostname guessing wherever an adapter gives it."""
    if doc.channel == "prior_check":
        return "prior_check"

    if doc.structured_type:
        mapped = tiers().get("publication_type_tier", {}).get(doc.structured_type)
        if mapped:
            return mapped  # type: ignore[return-value]
        if doc.channel == "literature":
            return "peer_reviewed"

    pack = pack_for(category, jurisdiction)
    host = doc.source_domain
    for key, tier in (("definitional", "definitional"),
                      ("independent_analysis", "independent_analysis"),
                      ("authority", "authority")):
        if any(host == d or host.endswith("." + d) for d in pack.get(key, [])):
            return tier  # type: ignore[return-value]

    if doc.channel == "literature":
        return "peer_reviewed"

    # BUG FIX: tier is earned by the DOMAIN, never by the channel that fetched
    # it. The authority channel previously granted `authority` to anything it
    # returned — a supplement retailer (metagenics.ie) was tiered as a medical
    # authority, cited as the sole support, and satisfied the expected-tier
    # check that should have flagged the claim as overstated.
    #
    # A credible outlet from ANOTHER category's pack (Reuters on a medical
    # claim) is real journalism but not a domain authority here: `reporting`.
    if _in_any_pack(host):
        return "reporting"
    return "reporting" if re.search(r"\.(gov|edu|ac\.uk)$", host) else "low"


def _in_any_pack(host: str) -> bool:
    from ..config import source_packs

    for pack in source_packs().values():
        lists = [pack.get(k, []) for k in (
            "authority", "definitional", "independent_analysis")]
        for juris in (pack.get("jurisdictions") or {}).values():
            lists += [juris.get(k, []) for k in ("authority",
                                                 "definitional", "independent_analysis")]
        for domains in lists:
            if any(host == d or host.endswith("." + d) for d in domains):
                return True
    return False


def assign_role(text: str) -> Role:
    if _ASSESSING.search(text):
        return "assesses"
    if _REPORTING.search(text):
        return "reports"
    return "asserts"


# A fact-checker's published rating is authoritative. Their ARTICLE TEXT is
# not: it quotes the falsehood verbatim in order to debunk it, so running NLI
# over it reads the debunked claim as the source's own position. Item 7 of the
# COVID trace did exactly that — the stored quote was "This virus ... is not
# from nature", which is the thing being refuted.
_RATING_REFUTES = (
    "pants on fire", "no evidence", "baseless", "fabricated", "debunked",
    "not true", "incorrect", "altered", "miscaptioned", "unfounded",
    "unsupported", "scam", "hoax", "false",
)
_RATING_SUPPORTS = ("mostly true", "accurate", "correct",
                    "verified", "legit", "true")
# "Misleading" and "Distorts the Facts" are NOT clean refutations. They say the
# underlying facts are partly present and the FRAMING is wrong — which is axis
# 2 (evidence_standing), not axis 1. Treating them as `refuted` is how "Covid
# originated from China" came back refuted on the strength of a fact-check
# about gain-of-function research.
_RATING_NUANCED = (
    "mixture", "half true", "partly", "partially", "unproven", "undetermined",
    "unverified", "outdated", "needs context", "lacks context", "exaggerat",
    "research in progress", "labeled satire", "misleading", "distorts",
    "missing context", "out of context",
)


def _stance_from_rating(rating: str) -> tuple[str, float] | None:
    """Map a published rating to a stance. Returns None when the rating is not
    recognised — in which case the fact-check becomes context, never a verdict."""
    low = rating.lower().strip()
    for word in _RATING_NUANCED:
        if word in low:
            return None
    for word in _RATING_SUPPORTS:
        if word in low and "not " + word not in low:
            return "supports", 0.95
    for word in _RATING_REFUTES:
        if word in low:
            return "refutes", 0.95
    return None


# Hedged evidence cannot fully support an unhedged claim. "Supplements MAY
# HELP to prevent" was graded as supporting "supplements PREVENT"; "some trials
# found a protective effect and others did not" was graded as support at 0.886.
_HEDGE = re.compile(
    r"\b(may|might|could|can help|possibly|potentially|appears? to|suggests?|"
    r"suggested|some (studies|trials|research|evidence)|mixed|inconclusive|"
    r"preliminary|limited evidence|associated with|linked to|others (did|do) not|"
    r"not all|in some)\b", re.I)


# Categorical claims versus partial effects. "Reduces risk by up to 12%" was
# graded as supporting "prevents" at 0.887. The hedge check caught "may" but
# not this: the evidence is unhedged and still describes a much weaker effect
# than the claim asserts. A 12% reduction is not prevention.
_CATEGORICAL = re.compile(
    r"\b(prevents?|prevented|cures?|cured|stops?|eliminates?|eradicates?|"
    r"guarantees?|always|never|completely|entirely|100\s?%|proven to)\b", re.I)
_PARTIAL = re.compile(
    r"\b(reduc\w*|lower\w*|decreas\w*|cut)\b[^.]{0,40}\b(risk|rate|odds|incidence|chance)\b"
    r"|\bby up to\b|\b\d{1,2}(?:\.\d+)?\s?(?:%|per ?cent)|\b(modest|small|slight|marginal|"
    r"partial|some protection|minor)\b", re.I)


def hedge_mismatch(premise: str, claim: str) -> bool:
    """Evidence is weaker than the claim, in either of two ways:

    - it HEDGES ("may help", "some trials found") where the claim does not;
    - it describes a PARTIAL effect ("reduced risk by 12%") where the claim is
      categorical ("prevents").
    """
    hedged = bool(_HEDGE.search(premise)) and not _HEDGE.search(claim)
    partial = bool(_CATEGORICAL.search(claim)) and bool(
        _PARTIAL.search(premise))
    return hedged or partial


# Headlines overclaim. "Study CONFIRMS vitamin D prevents..." heads an article
# whose body reports mixed results. A headline is still worth scoring — it is
# often the clearest statement of an article's point — but at a discount.
HEADLINE_DISCOUNT = 0.75

# Listing findings sentences first was not enough: selection picks the highest
# NLI score, not the first candidate, so a background sentence with good word
# overlap still won. A systematic review's RESULTS and CONCLUSIONS are where
# its verdict lives, so they are boosted in the selection.
FINDINGS_BOOST = 1.4

# PubMed files sample sizes under RESULTS, so the findings boost lifted "The
# analysis included 43 RCTs with 49320 participants" — a description of the
# study, not of what it found.
_STUDY_DESCRIPTION = re.compile(
    r"\b(?:we (?:searched|included|screened|identified)|(?:were|was) included"
    r"|the (?:analysis|review|search|study) included|comprised"
    r"|\d+\s+(?:RCTs?|trials?|studies|participants|patients|articles))\b", re.I)
# ...unless the sentence also says what happened.
_EFFECT_LANGUAGE = re.compile(
    r"\b(effect|impact|risk|incidence|reduc\w*|prevent\w*|associat\w*|significant"
    r"|outcome|benefit|protect\w*|increase[ds]?|decrease[ds]?|odds|hazard)\b", re.I)
STUDY_DESCRIPTION_PENALTY = 0.5


def is_study_description(sentence: str) -> bool:
    """Describes the study rather than its result."""
    return bool(_STUDY_DESCRIPTION.search(sentence)) and not _EFFECT_LANGUAGE.search(sentence)


# A statistically null finding is not a logical contradiction — "no
# significant effect" is the absence of a demonstrated effect, and an
# MNLI-trained model reasonably scores it as neutral. But against a
# CATEGORICAL efficacy claim ("prevents", "cures"), a null result in the
# peer-reviewed literature is precisely the evidence that matters. A 2024
# meta-analysis reporting "no significant effect of vitamin D supplementation
# on ARTI incidence" scored `insufficient` at 0.104.
# "no Vitamin D impact on outcomes including respiratory infections" did not
# match, because the pattern required the literal word "significant". Up to
# three words may now sit between the negation and the outcome noun — but not
# safety words, since "no adverse effects" is a finding about harm, not about
# efficacy, and must never be read as refuting an efficacy claim.
_NULL_RESULT = re.compile(
    r"\b(?:no|not)\s+"
    r"(?:(?!adverse|side|harmful|serious|safety|toxic|unexpected)\w+\s+){0,3}"
    r"(?:significant\s+)?"
    r"(?:effect|impact|difference|benefit|reduction|association|protection|protective)\b"
    r"|\bdid not (?:significantly )?(?:reduce|prevent|lower|decrease|affect|improve|protect)"
    r"|\b(?:was|were) not (?:significantly )?(?:associated|effective|protective)"
    r"|\bnot statistically significant"
    r"|\bfailed to (?:reduce|prevent|show|demonstrate)"
    r"|\b(?:revealed|showed|found|reported) no\b", re.I)

# A null result in a SUBGROUP says less about the general claim.
_SUBGROUP = re.compile(
    r"\b(subgroups?|in (ethnic|older|younger|elderly)|among (those|patients|participants) with)\b", re.I)

NULL_RESULT_CONFIDENCE = 0.70
NULL_SUBGROUP_CONFIDENCE = 0.58


def null_result_against(sentence: str, claim: str) -> float | None:
    """Confidence that a literature finding refutes a categorical claim, or None.

    Deliberately narrow: only a categorical claim, only an explicit null
    phrase. Everything else stays with the NLI model. This is a rule, not a
    measurement, and should be validated on the claim set like the weights."""
    if not _CATEGORICAL.search(claim) or not _NULL_RESULT.search(sentence):
        return None
    return NULL_SUBGROUP_CONFIDENCE if _SUBGROUP.search(sentence) else NULL_RESULT_CONFIDENCE


def _stance_from_probs(p: dict[str, float], min_conf: float) -> tuple[Stance, float]:
    ent, con = p.get("entailment", 0.0), p.get("contradiction", 0.0)
    if ent >= min_conf and ent > con:
        return "supports", ent
    if con >= min_conf and con > ent:
        return "refutes", con
    return "insufficient", max(ent, con)


# ------------------------------------------------- reference and duplicates
#
# NLI training data (SNLI, MNLI) is labelled on the assumption that premise and
# hypothesis describe the SAME situation. Given a sentence about a different
# event, a model trained that way tends to answer "contradiction" — Chen et al.,
# "On Reference (In-)Determinacy in Natural Language Inference" (NAACL Findings
# 2025) report over 80% false contradictions on such pairs. In retrieved
# evidence that is the common case: on the Dataset A packs, about half of the
# sentences graded `refutes` shared fewer than one in five of the claim's
# content words ("Two-year-old Katrice Lee went missing..." refuted a claim
# about the Black Forest at 0.99). Every committed FactShield verdict on those
# packs was `refuted`, and almost all were wrong.
#
# So a sentence must be ABOUT the claim before its stance counts. Lexical
# overlap is a crude test of that, but it is transparent and cannot fabricate.
# The threshold is a config value (nli.reference_min_overlap), chosen on the
# dev split only; see eval/from_packs.py.

_STOP = frozenset(
    "the a an and or of to in on for with was were is are be been that this from by "
    "at as it its his her their who said says have has had will would could".split())


def _fold(text: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def content_terms(text: str) -> set[str]:
    """Lower-cased content words, accents folded (Pokémon == Pokemon),
    possessives stripped (Brennan's == Brennan)."""
    low = _fold((text or "").replace("\u2019s", "").replace("'s", "")).lower()
    # Crude stemming by prefix, so "supplements"/"supplementation" and
    # "stabbed"/"stabbing" meet. It does not make paraphrases meet ("colds" vs
    # "respiratory infection"): that is the known limit of a lexical test.
    return {w[:6] for w in re.findall(r"\w+", low) if len(w) > 3 and w not in _STOP}


def claim_overlap(sentence: str, claim: str) -> float:
    ct = content_terms(claim)
    return len(ct & content_terms(sentence)) / len(ct) if ct else 1.0


def claim_names(claim: str) -> set[str]:
    """Terms from the claim's capitalised words (not the first word): the
    people, places and things it is about. "Germany's Black Forest caused..."
    -> {black, forest}."""
    toks = re.findall(r"[\w'\u2019-]+", claim or "")
    out: set[str] = set()
    for k, t in enumerate(toks):
        if k > 0 and t[:1].isupper() and t.lower() not in _STOP:
            out |= content_terms(t)
    return out


def about_claim(sentence: str, title: str, claim: str, min_overlap: float) -> bool:
    """Two lexical tests that a sentence is about THIS claim:
      * it shares at least `min_overlap` of the claim's content words, and
      * if the claim names anything, the sentence or its headline mentions one
        of those names. "Two-year-old Katrice Lee went missing" shares
        "missing" and "year" with a claim about the Black Forest, so overlap
        alone let it through at 0.99 `refutes`.
    """
    if claim_overlap(sentence, claim) < min_overlap:
        return False
    names = claim_names(claim)
    return not names or bool(names & content_terms(f"{title} {sentence}"))


def dedupe(items: list[EvidenceItem], min_jaccard: float = 0.8) -> list[EvidenceItem]:
    """Keep one item per near-identical quote.

    The same article reached the packs as bbc.com and news.bbc.co.uk, and the
    same Reuters story under two URLs. Counting each copy as a separate source
    is the error the data-fusion literature calls ignoring source dependence
    (Dong, Berti-Equille & Srivastava, VLDB 2009): agreement between copies is
    not independent confirmation. Identical wording is the cheapest copy signal;
    it misses rewrites, which is a stated limitation.
    """
    kept: list[EvidenceItem] = []
    seen: list[set[str]] = []
    for it in items:
        if it.tier == "prior_check":
            kept.append(it)
            continue
        t = content_terms(it.quote)
        if t and any(len(t & s) / len(t | s) >= min_jaccard for s in seen):
            continue
        seen.append(t)
        kept.append(it)
    return kept


async def grade(
    docs: list[RawDocument],
    claim: str,
    sub_claim_id: int,
    category: str,
    jurisdiction: str | None = None,
    start_id: int = 1,
) -> list[EvidenceItem]:
    """Grade every document against one sub-claim.

    For each document the best-matching sentence is selected by entailment or
    contradiction strength, and that sentence becomes the stored quote. A quote
    can be string-matched back against the source; a paraphrase cannot.
    """
    from ..config import models

    min_conf = models().get("nli", {}).get("min_confidence", 0.55)

    max_per_doc = models().get("nli", {}).get("max_sentences_per_doc", 4)
    min_overlap = models().get("nli", {}).get("reference_min_overlap", 0.0)
    # Stance is judged against the assertion; the date is checked separately in
    # evidence_state by comparing publication dates.
    stance_claim = strip_explicit_date(claim)

    pairs: list[tuple[str, str]] = []
    index: list[tuple[int, str]] = []
    findings_sents: set[tuple[int, str]] = set()
    for di, doc in enumerate(docs):
        from_findings = set(sentences(doc.key_findings)
                            ) if doc.key_findings else set()
        for sent in _candidate_sentences(doc.title, doc.text, stance_claim, max_per_doc,
                                         key_findings=doc.key_findings):
            if sent in from_findings:
                findings_sents.add((di, sent))
            # The premise carries three things the sentence alone does not.
            #
            # TITLE gives scope. A WHO page titled "first cases confirmed in
            # Europe" contains "The first cases of 2019-nCoV have been reported
            # in the European Region" — read without its title that appears to
            # contradict "first detected in Wuhan". With it, plainly not.
            #
            # DATE resolves relative time. A WSJ article saying "the Federal
            # Reserve raised rates Wednesday" scored `insufficient` against the
            # claim "raised rates in the week of September 17, 2026", because
            # nothing in the sentence connects "Wednesday" to that week. Three
            # strong sources contributed zero weight as a result.
            parts = []
            if doc.published_date:
                parts.append(f"Published {doc.published_date.isoformat()}.")
            if doc.title:
                parts.append(f"{doc.title.strip()}.")
            parts.append(sent)
            premise = " ".join(parts)
            pairs.append((premise[:1200], stance_claim))
            index.append((di, sent))

    if not pairs:
        return []

    # Does each fact-check actually review OUR claim? One extra pair per
    # prior_check document, and it is what stops a rating for a different
    # claim being inherited as a verdict.
    prior_match_min = models().get("nli", {}).get("prior_claim_match_min", 0.6)
    match_index = [
        (di, d.prior_claim_text)
        for di, d in enumerate(docs)
        if d.channel == "prior_check" and d.prior_claim_text
    ]
    match_scores: dict[int, float] = {}
    if match_index:
        mp = await nli.classify_async([(t, claim) for _, t in match_index])
        for (di, _), p in zip(match_index, mp):
            # Symmetric-ish: entailment either way means "same claim".
            match_scores[di] = max(p.get("entailment", 0.0),
                                   p.get("contradiction", 0.0))

    probs = await nli.classify_async(pairs)

    # Keep the single strongest sentence per document — with headlines
    # discounted, so a punchy title cannot outrank a hedged body sentence that
    # actually describes the evidence.
    best: dict[int, tuple[float, str, dict[str, float]]] = {}
    for (di, sent), p in zip(index, probs):
        strength = max(p.get("entailment", 0.0), p.get("contradiction", 0.0))
        if (di, sent) in findings_sents:
            strength *= FINDINGS_BOOST
        elif sent.strip() == (docs[di].title or "").strip():
            strength *= HEADLINE_DISCOUNT
        if is_study_description(sent):
            strength *= STUDY_DESCRIPTION_PENALTY
        if di not in best or strength > best[di][0]:
            best[di] = (strength, sent, p)

    items: list[EvidenceItem] = []
    next_id = start_id
    for di, (_, sent, p) in sorted(best.items()):
        doc = docs[di]
        stance, conf = _stance_from_probs(p, min_conf)

        if sent.strip() == (doc.title or "").strip():
            conf *= HEADLINE_DISCOUNT

        # Literature only: a null finding against a categorical efficacy claim.
        if doc.channel == "literature" and stance != "refutes":
            null_conf = null_result_against(sent, stance_claim)
            if null_conf is not None:
                stance, conf = "refutes", null_conf

        # A hedged premise supports a hedged claim, not a categorical one.
        if stance == "supports" and hedge_mismatch(sent, stance_claim):
            conf *= 0.5
        if stance in ("supports", "refutes") and conf < min_conf:
            stance = "insufficient"

        # Not about this claim: the stance is the NLI artefact described above.
        # Journal abstracts are exempt: they were retrieved by the claim's own
        # concept query and state findings in technical paraphrase ("respiratory
        # infection" for "colds"), which a word-overlap test cannot match.
        if (stance in ("supports", "refutes") and min_overlap
                and doc.channel != "literature"
                and not about_claim(sent, doc.title or "", stance_claim, min_overlap)):
            stance = "insufficient"

        # A fact-checker's published rating is a direct verdict — but ONLY if
        # they reviewed the same claim. ClaimReview search returns loosely
        # related claims: searching "Covid originated from China" returns
        # fact-checks of "Wuhan Lab Leak Theory CONFIRMED" and "COVID stands
        # for Chinese-Originated Viral Infectious Disease". Inheriting a
        # rating from a different claim produces a confident wrong verdict,
        # which is the worst failure this system can have.
        claim_match: float | None = None   # only meaningful for fact-checks
        if doc.channel == "prior_check":
            claim_match = round(float(match_scores.get(di, 0.0)), 3)
            rated = _stance_from_rating(doc.prior_rating or "")
            if rated and claim_match >= prior_match_min:
                stance, conf = rated
            else:
                # Either the fact-checker reviewed a DIFFERENT claim, or their
                # rating is nuanced ("Distorts the Facts", "Mixture"). Both are
                # context, never a verdict. Crucially we do NOT fall back to
                # NLI over the article prose, because that prose contains the
                # debunked claim verbatim.
                stance, conf = "discusses", claim_match

            # The quote must make clear this is a RATING, not the source
            # asserting the claim.
            publisher = doc.prior_publisher or doc.source_domain
            reviewed = (doc.prior_claim_text or "").strip().rstrip(".")
            if reviewed:
                sent = f'{publisher} rated the claim "{reviewed}" as {doc.prior_rating}.'
            else:
                sent = f"{publisher} rating: {doc.prior_rating}."

        items.append(
            EvidenceItem(
                id=next_id,
                sub_claim_id=sub_claim_id,
                url=doc.url,
                title=doc.title,
                # A truncated quote is marked as such rather than presented as
                # a whole sentence.
                quote=(sent[:500] if re.search(r"[.!?\"\u201d')\]]\s*$", sent[:500])
                       else sent[:500].rstrip(" ,;:-[") + " …"),
                published_date=doc.published_date,
                source_domain=doc.source_domain,
                channel=doc.channel,
                tier=assign_tier(doc, category, jurisdiction),
                stance=stance,
                # Role is judged on the SENTENCE being used as evidence, not
                # the whole document. A long article contains "said" somewhere,
                # so scanning the full text marked almost everything as
                # `reports` — which then tripped the "everyone is just
                # repeating this" abstention and killed valid claims.
                # Peer-reviewed research evaluates a hypothesis — that is what
                # the genre IS. Every paper was being scored `asserts` (weight
                # 0.4) because its findings sentence contains no phrase like
                # "our analysis found", so a meta-analysis directly on the
                # question counted for less than a news article.
                role=("assesses" if doc.channel in ("prior_check", "literature")
                      else assign_role(sent)),
                stance_confidence=round(float(conf), 3),
                claim_match=claim_match,
                retracted=doc.retracted,
            )
        )
        next_id += 1

    if models().get("nli", {}).get("dedupe_quotes", False):
        items = dedupe(items)
    return items
