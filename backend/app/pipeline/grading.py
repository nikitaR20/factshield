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
        if len(s.strip()) > 25 and not _JUNK.search(s)
    ]
    return parts[:limit] or ([text.strip()] if text and text.strip() else [])


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
        scored.append((overlap, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for score, s in scored[:top_k] if score > 0] or sentences(text)[:1]


def _candidate_sentences(doc_title: str, text: str, claim: str, top_k: int) -> list[str]:
    """Headline first, then body sentences.

    A Reuters article titled "Fed forecasts see latest hike followed by another
    before end year" is the clearest statement of the claim in the whole
    document. It was previously used only as premise context, while the
    sentence actually scored was about 2029 rates — and came back `refutes` at
    0.884, flipping the verdict.
    """
    out: list[str] = []
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
    if doc.channel == "authority":
        return "authority"
    return "reporting" if re.search(r"\.(gov|edu|ac\.uk)$", host) else "low"


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


def _stance_from_probs(p: dict[str, float], min_conf: float) -> tuple[Stance, float]:
    ent, con = p.get("entailment", 0.0), p.get("contradiction", 0.0)
    if ent >= min_conf and ent > con:
        return "supports", ent
    if con >= min_conf and con > ent:
        return "refutes", con
    return "insufficient", max(ent, con)


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
    # Stance is judged against the assertion; the date is checked separately in
    # evidence_state by comparing publication dates.
    stance_claim = strip_explicit_date(claim)

    pairs: list[tuple[str, str]] = []
    index: list[tuple[int, str]] = []
    for di, doc in enumerate(docs):
        for sent in _candidate_sentences(doc.title, doc.text, stance_claim, max_per_doc):
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

    # Keep the single strongest sentence per document.
    best: dict[int, tuple[float, str, dict[str, float]]] = {}
    for (di, sent), p in zip(index, probs):
        strength = max(p.get("entailment", 0.0), p.get("contradiction", 0.0))
        if di not in best or strength > best[di][0]:
            best[di] = (strength, sent, p)

    items: list[EvidenceItem] = []
    next_id = start_id
    for di, (_, sent, p) in sorted(best.items()):
        doc = docs[di]
        stance, conf = _stance_from_probs(p, min_conf)

        # A fact-checker's published rating is a direct verdict — but ONLY if
        # they reviewed the same claim. ClaimReview search returns loosely
        # related claims: searching "Covid originated from China" returns
        # fact-checks of "Wuhan Lab Leak Theory CONFIRMED" and "COVID stands
        # for Chinese-Originated Viral Infectious Disease". Inheriting a
        # rating from a different claim produces a confident wrong verdict,
        # which is the worst failure this system can have.
        claim_match = 1.0
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
                quote=sent[:500],
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
                role="assesses" if doc.channel == "prior_check" else assign_role(
                    sent),
                stance_confidence=round(float(conf), 3),
                claim_match=claim_match,
                retracted=doc.retracted,
            )
        )
        next_id += 1

    return items
