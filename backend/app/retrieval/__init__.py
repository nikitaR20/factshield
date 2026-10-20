"""Retrieval layer.

Two mechanisms, and the distinction is index versus site:

  * ADAPTERS wrap structured APIs that index many sources. PubMed indexes
    thousands of journals; one adapter covers the medical literature. They are
    preferred where they exist because they return structured metadata —
    publication type, date, retraction status — which populates `tier` directly
    instead of guessing it from a URL.

  * DOMAIN SEARCH covers authoritative sites with no API (Mayo Clinic, NHS,
    wire services). ONE call per category, passing the whole authority pack as
    a domain filter. The provider ranks across the pack, so no per-claim source
    routing is needed.

Cost note: the adapters below are free. Only `domain_search` and `open_web`
consume paid search credits, and open_web is disabled for medical and science
where the adapters already cover the expected tiers.
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime

import httpx

from ..config import env, models, pack_for
from ..schemas import Channel, RawDocument

# Never evidence, whatever the search engine returns. Social platforms are
# user-generated; prediction markets publish trading odds, which are a measure
# of what bettors expect, not of what happened. A Polymarket page and a
# Facebook video both appeared as "evidence" for a Federal Reserve decision.
DENY_DOMAINS = {
    "facebook.com", "instagram.com", "twitter.com", "x.com", "tiktok.com",
    "youtube.com", "youtu.be", "reddit.com", "pinterest.com", "quora.com",
    "medium.com", "substack.com", "linkedin.com", "threads.net",
    "polymarket.com", "kalshi.com", "predictit.org", "metaculus.com",
    "answers.com", "ask.com", "wikihow.com", "scribd.com", "issuu.com",
}


def _denied(domain: str) -> bool:
    d = domain.lower().removeprefix("www.")
    return any(d == bad or d.endswith("." + bad) for bad in DENY_DOMAINS)


_UA = {
    "User-Agent": "FactShield/2.0 (academic research; contact in repo README)"}


def _domain(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).netloc.lower().removeprefix("www.")


_ABSTRACT = re.compile(
    r"<PMID[^>]*>(\d+)</PMID>(.*?)(?=<PubmedArticle|\Z)", re.S)
_ABSTRACT_TEXT = re.compile(r"<AbstractText([^>]*)>(.*?)</AbstractText>", re.S)
_TAGS = re.compile(r"<[^>]+>")
_LABEL = re.compile(r'(?:Label|NlmCategory)="([^"]+)"', re.I)

# Sections of a structured abstract that carry the finding.
# Ordered: a conclusion states the verdict, results state the numbers, and
# both beat everything else. Conclusions are placed first so they are scored
# first when only a few candidates fit.
_CONCLUSION_SECTIONS = {"conclusion", "conclusions", "interpretation",
                        "authors' conclusions", "discussion"}
_RESULT_SECTIONS = {"results", "result", "findings", "main outcome",
                    "main results", "outcomes"}
_FINDING_SECTIONS = _CONCLUSION_SECTIONS | _RESULT_SECTIONS


def _clean(fragment: str) -> str:
    """Strip tags and DECODE ENTITIES. PubMed's XML encodes the middle dot used
    as a decimal separator as `&#xb7;`, so "OR 1·40; 95% CI 0·70, 2·79" reached
    the model as "OR 1&#xb7;40; 95 % CI 0&#xb7;70" — the numbers that carry the
    finding were unreadable."""
    import html

    text = html.unescape(_TAGS.sub("", fragment))
    text = text.replace("\u00b7", ".")          # middle dot as decimal point
    return re.sub(r"\s+", " ", text).strip()


def _parse_abstracts(xml: str) -> dict[str, tuple[str, str | None]]:
    """Return {pmid: (full_abstract, key_findings)}.

    The previous regex matched `<AbstractText[^>]*>` and discarded the
    attributes — which is exactly where PubMed puts `Label="CONCLUSIONS"`. The
    structure was delivered and thrown away.

    Unstructured abstracts have no labels; for those the final two sentences
    are taken as findings, since abstracts conventionally close on them.
    """
    out: dict[str, tuple[str, str | None]] = {}
    for pmid, block in _ABSTRACT.findall(xml or ""):
        parts: list[str] = []
        conclusions: list[str] = []
        results: list[str] = []
        for attrs, body in _ABSTRACT_TEXT.findall(block):
            text = _clean(body)
            if not text:
                continue
            parts.append(text)
            label = _LABEL.search(attrs or "")
            section = label.group(1).strip().lower() if label else ""
            if section in _CONCLUSION_SECTIONS:
                conclusions.append(text)
            elif section in _RESULT_SECTIONS:
                results.append(text)
        if not parts:
            continue
        full = " ".join(parts)
        findings = conclusions + results
        if not findings:
            tail = re.split(r"(?<=[.!?])\s+", full)
            findings = [" ".join(tail[-2:])] if len(tail) > 2 else []
        out[pmid] = (full, " ".join(findings) or None)
    return out


_URL_DATE = re.compile(
    r"(?:^|[/\-_])(20\d{2})[/\-_]?(0[1-9]|1[0-2])[/\-_]?(0[1-9]|[12]\d|3[01])(?:[/\-_]|$)")
_URL_YM = re.compile(r"(?:^|[/\-_])(20\d{2})[/\-_](0[1-9]|1[0-2])(?:[/\-_]|$)")


def _date_from_url(url: str) -> date | None:
    """Recover a publication date from the URL when the search API gives none.

    Institutional pages routinely expose no date, so a 2026-04 FOMC minutes
    page and this week's statement looked equally current — and the stale one
    contradicted the claim with no recency penalty applied.
    """
    m = _URL_DATE.search(url)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = _URL_YM.search(url)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            pass
    # Bare 8-digit form with no separators, e.g. fomcminutes20260429.htm
    m = re.search(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", url)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _canonical(url: str) -> str:
    """Scheme- and www-insensitive key, trailing slash removed."""
    from urllib.parse import urlparse

    u = urlparse(url)
    host = u.netloc.lower().removeprefix("www.")
    return f"{host}{u.path.rstrip('/')}"


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _parse_date(raw) -> date | None:
    """Parse the date formats search and index APIs actually return.

    BUG FIX: the previous version sliced the input to a length derived from
    the FORMAT string, so PubMed's usual "2008 Mar" matched nothing and became
    None. Month-precision dates were silently discarded — which starved the
    outdated-claim detection that depends on comparing them.

    Handles: ISO dates and datetimes, "2008 Mar 15", "2008 Mar", "2008 Mar-Apr",
    "2008 Spring", and a bare "2008".
    """
    if not raw:
        return None
    if isinstance(raw, date):
        return raw
    text = str(raw).strip()

    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    m = re.match(
        r"(\d{4})(?:[\s/-]+([A-Za-z]{3})[A-Za-z]*)?(?:[\s/-]+(\d{1,2}))?", text)
    if m:
        year = int(m.group(1))
        month = _MONTHS.get((m.group(2) or "").lower()[:3], 1)
        day = int(m.group(3)) if m.group(3) else 1
        try:
            return date(year, month, day)
        except ValueError:
            return date(year, month, 1)
    return None


# ------------------------------------------------------------ fact-check API

async def factcheck(claim: str, limit: int = 5, **_) -> list[RawDocument]:
    """Google Fact Check Tools API — live ClaimReview markup, free.

    This is a query against publishers' markup as they publish it, not a
    stored corpus. It resolves viral claims outright and is the single
    highest-value channel per unit cost.
    """
    key = env("FACTCHECK_API_KEY")
    if not key:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    try:
        async with httpx.AsyncClient(timeout=8, headers=_UA) as c:
            r = await c.get(url, params={"query": claim, "key": key, "pageSize": limit})
            r.raise_for_status()
            data = r.json()
    except Exception:
        return []

    docs: list[RawDocument] = []
    for item in data.get("claims", [])[:limit]:
        for review in item.get("claimReview", [])[:1]:
            link = review.get("url")
            if not link:
                continue
            docs.append(
                RawDocument(
                    url=link,
                    title=review.get("title") or item.get("text", "")[:200],
                    text=f"{item.get('text', '')} — Rating: {review.get('textualRating', '')}",
                    published_date=_parse_date(review.get("reviewDate")),
                    source_domain=_domain(link),
                    channel="prior_check",
                    prior_rating=review.get("textualRating"),
                    prior_publisher=(review.get("publisher")
                                     or {}).get("name"),
                    prior_claim_text=item.get("text"),
                )
            )
    return docs


# ------------------------------------------------------------------- PubMed

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "that", "this", "with", "by", "as", "at",
    "from", "it", "its", "can", "will", "does", "do", "has", "have", "had",
    "their", "they", "some", "any", "all", "more", "most", "than", "which",
    "who", "people", "prevent", "prevents", "cause", "causes", "reduce",
    "reduces", "help", "helps", "study", "studies", "shows", "show",
}


def _pubmed_terms(claim: str) -> str:
    """Keywords, not the sentence. Searching PubMed with the raw claim returned
    one 2017 dose-comparison trial in children and missed every meta-analysis
    on the question. Relation words ("prevents", "causes") are dropped: they
    are what we are testing, and requiring them filters out null results."""
    def keep(w: str) -> bool:
        # Short tokens are usually noise — except the ones that ARE the claim:
        # "Vitamin D", "5G", "B12", "HIV". Dropping the D turned a query about
        # vitamin D into a query about vitamins.
        if any(ch.isdigit() for ch in w) or (w.isupper() and len(w) <= 5):
            return True
        return len(w) >= 3 and w.lower() not in _STOP

    words = [w for w in re.findall(
        r"[A-Za-z0-9][A-Za-z0-9\-]*", claim) if keep(w)]
    return " ".join(words[:8])


def _concept_query(concepts: list[list[str]]) -> str:
    """(synonyms of concept A) AND (synonyms of concept B).

    Each concept is ORed internally so wording variation doesn't lose papers —
    "supplements" versus "supplementation", "respiratory infection" versus
    "acute respiratory tract infection". Concepts are ANDed so the result is
    about the relationship, not either topic alone."""
    groups = []
    for synonyms in concepts:
        terms = [
            f'"{t}"' if " " in t else t for t in synonyms if t and t.strip()]
        if terms:
            groups.append("(" + " OR ".join(terms) + ")")
    return " AND ".join(groups)


async def pubmed(claim: str, limit: int = 5, concepts: list[list[str]] | None = None,
                 **_) -> list[RawDocument]:
    """Syntheses first, NEWEST first, then primary studies.

    When triage supplies concepts (the intervention and the outcome), the query
    is built from them. Otherwise it falls back to keyword extraction. The
    raw-sentence and plain-keyword queries both missed every meta-analysis on
    vitamin D and respiratory infection, and returned a 2017 dose-comparison
    trial that does not test the claim at all.

    Newest first matters for syntheses specifically: a 2025 meta-analysis
    supersedes the 2021 one it updates, and the pilot's gold label changed for
    exactly that reason.

    A systematic review or meta-analysis is the evidence that settles a
    medical question; a single trial rarely does. Querying for syntheses
    explicitly, then filling with primary studies, puts the right tier of
    evidence in front of the grader."""
    terms = _concept_query(concepts) if concepts else _pubmed_terms(claim)
    if not terms:
        return []
    syntheses = await _pubmed_search(
        f"({terms}) AND (systematic review[pt] OR meta-analysis[pt])", limit,
        sort="pub_date",
    )
    seen = {str(d.url) for d in syntheses}
    primary = [d for d in await _pubmed_search(terms, limit) if str(d.url) not in seen]
    return (syntheses + primary)[:limit]


async def _pubmed_search(query: str, limit: int, sort: str = "relevance") -> list[RawDocument]:
    """NCBI E-utilities — free, no key needed at modest volume.

    Returns publication type as structured metadata, which maps straight onto
    tier. "Systematic Review" as a field beats inferring tier from a hostname.
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        async with httpx.AsyncClient(timeout=8, headers=_UA) as c:
            s = await c.get(
                f"{base}/esearch.fcgi",
                params={"db": "pubmed", "term": query, "retmax": limit,
                        "retmode": "json", "sort": sort},
            )
            s.raise_for_status()
            ids = s.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
            f = await c.get(
                f"{base}/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(
                    ids), "retmode": "json"},
            )
            f.raise_for_status()
            summaries = f.json().get("result", {})

            # Abstracts. A title alone gives the stance model nothing to judge.
            ab = await c.get(
                f"{base}/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(ids),
                        "retmode": "xml", "rettype": "abstract"},
            )
            abstracts = _parse_abstracts(ab.text)
    except Exception:
        return []

    docs: list[RawDocument] = []
    for pid in ids:
        rec = summaries.get(pid)
        if not rec:
            continue
        pubtypes = [p for p in rec.get("pubtype", [])]
        full, findings = abstracts.get(pid) or (rec.get("title", ""), None)
        title = _clean(rec.get("title", ""))
        docs.append(
            RawDocument(
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                title=title[:300],
                text=f"{title} {full}"[:6000],
                key_findings=findings,
                published_date=_parse_date(rec.get("pubdate")),
                source_domain="pubmed.ncbi.nlm.nih.gov",
                channel="literature",
                structured_type=pubtypes[0] if pubtypes else None,
                retracted=any("retract" in p.lower() for p in pubtypes),
            )
        )
    return docs


# ----------------------------------------------------------------- CrossRef

async def crossref(claim: str, limit: int = 5, **_) -> list[RawDocument]:
    """CrossRef — free, covers academic publishing broadly, exposes retractions."""
    try:
        async with httpx.AsyncClient(timeout=8, headers=_UA) as c:
            r = await c.get(
                "https://api.crossref.org/works",
                params={"query.bibliographic": claim, "rows": limit,
                        "select": "title,URL,issued,type,abstract,update-to"},
            )
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
    except Exception:
        return []

    docs: list[RawDocument] = []
    for it in items:
        title = (it.get("title") or [""])[0]
        link = it.get("URL")
        if not title or not link:
            continue
        parts = (it.get("issued", {}).get("date-parts") or [[None]])[0]
        pub = date(parts[0], parts[1] if len(parts) > 1 else 1,
                   parts[2] if len(parts) > 2 else 1) if parts and parts[0] else None
        docs.append(
            RawDocument(
                url=link, title=title[:300],
                text=(it.get("abstract") or title)[:2000],
                published_date=pub, source_domain=_domain(link),
                channel="literature", structured_type=it.get("type"),
                retracted=any(u.get("type") ==
                              "retraction" for u in it.get("update-to", [])),
            )
        )
    return docs


# ------------------------------------------------------------ search wrapper

async def _tavily(claim: str, limit: int, include: list[str] | None, channel: Channel):
    key = env("TAVILY_API_KEY")
    if not key:
        return []
    payload = {
        "api_key": key, "query": claim, "max_results": limit,
        "search_depth": "advanced",  # 2 credits, but returns real page text
    }
    if include:
        payload["include_domains"] = include
    try:
        async with httpx.AsyncClient(timeout=10, headers=_UA) as c:
            r = await c.post("https://api.tavily.com/search", json=payload)
            r.raise_for_status()
            results = r.json().get("results", [])
    except Exception:
        return []

    return [
        RawDocument(
            url=x["url"], title=x.get("title", "")[:300],
            text=x.get("content", "")[:4000],
            published_date=_parse_date(
                x.get("published_date")) or _date_from_url(x["url"]),
            source_domain=_domain(x["url"]), channel=channel,
        )
        for x in results if x.get("url")
    ]


async def domain_search(claim: str, limit: int = 5, category: str = "general",
                        jurisdiction: str | None = None, **_) -> list[RawDocument]:
    pack = pack_for(category, jurisdiction)
    domains = list(pack.get("definitional", [])) + list(pack.get("authority", [])) \
        + list(pack.get("independent_analysis", []))
    docs = await _tavily(claim, limit, domains or None, "authority")

    # The provider's domain filter is not strictly enforced — metagenics.ie,
    # news.harvard.edu and bcmj.org all came back from a medical-pack search.
    # Anything outside the pack is re-labelled as open web rather than
    # dropped: it may still be useful context, but it must not ride in on the
    # authority channel.
    for d in docs:
        if domains and not any(
            d.source_domain == x or d.source_domain.endswith("." + x) for x in domains
        ):
            d.channel = "open_web"
    return docs


async def open_web(claim: str, limit: int = 5, **_) -> list[RawDocument]:
    return await _tavily(claim, limit, None, "open_web")


ADAPTERS = {
    "factcheck": factcheck,
    "pubmed": pubmed,
    "crossref": crossref,
    "federal_register": domain_search,
    "govuk": domain_search,
}


async def retrieve(claim: str, category: str, jurisdiction: str | None = None,
                   concepts: list[list[str]] | None = None) -> list[RawDocument]:
    """Fan out across every channel concurrently.

    Sequential execution would triple latency for no benefit. Per-channel
    failures are swallowed: a dead adapter means fewer sources, never a dead
    request.
    """
    cfg = models().get("retrieval", {})
    limit = cfg.get("max_docs_per_channel", 6)
    pack = pack_for(category, jurisdiction)

    tasks = [ADAPTERS[name](claim, limit=limit, category=category,
                            jurisdiction=jurisdiction, concepts=concepts)
             for name in pack.get("adapters", ["factcheck"]) if name in ADAPTERS]
    tasks.append(domain_search(
        claim, limit, category=category, jurisdiction=jurisdiction))
    if pack.get("use_open_web", True):
        tasks.append(open_web(claim, limit))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    docs, seen = [], set()
    for res in results:
        if isinstance(res, Exception):
            continue
        for d in res:
            if _denied(d.source_domain):
                continue
            # http/https/www variants of one page are one source, not three.
            key = _canonical(str(d.url))
            if key in seen:
                continue
            seen.add(key)
            docs.append(d)
    return docs
