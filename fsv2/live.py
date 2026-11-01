"""Stages 1–2 for the browser demo, feeding the same verdict core (stages 3–6).

    extract()      highlighted text -> claims to check ("Checking as: ...")
    check_claim()  one claim -> searches -> passages -> panel verdict -> explanation

Evidence is turned into passages with the SAME cleaning and sentence
selection as the thesis packs (study/passages.py), so the demo shows the kind
of evidence the thesis evaluated. Two deliberate differences from the packs:

  * Fact-check sites are NOT excluded. A published fact-check of the same
    claim is the best evidence a reader can get. Ratings returned by the
    Google Fact Check Tools API are used only after a model judges that the
    fact-checker reviewed the SAME claim (design 3e); others are shown as
    "fact-checks of related claims", never used as evidence.
  * Search errors are recorded and shown, never swallowed. (The thesis
    retrieval code returns "no results" on any HTTP error; that silently
    produced empty packs once.)

Deviation from the design, stated: passages are chosen by the packs' lexical
sentence selection, not a local embedding model (design 3c). It keeps the
demo's evidence identical in form to the thesis evidence and adds no new
dependency.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import date
from typing import Awaitable, Callable
from urllib.parse import urlparse

import httpx

from app.config import env, pack_for
from study import passages as P

from . import decide, evidence as ev, llm
from .settings import cfg, prompt

Progress = Callable[[str], Awaitable[None] | None]

DENY = {"facebook.com", "instagram.com", "twitter.com", "x.com", "tiktok.com", "youtube.com",
        "youtu.be", "reddit.com", "pinterest.com", "quora.com", "medium.com", "substack.com",
        "linkedin.com", "threads.net", "polymarket.com", "kalshi.com", "predictit.org",
        "metaculus.com"}

LABEL_TEXT = {"supported": "Supported", "refuted": "Refuted",
              "not_enough_evidence": "Not enough evidence",
              "conflicting_evidence": "Conflicting evidence"}


def _live() -> dict:
    return cfg()["live"]


def words(text: str) -> int:
    return len((text or "").split())


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _denied(dom: str) -> bool:
    return any(dom == d or dom.endswith("." + d) for d in DENY)


# ------------------------------------------------------------------ helper LLM

async def _helper_json(user: str, stub: bool, client: httpx.AsyncClient) -> tuple[dict | None, dict]:
    system = "You answer only with the JSON object requested."
    c = llm.panel_cfg(_live()["helper_model"], stub=stub)
    r = await llm.call(c, system, user, client)
    meta = {"model_version": r.model_version, "error": r.error, "tokens": r.tokens}
    return (llm.extract_json(r.text) if r.error is None else None), meta


# --------------------------------------------------------------- 1. extract

class TooLong(ValueError):
    def __init__(self, n: int, cap: int):
        super().__init__(f"{n} words; the limit is {cap}")
        self.n, self.cap = n, cap


async def extract(payload: dict, *, stub: bool, client: httpx.AsyncClient) -> dict:
    sel = (payload.get("selection") or "").strip()
    n, cap = words(sel), _live()["word_cap"]
    if n > cap:
        raise TooLong(n, cap)
    tmpl, sha = prompt("extract")
    user = (tmpl.replace("{title}", payload.get("page_title") or "(no title)")
                .replace("{domain}", payload.get("page_domain") or "unknown site")
                .replace("{date}", payload.get("page_published") or "unknown date")
                .replace("{context}", (payload.get("surrounding_text") or "")[:6000] or "(none)")
                .replace("{selection}", sel))
    obj, meta = await _helper_json(user, stub, client)
    if obj is None:
        return {"error": meta["error"] or "could not read the claim list", "claims": [],
                "skipped": [], "meta": meta}
    claims = []
    for i, c in enumerate((obj.get("claims") or [])[:8]):
        text = " ".join(str(c.get("checking_as") or "").split())
        if not text:
            continue
        queries = [str(q).strip() for q in (c.get("queries") or []) if str(q).strip()][:2]
        claims.append({"id": i + 1, "original": str(c.get("original") or "").strip(),
                       "checking_as": text, "resolved": bool(c.get("resolved", True)),
                       "unclear": c.get("unclear"), "queries": queries or [text]})
    skipped = [{"text": str(s.get("text", "")).strip(), "reason": str(s.get("reason", "")).strip()}
               for s in (obj.get("skipped") or []) if isinstance(s, dict) and s.get("text")]
    return {"error": None, "claims": claims, "skipped": skipped, "words": n,
            "meta": {**meta, "prompt_sha": sha}}


# ---------------------------------------------------------------- 2. search

class Search:
    """Real searches. Every call returns (results, error_or_None)."""

    async def factchecks(self, query: str, client: httpx.AsyncClient) -> tuple[list[dict], str | None]:
        key = env("FACTCHECK_API_KEY")
        if not key:
            return [], "FACTCHECK_API_KEY not set"
        try:
            r = await client.get("https://factchecktools.googleapis.com/v1alpha1/claims:search",
                                 params={"query": query, "key": key, "pageSize": 8}, timeout=15)
            if r.status_code >= 400:
                return [], f"fact-check search HTTP {r.status_code}"
            data = r.json()
        except Exception as exc:
            return [], f"fact-check search failed: {type(exc).__name__}"
        out = []
        for item in data.get("claims", []):
            for rev in (item.get("claimReview") or [])[:1]:
                if rev.get("url"):
                    out.append({"url": rev["url"], "reviewed_claim": item.get("text", ""),
                                "rating": rev.get("textualRating", ""),
                                "publisher": (rev.get("publisher") or {}).get("name") or _domain(rev["url"]),
                                "date": (rev.get("reviewDate") or "")[:10] or None,
                                "title": rev.get("title") or ""})
        return out, None

    async def web(self, query: str, include: list[str] | None, client: httpx.AsyncClient
                  ) -> tuple[list[dict], str | None]:
        key = env("TAVILY_API_KEY")
        if not key:
            return [], "TAVILY_API_KEY not set"
        body = {"api_key": key, "query": query, "max_results": _live()["results_per_search"],
                "search_depth": "advanced"}
        if include:
            body["include_domains"] = include
        try:
            r = await client.post("https://api.tavily.com/search", json=body, timeout=30)
            if r.status_code >= 400:
                why = {432: "Tavily plan credits used up", 433: "Tavily pay-as-you-go limit reached",
                       429: "Tavily rate limit", 401: "Tavily key rejected"}.get(r.status_code, "")
                return [], f"web search HTTP {r.status_code} {why}".strip()
            res = r.json().get("results", [])
        except Exception as exc:
            return [], f"web search failed: {type(exc).__name__}"
        return [{"url": x["url"], "title": (x.get("title") or "")[:300],
                 "text": (x.get("content") or "")[:4000],
                 "published_date": (x.get("published_date") or "")[:10] or None,
                 "source_domain": _domain(x["url"])} for x in res if x.get("url")], None


class StubSearch(Search):
    """Offline stand-in: canned results built from the query. No network."""

    async def factchecks(self, query, client):
        return [{"url": "https://factcheck.example/review-1", "reviewed_claim": query,
                 "rating": "False", "publisher": "Example Fact Check", "date": "2026-10-01",
                 "title": "stub"},
                {"url": "https://factcheck.example/related", "reviewed_claim": "A different claim",
                 "rating": "True", "publisher": "Example Fact Check", "date": "2025-01-01",
                 "title": "stub"}], None

    async def web(self, query, include, client):
        h = int(hashlib.sha256(query.encode()).hexdigest(), 16)
        doms = ["apnews.com", "bbc.com", "reuters.com", "localnews.example", "bbc.co.uk"]
        return [{"url": f"https://{d}/story-{h % 997}-{i}", "title": f"Report on {query[:40]}",
                 "text": (f"{query}. Officials described the events in detail on Monday. "
                          f"Further reporting by {d} added context about what happened next. ") * 3,
                 "published_date": "2026-10-20", "source_domain": d}
                for i, d in enumerate(doms[: 2 + h % 4])], None


# ------------------------------------------------------- 3. passages + match

async def match_factchecks(claim: str, fcs: list[dict], stub: bool, client) -> tuple[list[int], dict]:
    if not fcs:
        return [], {}
    tmpl, _ = prompt("match")
    items = "\n".join(f"{i + 1}. {f['publisher']} ({f.get('date') or 'no date'}) reviewed: "
                      f"\"{f['reviewed_claim']}\" — rating: {f['rating']}" for i, f in enumerate(fcs))
    obj, meta = await _helper_json(tmpl.replace("{claim}", claim).replace("{items}", items), stub, client)
    same = []
    for x in (obj or {}).get("same") or []:
        try:
            i = int(x) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(fcs) and i not in same:
            same.append(i)
    return same, meta


def web_passages(claim: str, docs: list[dict], seen_urls: set[str]) -> tuple[list[dict], list[dict]]:
    """Same cleaning, sentence selection, off-topic and near-duplicate rules as
    the packs, minus the fact-check-mention rule (see module docstring)."""
    max_chars = _live()["max_chars_per_passage"]
    pc = P._cfg()
    kept, dropped = [], []
    for d in docs:
        url = d["url"]
        if url in seen_urls:
            continue
        if _denied(d["source_domain"]):
            dropped.append({"url": url, "because": "social or prediction site"})
            continue
        cleaned, _ = P.clean(d.get("text") or "")
        shown = P.select(claim, cleaned, max_chars)
        if len(shown) < pc.get("min_chars", 60):
            dropped.append({"url": url, "because": "too little article text"})
            continue
        ov = P.overlap(claim, shown)
        if ov < pc.get("min_claim_overlap", 0.15):
            dropped.append({"url": url, "because": f"off-topic ({ov:.0%} of the claim's words)"})
            continue
        if any(P.near_duplicate(k["text"], shown) for k in kept):
            dropped.append({"url": url, "because": "near-duplicate"})
            continue
        seen_urls.add(url)
        kept.append({**d, "text": shown, "channel": "web", "claim_overlap": round(ov, 3)})
    kept.sort(key=lambda k: -k["claim_overlap"])
    return kept, dropped


def fc_passage(f: dict) -> dict:
    return {"url": f["url"], "title": f.get("title") or f"Fact-check by {f['publisher']}",
            "source_domain": _domain(f["url"]), "published_date": f.get("date"),
            "channel": "factcheck",
            "text": (f"{f['publisher']} published a fact-check of the claim \"{f['reviewed_claim']}\" "
                     f"and rated it \"{f['rating']}\"" + (f" ({f['date']})." if f.get("date") else "."))}


def as_pack(claim: str, fc: list[dict], web: list[dict]) -> dict:
    lv = _live()
    chosen = fc[: lv["max_factcheck_passages"]] + web
    chosen = chosen[: lv["max_passages"]]
    return {"claim": claim, "passages": [{"pid": f"P{i}", **{k: p.get(k) for k in
            ("url", "title", "source_domain", "published_date", "text", "channel")}}
            for i, p in enumerate(chosen, 1)]}


# ------------------------------------------------------------- 4–6. verdict

def _best_sentence(text: str, statement: str) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if len(s.strip()) > 20] or [text]
    st = P.terms(statement)
    return max(sents, key=lambda s: len(st & P.terms(s)))[:400]


def _reason_text(res: dict, n_sources: int, n_orgs: int) -> str:
    final, reason = res["final"], res.get("reason")
    if final in ("supported", "refuted"):
        return (f"All 3 checking models agreed · {n_orgs} independent "
                f"source{'s' if n_orgs != 1 else ''} cited")
    if final == "conflicting_evidence":
        return "The relevant sources disagree with each other, and I can't settle it from them."
    if reason == "models_disagreed":
        return "The checking models disagreed, so I won't give a verdict."
    if reason == "citation_check_failed":
        return "The models agreed, but the passage they cited didn't clearly say so on checking."
    if n_sources == 0:
        return "I couldn't find sources about this claim."
    return "The sources I found don't settle this claim."


def build_result(claim: str, pack: dict, res: dict, related: list[dict], errors: list[str],
                 rounds: int) -> dict:
    items = ev.prepare(pack)
    table = ev.by_pid(items)
    by = {p["pid"]: p for p in pack["passages"]}
    quotes, seen = [], set()
    for c in res.get("checks") or []:
        if not c["confirmed"]:
            continue
        for pid in c["passages"]:
            it = table.get(pid)
            if it and it.same_text_as:
                pid = it.same_text_as
            if pid in seen or pid not in by:
                continue
            seen.add(pid)
            p = by[pid]
            quotes.append({"pid": pid, "quote": _best_sentence(p["text"], c["statement"]),
                           "statement": c["statement"]})
    quotes = quotes[:3]
    cited = res.get("cited_pids") or []
    orgs = {ev.organisation((table[p].domain if table.get(p) else "")) for p in cited} - {""}
    sources = [{"pid": it.pid, "url": it.url, "title": it.title, "domain": it.domain,
                "date": it.date, "source_type": ("fact-check" if by[it.pid].get("channel") == "factcheck"
                                                 else it.source_type),
                "rating": it.rating, "copy_of": it.same_text_as, "same_org_as": it.same_org_as,
                "text": by[it.pid]["text"]} for it in items]
    expl = res.get("explanation") or {}
    return {"checking_as": claim, "label": res["final"], "label_text": LABEL_TEXT[res["final"]],
            "reason": res.get("reason"),
            "reason_text": _reason_text(res, len(sources), len(orgs)),
            "key_quotes": quotes, "explanation": expl.get("sentences") or [],
            "quotes_only": bool(expl.get("quotes_only")), "sources": sources,
            "related_factchecks": related, "n_sources": len(sources),
            "checked_on": date.today().isoformat(), "search_rounds": rounds,
            "search_errors": errors, "path": res.get("path"), "n_model_calls": res.get("n_calls"),
            "models": sorted({f"{s['model_key']}: {s['model_version']}" for s in res.get("samples", [])
                              if s.get("model_version")})}


async def check_claim(claim: str, queries: list[str], *, checker, search: Search, stub: bool,
                      client: httpx.AsyncClient, progress: Progress) -> dict:
    lv = _live()
    errors: list[str] = []

    async def step(msg: str):
        r = progress(msg)
        if asyncio.iscoroutine(r):
            await r

    await step("Searching published fact-checks")
    fcs, err = await search.factchecks(claim, client)
    if err:
        errors.append(err)
    same, _ = await match_factchecks(claim, fcs, stub, client)
    fc_used = [fc_passage(fcs[i]) for i in same]
    related = [{"publisher": f["publisher"], "reviewed_claim": f["reviewed_claim"], "rating": f["rating"],
                "url": f["url"], "date": f.get("date")} for i, f in enumerate(fcs) if i not in same]

    await step("Searching news sources")
    q1 = queries[0] if queries else claim
    authority = list(pack_for("general").get("authority", []))
    (a, e1), (b, e2) = await asyncio.gather(search.web(q1, authority, client),
                                            search.web(q1, None, client))
    errors += [e for e in (e1, e2) if e]
    seen: set[str] = {p["url"] for p in fc_used}
    web, _ = web_passages(claim, a + b, seen)
    pack = as_pack(claim, fc_used, web)
    rounds = 1

    async def verdict(pk):
        await step(f"3 models reading {len(pk['passages'])} sources")
        return await decide.check_claim(claim, pk, checker=checker, stub=stub, client=client,
                                        explain=False)

    if not pack["passages"]:
        res = {"error": None, "final": "not_enough_evidence", "reason": "no_sources", "checks": [],
               "cited_pids": [], "samples": [], "n_calls": 0, "path": "no_sources"}
    else:
        res = await verdict(pack)
    if res.get("error"):
        return {"error": res["error"], "checking_as": claim, "search_errors": errors}

    if (res["final"] == "not_enough_evidence" and lv["max_search_rounds"] > 1
            and len(queries) > 1 and len(pack["passages"]) < lv["max_passages"]):
        await step("Not settled yet: searching again with a different query")
        (a2, e3), (b2, e4) = await asyncio.gather(search.web(queries[1], authority, client),
                                                  search.web(queries[1], None, client))
        errors += [e for e in (e3, e4) if e]
        more, _ = web_passages(claim, a2 + b2, seen)
        if more:
            rounds = 2
            pack = as_pack(claim, fc_used, web + more)
            res2 = await verdict(pack)
            if res2.get("error"):
                return {"error": res2["error"], "checking_as": claim, "search_errors": errors}
            res = res2

    if lv.get("explain") and pack["passages"]:
        await step("Writing the explanation and checking every sentence")
        items = ev.prepare(pack)
        res["explanation"] = await decide.explain_verdict(
            claim, res["final"], res.get("reason"), items, res.get("samples", []),
            res.get("cited_pids") or [], checker, stub, client, {})
    out = build_result(claim, pack, res, related, errors, rounds)
    out["error"] = None
    return out
