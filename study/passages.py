"""Turning a retrieved web page into the passage a model sees.

Search results arrive as scraped page text: article prose mixed with site
menus, headline lists, sign-in prompts and adverts, and sometimes with a
fact-checker's verdict quoted inside an ordinary news story. Every rule here
is deterministic, configured in study/config.yaml (packs.processing), and
records what it removed, so the method can be published and re-run.

Order, per document:
  1. clean     drop navigation and boilerplate lines
  2. select    keep the sentences most about the claim (with neighbours),
               in their original order, up to max_chars
  3. checks    on the text a model would see:
                 - attributes a finding to a named fact-checker -> drop
                 - shares too few content words with the claim  -> drop
               and across the pack:
                 - near-duplicate of a passage already kept      -> drop

None of this judges whether a passage supports or refutes the claim. That
stays a human judgement in the audit.
"""

from __future__ import annotations

import re

from .common import config

VERSION = "passages-v2"

_STOP = set("""
a an the of to in on for and or but nor is are was were be been being that this these those
with by as at from it its it's has have had having his her hers their theirs they them he she
him we our you your i me my not no yes will would can could should may might must shall do does
did done about into over after before than then more most less least very so such who whom whose
which what when where why how also just only any some all each every other another there here
said says say claim claims claimed one two new
""".split())

_NAV = re.compile(
    r"^(?:top stories|newsletters?|see all newsletters|sections?\b|skip to|sign in|sign up|log in|"
    r"subscribe|advertisement|report ad|menu|most read|most watched|trending|related(?: topics| stories)?|"
    r"share this|follow us|read more|continue with|opens in new tab|enable accessibility|"
    r"accessibility links|cookie|privacy policy|terms of (?:use|service)|all rights reserved|"
    r"watch live|on now|image \d+|getty images|photo:|video:|copyright|©|load more|"
    r"breaking news|languages|previous\s+next|donate now)\b",
    re.I)
_HEADLINE_PREFIX = re.compile(r"^(?:FACT FOCUS|LIVE|WATCH|VIDEO|PHOTOS?|OPINION|ANALYSIS)\s*:", re.I)
_SENT = re.compile(r"(?<=[.!?\"”])\s+(?=[A-Z0-9\"“'(])")


def _cfg() -> dict:
    return config()["packs"].get("processing") or {}


def terms(text: str) -> set[str]:
    """Content words, possessives stripped ("Exampleton's" -> "exampleton")."""
    out = set()
    for w in re.findall(r"[a-z0-9][a-z0-9'’-]*", text.lower()):
        w = re.sub(r"['’]s?$", "", w)
        if len(w) > 2 and w not in _STOP:
            out.add(w)
    return out


def overlap(claim: str, text: str) -> float:
    """Share of the claim's content words that appear in the text."""
    c = terms(claim)
    return len(c & terms(text)) / len(c) if c else 0.0


def _is_boilerplate(line: str) -> bool:
    s = line.strip().strip("#*>-•|[]() \t").strip()
    if not s:
        return True
    if _NAV.match(s) or _HEADLINE_PREFIX.match(s):
        return True
    words = s.split()
    # Menus and section labels: a few words, no sentence punctuation.
    if len(words) < _cfg().get("min_words_per_line", 6) and not re.search(r"[.!?]$", s):
        return True
    # Strings of glued-together menu items ("MoviesFashionTelevision...").
    if re.search(r"(?:[a-z][A-Z][a-z]+){3,}", s) and not re.search(r"[.!?]", s):
        return True
    # "Most read" lists flattened onto one line: "1. Headline 4 MIN READ 669 2. ..."
    if len(re.findall(r"\b\d+\s+MIN READ\b", s, re.I)) >= 2:
        return True
    # Lines that are mostly links or markup.
    if s.count("](") + s.count("http") >= max(2, len(words) // 4):
        return True
    return False


def clean(text: str) -> tuple[str, int]:
    """Drop boilerplate lines. Returns (text, n_lines_removed)."""
    pieces = re.split(r"\n|\s\[\.\.\.\]\s|\s\[…\]\s", text or "")
    kept, removed = [], 0
    for p in pieces:
        if _is_boilerplate(p):
            removed += 1 if p.strip() else 0
            continue
        kept.append(" ".join(p.split()))
    return "\n".join(kept), removed


def select(claim: str, text: str, max_chars: int) -> str:
    """Keep the sentences most about the claim, each with its neighbours, in
    original order, joined with ' … ' where text was skipped."""
    sents = [s.strip() for line in text.split("\n") for s in _SENT.split(line) if s.strip()]
    if not sents:
        return ""
    if sum(len(s) + 1 for s in sents) <= max_chars:
        return " ".join(sents)
    c = terms(claim)
    scored = sorted(range(len(sents)), key=lambda i: (-len(c & terms(sents[i])), i))
    window = _cfg().get("context_sentences", 1)
    chosen: set[int] = set()
    used = 0
    for i in scored:
        block = [j for j in range(i - window, i + window + 1) if 0 <= j < len(sents) and j not in chosen]
        extra = sum(len(sents[j]) + 1 for j in block)
        if used + extra > max_chars and chosen:
            continue
        chosen.update(block)
        used += extra
        if used >= max_chars:
            break
    out, last = [], None
    for j in sorted(chosen):
        if last is not None and j != last + 1:
            out.append("…")
        out.append(sents[j])
        last = j
    return " ".join(out)[:max_chars]


def _factcheckers() -> list[str]:
    return [n.lower() for n in _cfg().get("factchecker_names") or []]


_VERB = (r"(?:rated|rates|rating|found|finds|reported|reports|examined|examines|investigated|"
         r"investigates|verified|verifies|debunked|debunks|concluded|concludes|determined|labell?ed|"
         r"checked|confirmed|confirms|said|says|wrote|noted|traced|analy[sz]ed|analysis|"
         r"looked into|fact-?checked)")


def factcheck_mention(text: str) -> str | None:
    """The phrase that attributes a finding to a named fact-checker, else None.

    Matches "<fact-checker> ... <finding verb>" within a short span, "according
    to <fact-checker>", "as verified by <fact-checker>", a fact-check segment
    ("Fact or Fiction"), or a published rating ("Rating: False"). A bare
    mention ("fact-checkers often debunk hoaxes") does not match; a menu link
    has already been removed by clean().
    """
    t = " ".join((text or "").split())
    for name in _factcheckers():
        n = re.escape(name)
        for pat in (rf"\b(?:according to|as (?:verified|reported|confirmed|noted) by|per)\s+(?:the\s+)?{n}\b",
                    rf"\b{n}(?:'s|’s)?\b[^.]{{0,80}}?\b{_VERB}\b",
                    rf"\b{_VERB}\b[^.]{{0,40}}?\bby\s+(?:the\s+)?{n}\b"):
            m = re.search(pat, t, re.I)
            if m:
                return m.group(0)[:160]
    for pat in (r"\bfact or fictions?\b", r"\brating\s*:\s*(?:true|false|mostly|half|pants|unproven|mixture)",
                r"\bin this case,? (?:their|the fact-?checkers'?) analysis\b"):
        m = re.search(pat, t, re.I)
        if m:
            return m.group(0)[:160]
    return None


def near_duplicate(a: str, b: str) -> bool:
    ta, tb = terms(a), terms(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= _cfg().get("duplicate_jaccard", 0.8)


def process(claim: str, docs: list[dict], max_chars: int) -> tuple[list[dict], list[dict]]:
    """Apply every rule to already-fetched documents (after URL/domain
    exclusions). Returns (kept, dropped); each dropped item has `because`."""
    cfg = _cfg()
    enabled = cfg.get("enabled", True)
    kept, dropped = [], []
    for d in docs:
        raw = d.get("text") or ""
        if not enabled:
            kept.append({**d, "shown_text": raw[:max_chars], "lines_removed": 0,
                         "claim_overlap": round(overlap(claim, raw[:max_chars]), 3)})
            continue
        cleaned, n_removed = clean(raw)
        shown = select(claim, cleaned, max_chars)
        ov = overlap(claim, shown)
        base = {"url": str(d.get("url")), "channel": d.get("channel"),
                "source_domain": d.get("source_domain")}
        if len(shown) < cfg.get("min_chars", 60):
            dropped.append({**base, "because": f"too little article text after cleaning ({len(shown)} chars)"})
            continue
        fc = factcheck_mention(shown)
        if fc and cfg.get("drop_factcheck_mentions", True):
            dropped.append({**base, "because": f"attributes a finding to a fact-checker: {fc!r}"})
            continue
        if ov < cfg.get("min_claim_overlap", 0.15):
            dropped.append({**base, "because": f"off-topic (shares {ov:.0%} of the claim's content words)"})
            continue
        dup = next((k for k in kept if near_duplicate(k["shown_text"], shown)), None)
        if dup:
            dropped.append({**base, "because": f"near-duplicate of {dup['url']}"})
            continue
        kept.append({**d, "shown_text": shown, "lines_removed": n_removed,
                     "claim_overlap": round(ov, 3)})
    return kept, dropped
