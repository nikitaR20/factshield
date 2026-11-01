"""Stage 3: prepare the pack passages before any model reads them.

Nothing here removes a passage because of what it says. Three things happen:

  * copies: a passage whose text is near-identical to an earlier one (from a
    different URL) is shown once, with the other publishers named;
  * shared origin: passages from the same organisation (bbc.com, bbc.co.uk,
    news.bbc.co.uk) and passages that credit a wire service are labelled, so
    the model is told they may not be independent (design 3a);
  * credibility: each passage carries its source type and, where the domain is
    in Lin et al. (2023), its 0–1 news-quality rating (design 3d). Unrated
    domains are labelled "unrated", never penalised, never filtered.

Relevance ("is this about the same event?") is NOT filtered here; the verdict
model marks it per passage (design 3b).
"""

from __future__ import annotations

import csv
import functools
import re
from dataclasses import dataclass, field
from pathlib import Path

from .settings import cfg

DATA = Path(__file__).resolve().parent / "data"

# Public suffixes with two labels that appear in news URLs. A short explicit
# list instead of a dependency: it only decides which label names the
# organisation, and a miss just means two domains are not grouped.
_TWO_LEVEL = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "gov.au",
    "co.nz", "govt.nz", "co.in", "gov.in", "co.za", "com.br", "com.mx", "co.jp",
    "com.sg", "com.hk", "com.ng", "com.pk", "co.ke", "com.ph", "co.il",
}

_WIRE = [
    ("AP", re.compile(r"\(AP\)|\bAssociated Press\b")),
    ("Reuters", re.compile(r"\(Reuters\)|\bReuters\b")),
    ("AFP", re.compile(r"\(AFP\)|\bAgence France-Presse\b|\bAFP\b")),
]

_WORD = re.compile(r"[a-z0-9]+")


def host(domain: str) -> str:
    return (domain or "").lower().strip().removeprefix("www.")


def registrable(domain: str) -> str:
    """news.bbc.co.uk -> bbc.co.uk; edition.cnn.com -> cnn.com."""
    parts = host(domain).split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _TWO_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def organisation(domain: str) -> str:
    """The name label of the registrable domain: bbc.com and bbc.co.uk -> bbc."""
    return registrable(domain).split(".")[0]


@functools.lru_cache(maxsize=1)
def _ratings() -> dict[str, float]:
    with (DATA / "domain_pc1.csv").open(encoding="utf-8") as fh:
        return {host(r["domain"]): float(r["pc1"]) for r in csv.DictReader(fh) if r.get("pc1")}


def rating(domain: str) -> float | None:
    """Lin et al. (2023) aggregate rating, matching the longest listed suffix."""
    parts = host(domain).split(".")
    table = _ratings()
    for i in range(len(parts) - 1):
        cand = ".".join(parts[i:])
        if cand in table:
            return table[cand]
    return None


def source_type(domain: str) -> str:
    h = host(domain)
    if h.endswith("wikipedia.org"):
        return "encyclopedia (user-edited)"
    if re.search(r"(^|\.)gov(\.[a-z]{2})?$|\.mil$|(^|\.)gov\.[a-z]{2}$", h):
        return "government"
    if re.search(r"\.edu$|\.ac\.[a-z]{2}$", h):
        return "academic"
    if rating(h) is not None:
        return "news outlet"
    return "other website"


def shingles(text: str, k: int) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return {tuple(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


@dataclass
class Prepared:
    pid: str
    url: str
    title: str
    domain: str
    date: str | None
    text: str
    source_type: str
    rating: float | None
    same_text_as: str | None = None              # shown once: points to the first copy
    also_published_by: list[str] = field(default_factory=list)
    same_org_as: list[str] = field(default_factory=list)
    credits_wire: list[str] = field(default_factory=list)


def prepare(pack: dict) -> list[Prepared]:
    ev = cfg()["evidence"]
    out: list[Prepared] = []
    sh: dict[str, set] = {}
    for p in pack.get("passages", []):
        dom = host(p.get("source_domain") or "")
        item = Prepared(
            pid=p["pid"], url=p["url"], title=p.get("title") or "(untitled)", domain=dom,
            date=(str(p["published_date"])[:10] if p.get("published_date") else None),
            text=p.get("text") or "",
            source_type=("published fact-check" if p.get("channel") == "factcheck" else source_type(dom)),
            rating=rating(dom),
            credits_wire=[name for name, rx in _WIRE if rx.search(p.get("text") or "")],
        )
        sh[item.pid] = shingles(item.text, ev["shingle_words"])
        out.append(item)

    for i, a in enumerate(out):
        for b in out[:i]:
            if b.same_text_as is None and a.same_text_as is None and \
                    jaccard(sh[a.pid], sh[b.pid]) >= ev["near_duplicate_jaccard"]:
                a.same_text_as = b.pid
                if a.domain and a.domain not in b.also_published_by and a.domain != b.domain:
                    b.also_published_by.append(a.domain)
        for b in out:
            if b.pid != a.pid and organisation(a.domain) and \
                    organisation(a.domain) == organisation(b.domain):
                a.same_org_as.append(b.pid)
    return out


def render(items: list[Prepared]) -> str:
    """The evidence block every panel model reads. Each text appears once."""
    if not items:
        return "(none)"
    blocks = []
    for it in items:
        meta = ", ".join(x for x in (it.domain, it.date) if x)
        head = f"[{it.pid}] {it.title}" + (f" ({meta})" if meta else "")
        if it.same_text_as:
            blocks.append(f"{head}\nSame text as [{it.same_text_as}]; not repeated. "
                          f"It is a copy, not a separate confirmation.")
            continue
        notes = [f"Source type: {it.source_type}",
                 (f"news-quality rating {it.rating:.2f} on a 0–1 scale (expert ratings, Lin et al. 2023)"
                  if it.rating is not None else "news-quality rating: unrated (not in the list; not a judgement)")]
        if it.also_published_by:
            notes.append("also published by " + ", ".join(it.also_published_by))
        if it.same_org_as:
            notes.append("same organisation as " + ", ".join(f"[{x}]" for x in it.same_org_as))
        if it.credits_wire:
            notes.append("mentions " + "/".join(it.credits_wire) +
                         " (may repeat that agency's report rather than report independently)")
        blocks.append(f"{head}\n" + " · ".join(notes) + f"\nURL: {it.url}\n{it.text}")
    return "\n\n".join(blocks)


def by_pid(items: list[Prepared]) -> dict[str, Prepared]:
    return {it.pid: it for it in items}


def text_for_check(items: list[Prepared], pids: list[str]) -> str:
    """Text of the cited passages (copies resolve to their original)."""
    table = by_pid(items)
    seen, parts = set(), []
    for pid in pids:
        it = table.get(pid)
        if it is None:
            continue
        if it.same_text_as:
            it = table[it.same_text_as]
        if it.pid not in seen:
            seen.add(it.pid)
            parts.append(it.text)
    return "\n\n".join(parts)
