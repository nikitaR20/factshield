"""The three conditions, the closed-book probe, and the verdict parser.

Prompt text lives in study/prompts/<name>.<version>.txt so it can be quoted
verbatim in the thesis. The version is pinned in config.yaml; nothing here
picks "the latest" file.

What is held constant across `free`, `format` and `citation`:
  * the system message (system.<v>.txt)
  * the claim, the evidence pack and the three verdict options, rendered by one
    template (material.<v>.txt) in one fixed position, after the instruction
Only the instruction file differs.
"""

from __future__ import annotations

import functools
import re

from .common import STUDY_DIR, config, sha256_text

PROMPT_DIR = STUDY_DIR / "prompts"
CONDITIONS = ("free", "format", "citation")
VERDICTS = ("supported", "refuted", "not_enough_evidence")
UNPARSEABLE = "unparseable"


@functools.lru_cache(maxsize=None)
def load(name: str, version: str | None = None) -> tuple[str, str]:
    """(text, sha256). Trailing newline stripped; nothing else touched."""
    v = version or config()["prompt_version"]
    p = PROMPT_DIR / f"{name}.{v}.txt"
    if not p.exists():
        raise FileNotFoundError(f"prompt {p} does not exist (prompt_version={v})")
    text = p.read_text(encoding="utf-8").rstrip("\n")
    return text, sha256_text(text)


def render_passages(pack: dict | None) -> str:
    if not pack or not pack.get("passages"):
        return "(none)"
    blocks = []
    for ps in pack["passages"]:
        meta = ", ".join(x for x in (ps.get("source_domain"), ps.get("published_date")) if x)
        head = f"[{ps['pid']}] {ps.get('title') or '(untitled)'}"
        if meta:
            head += f" ({meta})"
        blocks.append(f"{head}\nURL: {ps['url']}\n{ps['text']}")
    return "\n\n".join(blocks)


def _labels() -> dict:
    lab = config()["labels"]
    return {"label_supported": lab["supported"], "label_refuted": lab["refuted"],
            "label_nei": lab["nei"]}


def build(condition: str, claim: str, pack: dict | None) -> tuple[str, str, dict]:
    """Return (system, user, meta). `condition` is a study condition or
    `closed_book`. For the search-leak check, pass a study condition with an
    empty pack."""
    system, sys_sha = load("system")
    if condition == "closed_book":
        instr, i_sha = load("closed_book")
        material_t, m_sha = load("material_closed_book")
        material = material_t.format(claim=claim, **_labels())
    elif condition in CONDITIONS:
        instr, i_sha = load(condition)
        material_t, m_sha = load("material")
        material = material_t.format(claim=claim, passages=render_passages(pack), **_labels())
    else:
        raise ValueError(f"unknown condition {condition!r}")
    user = f"{instr}\n\n{material}"
    meta = {"prompt_version": config()["prompt_version"],
            "prompt_sha": {"system": sys_sha, "instruction": i_sha, "material": m_sha}}
    return system, user, meta


# ------------------------------------------------------------------ parsing

_MD = re.compile(r"[*_`#>]")
_BRACKET_CITE = re.compile(r"\[[^\]]*\]")
_VERDICT_LINE = re.compile(r"^\s*verdict\s*[:\-–]\s*(.+?)\s*$", re.I | re.M)
_CONF_LINE = re.compile(r"^\s*confidence\s*[:\-–]\s*(.+?)\s*$", re.I | re.M)
_REASON_LINE = re.compile(r"^\s*reasoning\s*[:\-–]", re.I | re.M)
_NEI = re.compile(r"\bnot\s+enough\s+evidence\b")
_LABEL = re.compile(r"\b(supported|refuted)\b")
# A label word with a qualifier in front is a different claim ("not supported",
# "partly supported"). The parser does not decide what those mean.
_QUALIFIED = re.compile(
    r"\b(not|un|partly|partially|mostly|largely|somewhat|half|neither|nor|weakly|"
    r"strongly|well|poorly|insufficiently|cannot\s+be|can't\s+be|isn't|is\s+not)\s+"
    r"(supported|refuted)\b")


def _clean(s: str) -> str:
    s = _BRACKET_CITE.sub(" ", s)
    s = _MD.sub("", s)
    return " ".join(s.lower().split())


def label_in(fragment: str) -> str | None:
    """Exactly one unqualified verdict label in `fragment`, else None."""
    s = _clean(fragment)
    if _QUALIFIED.search(s):
        return None
    found = set()
    if _NEI.search(s):
        found.add("not_enough_evidence")
        s = _NEI.sub(" ", s)
    found |= set(_LABEL.findall(s))
    return found.pop() if len(found) == 1 else None


def parse(text: str | None, condition: str) -> dict:
    """Parse a response into a verdict or `unparseable`. Never guesses.

    Rules, in order:
      1. If one or more `Verdict:` lines exist, they decide. All must name the
         same single label; otherwise unparseable.
      2. In `format`, a `Verdict:` line is required. Its absence is
         unparseable, not a cue to read the prose.
      3. Otherwise (free, citation, closed_book), the first non-empty line must
         contain exactly one unqualified label.
    Nothing else is read for the verdict. A response that states its verdict
    only in paragraph three is recorded as unparseable and counted.
    """
    out = {"parsed_verdict": UNPARSEABLE, "parse_rule": None, "confidence": None,
           "format_ok": None, "cited_ids": [], "cited_urls": []}
    if not text:
        out["parse_rule"] = "empty"
        return out

    out["cited_ids"] = cited_ids(text)
    out["cited_urls"] = cited_urls(text)

    vlines = _VERDICT_LINE.findall(text)
    if vlines:
        labs = {label_in(v) for v in vlines}
        if len(labs) == 1 and None not in labs:
            out["parsed_verdict"] = labs.pop()
            out["parse_rule"] = "verdict_line"
        else:
            out["parse_rule"] = "verdict_line_ambiguous"
    elif condition == "format":
        out["parse_rule"] = "format_missing_verdict_line"
    else:
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        lab = label_in(first)
        if lab:
            out["parsed_verdict"] = lab
            out["parse_rule"] = "first_line"
        else:
            out["parse_rule"] = "first_line_no_single_label"

    if condition == "format":
        cm = _CONF_LINE.findall(text)
        confs = {c for c in (_clean(x).strip(" .") for x in cm) if c in ("low", "medium", "high")}
        out["confidence"] = confs.pop() if len(confs) == 1 and len(cm) == 1 else None
        r, v, c = _REASON_LINE.search(text), _VERDICT_LINE.search(text), _CONF_LINE.search(text)
        out["format_ok"] = bool(r and v and c and r.start() < v.start() < c.start())
    return out


# ------------------------------------------------------------------ citations

_PID = re.compile(r"\bP(\d{1,3})\b")
_URL = re.compile(r"https?://[^\s\])>,\"']+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\"'(])|\n+")


def cited_ids(text: str) -> list[str]:
    seen, out = set(), []
    for m in _PID.finditer(text or ""):
        pid = f"P{int(m.group(1))}"
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def cited_urls(text: str) -> list[str]:
    seen, out = set(), []
    for u in _URL.findall(text or ""):
        u = u.rstrip(".")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def citation_units(text: str) -> list[dict]:
    """Split into sentences and list what each cites. One unit per (sentence,
    cited id or url). Sentences with no citation are not units."""
    units = []
    for i, sent in enumerate(s.strip() for s in _SENT.split(text or "")):
        if not sent:
            continue
        for pid in cited_ids(sent):
            units.append({"sentence_index": i, "sentence": sent, "cited": pid, "kind": "pid"})
        for u in cited_urls(sent):
            units.append({"sentence_index": i, "sentence": sent, "cited": u, "kind": "url"})
    return units


def strip_citations(sentence: str) -> str:
    """For the coding worksheet: remove [P3]-style markers so the sentence's
    form does not reveal its condition to the coder."""
    return " ".join(_BRACKET_CITE.sub("", sentence).split())
