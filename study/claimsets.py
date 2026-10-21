"""Claim sets from CSV (or TSV) files, e.g. exported fact-check tables.

    python -m study.claimsets check                 # what would be loaded, and why not
    python -m study.claimsets check --folder D:/thesis/claims

Expected columns (names are configurable in config.yaml, `claims.csv.columns`):

    question  comment  claim  rate  what's true  what's false  what's unknown
    origin    summary

What each column becomes:

    claim            -> `claim`, the text every model sees. Nothing else is shown
                        to a model.
    rate             -> `factcheck_rating` (verbatim) and, through the table in
                        config.yaml, the gold `label`. A rating that does not
                        clearly mean supported / refuted / not enough evidence
                        ("Mostly True", "Mixture") is EXCLUDED and reported.
                        Nothing is guessed.
    what's true/false/unknown, summary, origin, question, comment
                     -> kept under `factcheck` for auditing gold labels and for
                        human coders. Never shown to a model.
    optional `id`, `domain`, `set`, `source_url`, `source_published`,
    `surface_detectable` columns are used when present.

Without an `id` column, ids are `<file stem>-<row number>`. Row numbers change
if rows are reordered or deleted, so freeze the files before building packs;
the runner refuses a pack whose stored claim text no longer matches its id.

Which set a file belongs to (e.g. A = before model cutoffs, B = after) comes
from `claims.csv.set_by_file`, or a `set` column, or else the file name.
"""

from __future__ import annotations

import argparse
import csv
import functools
import io
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .common import GOLD_LABELS, config, resolve, sha256_text

CSV_SUFFIXES = (".csv", ".tsv")


def _norm_header(h: str) -> str:
    h = (h or "").replace("\ufeff", "").replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", h).strip().lower()


def norm_rating(r: str) -> str:
    r = (r or "").replace("\u2019", "'").strip().lower()
    r = re.sub(r"^rating\s*:\s*", "", r)
    r = re.sub(r"[\s_\-]+", " ", r).strip(" .!\"'")
    return r


@functools.lru_cache(maxsize=None)
def _rating_table() -> tuple[dict[str, str], frozenset[str]]:
    cfg = config()["claims"]["csv"]
    table: dict[str, str] = {}
    for label, values in (cfg["rating_to_label"] or {}).items():
        if label not in GOLD_LABELS:
            raise ValueError(f"config claims.csv.rating_to_label: {label!r} is not one of {GOLD_LABELS}")
        for v in values or []:
            key = norm_rating(v)
            if key in table and table[key] != label:
                raise ValueError(f"rating {v!r} is mapped to both {table[key]} and {label}")
            table[key] = label
    excluded = frozenset(norm_rating(v) for v in cfg.get("excluded_ratings") or [])
    clash = excluded & set(table)
    if clash:
        raise ValueError(f"ratings both mapped and excluded: {sorted(clash)}")
    return table, excluded


def _read_text(p: Path) -> str:
    raw = p.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Excel on Windows saves "CSV" as cp1252 unless told otherwise.
        return raw.decode("cp1252")


def _dialect(text: str, suffix: str) -> csv.Dialect | type[csv.Dialect]:
    if suffix == ".tsv":
        return csv.excel_tab
    head = text.split("\n", 1)[0]
    if head.count("\t") > max(head.count(","), head.count(";")):
        return csv.excel_tab
    if head.count(";") > head.count(","):
        class Semi(csv.excel):
            delimiter = ";"
        return Semi
    return csv.excel


def _truthy(v: str) -> bool | None:
    v = (v or "").strip().lower()
    if v in {"true", "yes", "1", "y"}:
        return True
    if v in {"false", "no", "0", "n"}:
        return False
    return None


def read_csv_file(p: Path) -> tuple[list[dict], list[dict]]:
    """Return (claims, excluded). `excluded` rows carry a reason and are never
    silently dropped: `check` prints them."""
    cfg = config()["claims"]["csv"]
    cols = {k: _norm_header(v) for k, v in cfg["columns"].items()}
    text = _read_text(p)
    reader = csv.DictReader(io.StringIO(text), dialect=_dialect(text, p.suffix.lower()))
    if not reader.fieldnames:
        raise ValueError(f"{p}: empty file")
    headers = {_norm_header(h): h for h in reader.fieldnames}
    for need in ("claim", "rating"):
        if cols[need] not in headers:
            raise ValueError(f"{p}: no {cols[need]!r} column. Found: {list(reader.fieldnames)}. "
                             f"Set claims.csv.columns.{need} in study/config.yaml.")

    def get(row: dict, key: str) -> str:
        h = headers.get(cols.get(key, key)) or headers.get(key)
        return (row.get(h) or "").strip() if h else ""

    table, excluded_ratings = _rating_table()
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", p.stem).strip("_") or "claims"
    set_name = (cfg.get("set_by_file") or {}).get(p.name)
    file_sha = sha256_text(text)
    claims, excluded = [], []
    for n, row in enumerate(reader, 1):
        cid = get(row, "id") or f"{stem}-{n:04d}"
        claim = " ".join(get(row, "claim").split())
        raw_rating = get(row, "rating")
        rating = norm_rating(raw_rating)
        base = {"file": p.name, "row": n, "id": cid, "claim": claim[:120], "rate": raw_rating}
        if not claim:
            excluded.append({**base, "reason": "empty claim"})
            continue
        if not rating:
            excluded.append({**base, "reason": "empty rate"})
            continue
        label = table.get(rating)
        if label is None:
            why = ("rating excluded by config (not clearly supported/refuted/NEI)"
                   if rating in excluded_ratings else "rating not in config table")
            excluded.append({**base, "reason": why})
            continue
        domain = get(row, "domain") or cfg["default_domain"]
        claims.append({
            "id": cid,
            "claim": claim,
            "set": get(row, "set") or set_name or stem,
            "label": label,
            "domain": domain,
            # Each fact-check is its own source unless a source_id column says otherwise.
            "source_id": get(row, "source_id") or cid,
            "source_url": get(row, "source_url") or None,
            "source_published": get(row, "source_published") or None,
            "surface_detectable": _truthy(get(row, "surface_detectable")),
            "factcheck_rating": raw_rating,
            "factcheck": {k: get(row, k) for k in
                          ("question", "comment", "whats_true", "whats_false",
                           "whats_unknown", "origin", "summary") if get(row, k)},
            "retrievable": None,
            "claims_file": p.name,
            "claims_file_row": n,
            "claims_file_sha256": file_sha,
        })
    return claims, excluded


def csv_files(folder: Path) -> list[Path]:
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in CSV_SUFFIXES and f.is_file())


def load_csv_claims(folder: Path) -> tuple[list[dict], list[dict]]:
    claims, excluded, seen_text = [], [], {}
    for f in csv_files(folder):
        c, e = read_csv_file(f)
        excluded.extend(e)
        for row in c:
            key = row["claim"].lower()
            if key in seen_text:
                excluded.append({"file": row["claims_file"], "row": row["claims_file_row"],
                                 "id": row["id"], "claim": row["claim"][:120],
                                 "rate": row["factcheck_rating"],
                                 "reason": f"duplicate claim text (first: {seen_text[key]})"})
                continue
            seen_text[key] = row["id"]
            claims.append(row)
    return claims, excluded


# ------------------------------------------------------------------ CLI

def check(folder: Path) -> int:
    files = csv_files(folder)
    print(f"claims folder: {folder}")
    if not files:
        print("  no .csv/.tsv files found")
        return 1
    table, excluded_ratings = _rating_table()
    claims, excluded = load_csv_claims(folder)

    seen = Counter()
    for f in files:
        text = _read_text(f)
        rdr = csv.DictReader(io.StringIO(text), dialect=_dialect(text, f.suffix.lower()))
        hdr = {_norm_header(h): h for h in rdr.fieldnames or []}
        col = hdr.get(_norm_header(config()["claims"]["csv"]["columns"]["rating"]))
        for r in rdr:
            seen[(r.get(col) or "").strip()] += 1

    print(f"\n{'rate value in file':<32}{'rows':>6}   becomes")
    for raw, n in seen.most_common():
        k = norm_rating(raw)
        dest = table.get(k) or ("EXCLUDED (config)" if k in excluded_ratings
                                else "EXCLUDED (empty)" if not k else "EXCLUDED (not in table)")
        print(f"  {raw[:30]!s:<30}{n:>6}   {dest}")

    by_set = defaultdict(Counter)
    for c in claims:
        by_set[c["set"]][c["label"]] += 1
    print(f"\n{len(claims)} claims kept, {len(excluded)} rows excluded")
    for s, cnt in sorted(by_set.items()):
        print(f"  set {s}: " + ", ".join(f"{l}={cnt.get(l, 0)}" for l in GOLD_LABELS))
    reasons = Counter(e["reason"] for e in excluded)
    for why, n in reasons.most_common():
        print(f"  excluded: {n} x {why}")
    unknown = sorted({e["rate"] for e in excluded if e["reason"] == "rating not in config table"})
    if unknown:
        print("\nRatings not in the table (add each to rating_to_label or excluded_ratings "
              "in study/config.yaml):\n  " + "\n  ".join(repr(u) for u in unknown))
    domains = Counter(c["domain"] for c in claims)
    print(f"\ndomains: {dict(domains)}")
    if set(domains) == {config()['claims']['csv']['default_domain']}:
        print("  (no `domain` column: every claim uses the default retrieval category)")
    sets = sorted(by_set)
    if set(sets) - set(config()["claims"]["csv"].get("set_by_file", {}).values()):
        print(f"\nsets are named from file names: {sets}. Map files to sets (e.g. A/B) in "
              f"claims.csv.set_by_file if they encode the pre/post-cutoff split.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check"])
    ap.add_argument("--folder", default=None, help="defaults to paths.claims in config")
    args = ap.parse_args()
    folder = resolve(args.folder) if args.folder else claims_folder()
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")
    sys.exit(check(folder))


def claims_folder() -> Path:
    from .common import path
    return path("claims")


if __name__ == "__main__":
    main()
