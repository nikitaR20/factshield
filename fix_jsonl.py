"""Repair a JSONL file whose records ran together onto one line.

JSONL needs exactly one JSON object per line. Editors, copy-paste and some
Windows tools will silently join them, which produces:

    json.decoder.JSONDecodeError: Extra data: line 1 column 269

This reads the file as a stream, splits it back into objects by tracking brace
depth, and rewrites it one per line. Also reports anything it could not parse
rather than writing a file that is quietly still broken.

    python fix_jsonl.py eval/dataset/claims.jsonl
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def split_objects(text: str) -> list[str]:
    """Split concatenated JSON objects by brace depth, ignoring braces that
    appear inside strings or are backslash-escaped."""
    chunks: list[str] = []
    depth = 0
    start = None
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunks.append(text[start: i + 1])
                start = None
    return chunks


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) >
                1 else "eval/dataset/claims.jsonl")
    if not path.exists():
        print(f"Not found: {path}")
        return

    text = path.read_text(encoding="utf-8")
    original_lines = len([l for l in text.splitlines() if l.strip()])

    chunks = split_objects(text)
    good, bad = [], []
    for chunk in chunks:
        try:
            good.append(json.loads(chunk))
        except json.JSONDecodeError as exc:
            bad.append((chunk[:90], str(exc)))

    if not good:
        print("No valid JSON objects found. The file may be corrupted beyond "
              "a line-break problem — check it opens as text.")
        return

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)

    with path.open("w", encoding="utf-8") as fh:
        for obj in good:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"lines before : {original_lines}")
    print(f"records found: {len(good)}")
    print(f"backup       : {backup.name}")

    if bad:
        print(f"\n{len(bad)} chunk(s) could not be parsed and were DROPPED:")
        for snippet, err in bad:
            print(f"  {err}\n    {snippet}...")

    # Claim files have required fields; flag anything missing rather than
    # letting the benchmark fail halfway through.
    if good and "claim" in good[0]:
        missing = [o.get("id", "?") for o in good
                   if not all(k in o for k in ("id", "claim", "label", "category"))]
        if missing:
            print(
                f"\nrecords missing a required field (id/claim/label/category): {missing}")
        ids = [o.get("id") for o in good]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            print(f"duplicate ids: {sorted(dupes)}")


if __name__ == "__main__":
    main()
