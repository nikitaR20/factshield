"""Stratified draw of citations for human coding.

    python -m study.sample_citations [--nli]

Unit: one (sentence, cited passage id) pair from a main-run response. The draw
is stratified evenly across condition x model, with a fixed seed, so the
citation condition — which produces most citations — cannot dominate.

If a cell has fewer units than its share, it contributes all it has and the
shortfall is REPORTED, not refilled from other cells: refilling is exactly
the imbalance stratification exists to prevent. Expect the free and format
cells to be short, since those prompts do not ask for citations.

Two files:
  citation_worksheet.csv  what the coder sees: claim, sentence (citation markers
                          removed), the cited passage text, blank judgement and
                          notes. No model, no condition, rows shuffled.
  citation_key.csv        sample_id -> claim, model, condition, run, cited id,
                          whether the id is in the pack, and (with --nli) the
                          NLI screen. Keep it closed while coding.

The NLI column is a triage aid and a convergent measure, never ground truth:
the product's NLI stage scored 0.000 macro-F1 on hedged claims in its own
benchmark, and without transformers installed it silently runs a keyword stub.
The backend in use is written into the key file.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from .common import config, load_claims, path, read_jsonl
from .packs import load_pack
from .prompts import citation_units, strip_citations
from .score import latest_rows, write_csv

CODING_GUIDE = (
    "judgement: one of SUPPORTS (the passage states what the sentence says), "
    "PARTIAL (some of it), NOT_SUPPORTED (the passage does not state it), "
    "CONTRADICTS (the passage says otherwise), NO_PASSAGE (cited id not in pack). "
    "Judge only the sentence against the passage shown."
)


def collect_units(rows: list[dict], claims: dict) -> list[dict]:
    units = []
    for r in rows:
        if r.get("kind", "main") != "main" or not r.get("response_text"):
            continue
        pk = load_pack(r["claim_id"])
        passages = {p["pid"]: p for p in pk["passages"]}
        for u in citation_units(r["response_text"]):
            if u["kind"] != "pid":
                continue
            p = passages.get(u["cited"])
            units.append({
                "claim_id": r["claim_id"], "claim": claims[r["claim_id"]]["claim"],
                "model": r["model"], "condition": r["condition"], "run_index": r["run_index"],
                "sentence_index": u["sentence_index"], "sentence": u["sentence"],
                "cited": u["cited"], "in_pack": int(p is not None),
                "passage_text": p["text"] if p else "(THIS ID IS NOT IN THE PACK)",
            })
    return units


def stratified(units: list[dict], total: int, seed: int) -> tuple[list[dict], list[dict]]:
    cells = defaultdict(list)
    for u in units:
        cells[(u["condition"], u["model"])].append(u)
    conds = config()["conditions"]
    models = sorted({u["model"] for u in units})
    keys = [(c, m) for c in conds for m in models]
    base, extra = divmod(total, len(keys)) if keys else (0, 0)
    rng = random.Random(seed + 4)
    drawn, report = [], []
    for i, k in enumerate(keys):
        want = base + (1 if i < extra else 0)
        pool = sorted(cells.get(k, []), key=lambda u: (u["claim_id"], u["run_index"],
                                                       u["sentence_index"], u["cited"]))
        take = rng.sample(pool, min(want, len(pool)))
        drawn.extend(take)
        report.append({"condition": k[0], "model": k[1], "available": len(pool),
                       "target": want, "drawn": len(take), "shortfall": want - len(take)})
    return drawn, report


def main() -> None:
    cfg = config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", default=None)
    ap.add_argument("--n", type=int, default=cfg["citations"]["sample_size"])
    ap.add_argument("--nli", action="store_true", help="add the NLI screen to the KEY file only")
    args = ap.parse_args()

    src = Path(args.responses) if args.responses else path("responses") / "main.jsonl"
    rows, _ = latest_rows(read_jsonl(src))
    claims = {c["id"]: c for c in load_claims()}
    units = collect_units(rows, claims)
    seed = cfg["seed"]
    drawn, report = stratified(units, args.n, seed)

    rng = random.Random(seed + 5)
    rng.shuffle(drawn)
    for i, u in enumerate(drawn, 1):
        u["sample_id"] = f"S{i:03d}"

    nli_backend = None
    if args.nli and drawn:
        from app.providers import nli
        nli_backend = nli.backend_name()
        scores = nli.classify([(u["passage_text"], strip_citations(u["sentence"])) for u in drawn])
        for u, s in zip(drawn, scores):
            u["nli_entailment"] = round(s["entailment"], 3)
            u["nli_neutral"] = round(s["neutral"], 3)
            u["nli_contradiction"] = round(s["contradiction"], 3)

    sheet = sorted(({"sample_id": u["sample_id"], "claim": u["claim"],
                     "sentence": strip_citations(u["sentence"]),
                     "cited_passage": u["passage_text"], "judgement": "", "notes": ""}
                    for u in drawn), key=lambda r: r["sample_id"])
    key = sorted(({"sample_id": u["sample_id"], "claim_id": u["claim_id"], "model": u["model"],
                   "condition": u["condition"], "run_index": u["run_index"], "cited": u["cited"],
                   "in_pack": u["in_pack"], "sentence_raw": u["sentence"],
                   **{k: u[k] for k in ("nli_entailment", "nli_neutral", "nli_contradiction") if k in u}}
                  for u in drawn), key=lambda r: r["sample_id"])

    res = path("results")
    write_csv(res / "citation_worksheet.csv", sheet, coding_guide=CODING_GUIDE, sample_seed=seed + 4,
              shuffle_seed=seed + 5, blind=True)
    write_csv(res / "citation_key.csv", key, nli_backend=nli_backend, strata=report,
              note="do not open while coding")

    print(f"{len(units)} citation units available; drew {len(drawn)} (target {args.n})")
    for r in report:
        flag = f"  SHORT by {r['shortfall']}" if r["shortfall"] else ""
        print(f"  {r['condition']:<9} {r['model']:<8} available {r['available']:>4}  "
              f"drawn {r['drawn']:>3}/{r['target']}{flag}")
    if nli_backend == "lexical-stub":
        print("WARNING: NLI ran on the lexical stub (transformers not installed). "
              "Its column is keyword overlap, not inference.")
    print(f"-> {res / 'citation_worksheet.csv'} (give to coder)\n-> {res / 'citation_key.csv'} (keep closed)")


if __name__ == "__main__":
    main()
