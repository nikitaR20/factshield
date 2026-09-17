"""Local natural-language inference.

This runs 10-15 times per request for stance, then again for grounding
verification. Doing it via an API is where cost and latency would actually
come from — so it runs locally, batched, on CPU.

No training is required to start: an off-the-shelf FEVER-tuned model maps
directly onto supports / refutes / insufficient.

The model loads lazily. If transformers or the weights are unavailable, a
deterministic lexical stub takes over so the pipeline, its tests and the
evaluation harness all still run. The stub is never silently used in
production — `backend_name()` reports which path is live.
"""

from __future__ import annotations

import asyncio
import functools
import re

from ..config import models

_LABELS = ("entailment", "neutral", "contradiction")


@functools.lru_cache(maxsize=1)
def _load():
    cfg = models().get("nli", {})
    try:
        import torch  # noqa: F401
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except Exception as exc:
        print(
            f"[factshield] NLI UNAVAILABLE - transformers/torch import failed: {exc}")
        print("[factshield] falling back to the lexical stub. Stance scores will be"
              " keyword-based and mostly wrong. Fix with:")
        print("[factshield]     pip install transformers torch sentencepiece")
        return None

    import os

    # Try candidates in order and use the first that loads with 3 labels.
    # Hardcoding one name has bitten this project twice: a name that no longer
    # exists returns None and the pipeline quietly runs on the keyword stub.
    # Fastest first — they are tried in order and the winner is printed.
    candidates = cfg.get("model_candidates") or [
        cfg.get("model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ]
    if isinstance(candidates, str):
        candidates = [candidates]

    threads = cfg.get("torch_threads") or max(1, (os.cpu_count() or 4))
    torch.set_num_threads(int(threads))

    for name in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForSequenceClassification.from_pretrained(name)
            mdl.eval()
            n_labels = getattr(mdl.config, "num_labels", 3)
            if n_labels != 3:
                # A binary entailment head cannot express "refutes". Using one
                # would collapse every contradiction into "insufficient".
                print(
                    f"[factshield] skipping {name!r}: {n_labels} labels, need 3")
                continue
            print(f"[factshield] NLI model loaded: {name} ({threads} threads)")
            return tok, mdl
        except Exception as exc:
            print(
                f"[factshield] could not load {name!r}: {type(exc).__name__}: {exc}")

    print("[factshield] NO NLI MODEL LOADED - falling back to the lexical stub.")
    print("[factshield] Stance scores will be keyword-based and mostly wrong.")
    return None


def backend_name() -> str:
    return "transformers" if _load() else "lexical-stub"


_NEG = re.compile(
    r"\b(no|not|never|false|fake|hoax|debunk\w*|myth|untrue|incorrect|"
    r"misleading|unfounded|baseless|refut\w*|disprov\w*)\b",
    re.I,
)
_POS = re.compile(
    r"\b(confirm\w*|verified|accurate|correct|true|established|"
    r"demonstrat\w*|shows?|found)\b",
    re.I,
)


def _stub(premise: str, hypothesis: str) -> dict[str, float]:
    """Deterministic lexical fallback. Crude by design — it exists so the
    pipeline is runnable and testable offline, not to be good."""
    h_terms = {w for w in re.findall(r"\w+", hypothesis.lower()) if len(w) > 3}
    p_terms = {w for w in re.findall(r"\w+", premise.lower()) if len(w) > 3}
    overlap = len(h_terms & p_terms) / max(1, len(h_terms))

    neg = bool(_NEG.search(premise))
    pos = bool(_POS.search(premise))

    if overlap < 0.2:
        return {"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1}
    if neg and not pos:
        return {"entailment": 0.1, "neutral": 0.25, "contradiction": 0.65}
    if pos and not neg:
        return {"entailment": 0.65, "neutral": 0.25, "contradiction": 0.1}
    return {"entailment": 0.3, "neutral": 0.5, "contradiction": 0.2}


def classify(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    """Batch classify (premise, hypothesis) pairs into label probabilities."""
    loaded = _load()
    if loaded is None:
        return [_stub(p, h) for p, h in pairs]

    import torch

    tok, mdl = loaded
    cfg = models().get("nli", {})
    bs = cfg.get("batch_size", 16)
    id2label = {int(k): v.lower() for k, v in mdl.config.id2label.items()}

    # Batches pad to their longest member, so mixing a 1000-character premise
    # with five short ones wastes most of the compute. Sorting by length groups
    # similar sizes together; the original order is restored afterwards.
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))
    ordered = [pairs[i] for i in order]
    scored_out: list[dict[str, float]] = [None] * len(pairs)  # type: ignore

    for i in range(0, len(ordered), bs):
        chunk = ordered[i: i + bs]
        enc = tok(
            [p for p, _ in chunk],
            [h for _, h in chunk],
            return_tensors="pt",
            truncation=True,
            max_length=int(cfg.get("max_length", 256)),
            padding=True,
        )
        with torch.no_grad():
            probs = torch.softmax(mdl(**enc).logits, dim=-1).tolist()
        for k, row in enumerate(probs):
            scored = {id2label.get(j, _LABELS[j]): float(v)
                      for j, v in enumerate(row)}
            scored_out[order[i + k]
                       ] = {k2: scored.get(k2, 0.0) for k2 in _LABELS}

    return scored_out


async def classify_async(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    """Run inference in a worker thread.

    CPU inference is synchronous and takes seconds. Calling `classify` directly
    from an async handler blocks the event loop, which freezes every other
    in-flight request — during a study that means participants stalling each
    other.
    """
    return await asyncio.to_thread(classify, pairs)


def warmup() -> None:
    """Load and run the model once at startup.

    Weights load lazily on first use, so without this the FIRST request of the
    process pays 10-30 seconds of model loading and looks like a hang.
    """
    try:
        classify([("warmup premise sentence", "warmup hypothesis")])
    except Exception:
        pass


def entails(premise: str, hypothesis: str, threshold: float | None = None) -> bool:
    """Used by the grounding layer: is this generated sentence supported by the
    evidence it cites?

    The bar here is deliberately lower than for stance classification. A valid
    summary sentence paraphrases across several quotes, and NLI routinely
    scores a correct paraphrase as `neutral` rather than `entailment`. Holding
    it to the stance threshold deleted every explanation the model produced.

    The safety property that matters is retained: a sentence CONTRADICTED by
    its own cited evidence never survives.
    """
    cfg = models().get("nli", {})
    t = threshold if threshold is not None else cfg.get("grounding_min", 0.30)
    p = classify([(premise, hypothesis)])[0]
    ent, con = p.get("entailment", 0.0), p.get("contradiction", 0.0)
    if con > ent:
        return False
    return ent >= t
