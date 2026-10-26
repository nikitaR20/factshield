"""The local citation checker (design 5b and 6).

Real checker: MiniCheck-Flan-T5-Large through the authors' own package
(Apache-2.0):

    pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
    python -c "import nltk; nltk.download('punkt_tab')"

It answers one question: does this text support this statement? Probability
above 0.5 (the package's threshold) counts as supported. It cannot say
"contradicts"; refuted verdicts are checked by confirming the passage supports
the stated counter-fact (see decide.py).

Stub checker: content-word overlap. It is only for testing the plumbing
offline; every row records which checker produced it, and run_packs refuses
to combine real models with the stub checker.
"""

from __future__ import annotations

import re
import time

from .settings import cfg

_STOP = set("""a an the of to in on for and or but is are was were be been being it its this that
these those with as at by from has have had not no do does did will would can could
should may might than then there their they he she his her them we you i our your""".split())


class StubChecker:
    name = "stub-overlap (NOT MiniCheck)"

    def score(self, docs: list[str], statements: list[str]) -> list[float]:
        out = []
        for d, s in zip(docs, statements):
            dw = set(re.findall(r"[a-z0-9]+", d.lower()))
            sw = [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]
            share = sum(w in dw for w in sw) / len(sw) if sw else 0.0
            out.append(0.9 if share >= 0.6 else 0.1)
        return out


class MiniCheckChecker:
    def __init__(self) -> None:
        from minicheck.minicheck import MiniCheck  # needs the package above
        c = cfg()["checker"]
        self.name = f"minicheck:{c['model']}"
        self._chunk = c.get("chunk_size")
        t0 = time.perf_counter()
        self._m = MiniCheck(model_name=c["model"], cache_dir="./ckpts")
        self.load_seconds = round(time.perf_counter() - t0, 1)

    def score(self, docs: list[str], statements: list[str]) -> list[float]:
        if not docs:
            return []
        _, probs, _, _ = self._m.score(docs=docs, claims=statements, chunk_size=self._chunk)
        return [float(p) for p in probs]


def load(stub: bool):
    return StubChecker() if stub else MiniCheckChecker()


def supported(prob: float) -> bool:
    return prob > cfg()["checker"]["threshold"]
