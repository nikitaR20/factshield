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
        with _load_on_cpu():
            self._m = MiniCheck(model_name=c["model"], cache_dir="./ckpts")
        self.load_seconds = round(time.perf_counter() - t0, 1)

    def score(self, docs: list[str], statements: list[str]) -> list[float]:
        if not docs:
            return []
        _, probs, _, _ = self._m.score(docs=docs, claims=statements, chunk_size=self._chunk)
        return [float(p) for p in probs]


class _load_on_cpu:
    """Load MiniCheck's model entirely into ordinary memory.

    MiniCheck's own code passes device_map="auto" to from_pretrained. That lets
    the accelerate library place weights by its estimate of free RAM, and when
    the machine is busy (a browser open next to the server) it tries to offload
    part of the model to disk and fails: "weights offloaded to the disk ...
    provide an offload_folder" (seen 2026-10-26). Flan-T5-Large needs about
    3.2 GB, which fits comfortably in 16 GB, so for the duration of the load
    device_map is dropped and the model is read straight into CPU memory.
    Nothing about the model or its outputs changes; only where it is loaded.
    """

    _classes = ("AutoModelForSeq2SeqLM", "AutoModelForSequenceClassification")

    def __enter__(self):
        import transformers
        self._patched = []
        for name in self._classes:
            cls = getattr(transformers, name, None)
            if cls is None or "from_pretrained" in cls.__dict__:
                continue
            original = cls.from_pretrained

            def patched(*args, _orig=original, **kwargs):
                kwargs.pop("device_map", None)
                kwargs.setdefault("low_cpu_mem_usage", True)
                return _orig(*args, **kwargs)

            cls.from_pretrained = patched
            self._patched.append(cls)
        return self

    def __exit__(self, *exc):
        for cls in self._patched:
            del cls.from_pretrained  # back to the inherited classmethod
        return False


def load(stub: bool):
    return StubChecker() if stub else MiniCheckChecker()


def supported(prob: float) -> bool:
    return prob > cfg()["checker"]["threshold"]
