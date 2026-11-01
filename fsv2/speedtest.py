"""Load MiniCheck-Flan-T5-Large and time it on real pack passages.

    python -m fsv2.speedtest            # 20 (passage, statement) pairs

Statements are each pack's own claim text, so the probabilities printed are
NOT results of anything; this only measures that the checker loads and how
long one check takes on this machine.
"""

from __future__ import annotations

import glob
import json
import time

from . import checker as chk


def main(n: int = 20) -> None:
    pairs = []
    for p in sorted(glob.glob("study/packs/claims_*.json")):
        pk = json.load(open(p, encoding="utf-8"))
        for ps in pk["passages"][:1]:
            pairs.append((ps["text"], pk["claim"]))
        if len(pairs) >= n:
            break
    c = chk.MiniCheckChecker()
    print(f"loaded {c.name} in {c.load_seconds}s")
    c.score([pairs[0][0]], [pairs[0][1]])  # warm-up
    t0 = time.perf_counter()
    c.score([d for d, _ in pairs], [s for _, s in pairs])
    dt = time.perf_counter() - t0
    print(f"{len(pairs)} checks in {dt:.1f}s -> {dt / len(pairs):.2f}s per check")


if __name__ == "__main__":
    main()
