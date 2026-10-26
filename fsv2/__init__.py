"""FactShield v2 — thesis core (stages 3–6 on frozen packs).

Design: claude/factshield-v2-design.md in the Factshield project.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT, _ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
