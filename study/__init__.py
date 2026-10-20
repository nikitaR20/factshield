"""The thesis study: how commercial models fact-check one claim three ways.

This package is deliberately separate from the FactShield product. It may
reuse `app.retrieval`, `app.providers.nli`, `app.config` and `app.schemas`,
and nothing else from `backend/app`. `study/tests/test_import_boundary.py`
enforces that on every test run.

Putting `backend/` on the path here mirrors what `eval/` does, so
`python -m study.<module>` works from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
