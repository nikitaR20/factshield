"""The study must never import FactShield's verdict logic.

Two independent checks:

1. STATIC. Parse every .py file under study/ (tests included), resolve each
   import to a module, and follow imports into backend/app transitively. Fail
   if a forbidden module is reachable, if any backend module outside the
   allowlist is reachable, or if a study module imports dynamically (which no
   static check can see through).

2. RUNTIME. Import every study module in a fresh interpreter and list which
   `app.*` modules actually ended up in sys.modules.

The static check is the one to cite: it covers code paths a runtime import
never executes (imports inside functions, rarely-used branches). The runtime
check guards against the static resolver itself being wrong.

Run with `-s` to print the resolved graph for the thesis appendix.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parent
BACKEND = REPO / "backend"

# The four modules the spec names, plus two it does not:
#   app.pipeline.grading  assigns stance and tier to evidence — FactShield's own
#                         judgement of what a source says. Not named in the spec,
#                         but it is verdict logic in everything but name.
#   app.providers.llm     the product's model client. The study has its own.
FORBIDDEN = {
    "app.pipeline.scoring",
    "app.pipeline.evidence_state",
    "app.pipeline.stages",
    "app.pipeline.grading",
    "app.providers.llm",
}
# Names inside app.pipeline.stages the spec lists explicitly. Importing any of
# them requires importing stages, so the module rule already covers them; they
# are checked by name as well so a failure message names the function.
FORBIDDEN_NAMES = {"triage", "fast_path_verdict", "explain", "ground", "describe_evidence"}

# Everything from backend/app the study may reach, directly or transitively.
# Packages are listed because importing app.providers.nli executes
# app/__init__.py and app/providers/__init__.py.
ALLOWED_APP = {
    "app",
    "app.config",
    "app.schemas",
    "app.retrieval",
    "app.providers",
    "app.providers.nli",
}

DYNAMIC_IMPORT_CALLS = {"__import__", "import_module", "exec", "eval", "run_module", "run_path"}


# ------------------------------------------------------------------ resolver

def _module_file(name: str) -> Path | None:
    """Map a dotted name to its source file, for modules we own."""
    parts = name.split(".")
    if parts[0] == "app":
        base = BACKEND
    elif parts[0] == "study":
        base = REPO
    else:
        return None
    p = base.joinpath(*parts)
    if (p / "__init__.py").exists():
        return p / "__init__.py"
    if p.with_suffix(".py").exists():
        return p.with_suffix(".py")
    return None


def _module_name(file: Path) -> str:
    if file.is_relative_to(BACKEND / "app"):
        rel = file.relative_to(BACKEND)
    else:
        rel = file.relative_to(REPO)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_of(name: str, file: Path) -> str:
    return name if file.name == "__init__.py" else name.rpartition(".")[0]


def _imports(file: Path) -> tuple[list[tuple[str, int, str]], list[tuple[int, str]]]:
    """Return ([(module, line, imported_name)], [(line, dynamic_call)])."""
    tree = ast.parse(file.read_text(), filename=str(file))
    me = _module_name(file)
    pkg = _package_of(me, file)
    found: list[tuple[str, int, str]] = []
    dynamic: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                # `import a.b.c` executes a, a.b and a.b.c
                parts = a.name.split(".")
                for i in range(1, len(parts) + 1):
                    found.append((".".join(parts[:i]), node.lineno, a.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = pkg.split(".")
                if node.level > 1:
                    base_parts = base_parts[: -(node.level - 1)]
                base = ".".join(base_parts)
                mod = f"{base}.{node.module}" if node.module else base
            else:
                mod = node.module or ""
            parts = mod.split(".")
            for i in range(1, len(parts) + 1):
                found.append((".".join(parts[:i]), node.lineno, mod))
            for a in node.names:
                # `from app.pipeline import stages` imports a submodule
                sub = f"{mod}.{a.name}"
                if _module_file(sub) is not None:
                    found.append((sub, node.lineno, sub))
                found.append((f"{mod}:{a.name}", node.lineno, a.name))
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
            if name in DYNAMIC_IMPORT_CALLS:
                dynamic.append((node.lineno, name))
    return found, dynamic


def build_graph() -> tuple[dict[str, set[str]], dict[str, list[tuple[int, str]]], dict[str, set[str]]]:
    """Walk from every study module. Returns (edges, dynamic_calls, names_imported)."""
    edges: dict[str, set[str]] = {}
    dynamic: dict[str, list[tuple[int, str]]] = {}
    names: dict[str, set[str]] = {}
    queue = [f for f in STUDY.rglob("*.py") if "__pycache__" not in f.parts]
    seen: set[Path] = set()
    while queue:
        f = queue.pop()
        if f in seen:
            continue
        seen.add(f)
        me = _module_name(f)
        found, dyn = _imports(f)
        if dyn and me.startswith("study"):
            dynamic[me] = dyn
        for target, _line, _raw in found:
            if ":" in target:
                mod, _, nm = target.partition(":")
                names.setdefault(me, set()).add(f"{mod}.{nm}")
                continue
            edges.setdefault(me, set()).add(target)
            tf = _module_file(target)
            if tf is not None and tf not in seen:
                queue.append(tf)
    return edges, dynamic, names


def reachable_from_study(edges: dict[str, set[str]]) -> dict[str, list[str]]:
    """Every module reachable from study/, with one shortest import chain each."""
    chains: dict[str, list[str]] = {}
    frontier = [m for m in edges if m.startswith("study")]
    for m in frontier:
        chains[m] = [m]
    i = 0
    while i < len(frontier):
        cur = frontier[i]
        i += 1
        for nxt in sorted(edges.get(cur, ())):
            if nxt not in chains:
                chains[nxt] = chains[cur] + [nxt]
                frontier.append(nxt)
    return chains


# ------------------------------------------------------------------ tests

def test_no_forbidden_module_reachable():
    edges, _, _ = build_graph()
    chains = reachable_from_study(edges)
    app_reached = {m: c for m, c in chains.items() if m == "app" or m.startswith("app.")}

    print("\n=== backend modules reachable from study/ ===")
    for m in sorted(app_reached):
        print(f"  {m:<22} via {' -> '.join(app_reached[m])}")

    hits = {m: c for m, c in app_reached.items()
            if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)}
    assert not hits, "study/ reaches forbidden verdict logic:\n" + "\n".join(
        f"  {m}: {' -> '.join(c)}" for m, c in hits.items())

    outside = {m: c for m, c in app_reached.items() if m not in ALLOWED_APP}
    assert not outside, "study/ reaches backend modules outside the allowlist:\n" + "\n".join(
        f"  {m}: {' -> '.join(c)}" for m, c in outside.items())
    print(f"PASS: {len(app_reached)} backend modules reached, all in the allowlist; "
          f"none of {sorted(FORBIDDEN)} reachable.")


def test_no_forbidden_names_imported():
    _, _, names = build_graph()
    bad = {m: sorted(n for n in ns if n.rsplit(".", 1)[-1] in FORBIDDEN_NAMES and n.startswith("app."))
           for m, ns in names.items() if m.startswith("study")}
    bad = {m: v for m, v in bad.items() if v}
    assert not bad, f"study/ imports forbidden pipeline names: {bad}"


def test_no_dynamic_imports_in_study():
    """A dynamic import is invisible to the static check, so it is not allowed."""
    _, dynamic, _ = build_graph()
    assert not dynamic, f"dynamic import/exec in study/ defeats the boundary check: {dynamic}"


def test_runtime_sys_modules():
    mods = sorted(_module_name(f) for f in STUDY.glob("*.py") if f.name != "__init__.py")
    code = (
        "import importlib, json, sys\n"
        f"for m in {mods!r}: importlib.import_module(m)\n"
        "print(json.dumps(sorted(k for k in sys.modules if k == 'app' or k.startswith('app.'))))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    print(f"\n=== runtime: app modules in sys.modules after importing {len(mods)} study modules ===")
    print("  " + ", ".join(sorted(loaded)))
    assert not loaded & FORBIDDEN, f"forbidden modules loaded at runtime: {loaded & FORBIDDEN}"
    assert loaded <= ALLOWED_APP, f"modules outside allowlist loaded: {loaded - ALLOWED_APP}"


def test_checker_catches_a_violation(tmp_path, monkeypatch):
    """The check must be able to fail. Plant a transitive violation and see it caught."""
    planted_a = STUDY / "_planted_a.py"
    planted_b = STUDY / "_planted_b.py"
    try:
        planted_a.write_text("from . import _planted_b\n")
        planted_b.write_text("from app.pipeline import stages\n")
        edges, _, _ = build_graph()
        chains = reachable_from_study(edges)
        assert "app.pipeline.stages" in chains
        assert chains["app.pipeline.stages"][-2:] == ["study._planted_b", "app.pipeline.stages"]
    finally:
        planted_a.unlink(missing_ok=True)
        planted_b.unlink(missing_ok=True)
