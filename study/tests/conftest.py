"""Shared fixtures: point every study path at a temp dir and use fixture claims."""

from __future__ import annotations

from pathlib import Path

import pytest

from study.common import config

FIXTURES = Path(__file__).parent / "fixtures"



def fake_docs(claim: str):
    return [
        {"url": "https://news.example/exampleton-tax", "title": "Exampleton raises sales tax",
         "text": "Exampleton's council voted on 23 September 2026 to raise the sales tax to 7 percent. " * 3,
         "published_date": "2026-09-24", "source_domain": "news.example", "channel": "open_web"},
        {"url": "https://gov.example/budget", "title": "Budget statement",
         "text": "The council budget statement lists the new 7 percent rate.",
         "published_date": None, "source_domain": "gov.example", "channel": "authority"},
    ]


@pytest.fixture
def study_paths(tmp_path, monkeypatch):
    paths = config()["paths"]
    claims_dir = tmp_path / "claims"
    claims_dir.mkdir()
    (claims_dir / "fixture.jsonl").write_text((FIXTURES / "claims.jsonl").read_text())
    monkeypatch.setitem(paths, "claims", str(claims_dir))
    monkeypatch.setitem(paths, "packs", str(tmp_path / "packs"))
    monkeypatch.setitem(paths, "responses", str(tmp_path / "responses"))
    monkeypatch.setitem(paths, "results", str(tmp_path / "results"))
    monkeypatch.setenv("TAVILY_API_KEY", "fixture")
    return tmp_path


@pytest.fixture
def fake_retriever():
    async def r(claim, category, jurisdiction, concepts=None):
        r.calls.append({"claim": claim, "category": category, "concepts": concepts})
        return fake_docs(claim)
    r.calls = []
    return r


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: fits a mixed model (~5 s)")
