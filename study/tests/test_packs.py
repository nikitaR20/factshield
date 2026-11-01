from __future__ import annotations

import asyncio
import json
import os

import pytest

from study import packs


def test_build_freeze_load(study_paths, fake_retriever):
    asyncio.run(packs.build_all(None, None, fake_retriever))
    p = packs.load_pack("t001")
    assert [x["pid"] for x in p["passages"]] == ["P1", "P2"]
    assert p["category"] == "financial"            # economics -> financial, from config
    assert p["passages"][0]["text_truncated"] in (True, False)
    assert {"url", "title", "published_date", "text", "channel"} <= set(p["passages"][0])
    assert not os.access(packs.pack_path("t001"), os.W_OK) or os.geteuid() == 0


def test_unretrievable_claim_is_flagged(study_paths, fake_retriever):
    asyncio.run(packs.build_all(None, None, fake_retriever))
    assert "unretrievable_by_all_providers" in packs.load_pack("t003")["flags"]


def test_empty_retrieval_is_flagged(study_paths):
    async def nothing(*a, **k):
        return []
    asyncio.run(packs.build_all({"t001"}, None, nothing))
    assert packs.load_pack("t001")["flags"] == ["empty_pack"]


def test_missing_search_key_is_flagged(study_paths, fake_retriever, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY")
    asyncio.run(packs.build_all({"t001"}, None, fake_retriever))
    assert "tavily_key_missing" in packs.load_pack("t001")["flags"]


def test_existing_pack_is_skipped_and_rebuild_refused(study_paths, fake_retriever):
    asyncio.run(packs.build_all({"t001"}, None, fake_retriever))
    h = packs.load_pack("t001")["pack_hash"]
    asyncio.run(packs.build_all({"t001"}, None, fake_retriever))   # no-op
    assert len(packs._manifest_rows("t001")) == 1
    with pytest.raises(packs.PackError, match="delete"):
        asyncio.run(packs.build_all(None, "t001", fake_retriever))
    assert packs.load_pack("t001")["pack_hash"] == h


def test_deleted_pack_needs_explicit_rebuild(study_paths, fake_retriever):
    asyncio.run(packs.build_all({"t001"}, None, fake_retriever))
    old = packs.load_pack("t001")["pack_hash"]
    packs.pack_path("t001").unlink()
    with pytest.raises(packs.PackError, match="--rebuild"):
        asyncio.run(packs.build_all({"t001"}, None, fake_retriever))
    asyncio.run(packs.build_all(None, "t001", fake_retriever))
    rows = packs._manifest_rows("t001")
    assert rows[-1]["rebuild"] is True and rows[-1]["previous_hash"] == old


def test_tampering_is_detected(study_paths, fake_retriever):
    asyncio.run(packs.build_all({"t001"}, None, fake_retriever))
    f = packs.pack_path("t001")
    os.chmod(f, 0o644)
    data = json.loads(f.read_text())
    data["passages"][0]["text"] = "edited"
    f.write_text(json.dumps(data))
    with pytest.raises(packs.PackError, match="modified"):
        packs.load_pack("t001")
    assert packs.verify() == 1


def test_search_concepts_passed_through_and_recorded(study_paths, fake_retriever):
    from study.common import path
    f = path("claims") / "fixture.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    rows[0]["search_concepts"] = [["sales tax", "VAT"], ["Exampleton"]]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    asyncio.run(packs.build_all({"t001", "t002"}, None, fake_retriever))
    by_claim = {c["claim"][:60]: c["concepts"] for c in fake_retriever.calls}
    assert [["sales tax", "VAT"], ["Exampleton"]] in by_claim.values()
    assert None in by_claim.values()                 # claim without the field
    assert packs.load_pack("t001")["retrieval"]["search_concepts"][1] == ["Exampleton"]


def test_truncated_key_findings_warned_not_excluded(study_paths, monkeypatch):
    long_text = "Background sentence. " * 120 + "CONCLUSION: the effect was null."

    async def pubmed_like(*a, **k):
        return [{"url": "https://pubmed.ncbi.nlm.nih.gov/1/", "title": "Review",
                 "text": long_text, "key_findings": "CONCLUSION: the effect was null.",
                 "published_date": "2025-01-01", "source_domain": "pubmed.ncbi.nlm.nih.gov",
                 "channel": "literature"},
                {"url": "https://pubmed.ncbi.nlm.nih.gov/2/", "title": "Short",
                 "text": "Short abstract. It worked.", "key_findings": "It worked.",
                 "published_date": "2025-01-01", "source_domain": "pubmed.ncbi.nlm.nih.gov",
                 "channel": "literature"}]

    asyncio.run(packs.build_all({"t001"}, None, pubmed_like))
    p = packs.load_pack("t001")
    assert [x["key_findings_in_text"] for x in p["passages"]] == [False, True]
    assert p["warnings"] == ["key_findings_truncated:P1"] and p["flags"] == []


def _doc(url, channel, domain):
    from app.schemas import RawDocument
    return RawDocument(url=url, title="t", text="x", source_domain=domain, channel=channel)


def test_study_fanout_limit_skip_and_record(study_paths, monkeypatch):
    from app import retrieval as R
    calls = {}

    async def fake_factcheck(claim, limit=5, **k):
        calls["factcheck"] = limit
        return [_doc("https://snopes.com/a", "prior_check", "snopes.com")]

    async def fake_domain(claim, limit=5, **k):
        calls["domain_search"] = limit
        return [_doc("https://www.snopes.com/fact-check/a", "authority", "snopes.com"),
                _doc("https://reuters.com/x", "authority", "reuters.com"),
                _doc("https://facebook.com/p", "authority", "facebook.com")]

    async def fake_web(claim, limit=5, **k):
        calls["open_web"] = limit
        return [_doc("http://reuters.com/x/", "open_web", "reuters.com"),   # duplicate
                _doc("https://bbc.com/y", "open_web", "bbc.com")]

    monkeypatch.setitem(R.ADAPTERS, "factcheck", fake_factcheck)
    monkeypatch.setattr(R, "domain_search", fake_domain)
    monkeypatch.setattr(R, "open_web", fake_web)
    monkeypatch.setitem(packs.config()["packs"], "fetch_per_channel", 8)
    monkeypatch.setitem(packs.config()["packs"], "exclude_channels", ["prior_check"])
    monkeypatch.setitem(packs.config()["packs"], "exclude_domains", ["snopes.com"])

    asyncio.run(packs.build_all({"t001"}, None))
    p = packs.load_pack("t001")
    assert calls == {"domain_search": 8, "open_web": 8}            # factcheck never called
    assert [x["url"] for x in p["passages"]] == ["https://reuters.com/x", "https://bbc.com/y"]
    f = p["retrieval"]["fanout"]
    assert f["mode"] == "study_fanout" and f["returned_per_channel"] == {"domain_search": 3, "open_web": 2}
    assert p["retrieval"]["excluded_by_config"][0]["source_domain"] == "snopes.com"


def test_fanout_null_uses_product_retrieve(study_paths, monkeypatch):
    from app import retrieval as R
    seen = {}

    async def fake_retrieve(claim, category, jurisdiction=None, concepts=None):
        seen["called"] = True
        return []

    monkeypatch.setattr(R, "retrieve", fake_retrieve)
    monkeypatch.setitem(packs.config()["packs"], "fetch_per_channel", None)
    asyncio.run(packs.build_all({"t001"}, None))
    assert seen and packs.load_pack("t001")["retrieval"]["fanout"]["mode"] == "app.retrieval.retrieve"


def test_one_failing_claim_does_not_stop_build(study_paths, fake_retriever, capsys):
    async def flaky(claim, *a, **k):
        if "9 percent" in claim:
            raise RuntimeError("search timed out")
        return await fake_retriever(claim, *a, **k)

    asyncio.run(packs.build_all(None, None, flaky))
    assert packs.pack_path("t001").exists() and packs.pack_path("t003").exists()
    assert not packs.pack_path("t002").exists()
    assert "1 claims FAILED" in capsys.readouterr().out
    asyncio.run(packs.build_all(None, None, fake_retriever))          # retry fills the gap
    assert packs.pack_path("t002").exists()


def test_summary(study_paths, fake_retriever, capsys):
    async def none_for_refuted(claim, *a, **k):
        return [] if "9 percent" in claim else await fake_retriever(claim, *a, **k)
    asyncio.run(packs.build_all(None, None, none_for_refuted))
    assert packs.summary() == 0
    out = capsys.readouterr().out
    assert "refuted                     1     0     0      1        0" in out
    assert "empty packs: t002" in out


def test_duplicate_config_key_is_refused(tmp_path):
    from study.common import config as cfg_fn
    f = tmp_path / "c.yaml"
    f.write_text("packs:\n  exclude_channels: [prior_check]\n  exclude_channels: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="appears twice"):
        cfg_fn(str(f))


def test_committed_config_has_the_exclusions():
    p = packs.config()["packs"]
    assert p["exclude_channels"] == ["prior_check"]
    assert "snopes.com" in p["exclude_domains"] and "fact-check" in p["exclude_url_patterns"]


def test_url_pattern_and_listed_url_exclusion(study_paths, monkeypatch):
    monkeypatch.setitem(packs.config()["packs"], "exclude_urls", ["https://www.bbc.com/judged-out/"])

    async def docs(*a, **k):
        return [{"url": "https://www.reuters.com/article/world/fact-check-guard-idUS1", "title": "t",
                 "text": "x", "channel": "authority", "source_domain": "reuters.com"},
                {"url": "https://apnews.com/article/archive-fact-checking-8336120181", "title": "t",
                 "text": "x", "channel": "authority", "source_domain": "apnews.com"},
                {"url": "http://bbc.com/judged-out", "title": "t", "text": "x",
                 "channel": "open_web", "source_domain": "bbc.com"},
                {"url": "https://bbc.com/news/keep", "title": "t", "text": "x",
                 "channel": "open_web", "source_domain": "bbc.com"}]

    asyncio.run(packs.build_all({"t001"}, None, docs))
    p = packs.load_pack("t001")
    assert [x["url"] for x in p["passages"]] == ["https://bbc.com/news/keep"]
    why = [e["because"] for e in p["retrieval"]["excluded_by_config"]]
    assert why == ["url contains 'fact-check'", "url contains 'fact-check'", "url listed in exclude_urls"]


def test_audit_sheet_and_targeted_rebuild(study_paths, monkeypatch, capsys):
    import csv
    for k in ("exclude_url_patterns", "exclude_urls"):
        monkeypatch.setitem(packs.config()["packs"], k, [])

    async def docs(claim, *a, **k):
        return [{"url": "https://reuters.com/article/fact-check-x", "title": "t",
                 "text": "Reuters Fact Check rated this claim false.", "channel": "authority",
                 "source_domain": "reuters.com"},
                {"url": "https://bbc.com/news/y", "title": "t", "text": "Plain news.",
                 "channel": "open_web", "source_domain": "bbc.com"}]

    asyncio.run(packs.build_all(None, None, docs))
    monkeypatch.setitem(packs.config()["packs"], "exclude_url_patterns", ["fact-check"])
    assert packs.audit() == 0
    out = capsys.readouterr().out
    assert "--rebuild t001,t002,t003" in out
    with (packs.path("results") / "pack_audit.csv").open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    first = rows[0]
    assert first["suspect"] == "url:fact-check" and first["removed_by_current_config"] == "url contains 'fact-check'"
    assert rows[1]["suspect"] == "" and rows[1]["removed_by_current_config"] == ""
    assert {"claim", "gold_label", "passage_text", "keep_passage", "label_ok"} <= set(first)

    with pytest.raises(packs.PackError, match="delete these packs first"):
        asyncio.run(packs.build_all(None, "t001,t002", docs))
    packs.pack_path("t001").unlink()
    packs.pack_path("t002").unlink()
    asyncio.run(packs.build_all(None, "t001,t002", docs))
    assert [x["url"] for x in packs.load_pack("t001")["passages"]] == ["https://bbc.com/news/y"]
    assert packs._manifest_rows("t002")[-1]["rebuild"] is True
