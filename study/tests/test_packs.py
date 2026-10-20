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
    async def nothing(*a):
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
