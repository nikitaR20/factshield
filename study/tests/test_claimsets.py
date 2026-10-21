"""CSV claim import. Fixture rows are invented."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from study import claimsets, packs, run
from study.common import config, load_claims, path

FIX = Path(__file__).parent / "fixtures" / "factchecks.csv"


@pytest.fixture
def csv_folder(study_paths):
    folder = path("claims")
    for f in folder.iterdir():
        f.unlink()
    shutil.copy(FIX, folder / "factchecks.csv")
    return folder


def test_columns_rating_map_and_exclusions(csv_folder):
    claims, excluded = claimsets.load_csv_claims(csv_folder)
    assert [c["label"] for c in claims] == ["refuted", "supported", "not_enough_evidence"]
    assert claims[0]["id"] == "factchecks-0001" and claims[1]["id"] == "factchecks-0002"
    assert claims[1]["factcheck"]["whats_true"] == "Council approved it."   # curly apostrophe header
    assert claims[0]["factcheck_rating"] == "False"
    assert claims[0]["set"] == "factchecks" and claims[0]["domain"] == "general"
    assert claims[0]["source_id"] == claims[0]["id"]
    reasons = sorted(e["reason"] for e in excluded)
    assert reasons == sorted([
        "rating excluded by config (not clearly supported/refuted/NEI)",   # Mostly True
        "empty claim",
        "rating not in config table",                                      # Legend-ish
        "duplicate claim text (first: factchecks-0001)",
    ])


def test_set_by_file_and_load_claims(csv_folder, monkeypatch):
    monkeypatch.setitem(config()["claims"]["csv"], "set_by_file", {"factchecks.csv": "B"})
    claims = load_claims()
    assert {c["set"] for c in claims} == {"B"} and len(claims) == 3


def test_check_command_reports(csv_folder, capsys):
    assert claimsets.check(csv_folder) == 0
    out = capsys.readouterr().out
    assert "3 claims kept, 4 rows excluded" in out
    assert "'Rating: Legend-ish'" in out and "EXCLUDED (config)" in out


def test_comma_csv_and_missing_column(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text('claim,rate\n"Exampleton has 3 parks, not 4.",false\n')
    claims, _ = claimsets.read_csv_file(f)
    assert claims[0]["claim"] == "Exampleton has 3 parks, not 4." and claims[0]["label"] == "refuted"
    g = tmp_path / "g.csv"
    g.write_text("statement,verdict\nx,true\n")
    with pytest.raises(ValueError, match="no 'claim' column"):
        claimsets.read_csv_file(g)


def test_runner_refuses_pack_for_changed_claim(csv_folder, fake_retriever):
    asyncio.run(packs.build_all(None, None, fake_retriever))
    kept, _, excluded = run.select_claims(None, include_flagged=True)
    assert len(kept) == 3 and not excluded
    f = csv_folder / "factchecks.csv"
    lines = f.read_text(encoding="utf-8-sig").splitlines()
    lines[1], lines[2] = lines[2], lines[1]          # reorder rows -> ids move
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    kept, _, excluded = run.select_claims(None, include_flagged=True)
    assert {e[0] for e in excluded} == {"factchecks-0001", "factchecks-0002"}
    assert all("different claim text" in e[1] for e in excluded)


def test_pack_verdict_warning_and_channel_exclusion(csv_folder, monkeypatch):
    async def with_factcheck(*a, **k):
        return [{"url": "https://factcheck.example/x", "title": "Fact check",
                 "text": "Exampleton banned cars — Rating: False", "channel": "prior_check",
                 "source_domain": "factcheck.example", "published_date": None},
                {"url": "https://news.example/y", "title": "News", "text": "Some news.",
                 "channel": "open_web", "source_domain": "news.example", "published_date": None}]

    asyncio.run(packs.build_all({"factchecks-0001"}, None, with_factcheck))
    p = packs.load_pack("factchecks-0001")
    assert "contains_factcheck_verdict:P1" in p["warnings"]

    monkeypatch.setitem(config()["packs"], "exclude_channels", ["prior_check"])
    asyncio.run(packs.build_all({"factchecks-0002"}, None, with_factcheck))
    q = packs.load_pack("factchecks-0002")
    assert [x["channel"] for x in q["passages"]] == ["open_web"]
    assert q["retrieval"]["excluded_by_config"][0]["channel"] == "prior_check"
    assert not any(w.startswith("contains_factcheck") for w in q["warnings"])


def test_claims_dir_env_override(csv_folder, monkeypatch, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "x.csv").write_text("claim,rate\nA thing happened.,true\n")
    monkeypatch.setenv("STUDY_CLAIMS_DIR", str(other))
    from study.common import provenance
    assert [c["id"] for c in load_claims()] == ["x-0001"]
    assert provenance()["claims_dir"] == str(other)


def test_mixed_model_one_claim_per_source():
    import pandas as pd
    from study import analyse
    df = analyse.synthetic(1)
    df = df[df["run_index"] == 0].copy()
    df["source_id"] = df["claim_id"]
    m = analyse.fit_mixed(df, "correct", "laplace")
    assert "sd_claim" in set(m["term"]) and "sd_source" not in set(m["term"])
    assert isinstance(m, pd.DataFrame)


def test_mixed_encoding_file_decodes_and_reports(tmp_path, capsys):
    """The Windows error: 'charmap' codec can't decode byte 0x9d."""
    f = tmp_path / "mixed.csv"
    f.write_bytes(
        b"claim,rate\n"
        + "\u201cExampleton\u201d banned cars.,false\n".encode("utf-8")   # UTF-8 curly quotes (..9D)
        + "Exampleton\u2019s caf\xe9 closed.,true\n".encode("cp1252")       # a Windows row
        + b"Stray byte \x9d here.,false\n"                                  # invalid in both
    )
    claims, _ = claimsets.read_csv_file(f)
    assert claims[0]["claim"] == "\u201cExampleton\u201d banned cars."
    assert claims[1]["claim"] == "Exampleton\u2019s caf\xe9 closed."
    assert "\ufffd" in claims[2]["claim"]
    rep = claimsets._DECODE_REPORT[str(f)]
    assert rep == {"encoding": "mixed", "cp1252_lines": [3], "replaced_lines": [4]}
    claimsets.check(tmp_path)
    out = capsys.readouterr().out
    assert "not pure UTF-8" in out and "UNDECODABLE" in out and "mixed-0003" in out
