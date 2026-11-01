"""Tests for fsv2/overstatement.py (no model calls, no real data)."""

import csv

import fsv2  # noqa: F401  (sets sys.path)
from fsv2 import overstatement as ov

NEE, S, R = ov.NEE, ov.SUP, ov.REF


def test_os_counts_split_and_costs():
    gold = {"a": NEE, "b": NEE, "c": NEE, "d": S, "e": R}
    pred = {"a": S, "b": R, "c": NEE, "d": NEE, "e": S}
    s = ov.os_counts(pred, gold, list(gold))
    assert (s["os"], s["os_supported"], s["os_refuted"]) == (2, 1, 1)
    assert s["us"] == 1 and s["wd"] == 1 and s["correct"] == 1
    assert s["os_rate"] == round(2 / 3, 3)


def test_contrast_counts_discordant_nee_only():
    gold = {"a": NEE, "b": NEE, "c": NEE, "d": S}
    A = {"a": S, "b": S, "c": NEE, "d": NEE}
    B = {"a": S, "b": NEE, "c": R, "d": S}
    t = ov.contrast("x", A, B, gold, list(gold))
    assert (t["os_A"], t["os_B"], t["A_only"], t["B_only"]) == (2, 2, 1, 1)
    assert t["us_A"] == 1 and t["us_B"] == 0 and t["p_exact"] == 1.0


def test_ranks_ties_and_spearman():
    assert ov.ranks([3, 1, 1, 2]) == [4.0, 1.5, 1.5, 3.0]
    rho, p = ov.spearman_perm([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], reps=500)
    assert rho == 1.0 and p < 0.05
    rho0, _ = ov.spearman_perm([1, 1, 1], [1, 2, 3], reps=10)
    assert rho0 == 0.0


def test_load_codes_final_overrides_draft(tmp_path):
    f = tmp_path / "c.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["claim_id", "primary_code", "secondary_code", "self_source", "FINAL_code"])
        w.writerow(["x", "a", "e", "Y", ""])
        w.writerow(["y", "N", "", "n", "l"])
        w.writerow(["z", "", "", "", ""])
    c = ov.load_codes(f)
    assert c["x"]["code"] == "A" and c["x"]["secondary"] == "E" and c["x"]["self_source"] == "y"
    assert c["x"]["checked"] is False and c["x"]["draft"] == "A"
    assert c["y"]["code"] == "L" and c["y"]["checked"] is True and c["y"]["draft"] == "N"
    assert "z" not in c


def test_features_on_a_tiny_pack():
    pack = {"claim": "Acme announced a new phone", "passages": [
        {"pid": "P1", "url": "https://reuters.com/a", "source_domain": "reuters.com", "published_date": "2026-01-01",
         "text": "Acme announced a new phone on Monday, the company said."},
        {"pid": "P2", "url": "https://example.org/b", "source_domain": "example.org", "text": "Unrelated text about weather."}]}
    f = ov.features(pack)
    assert f["n_passages"] == 2 and f["n_orgs"] == 2
    assert f["attrib_share"] == 0.5 and f["dated_share"] == 0.5
    assert f["max_overlap"] > 0.5 and f["mean_rating"] is not None
