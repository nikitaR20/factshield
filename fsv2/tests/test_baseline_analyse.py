"""Tests for fsv2/baseline_analyse.py. Offline, synthetic rows."""

import json

import fsv2  # noqa: F401  (sets sys.path)
from fsv2 import baseline_analyse as ba
from fsv2 import fourth_model as fm

NEE = "not_enough_evidence"
V = "fsv2-0.2/minicheck:flan-t5-large"
IDS = ["claims_Dataset_A-9001", "claims_Dataset_A-9002", "claims_Dataset_A-9003"]
GOLD = {IDS[0]: NEE, IDS[1]: "refuted", IDS[2]: "supported"}


def fs_rows(cid, final, singles):
    rows = [{"claim_id": cid, "condition": "fs2", "error": None, "pack_hash": "h",
             "model_version": V, "factshield_v2": {"final": final, "samples": []}}]
    rows += [{"claim_id": cid, "condition": "fs2_single", "model": f"fs2:{m}", "model_version": f"{V}/x",
              "parsed_verdict": v, "error": None} for m, v in zip(fm.BASE3, singles)]
    return rows


def plain_row(cid, model, cond, run, text, error=None):
    return {"kind": "main", "claim_id": cid, "model": model, "condition": cond, "run_index": run,
            "live_search": False, "error": error, "response_text": text}


def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_end_to_end(tmp_path, capsys):
    res = tmp_path / "fs.jsonl"
    write(res, fs_rows(IDS[0], "supported", ["supported"] * 3)      # FactShield confident error
          + fs_rows(IDS[1], "refuted", ["refuted"] * 3)
          + fs_rows(IDS[2], NEE, ["supported", NEE, NEE]))
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps(GOLD))
    claude = tmp_path / "claude.jsonl"
    write(claude, [{"claim_id": c, "model_key": "claude", "verdict": v, "error": None}
                   for c, v in zip(IDS, [NEE, "refuted", "supported"])])
    rep = tmp_path / "rep.jsonl"
    write(rep, [{"claim_id": c, "model_key": "openai", "verdict": v, "error": None}
                for c, v in zip(IDS, ["supported", NEE, "supported"])])   # 1 flip on A-9002
    rows = []
    for m in ba.MODELS:
        for cond in ba.CONDITIONS:
            for run in (0, 1):
                rows += [plain_row(IDS[0], m, cond, run, "Reasoning: x\nVerdict: supported\nConfidence: high"),
                         plain_row(IDS[1], m, cond, run, "Reasoning: x\nVerdict: refuted\nConfidence: high"),
                         plain_row(IDS[2], m, cond, run, "Verdict: maybe"),        # unparseable
                         plain_row(IDS[2], m, cond, run, None, error="HTTP 429")]  # error row ignored
    plain = tmp_path / "main.jsonl"
    write(plain, rows)
    new = tmp_path / "new.txt"; new.write_text("")
    ex = tmp_path / "ex.txt"; ex.write_text("# comment, with a comma\n")
    rep_ = ba.analyse(ba.argparse.Namespace(
        results=[str(res)], plain=str(plain), claude_v2=str(claude),
        v2_repeat=[f"openai={rep}"], gold=str(gold), new_ids=str(new), exclude=str(ex),
        summary=str(tmp_path / "s.json")))
    allc = rep_["subsets"]["all"]
    assert allc["plain format openai"]["confidently_wrong"] == 1
    assert allc["plain format openai"]["unparseable"] == 1
    assert allc["FactShield v2 as run"]["confidently_wrong"] == 1
    assert allc["v2 single claude"]["correct"] == 3
    prim = rep_["primary"]["all"]
    assert prim[0]["test"].startswith("CW FactShield vs plain format openai")
    assert (prim[0]["a_only"], prim[0]["b_only"]) == (0, 0)
    assert rep_["noise"]["plain format openai"]["changed"] == 0
    assert rep_["noise"]["v2 single openai"]["changed"] == 1
    assert "without 0 audited-label claims" in rep_["subsets"]
    assert not rep_["missing"]["openai/format/r0"]


def test_load_plain_ignores_live_errors_and_other_models(tmp_path):
    p = tmp_path / "m.jsonl"
    write(p, [plain_row(IDS[0], "openai", "free", 0, "Supported. The passage says so."),
              {**plain_row(IDS[0], "openai", "free", 0, "Refuted."), "live_search": True},
              plain_row(IDS[0], "groq", "free", 0, "Refuted."),
              plain_row(IDS[1], "openai", "free", 0, "Refuted.", error="x")])
    out, rules, ver = ba.load_plain(p)
    assert out[("openai", "free", 0)] == {IDS[0]: "supported"}
    assert ver.startswith("parser")
