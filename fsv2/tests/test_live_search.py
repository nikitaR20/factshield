"""Tests for fsv2/live_search.py. No network: provider replies are mocked in the shapes
their docs describe (OpenAI Responses web_search, Claude web_search tool, Gemini grounding)."""

import asyncio
import csv
import json

import httpx
import pytest

import fsv2  # noqa: F401  (sets sys.path)
from fsv2 import live_search as ls

NEE = "not_enough_evidence"


def call(provider_key, cfg, handler, monkeypatch):
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(k, "k")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def go():
        try:
            return await ls.ask(provider_key, cfg, "SYS", "USER", client)
        finally:
            await client.aclose()
    return asyncio.run(go())


def conf(name):
    return ls.settings()["models"][name]


def test_openai_request_and_sources(monkeypatch):
    seen = {}

    def h(req):
        seen["url"] = str(req.url); seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "model": "gpt-6-luna-2026", "status": "completed",
            "output": [
                {"type": "web_search_call", "status": "completed",
                 "action": {"type": "search", "query": "measles 2026",
                            "sources": [{"type": "url", "url": "https://www.reuters.com/a"}]}},
                {"type": "web_search_call", "status": "completed",
                 "action": {"type": "open_page", "url": "https://www.reuters.com/a"}},
                {"type": "message", "content": [{"type": "output_text",
                    "text": '{"verdict": "supported", "sources": ["https://www.reuters.com/a"], "reason": "r"}',
                    "annotations": [{"type": "url_citation", "url": "https://apnews.com/b", "title": "AP"}]}]}],
            "usage": {"input_tokens": 9000, "output_tokens": 120, "total_tokens": 9120,
                      "output_tokens_details": {"reasoning_tokens": 0}}})
    r, info = call("openai", conf("openai"), h, monkeypatch)
    b = seen["body"]
    assert seen["url"] == "https://api.openai.com/v1/responses"
    assert b["tools"] == [{"type": "web_search"}] and b["tool_choice"] == "auto"
    assert b["include"] == ["web_search_call.action.sources"]
    assert b["max_tool_calls"] == 3 and b["reasoning"] == {"effort": "none"} and b["temperature"] == 0
    assert r.error is None and info["n_searches"] == 1 and info["queries"] == ["measles 2026"]
    assert {s["url"] for s in info["sources"]} == {"https://www.reuters.com/a", "https://apnews.com/b"}
    assert ls.parse_answer(r.text)["verdict"] == "supported"


def test_claude_request_pause_turn_and_sources(monkeypatch):
    bodies = []

    def h(req):
        bodies.append(json.loads(req.content))
        if len(bodies) == 1:
            return httpx.Response(200, json={
                "model": "claude-haiku-5-5", "stop_reason": "pause_turn",
                "content": [{"type": "server_tool_use", "id": "s1", "name": "web_search",
                             "input": {"query": "tiger shroff attack claim"}},
                            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                                {"type": "web_search_result", "url": "https://www.bbc.com/x", "title": "BBC",
                                 "page_age": "2026-10-20", "encrypted_content": "e"}]}],
                "usage": {"input_tokens": 5000, "output_tokens": 50,
                          "server_tool_use": {"web_search_requests": 1}}})
        return httpx.Response(200, json={
            "model": "claude-haiku-5-5", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": '{"verdict": "not_enough_evidence", ',
                         "citations": [{"type": "web_search_result_location", "url": "https://www.bbc.com/x",
                                        "title": "BBC", "cited_text": "c", "encrypted_index": "i"}]},
                        {"type": "text", "text": '"sources": [], "reason": "unverified"}'}],
            "usage": {"input_tokens": 7000, "output_tokens": 80,
                      "server_tool_use": {"web_search_requests": 1}}})
    r, info = call("claude", conf("claude"), h, monkeypatch)
    assert bodies[0]["tools"] == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
    assert "thinking" not in bodies[0] and bodies[0]["output_config"] == {"effort": "low"}
    assert "temperature" not in bodies[0]
    assert len(bodies) == 2 and bodies[1]["messages"][-1]["role"] == "assistant"   # continued
    assert r.error is None and info["n_searches"] == 2
    assert info["queries"] == ["tiger shroff attack claim"]
    assert r.tokens["input"] == 12000
    assert ls.parse_answer(r.text) == {"verdict": NEE, "sources_listed": [], "reason": "unverified"}


def test_claude_search_error_recorded(monkeypatch):
    def h(req):
        return httpx.Response(200, json={
            "stop_reason": "end_turn",
            "content": [{"type": "web_search_tool_result", "tool_use_id": "s",
                         "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}},
                        {"type": "text", "text": '{"verdict": "refuted", "sources": [], "reason": "x"}'}],
            "usage": {"input_tokens": 10, "output_tokens": 5}})
    r, info = call("claude", conf("claude"), h, monkeypatch)
    assert info["search_errors"] == ["max_uses_exceeded"] and r.error is None


def test_gemini_grounding(monkeypatch):
    seen = {}

    def h(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "modelVersion": "gemini-3.5-flash-lite",
            "candidates": [{"finishReason": "STOP", "content": {"parts": [
                {"text": "thinking", "thought": True},
                {"text": '```json\n{"verdict": "refuted", "sources": ["https://x.org"], "reason": "r"}\n```'}]},
                "groundingMetadata": {"webSearchQueries": ["q1", "q2"], "groundingChunks": [
                    {"web": {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc",
                             "title": "snopes.com"}}]}}],
            "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 40, "totalTokenCount": 340}})
    r, info = call("gemini", conf("gemini"), h, monkeypatch)
    assert seen["body"]["tools"] == [{"google_search": {}}]
    assert "temperature" not in seen["body"]["generationConfig"]
    assert info["n_searches"] == 2 and r.text.startswith("```json")
    assert ls.parse_answer(r.text)["verdict"] == "refuted"
    rules = ls.factcheck_rules()
    assert ls.is_factcheck(info["sources"][0]["url"], info["sources"][0]["title"], rules)


def test_factcheck_detection_and_parse():
    rules = ls.factcheck_rules()
    assert ls.is_factcheck("https://www.politifact.com/x", None, rules)
    assert ls.is_factcheck("https://www.reuters.com/fact-check/abc", None, rules)
    assert not ls.is_factcheck("https://www.reuters.com/world/abc", "Reuters", rules)
    assert ls.parse_answer("no json here")["verdict"] == "unparseable"
    assert ls.parse_answer('{"verdict": "conflicting evidence"}')["verdict"] == NEE


def _sheet(path, rows):
    cols = ["claim_id", "claim", "csv_rating", "pack_gold_main_study", "pack_override_reason", "check_note",
            "ORIGIN (real/modified)", "WHAT_CHANGED (if modified)",
            "WORLD_LABEL (supported/refuted/not_enough_evidence)", "EVIDENCE_URL", "CHECKED_ON (YYYY-MM-DD)", "NOTES"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for cid, claim, lab, org in rows:
            w.writerow([cid, claim, "", "", "", "", org, "", lab, "", "", ""])


def test_run_refuses_without_labels_and_runs_smoke(tmp_path, monkeypatch, capsys):
    sheet = tmp_path / "labels.csv"
    _sheet(sheet, [("claims_Dataset_B-9001", "c1", "", "real")])
    with pytest.raises(SystemExit, match="no valid WORLD_LABEL"):
        ls.main(["run", "--labels", str(sheet), "--out", str(tmp_path / "o.jsonl")])

    async def fake_ask(m, cfg, system, user, client):
        assert ls.SMOKE_CLAIM in user and "Today's date is 2026-11-01" in user
        r = ls.CallResult(provider=cfg["provider"], model_requested=cfg["model"])
        r.text = '{"verdict": "supported", "sources": [], "reason": "r"}'
        r.tokens = {"input": 10, "output": 5}
        return r, {"queries": ["q"], "sources": [], "n_searches": 1, "search_errors": []}
    monkeypatch.setattr(ls, "ask", fake_ask)
    out = tmp_path / "smoke.jsonl"
    ls.main(["run", "--smoke", "--out", str(out), "--date", "2026-11-01"])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert {r["model_key"] for r in rows} == {"openai", "claude", "gemini"}
    assert all(r["claim_id"] == "smoke-0001" for r in rows)


def test_full_run_hides_verdicts_resumes_and_analyses(tmp_path, monkeypatch, capsys):
    sheet = tmp_path / "labels.csv"
    _sheet(sheet, [("claims_Dataset_B-9001", "claim one", "supported", "real"),
                   ("claims_Dataset_B-9002", "claim two", "refuted", "modified"),
                   ("claims_Dataset_B-9003", "claim three", "not_enough_evidence", "real")])
    answers = {("claim one", "openai"): "supported", ("claim two", "openai"): "supported",
               ("claim three", "openai"): NEE}
    calls = []

    async def fake_ask(m, cfg, system, user, client):
        claim = user.split("CLAIM:\n", 1)[1].strip()
        calls.append((claim, m))
        r = ls.CallResult(provider=cfg["provider"], model_requested=cfg["model"])
        v = answers.get((claim, m), "supported")
        r.text = json.dumps({"verdict": v, "sources": ["https://www.snopes.com/x"] if claim == "claim two" else [],
                             "reason": "r"})
        r.tokens = {"input": 100, "output": 10}
        return r, {"queries": ["q"], "sources": [], "n_searches": 1, "search_errors": []}
    monkeypatch.setattr(ls, "ask", fake_ask)
    out = tmp_path / "live.jsonl"
    ls.main(["run", "--models", "openai", "--labels", str(sheet), "--out", str(out), "--limit", "2"])
    ls.main(["run", "--models", "openai", "--labels", str(sheet), "--out", str(out)])
    assert len(calls) == 3 and len(set(calls)) == 3
    printed = capsys.readouterr().out
    assert "supported" not in printed and "refuted" not in printed
    rep = ls.analyse(ls.argparse.Namespace(labels=str(sheet), live=str(out), results=None,
                                           claude_v2=None, summary=str(tmp_path / "s.json")))
    s = rep["models"]["openai"]["score"]
    assert (s["correct"], s["answered"], s["confidently_wrong"]) == (2, 2, 1)
    assert rep["models"]["openai"]["with_factcheck_source"] == 1
    assert rep["models"]["openai"]["by_origin"]["modified"]["confidently_wrong"] == 1
