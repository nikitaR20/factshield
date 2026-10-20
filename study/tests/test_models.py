"""Adapter behaviour, offline. httpx.MockTransport stands in for the providers."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from study import models


@pytest.fixture(autouse=True)
def _keys_and_fast_sleep(monkeypatch):
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(k, "test-key")
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(models.asyncio, "sleep", fake_sleep)
    cfg = models.config()
    for key in ("openai", "gemini", "groq"):
        monkeypatch.setitem(cfg["models"][key], "model", f"{key}-test-model")
    return sleeps


def run(coro):
    return asyncio.run(coro)


def client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def chat_ok(content="Verdict: supported", model="served-model-2026-10-01"):
    return {"model": model, "system_fingerprint": "fp_x",
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}}


def test_retries_429_then_succeeds_and_records_served_version(_keys_and_fast_sleep):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "1"}, text="slow down")
        return httpx.Response(200, json=chat_ok())

    r = run(models.complete("openai", "sys", "user", client=client_for(handler)))
    assert r.error is None
    assert r.attempts == 3
    assert len(_keys_and_fast_sleep) == 2 and all(s >= 1 for s in _keys_and_fast_sleep)
    assert r.model_requested == "openai-test-model"
    assert r.model_version == "served-model-2026-10-01"   # from the response, not the request
    assert r.tool_calls == []
    assert r.tools_attached == []


def test_404_fails_fast():
    def handler(req):
        return httpx.Response(404, json={"error": {"message": "model not found"}})

    r = run(models.complete("groq", "s", "u", client=client_for(handler)))
    assert r.attempts == 1
    assert r.http_status == 404
    assert "HTTP 404" in r.error


def test_5xx_exhausts_retries():
    def handler(req):
        return httpx.Response(503, text="unavailable")

    r = run(models.complete("groq", "s", "u", client=client_for(handler)))
    assert r.attempts == models.config()["retry"]["max_attempts"]
    assert "gave up" in r.error


def test_closed_book_request_sends_no_tools_and_temperature_recorded():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=chat_ok())

    r = run(models.complete("openai", "s", "u", client=client_for(handler)))
    assert "tools" not in seen
    assert seen["temperature"] == 0 and r.temperature_sent == 0


def test_null_temperature_is_not_sent(monkeypatch):
    monkeypatch.setitem(models.config()["models"]["openai"], "temperature", None)
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=chat_ok())

    r = run(models.complete("openai", "s", "u", client=client_for(handler)))
    assert "temperature" not in seen and r.temperature_sent is None


def test_gemini_grounding_metadata_is_flagged_and_thoughts_excluded():
    def handler(req):
        assert req.headers["x-goog-api-key"] == "test-key"
        assert "key=" not in str(req.url)
        return httpx.Response(200, json={
            "modelVersion": "gemini-served-001",
            "candidates": [{
                "content": {"parts": [{"text": "thinking...", "thought": True},
                                      {"text": "Verdict: refuted"}]},
                "finishReason": "STOP",
                "groundingMetadata": {"webSearchQueries": ["x"]},
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "thoughtsTokenCount": 7},
        })

    r = run(models.complete("gemini", "s", "u", client=client_for(handler)))
    assert r.text == "Verdict: refuted"
    assert r.reasoning_text == "thinking..."
    assert r.model_version == "gemini-served-001"
    assert any(t["type"] == "groundingMetadata" for t in r.tool_calls)
    assert r.tokens["reasoning"] == 7


def test_groq_executed_tools_flagged():
    body = chat_ok()
    body["choices"][0]["message"]["executed_tools"] = [{"type": "browser_search", "arguments": "{}"}]
    r = run(models.complete("groq", "s", "u", client=client_for(lambda req: httpx.Response(200, json=body))))
    assert any(t["type"] == "executed_tools" for t in r.tool_calls)


def test_search_model_refused_without_a_call(monkeypatch):
    monkeypatch.setitem(models.config()["models"]["openai"], "model", "gpt-5-search-api")

    def handler(req):
        raise AssertionError("must not be called")

    r = run(models.complete("openai", "s", "u", client=client_for(handler)))
    assert "search product" in r.error and r.attempts == 0


def test_change_me_refused(monkeypatch):
    monkeypatch.setitem(models.config()["models"]["gemini"], "model", "CHANGE_ME")
    r = run(models.complete("gemini", "s", "u"))
    assert "CHANGE_ME" in r.error


def test_missing_key_is_an_error_not_a_crash(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    r = run(models.complete("groq", "s", "u"))
    assert r.error == "GROQ_API_KEY not set" and r.temperature_sent is None


def test_live_search_attaches_tool_on_openai_responses_api():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "model": "served", "status": "completed",
            "output": [{"type": "web_search_call", "id": "ws1", "action": {"type": "search"}},
                       {"type": "message", "content": [{"type": "output_text", "text": "Verdict: supported",
                                                        "annotations": [{"type": "url_citation", "url": "https://x"}]}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    r = run(models.complete("openai", "s", "u", live_search=True, client=client_for(handler)))
    assert seen["url"].endswith("/v1/responses")
    assert seen["body"]["tools"] == [{"type": "web_search"}]
    assert r.tools_attached == [{"type": "web_search"}]
    assert {t["type"] for t in r.tool_calls} >= {"web_search_call", "annotations"}


def test_stub_is_deterministic_and_marked():
    a = run(models.complete("stub", "s", "claim [P1] [P2]"))
    b = run(models.complete("stub", "s", "claim [P1] [P2]"))
    assert a.text == b.text and a.model_version == "stub-offline-v1" and a.raw["stub"] is True
