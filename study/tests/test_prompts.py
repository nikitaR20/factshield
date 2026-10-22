"""Prompt assembly and the verdict parser. The parser cases are the published
parsing rules in executable form: when the pilot shows a new response shape,
add it here first and decide deliberately how it is treated."""

from __future__ import annotations

import pytest

from study import prompts

PACK = {"passages": [
    {"pid": "P1", "url": "https://a.example/x", "title": "A", "source_domain": "a.example",
     "published_date": "2026-09-24", "text": "First passage."},
    {"pid": "P2", "url": "https://b.example/y", "title": "B", "source_domain": "b.example",
     "published_date": None, "text": "Second passage."},
]}


def test_only_the_instruction_differs_between_conditions():
    built = {c: prompts.build(c, "The claim.", PACK) for c in prompts.CONDITIONS}
    systems = {b[0] for b in built.values()}
    assert len(systems) == 1
    tails = {b[1][b[1].index("CLAIM:"):] for b in built.values()}
    assert len(tails) == 1, "claim, pack and options must be byte-identical across conditions"
    instr = {b[1][:b[1].index("CLAIM:")] for b in built.values()}
    assert len(instr) == 3
    shas = {b[2]["prompt_sha"]["material"] for b in built.values()}
    assert len(shas) == 1


def test_closed_book_has_no_passages():
    _, user, _ = prompts.build("closed_book", "The claim.", PACK)
    assert "First passage" not in user and "EVIDENCE" not in user


def test_empty_pack_renders_none():
    _, user, _ = prompts.build("free", "The claim.", None)
    assert "EVIDENCE PASSAGES:\n(none)" in user


@pytest.mark.parametrize("text,cond,expected,rule", [
    ("Supported. The passage says so.", "free", "supported", "first_line"),
    ("**Refuted**\nBecause P1 says otherwise.", "free", "refuted", "first_line"),
    ("Not enough evidence.\nNothing in the pack.", "free", "not_enough_evidence", "first_line"),
    ("Verdict: refuted\nexplanation", "free", "refuted", "verdict_line"),
    ("The claim is not supported by the passages.", "free", "unparseable", "first_line_no_single_label"),
    ("Partly supported.", "free", "unparseable", "first_line_no_single_label"),
    ("Supported or refuted? Hard to say.", "free", "unparseable", "first_line_no_single_label"),
    ("Let me look.\nSupported.", "free", "unparseable", "first_line_no_single_label"),
    ("Verdict: supported\nVerdict: refuted", "free", "unparseable", "verdict_line_ambiguous"),
    ("Refuted [P1]. The passage gives a different figure [P2].", "citation", "refuted", "first_line"),
    ("Reasoning: x\nVerdict: Refuted\nConfidence: high", "format", "refuted", "verdict_line"),
    ("Reasoning: x\nThe answer is refuted.\nConfidence: high", "format", "unparseable", "format_missing_verdict_line"),
    ("", "free", "unparseable", "empty"),
    # Pilot 2026-10-22, gpt-oss: verdict line wrapped in bold, at the end
    ("The sources do not say this [P1][P3].\n\n**Verdict: not enough evidence**.", "citation",
     "not_enough_evidence", "verdict_line"),
    ("Long reasoning.\n**Verdict:** Refuted", "free", "refuted", "verdict_line"),
    ("## Verdict - Supported\nbecause", "free", "supported", "verdict_line"),
    # "verdict:" mid-sentence is not a Verdict field (rule is first_line, not verdict_line)
    ("Text mentioning the verdict: supported, but not as a field", "free", "supported", "first_line"),
    (None, "free", "unparseable", "empty"),
])
def test_parser(text, cond, expected, rule):
    out = prompts.parse(text, cond)
    assert out["parsed_verdict"] == expected
    assert out["parse_rule"] == rule


def test_format_confidence_and_order():
    ok = prompts.parse("Reasoning: r\nVerdict: supported\nConfidence: Medium", "format")
    assert ok["confidence"] == "medium" and ok["format_ok"] is True
    wrong_order = prompts.parse("Verdict: supported\nReasoning: r\nConfidence: low", "format")
    assert wrong_order["parsed_verdict"] == "supported"   # still parsed...
    assert wrong_order["format_ok"] is False               # ...but non-compliance recorded
    bad_conf = prompts.parse("Reasoning: r\nVerdict: supported\nConfidence: 80%", "format")
    assert bad_conf["confidence"] is None


def test_citations_extracted_and_units_split():
    text = "Refuted [P1]. The figure differs [P2, P7]. See https://c.example/z. Done."
    out = prompts.parse(text, "citation")
    assert out["cited_ids"] == ["P1", "P2", "P7"]
    assert out["cited_urls"] == ["https://c.example/z"]
    units = prompts.citation_units(text)
    assert [(u["sentence_index"], u["cited"]) for u in units] == [
        (0, "P1"), (1, "P2"), (1, "P7"), (2, "https://c.example/z")]
    assert prompts.strip_citations("The figure differs [P2, P7].") == "The figure differs ."


def test_prompt_files_are_pinned_not_latest(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        prompts.load("free", "v999")


def test_bold_format_fields():
    out = prompts.parse("**Reasoning:** r\n**Verdict:** supported\n**Confidence:** High", "format")
    assert out["parsed_verdict"] == "supported" and out["confidence"] == "high" and out["format_ok"] is True
