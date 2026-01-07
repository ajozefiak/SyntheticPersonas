import pytest

dspy = pytest.importorskip("dspy")

from persona_gepa.judge import parse_judge_output


def test_parse_judge_output_json():
    raw = '{"accuracy": 0.9, "faithfulness": 0.8, "tone": 0.7, "style": 0.6, "feedback": "Good."}'
    judgment = parse_judge_output(raw)
    assert judgment.accuracy == 0.9
    assert judgment.faithfulness == 0.8
    assert judgment.tone == 0.7
    assert judgment.style == 0.6
    assert judgment.feedback == "Good."


def test_parse_judge_output_with_text():
    raw = "Result: {\"accuracy\": 1, \"faithfulness\": 0.5, \"tone\": 0.5, \"style\": 0.5, \"feedback\": \"OK\"}"
    judgment = parse_judge_output(raw)
    assert judgment.accuracy == 1.0
    assert judgment.feedback == "OK"


def test_parse_judge_output_fallback():
    raw = "accuracy:0.2 faithfulness:0.3 tone:0.4 style:0.5 feedback:Needs work"
    judgment = parse_judge_output(raw)
    assert judgment.accuracy == 0.2
    assert judgment.faithfulness == 0.3
    assert judgment.tone == 0.4
    assert judgment.style == 0.5
