from types import SimpleNamespace

import pytest

dspy = pytest.importorskip("dspy")

from persona_gepa.program import PersonaAnswerProgram


def test_program_forward_returns_prediction():
    program = PersonaAnswerProgram()
    expected = SimpleNamespace(answer="ok")
    program.predict = lambda **_kwargs: expected

    result = program.forward("history", "question", "")

    assert result is expected
    assert result.answer == "ok"
