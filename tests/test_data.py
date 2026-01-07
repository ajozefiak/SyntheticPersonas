import pytest

dspy = pytest.importorskip("dspy")

from persona_gepa.data import build_examples, format_history


def test_format_history_and_examples():
    interviews = [
        [
            {"q": "Where were you born?", "a": "In Paris."},
            {"q": "Favorite food?", "a": "Baguettes."},
        ]
    ]

    history = format_history(interviews[0][:1])
    assert history == "Q: Where were you born?\nA: In Paris.\n"

    examples = build_examples(interviews)
    assert len(examples) == 2
    assert examples[0].history == ""
    assert examples[0].question == "Where were you born?"
    assert examples[0].answer == "In Paris."
    assert examples[1].history == history
    assert examples[1].question == "Favorite food?"
    assert examples[1].answer == "Baguettes."
