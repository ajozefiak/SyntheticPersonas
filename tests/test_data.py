import json

import pytest

dspy = pytest.importorskip("dspy")

from persona_gepa.data import build_examples, build_train_val_examples, load_interviews


def test_load_interviews_new_format(tmp_path):
    data = [
        {"interviewer_question": "Where were you born?", "respondent_answer": "Paris."},
        {"interviewer_question": "Favorite food?", "respondent_answer": "Baguettes."},
    ]
    path = tmp_path / "interview.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    interviews = load_interviews(str(path))
    assert len(interviews) == 1
    assert interviews[0][0]["q"] == "Where were you born?"
    assert interviews[0][0]["a"] == "Paris."


def test_load_interviews_missing_keys(tmp_path):
    data = [{"interviewer_question": "Where were you born?"}]
    path = tmp_path / "interview.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="respondent_answer"):
        load_interviews(str(path))


def test_temporal_split_correctness():
    interviews = [
        [
            {"interviewer_question": "Q1", "respondent_answer": "A1"},
            {"interviewer_question": "Q2", "respondent_answer": "A2"},
            {"interviewer_question": "Q3", "respondent_answer": "A3"},
            {"interviewer_question": "Q4", "respondent_answer": "A4"},
            {"interviewer_question": "Q5", "respondent_answer": "A5"},
        ]
    ]

    train, val = build_train_val_examples(interviews, val_ratio=0.4)

    assert [ex.question for ex in train] == ["Q1", "Q2", "Q3"]
    assert [ex.question for ex in val] == ["Q4", "Q5"]
    assert val[0].history == (
        "Q: Q1\nA: A1\n"
        "Q: Q2\nA: A2\n"
        "Q: Q3\nA: A3\n"
    )


def test_temporal_split_edge_cases():
    interviews = [
        [
            {"interviewer_question": "Q1", "respondent_answer": "A1"},
        ],
        [
            {"interviewer_question": "Q2", "respondent_answer": "A2"},
            {"interviewer_question": "Q3", "respondent_answer": "A3"},
        ],
        [
            {"interviewer_question": "Q4", "respondent_answer": "A4"},
            {"interviewer_question": "Q5", "respondent_answer": "A5"},
            {"interviewer_question": "Q6", "respondent_answer": "A6"},
        ],
    ]

    train, val = build_train_val_examples(interviews, val_ratio=0.2)
    assert len(train) == 4
    assert len(val) == 2


def test_build_examples_backwards_compat():
    interviews = [
        [
            {"q": "Where were you born?", "a": "Paris."},
            {"q": "Favorite food?", "a": "Baguettes."},
        ]
    ]

    examples = build_examples(interviews)
    assert len(examples) == 2
    assert examples[0].question == "Where were you born?"
