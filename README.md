# SyntheticPersonas

GEPA + DSPy utilities for optimizing synthetic persona system prompts from interview transcripts.

## Install

From the repo root:

```
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Set your API key:

```
export OPENAI_API_KEY=...
```

## Data Format

Training data expects a list of interviews. Each interview is a list of turns with `interviewer_question` and `respondent_answer` keys:

```json
[
  [
    {"interviewer_question": "Where did you grow up?", "respondent_answer": "I grew up in Austin."},
    {"interviewer_question": "What do you enjoy doing?", "respondent_answer": "I like hiking and reading."}
  ]
]
```

JSONL is also supported (one interview per line).

## Quickstart

Optimize a persona prompt:

```
python -m persona_gepa.optimize --data-path data/interviews.json --output-dir artifacts/persona_gepa
```

Run inference with an optimized artifact:

```
python -m persona_gepa.infer --artifact-path artifacts/persona_gepa/persona_gepa_artifact.json \
  --history "Q: Where did you grow up?\nA: I grew up in Austin.\n" \
  --question "What do you enjoy doing?"
```

## Databricks Notes

See `examples/databricks_demo.py` for a notebook-friendly flow:

1. Load processed interviews into Python.
2. Build train/val sets with `build_train_val_examples`.
3. Run `run_optimization` with `num_threads` for parallel evaluation.
4. Save artifacts and run inference with `run_inference`.

## Tests

```
pytest
```
