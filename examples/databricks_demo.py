from persona_gepa.config import PersonaGEPAConfig
from persona_gepa.data import build_train_val_examples, format_history
from persona_gepa.infer import run_inference
from persona_gepa.optimize import run_optimization


# In Databricks, load your processed interview data here.
interviews = [
    [
        {"q": "Where did you grow up?", "a": "I grew up in Austin."},
        {"q": "What do you enjoy doing?", "a": "I like hiking and reading."},
    ]
]

trainset, valset = build_train_val_examples(interviews, val_ratio=0.5, seed=7)

config = PersonaGEPAConfig(
    output_dir="artifacts/persona_gepa_demo",
    cache_dir=".cache/dspy",
    num_threads=4,
)

program, artifact_path, report = run_optimization(config, trainset, valset)
print("Saved artifact:", artifact_path)
print("Validation report:", report)

history = format_history(interviews[0][:1])
question = interviews[0][1]["q"]
answer = run_inference(config, artifact_path, history, question)
print("Answer:", answer)
