import os

from persona_gepa.config import PersonaGEPAConfig
from persona_gepa.data import build_train_val_examples, format_history
from persona_gepa.infer import run_inference
from persona_gepa.optimize import run_optimization


# In Databricks, load your processed interview data here.
interviews = [
    [
        {
            "interviewer_question": "Where did you grow up?",
            "respondent_answer": "I grew up in Austin.",
        },
        {
            "interviewer_question": "What do you enjoy doing?",
            "respondent_answer": "I like hiking and reading.",
        },
    ]
]

# In Databricks, set your OpenAI-compatible API credentials and base URL.
# Example (use your secret scope/key):
# api_key = dbutils.secrets.get(scope="your-scope", key="your-key")
# os.environ["OPENAI_API_KEY"] = api_key
# os.environ["OPENAI_API_BASE"] = "https://your-gateway.example.com/api/v2"

trainset, valset = build_train_val_examples(interviews, val_ratio=0.5, seed=7)

config = PersonaGEPAConfig(
    output_dir="artifacts/persona_gepa_demo",
    cache_dir=".cache/dspy",
    num_threads=4,
    api_base=os.getenv("OPENAI_API_BASE"),
)

program, artifact_path, report = run_optimization(config, trainset, valset)
print("Saved artifact:", artifact_path)
print("Validation report:", report)

history = format_history(interviews[0][:1])
question = interviews[0][1]["interviewer_question"]
answer = run_inference(config, artifact_path, history, question)
print("Answer:", answer)
