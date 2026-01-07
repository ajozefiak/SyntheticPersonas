from __future__ import annotations

import dspy


DEFAULT_INSTRUCTIONS = (
    "You are answering as the interviewee. Use the provided transcript history to "
    "stay accurate and faithful to what was said. Match the interviewee's tone and "
    "style. If the answer is not supported by the transcript, say you do not know."
)


class PersonaAnswerSignature(dspy.Signature):
    """You are answering as the interviewee. Use the provided transcript history to stay accurate and faithful to what was said. Match the interviewee's tone and style. If the answer is not supported by the transcript, say you do not know."""

    instructions = DEFAULT_INSTRUCTIONS

    history = dspy.InputField(desc="Transcript context in Q/A format.")
    question = dspy.InputField(desc="Current interview question.")
    persona_profile = dspy.InputField(desc="Optional persona profile.", default="")

    answer = dspy.OutputField(desc="Answer in the interviewee's voice.")


class PersonaAnswerProgram(dspy.Module):
    def __init__(self, lm=None):
        super().__init__()
        self.predict = dspy.Predict(PersonaAnswerSignature, lm=lm)

    def forward(self, history: str, question: str, persona_profile: str = ""):
        return self.predict(
            history=history, question=question, persona_profile=persona_profile
        )
