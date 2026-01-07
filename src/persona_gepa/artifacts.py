from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Optional

from persona_gepa.program import PersonaAnswerProgram


def extract_instructions(program: PersonaAnswerProgram) -> str:
    signature = getattr(program, "predict", None)
    if signature is None:
        return ""
    sig_obj = getattr(signature, "signature", None)
    if sig_obj is None:
        return ""
    instructions = getattr(sig_obj, "instructions", None)
    if instructions:
        return str(instructions).strip()
    doc = getattr(sig_obj, "__doc__", None)
    return str(doc or "").strip()


def apply_instructions(program: PersonaAnswerProgram, instructions: str) -> None:
    sig_obj = getattr(program.predict, "signature", None)
    if sig_obj is None:
        return
    if hasattr(sig_obj, "instructions"):
        setattr(sig_obj, "instructions", instructions)
    sig_obj.__doc__ = instructions


def save_artifact(
    program: PersonaAnswerProgram,
    path: str,
    metadata: Optional[Dict[str, object]] = None,
) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    artifact = {
        "instructions": extract_instructions(program),
        "metadata": metadata or {},
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    return path


def load_artifact(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_program(path: str, lm=None) -> PersonaAnswerProgram:
    artifact = load_artifact(path)
    program = PersonaAnswerProgram(lm=lm)
    instructions = artifact.get("instructions", "")
    if instructions:
        apply_instructions(program, instructions)
    return program
