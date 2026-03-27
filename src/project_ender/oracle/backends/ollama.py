"""OllamaBackend — calls Ollama's REST API via curl subprocess."""

from __future__ import annotations

import json
import subprocess

from .base import Backend

# Models that use extended thinking and need it disabled for reliable
# structured output during bulk labelling.
_THINKING_MODELS = frozenset({"deepseek-r1"})


def _is_thinking_model(model: str) -> bool:
    """Check if the model family uses a thinking/reasoning mode."""
    base = model.split(":")[0]
    return base in _THINKING_MODELS


class OllamaBackend(Backend):
    """
    Queries a locally-running Ollama instance via curl.

    The model name is passed at construction and used as both the Ollama
    model identifier and the teacher_id written to the labels table
    (with slashes and colons replaced by underscores).

    For thinking models (e.g. DeepSeek R1), thinking is disabled to get
    reliable structured JSON output within a reasonable token budget.

    Expects Ollama to be serving on localhost:11434 (the default).
    """

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self._model = model
        self._host = host
        self._teacher = model.replace("/", "_").replace(":", "_")
        self._disable_thinking = _is_thinking_model(model)

    @property
    def teacher_id(self) -> str:
        return self._teacher

    def call(self, prompt: str, valid_actions: list[int] | None = None) -> str:
        body: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
            },
        }
        if self._disable_thinking:
            body["think"] = False

        # Constrained decoding: restrict action_id to only valid values.
        if valid_actions:
            body["format"] = {
                "type": "object",
                "properties": {
                    "action_id": {"type": "integer", "enum": valid_actions},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"},
                },
                "required": ["action_id", "confidence", "reasoning"],
            }

        payload = json.dumps(body)
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-X",
                "POST",
                f"{self._host}/api/generate",
                "-H",
                "Content-Type: application/json",
                "-d",
                payload,
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        data = json.loads(result.stdout)
        response = data.get("response", "")
        # Fallback: if response is empty but thinking has content, extract
        # from there (safety net for models that ignore think=false).
        if not response.strip() and data.get("thinking"):
            response = data["thinking"]
        return response
