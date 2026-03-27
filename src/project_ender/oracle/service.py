"""ModelService — unified interface for querying oracle models via bash."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from project_ender.adapter import Action

from .backends.base import Backend
from .backends.claude_cli import ClaudeCliBackend
from .prompts import build_prompt


@dataclass
class OracleLabel:
    """One labelled decision from an oracle."""

    action_id: int
    confidence: float
    reasoning: str
    teacher: str


def _build_backend(spec: str) -> Backend:
    """
    Parse a backend spec string and return the appropriate Backend.

    Supported in M1:
        "claude"          → ClaudeCliBackend (claude CLI via subprocess)

    Planned for M2+:
        "ollama:<model>"  → OllamaBackend (curl to localhost:11434)
        "lmstudio:<model>" → LMStudioBackend (curl to localhost:1234, OpenAI-compat)
    """
    if spec == "claude":
        return ClaudeCliBackend()
    raise ValueError(
        f"Unknown backend spec: {spec!r}. "
        "Supported: 'claude'. "
        "Coming in M2: 'ollama:<model>', 'lmstudio:<model>'."
    )


class ModelService:
    """
    Routes oracle queries to the configured model backend.

    All backends are called via bash (subprocess or curl), so no Python SDK
    dependencies are required for any model.

    Usage:
        svc = ModelService("claude")
        label = svc.query(state_summary, action_space, valid_actions)
    """

    def __init__(self, backend: str = "claude") -> None:
        self._backend = _build_backend(backend)

    @property
    def teacher_id(self) -> str:
        return self._backend.teacher_id

    def query(
        self,
        state_summary: str,
        action_space: list[Action],
        valid_actions: list[int],
    ) -> OracleLabel:
        """Send a state to the oracle and parse the structured response."""
        prompt = build_prompt(state_summary, action_space, valid_actions)
        raw = self._backend.call(prompt)
        return self._parse(raw, valid_actions)

    def _parse(self, raw: str, valid_actions: list[int]) -> OracleLabel:
        text = raw.strip()

        # Strip markdown code fences if the model wrapped its output
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop the opening fence (```json or ```) and closing fence
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Oracle returned non-JSON output: {raw!r}") from exc

        try:
            action_id = int(data["action_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Oracle response missing valid 'action_id': {data!r}"
            ) from exc

        if action_id not in valid_actions:
            raise ValueError(
                f"Oracle returned action_id={action_id} "
                f"which is not in valid_actions={valid_actions}"
            )

        return OracleLabel(
            action_id=action_id,
            confidence=float(data.get("confidence", 1.0)),
            reasoning=str(data.get("reasoning", "")),
            teacher=self._backend.teacher_id,
        )
