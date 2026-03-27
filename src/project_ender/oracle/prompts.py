"""Prompt template for oracle queries."""

from __future__ import annotations

from project_ender.adapter import Action

_SYSTEM_INSTRUCTIONS = (
    "You are a tactical commander. "
    "Given the battlefield state and a numbered list of available actions, "
    "choose exactly one action.\n"
    "Respond ONLY with valid JSON matching exactly this schema:\n"
    '{"action_id": <int>, "confidence": <float 0-1>, "reasoning": "<string>"}\n'
    "Do not include any other text, markdown, or explanation outside the JSON object."
)


def build_prompt(
    state_summary: str,
    action_space: list[Action],
    valid_actions: list[int],
) -> str:
    """Combine system instructions, state summary, and valid actions into a prompt."""
    valid_set = set(valid_actions)
    action_lines = "\n".join(
        f"  {a.id}: {a.name} — {a.description}"
        for a in action_space
        if a.id in valid_set
    )
    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        f"---\n\n"
        f"{state_summary}\n\n"
        f"Available actions:\n{action_lines}"
    )
