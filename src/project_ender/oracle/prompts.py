"""Prompt template for oracle queries."""

from __future__ import annotations

from project_ender.adapter import Action

_SYSTEM_INSTRUCTIONS = (
    "You are a tactical commander optimising for victory. "
    "Given the battlefield state and a numbered list of available actions, "
    "choose exactly one action.\n"
    "Select the action with the highest expected strategic value. "
    "When an aggressive or high-risk action offers a meaningfully better outcome, "
    "prefer it over a cautious alternative — do not default to conservative play "
    "simply to avoid risk. Caution is only correct when the expected value "
    "genuinely favours it.\n"
    "Respond ONLY with valid JSON matching exactly this schema:\n"
    '{"action_id": <int>, "confidence": <float 0-1>, "reasoning": "<string>"}\n'
    "Do not include any other text, markdown, or explanation outside the JSON object."
)

DEFAULT_STYLE_HINT = (
    "Holding is not inherently safer than acting. "
    "Evaluate each action on its tactical merit for the current position. "
    "Passive play loses games."
)

AGGRESSIVE_STYLE_HINT = (
    "Play to win. Aggressive positioning and attacks are often correct. "
    "Do not default to holding unless you have a specific tactical reason to consolidate."
)


def build_prompt(
    state_summary: str,
    action_space: list[Action],
    valid_actions: list[int],
    style_hint: str = "",
) -> str:
    """Combine system instructions, state summary, and valid actions into a prompt.

    Args:
        state_summary: Summary of the game state.
        action_space: All available actions.
        valid_actions: IDs of legal actions in the current state.
        style_hint: Optional strategic guidance to combat mode collapse.
    """
    valid_set = set(valid_actions)
    action_lines = "\n".join(
        f"  {a.id}: {a.name} — {a.description}"
        for a in action_space
        if a.id in valid_set
    )

    prompt_parts = [_SYSTEM_INSTRUCTIONS]
    if style_hint:
        prompt_parts.append(style_hint)
    prompt_parts.extend(["---", state_summary, "Available actions:", action_lines])

    return "\n\n".join(prompt_parts)
