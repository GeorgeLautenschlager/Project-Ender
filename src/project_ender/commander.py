"""
Commander — decision-making layer sitting above the oracle.

In M1 (oracle-only mode) the Commander always delegates to the oracle.
The ``confidence_threshold`` parameter is a stub: when the oracle's confidence
falls below the threshold the Commander currently logs a warning but still uses
the oracle response.  In M2 this is the switch-point where the trained policy
takes over from the oracle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from project_ender.adapter import Action
from project_ender.oracle import ModelService

logger = logging.getLogger(__name__)


@dataclass
class CommanderDecision:
    """A decided action returned by the Commander."""

    action_id: int
    confidence: float
    reasoning: str
    source: str  # "oracle" in M1; "oracle" or "policy" in M2+


class Commander:
    """
    Routes each game state to the appropriate decision-maker.

    M1 — oracle-only mode:
        Always queries the oracle.  If ``confidence < confidence_threshold``
        a debug message is logged; the oracle answer is still used because
        there is no policy fallback yet.

    M2+ — blend mode (not yet implemented):
        When oracle confidence >= threshold, use the oracle decision.
        Below threshold, fall back to the trained policy network.
        See DELIVERY.md M2: "Commander blend (policy mode, threshold switching)".
    """

    def __init__(
        self,
        oracle_backend: str = "claude",
        confidence_threshold: float = 0.5,
    ) -> None:
        self._oracle = ModelService(oracle_backend)
        self.confidence_threshold = confidence_threshold

    def decide(
        self,
        state_summary: str,
        action_space: list[Action],
        valid_actions: list[int],
    ) -> CommanderDecision:
        """Return a decision for the given game state."""
        label = self._oracle.query(state_summary, action_space, valid_actions)

        if label.confidence < self.confidence_threshold:
            # M2: fall back to the policy network here.
            # M1: no policy available — log and use the oracle response anyway.
            logger.debug(
                "Oracle confidence %.2f below threshold %.2f"
                " — policy fallback not yet implemented (M2 stub)",
                label.confidence,
                self.confidence_threshold,
            )

        return CommanderDecision(
            action_id=label.action_id,
            confidence=label.confidence,
            reasoning=label.reasoning,
            source="oracle",
        )
