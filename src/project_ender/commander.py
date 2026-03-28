"""
Commander — decision-making layer sitting above the oracle and policy network.

Blend logic (M2):
    if policy is loaded and policy.confidence >= confidence_threshold:
        use policy   (source: "policy")
    else:
        use oracle   (source: "oracle", decision logged for future training)

Without a policy loaded the Commander always delegates to the oracle, which
was the M1 behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from project_ender.adapter import Action
from project_ender.oracle import ModelService

if TYPE_CHECKING:
    from project_ender.policy.inference import PolicyInference

logger = logging.getLogger(__name__)


@dataclass
class CommanderDecision:
    """A decided action returned by the Commander."""

    action_id: int
    confidence: float
    reasoning: str
    source: str  # "oracle" or "policy"


class Commander:
    """
    Routes each game state to the appropriate decision-maker.

    Oracle-only mode (M1, default):
        Always queries the oracle.

    Blend mode (M2):
        If a PolicyInference is provided, run the policy first.
        If policy confidence >= confidence_threshold, use the policy decision.
        Otherwise fall back to the oracle (and the oracle decision can be
        logged as new training data for the next corpus cycle).

    Args:
        oracle_backend:       Backend spec string passed to ModelService.
        confidence_threshold: Minimum policy confidence to use policy over oracle.
        policy:               Optional trained PolicyInference for blend mode.
    """

    def __init__(
        self,
        oracle_backend: str = "claude",
        confidence_threshold: float = 0.5,
        policy: PolicyInference | None = None,
    ) -> None:
        self._oracle = ModelService(oracle_backend)
        self.confidence_threshold = confidence_threshold
        self._policy = policy

    def decide(
        self,
        state_summary: str,
        action_space: list[Action],
        valid_actions: list[int],
        state_vector: list[float] | None = None,
    ) -> CommanderDecision:
        """
        Return a decision for the given game state.

        Args:
            state_summary: Human-readable state description for the oracle prompt.
            action_space:  Full action menu (used to build oracle prompt).
            valid_actions: Legal action ids for this state.
            state_vector:  Float feature vector (required for policy mode).
        """
        # Policy path — only available if a policy is loaded and state_vector provided.
        if self._policy is not None and state_vector is not None:
            action_id, conf = self._policy.act(state_vector, valid_actions)
            if conf >= self.confidence_threshold:
                return CommanderDecision(
                    action_id=action_id,
                    confidence=conf,
                    reasoning="",
                    source="policy",
                )
            logger.debug(
                "Policy confidence %.2f below threshold %.2f — falling back to oracle",
                conf,
                self.confidence_threshold,
            )

        # Oracle path (M1 behaviour, or policy fallback).
        if self._policy is not None and state_vector is None:
            logger.debug(
                "Policy loaded but state_vector not provided — using oracle only"
            )

        label = self._oracle.query(state_summary, action_space, valid_actions)
        return CommanderDecision(
            action_id=label.action_id,
            confidence=label.confidence,
            reasoning=label.reasoning,
            source="oracle",
        )
