"""Ender protocol message types.

All messages are JSON-serialisable dataclasses exchanged over a local socket.

Wire format: each message is a single UTF-8 JSON line terminated with '\\n'.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal


@dataclass
class StateRequest:
    """Game → Ender: ask for a decision given the current state."""

    state_vector: list[float]
    state_summary: str
    valid_actions: list[int]

    def to_json(self) -> str:
        return json.dumps(
            {
                "state_vector": self.state_vector,
                "state_summary": self.state_summary,
                "valid_actions": self.valid_actions,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> StateRequest:
        data = json.loads(raw)
        return cls(
            state_vector=data["state_vector"],
            state_summary=data["state_summary"],
            valid_actions=data["valid_actions"],
        )


@dataclass
class ActionResponse:
    """Ender → Game: the chosen action."""

    action_id: int
    confidence: float
    source: Literal["policy", "oracle"]

    def to_json(self) -> str:
        return json.dumps(
            {
                "action_id": self.action_id,
                "confidence": self.confidence,
                "source": self.source,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> ActionResponse:
        data = json.loads(raw)
        return cls(
            action_id=data["action_id"],
            confidence=data["confidence"],
            source=data["source"],
        )


@dataclass
class RewardEvent:
    """Game → Ender: report the outcome of the last action."""

    action_id: int
    reward: float
    terminal: bool

    def to_json(self) -> str:
        return json.dumps(
            {
                "action_id": self.action_id,
                "reward": self.reward,
                "terminal": self.terminal,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> RewardEvent:
        data = json.loads(raw)
        return cls(
            action_id=data["action_id"],
            reward=data["reward"],
            terminal=data["terminal"],
        )


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7373
