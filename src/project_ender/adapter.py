"""DomainAdapter abstract base class and Action definition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Action:
    """A single entry in a domain's action space."""

    id: int
    name: str
    description: str


class DomainAdapter(ABC):
    """
    Per-game interface that a game author implements.

    Three methods translate between the game's native representation and
    the Ender protocol's float vectors and integer action IDs.
    """

    @property
    @abstractmethod
    def action_space(self) -> list[Action]:
        """The fixed menu of decisions available to the commander."""
        ...

    @abstractmethod
    def encode_state(self, world: Any) -> list[float]:
        """
        Compress current world state into a fixed-length feature vector.

        The float vector is consumed by the policy network.
        """
        ...

    @abstractmethod
    def encode_state_summary(self, world: Any) -> str:
        """
        Produce a human-readable prose summary of the world state.

        This representation is sent to the oracle (LLM) in the prompt.
        """
        ...

    @abstractmethod
    def decode_action(self, action_id: int) -> Any:
        """Translate a network output integer into a concrete game command."""
        ...

    @abstractmethod
    def compute_reward(self, event: Any) -> float:
        """
        Map a game event to a scalar reward signal.

        Design intermediate signals — sparse rewards (win/loss only) stall training.
        """
        ...

    def valid_action_ids(self, world: Any) -> list[int]:
        """
        Return the list of legal action IDs for the current world state.

        Default: all actions in the action space are valid.
        Override to filter illegal moves.
        """
        return [a.id for a in self.action_space]
