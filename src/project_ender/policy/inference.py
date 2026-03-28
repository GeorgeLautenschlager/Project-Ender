"""PolicyInference — wraps a trained PolicyNetwork for live inference.

Provides a single `act()` method: given a state vector and the list of
valid action ids, return the highest-probability valid action and its
softmax confidence score.

Usage:
    inf = PolicyInference.from_checkpoint(
        "policy.pt", state_dim=18, action_dim=18
    )
    action_id, confidence = inf.act(state_vector, valid_actions)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from project_ender.policy.network import PolicyNetwork


class PolicyInference:
    """
    Wraps a trained PolicyNetwork for single-step inference.

    Args:
        network:    A trained (or untrained) PolicyNetwork.
        device:     Torch device string, default "cpu".
    """

    def __init__(
        self,
        network: PolicyNetwork,
        device: str = "cpu",
    ) -> None:
        self._net = network.to(device)
        self._net.eval()
        self._action_dim = network.action_dim
        self._device = device

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        device: str = "cpu",
    ) -> PolicyInference:
        """Load a PolicyNetwork from a saved state-dict checkpoint."""
        net = PolicyNetwork(state_dim, action_dim, hidden_sizes)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        net.load_state_dict(state_dict)
        return cls(net, device=device)

    def act(
        self,
        state_vector: list[float],
        valid_actions: list[int],
    ) -> tuple[int, float]:
        """
        Choose the highest-probability valid action.

        Invalid actions are masked to -inf before softmax so they receive
        zero probability mass.

        Args:
            state_vector:  Float list of length state_dim.
            valid_actions: Non-empty list of legal action ids.

        Returns:
            (action_id, confidence) where confidence is the softmax
            probability of the chosen action (0.0–1.0).
        """
        if not valid_actions:
            raise ValueError("valid_actions must be non-empty")

        x = torch.tensor(state_vector, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            logits = self._net(x).squeeze(0)  # (action_dim,)

        # Mask invalid actions with -inf so they get zero softmax mass.
        mask = torch.full((self._action_dim,), float("-inf"), device=self._device)
        for a in valid_actions:
            mask[a] = logits[a]

        probs = F.softmax(mask, dim=0)
        action_id = int(probs.argmax().item())
        confidence = float(probs[action_id].item())
        return action_id, confidence
