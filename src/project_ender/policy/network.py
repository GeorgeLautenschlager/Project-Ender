"""PolicyNetwork — a small configurable MLP for game decision-making.

Architecture:
    Input:  state_vector  (float[state_dim])
    Hidden: N layers of (Linear → ReLU), width configurable
    Output: logits over action_space (float[action_dim])

The network is intentionally simple. Domain adapter quality and reward
shaping matter more than network depth for games with small state/action
spaces like Skirmish.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Configurable MLP policy network.

    Args:
        state_dim:    Length of the input state vector.
        action_dim:   Number of actions (output logits).
        hidden_sizes: Sequence of hidden layer widths. Default: (128, 128).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must be non-empty")

        layers: list[nn.Module] = []
        in_size = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        layers.append(nn.Linear(in_size, action_dim))

        self.net = nn.Sequential(*layers)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape (batch, state_dim) or (state_dim,).
        Returns:
            Logits of shape (batch, action_dim) or (action_dim,).
        """
        return self.net(x)  # type: ignore[no-any-return]
