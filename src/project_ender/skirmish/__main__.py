"""
Run a Skirmish game from the command line.

    # Two humans, no server needed
    python -m project_ender.skirmish

    # One human (Red) vs Ender server (Blue)
    python -m project_ender.skirmish --blue ender

    # Fully automated: both sides use Ender server (validates full pipeline)
    python -m project_ender.skirmish --red ender --blue ender

    # Random bot vs random bot (no server, instant)
    python -m project_ender.skirmish --red random --blue random

    # Oracle (Claude CLI) vs random — watch the oracle play directly
    python -m project_ender.skirmish --red oracle --blue random

    # Override oracle backend (e.g. for M2+ local models)
    python -m project_ender.skirmish --blue oracle --oracle-backend claude

    # Set confidence threshold (logged when oracle falls below it; M2 stub)
    python -m project_ender.skirmish --red oracle --confidence-threshold 0.7

    # Trained policy vs random (M2)
    python -m project_ender.skirmish --red policy --policy-checkpoint policy.pt

    # Oracle vs trained policy — compare them head-to-head (M2)
    python -m project_ender.skirmish --red oracle --blue policy --policy-checkpoint policy.pt

    # Commander blend: policy with oracle fallback below threshold (M2)
    python -m project_ender.skirmish --red oracle --policy-checkpoint policy.pt --confidence-threshold 0.7
"""

from __future__ import annotations

import argparse
import random
import sys

from project_ender.client import EnderClient
from project_ender.commander import Commander
from project_ender.policy.inference import PolicyInference
from project_ender.protocol import DEFAULT_HOST, DEFAULT_PORT, StateRequest
from project_ender.skirmish.adapter import SkirmishAdapter
from project_ender.skirmish.game import (
    PlayerFn,
    SkirmishGame,
    SkirmishState,
)

_adapter = SkirmishAdapter()

# Skirmish-specific dimensions — must match SkirmishAdapter.encode_state().
_STATE_DIM = 18
_ACTION_DIM = 18


def _human_player(state: SkirmishState, valid: list[int]) -> int:
    """Prompt the human at the terminal to choose an action."""
    active = state.active_commander()
    print(f"\n  {active.value}'s turn. Valid actions:")
    for action_id in valid:
        print(f"    {action_id:2d} — {_adapter.decode_action(action_id)}")
    while True:
        raw = input("  Enter action id: ").strip()
        try:
            chosen = int(raw)
        except ValueError:
            print("  Please enter an integer.")
            continue
        if chosen not in valid:
            print(f"  {chosen} is not a valid action. Choose from: {valid}")
            continue
        return chosen


def _random_player(state: SkirmishState, valid: list[int]) -> int:
    action = random.choice(valid)
    active = state.active_commander()
    name = _adapter.decode_action(action)
    print(f"  {active.value} (random) plays: {action} -- {name}")
    return action


def _ender_player(host: str, port: int) -> PlayerFn:
    """
    Return a player function that delegates decisions to an Ender server.

    A single persistent connection is opened when the first decision is requested.
    """
    client: EnderClient | None = None

    def player(state: SkirmishState, valid: list[int]) -> int:
        nonlocal client
        if client is None:
            client = EnderClient(host=host, port=port)
            client.connect()

        req = StateRequest(
            state_vector=_adapter.encode_state(state),
            state_summary=_adapter.encode_state_summary(state),
            valid_actions=valid,
        )
        resp = client.request(req)
        active = state.active_commander()
        print(
            f"  {active.value} (ender) plays: {resp.action_id}"
            f" — {_adapter.decode_action(resp.action_id)}"
            f"  [conf={resp.confidence:.2f}, src={resp.source}]"
        )
        return resp.action_id

    return player


def _oracle_player(
    backend: str,
    confidence_threshold: float = 0.5,
    policy: PolicyInference | None = None,
) -> PlayerFn:
    """Return a player function that calls the Commander.

    If a policy is also provided the Commander runs in blend mode:
    policy decisions are used when confidence >= confidence_threshold,
    otherwise the oracle is queried.
    """
    commander = Commander(
        oracle_backend=backend,
        confidence_threshold=confidence_threshold,
        policy=policy,
    )

    def player(state: SkirmishState, valid: list[int]) -> int:
        decision = commander.decide(
            state_summary=_adapter.encode_state_summary(state),
            action_space=_adapter.action_space,
            valid_actions=valid,
            state_vector=_adapter.encode_state(state),
        )
        active = state.active_commander()
        print(
            f"  {active.value} ({decision.source}/{backend}) plays: {decision.action_id}"
            f" — {_adapter.decode_action(decision.action_id)}"
            f"  [conf={decision.confidence:.2f}]"
        )
        if decision.reasoning:
            print(f"    Reasoning: {decision.reasoning}")
        return decision.action_id

    return player


def _policy_player(
    checkpoint: str,
    hidden_sizes: tuple[int, ...] = (128, 128),
) -> PlayerFn:
    """Return a player function that uses only the trained policy (no oracle)."""
    policy = PolicyInference.from_checkpoint(
        checkpoint,
        state_dim=_STATE_DIM,
        action_dim=_ACTION_DIM,
        hidden_sizes=hidden_sizes,
    )

    def player(state: SkirmishState, valid: list[int]) -> int:
        sv = _adapter.encode_state(state)
        action_id, conf = policy.act(sv, valid)
        active = state.active_commander()
        print(
            f"  {active.value} (policy) plays: {action_id}"
            f" — {_adapter.decode_action(action_id)}"
            f"  [conf={conf:.2f}]"
        )
        return action_id

    return player


def _make_player(
    kind: str,
    host: str,
    port: int,
    oracle_backend: str = "claude",
    confidence_threshold: float = 0.5,
    policy_checkpoint: str | None = None,
    policy_hidden_sizes: tuple[int, ...] = (128, 128),
) -> PlayerFn:
    if kind == "human":
        return _human_player
    if kind == "random":
        return _random_player
    if kind == "ender":
        return _ender_player(host, port)
    if kind == "policy":
        if policy_checkpoint is None:
            raise ValueError(
                "--policy-checkpoint is required when using the 'policy' player type"
            )
        return _policy_player(policy_checkpoint, policy_hidden_sizes)
    if kind == "oracle":
        policy: PolicyInference | None = None
        if policy_checkpoint is not None:
            policy = PolicyInference.from_checkpoint(
                policy_checkpoint,
                state_dim=_STATE_DIM,
                action_dim=_ACTION_DIM,
                hidden_sizes=policy_hidden_sizes,
            )
        return _oracle_player(oracle_backend, confidence_threshold, policy)
    raise ValueError(f"Unknown player type: {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a game of Skirmish")
    parser.add_argument(
        "--red",
        choices=["human", "random", "ender", "oracle", "policy"],
        default="human",
        help="Red player type (default: human)",
    )
    parser.add_argument(
        "--blue",
        choices=["human", "random", "ender", "oracle", "policy"],
        default="human",
        help="Blue player type (default: human)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ender server host")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Ender server port"
    )
    parser.add_argument(
        "--oracle-backend",
        default="claude",
        help="Oracle backend spec for --red/--blue oracle (default: 'claude')",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help=(
            "Commander confidence threshold (default: 0.5). "
            "In blend mode, policy decisions below this confidence fall back to oracle."
        ),
    )
    parser.add_argument(
        "--policy-checkpoint",
        default=None,
        help=(
            "Path to a trained policy checkpoint (.pt file). "
            "Required for --red/--blue policy. "
            "If provided with --red/--blue oracle, enables Commander blend mode."
        ),
    )
    parser.add_argument(
        "--policy-hidden-sizes",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Policy network hidden layer widths (default: 128 128)",
    )
    args = parser.parse_args()

    hidden_sizes = tuple(args.policy_hidden_sizes)

    red = _make_player(
        args.red,
        args.host,
        args.port,
        args.oracle_backend,
        args.confidence_threshold,
        args.policy_checkpoint,
        hidden_sizes,
    )
    blue = _make_player(
        args.blue,
        args.host,
        args.port,
        args.oracle_backend,
        args.confidence_threshold,
        args.policy_checkpoint,
        hidden_sizes,
    )

    game = SkirmishGame(red_player=red, blue_player=blue)
    result = game.run(verbose=True)
    sys.exit(0 if result.winner is not None else 1)


if __name__ == "__main__":
    main()
