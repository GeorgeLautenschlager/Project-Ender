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
"""

from __future__ import annotations

import argparse
import random
import sys

from project_ender.client import EnderClient
from project_ender.protocol import DEFAULT_HOST, DEFAULT_PORT, StateRequest
from project_ender.skirmish.adapter import SkirmishAdapter
from project_ender.skirmish.game import (
    PlayerFn,
    SkirmishGame,
    SkirmishState,
)

_adapter = SkirmishAdapter()


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


def _ender_player(
    host: str, port: int
) -> PlayerFn:
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


def _make_player(kind: str, host: str, port: int) -> PlayerFn:
    if kind == "human":
        return _human_player
    if kind == "random":
        return _random_player
    if kind == "ender":
        return _ender_player(host, port)
    raise ValueError(f"Unknown player type: {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a game of Skirmish")
    parser.add_argument(
        "--red",
        choices=["human", "random", "ender"],
        default="human",
        help="Red player type (default: human)",
    )
    parser.add_argument(
        "--blue",
        choices=["human", "random", "ender"],
        default="human",
        help="Blue player type (default: human)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ender server host")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Ender server port"
    )
    args = parser.parse_args()

    red = _make_player(args.red, args.host, args.port)
    blue = _make_player(args.blue, args.host, args.port)

    game = SkirmishGame(red_player=red, blue_player=blue)
    result = game.run(verbose=True)
    sys.exit(0 if result.winner is not None else 1)


if __name__ == "__main__":
    main()
