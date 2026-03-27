"""Corpus generation — scripted self-play to populate the states table.

Usage:
    python -m project_ender.corpus
    python -m project_ender.corpus --games 500 --db corpus.db
"""

from __future__ import annotations

import argparse
import logging
import random
import uuid

from project_ender.logger import DecisionLogger
from project_ender.skirmish.adapter import SkirmishAdapter
from project_ender.skirmish.game import SkirmishGame, SkirmishState

log = logging.getLogger(__name__)
_adapter = SkirmishAdapter()


def _random_player(state: SkirmishState, valid: list[int]) -> int:
    return random.choice(valid)


def _play_and_capture(
    captured: list[tuple[SkirmishState, list[int]]],
) -> None:
    """Play one game with random players, appending every (state, valid) to captured."""

    def player(state: SkirmishState, valid: list[int]) -> int:
        captured.append((state, valid))
        return _random_player(state, valid)

    SkirmishGame(red_player=player, blue_player=player).run(verbose=False)


def generate_corpus(db_path: str, n_games: int = 500) -> int:
    """
    Play n_games of Skirmish with random players and log every state.

    Both players' decisions are logged so each game contributes states from
    both commanders' perspectives. Returns the total number of states written.
    """
    logger = DecisionLogger(db_path)
    total = 0

    for game_num in range(n_games):
        game_id = str(uuid.uuid4())
        captured: list[tuple[SkirmishState, list[int]]] = []
        _play_and_capture(captured)

        for turn_idx, (state, valid) in enumerate(captured):
            logger.write_state(
                game_id=game_id,
                turn=turn_idx,
                state_vector=_adapter.encode_state(state),
                state_summary=_adapter.encode_state_summary(state),
                valid_action_ids=valid,
            )
            total += 1

        if (game_num + 1) % 50 == 0:
            log.info("Game %d/%d. States so far: %d", game_num + 1, n_games, total)

    log.info("Corpus generation complete. Total states: %d", total)
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate a Skirmish corpus")
    parser.add_argument("--db", default="corpus.db", help="SQLite database path")
    parser.add_argument(
        "--games", type=int, default=500, help="Number of games to play"
    )
    args = parser.parse_args()
    total = generate_corpus(args.db, args.games)
    print(f"Done. {total} states written to {args.db}")


if __name__ == "__main__":
    main()
