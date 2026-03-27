"""Stratified corpus generator — balanced random-vs-random self-play.

Plays random-vs-random games and buckets decision states into 9 cells:

    phase       × perspective
    ─────────────────────────
    early (0-66)   winning
    mid  (67-133)  losing
    late (134-200) draw

Reservoir sampling (Algorithm R) ensures each bucket contains at most
``states_per_bucket`` states drawn uniformly from all qualifying states
seen across all games.

Usage:
    python -m project_ender.stratified_corpus
    python -m project_ender.stratified_corpus --per-bucket 200 --max-games 10000 --db corpus.db
"""

from __future__ import annotations

import argparse
import logging
import random
import uuid
from typing import TYPE_CHECKING

from project_ender.logger import DecisionLogger
from project_ender.skirmish.adapter import SkirmishAdapter
from project_ender.skirmish.game import (
    Commander,
    GameResult,
    MAX_TURNS,
    SkirmishGame,
    SkirmishState,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)
_adapter = SkirmishAdapter()

# Phase boundaries (inclusive on the lower end of each range)
_PHASE_MID_START = MAX_TURNS // 3 + 1   # 67
_PHASE_LATE_START = 2 * MAX_TURNS // 3 + 1  # 134

PHASES = ("early", "mid", "late")
PERSPECTIVES = ("winning", "losing", "draw")

# All 9 bucket keys in a stable order
BUCKET_KEYS: tuple[tuple[str, str], ...] = tuple(
    (phase, perspective) for phase in PHASES for perspective in PERSPECTIVES
)


def _phase(turn: int) -> str:
    """Classify a turn number into early / mid / late."""
    if turn < _PHASE_MID_START:
        return "early"
    if turn < _PHASE_LATE_START:
        return "mid"
    return "late"


def _perspective(active: Commander, winner: Commander | None) -> str:
    """Classify a decision from the active commander's point of view."""
    if winner is None:
        return "draw"
    return "winning" if active == winner else "losing"


def _play_one_game(
    rng: random.Random,
) -> tuple[list[tuple[SkirmishState, list[int]]], GameResult]:
    """Play one game with random players.

    Returns a list of ``(state_snapshot, valid_actions)`` — one entry per
    decision point — paired with the final ``GameResult``.
    """
    captured: list[tuple[SkirmishState, list[int]]] = []

    def player(state: SkirmishState, valid: list[int]) -> int:
        captured.append((state.copy(), list(valid)))
        return rng.choice(valid)

    result = SkirmishGame(red_player=player, blue_player=player).run(verbose=False)
    return captured, result


def generate_stratified_corpus(
    db_path: str,
    states_per_bucket: int = 200,
    max_games: int = 10_000,
    seed: int | None = None,
) -> dict[str, int]:
    """Run random-vs-random games and write a stratified state corpus to SQLite.

    Decision states are bucketed by game phase and the active commander's
    outcome perspective (winning / losing / draw).  Reservoir sampling keeps
    each of the 9 buckets at most ``states_per_bucket`` entries drawn
    uniformly from all qualifying states observed.

    The loop terminates early once every bucket has reached its quota, or
    after ``max_games`` games — whichever comes first.

    Args:
        db_path:          Path to the SQLite database (created if absent).
        states_per_bucket: Target number of states per bucket (default 200).
        max_games:        Hard cap on games to simulate (default 10 000).
        seed:             RNG seed for reproducibility (default None).

    Returns:
        A dict mapping ``"<phase>_<perspective>"`` → states written.
    """
    rng = random.Random(seed)

    # buckets[key] accumulates (game_id, state, valid_actions) triples
    BucketEntry = tuple[str, SkirmishState, list[int]]
    buckets: dict[tuple[str, str], list[BucketEntry]] = {k: [] for k in BUCKET_KEYS}
    seen: dict[tuple[str, str], int] = {k: 0 for k in BUCKET_KEYS}

    games_played = 0
    for games_played in range(1, max_games + 1):
        # Early-exit once every bucket is full
        if all(len(buckets[k]) >= states_per_bucket for k in BUCKET_KEYS):
            log.info("All buckets full after %d games.", games_played - 1)
            break

        game_id = str(uuid.uuid4())
        captured, result = _play_one_game(rng)
        winner = result.winner

        for state, valid in captured:
            key = (_phase(state.turn), _perspective(state.active_commander(), winner))
            seen[key] += 1
            bucket = buckets[key]
            entry: BucketEntry = (game_id, state, valid)

            if len(bucket) < states_per_bucket:
                bucket.append(entry)
            else:
                # Reservoir sampling: replace a random existing entry
                j = rng.randrange(seen[key])
                if j < states_per_bucket:
                    bucket[j] = entry

        if games_played % 500 == 0:
            filled = sum(1 for k in BUCKET_KEYS if len(buckets[k]) >= states_per_bucket)
            log.info(
                "Game %d/%d — %d/9 buckets full.",
                games_played,
                max_games,
                filled,
            )

    else:
        log.warning(
            "Reached max_games=%d before all buckets filled.", max_games
        )

    # Flush selected states to the database
    logger = DecisionLogger(db_path)
    stats: dict[str, int] = {}

    for (phase, perspective), entries in buckets.items():
        for game_id, state, valid in entries:
            logger.write_state(
                game_id=game_id,
                turn=state.turn,
                state_vector=_adapter.encode_state(state),
                state_summary=_adapter.encode_state_summary(state),
                valid_action_ids=valid,
            )
        key_str = f"{phase}_{perspective}"
        stats[key_str] = len(entries)
        log.info("  %-20s %d states", key_str, len(entries))

    total = sum(stats.values())
    log.info("Stratified corpus complete. %d states written to %s", total, db_path)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Generate a stratified Skirmish corpus via random self-play"
    )
    parser.add_argument("--db", default="corpus.db", help="SQLite database path")
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=200,
        dest="per_bucket",
        help="Target states per bucket (9 buckets total)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=10_000,
        dest="max_games",
        help="Hard cap on games to simulate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility",
    )
    args = parser.parse_args()

    stats = generate_stratified_corpus(
        db_path=args.db,
        states_per_bucket=args.per_bucket,
        max_games=args.max_games,
        seed=args.seed,
    )

    total = sum(stats.values())
    print(f"\nStratified corpus written to {args.db}")
    print(f"{'Bucket':<25} {'States':>6}")
    print("-" * 33)
    for key in sorted(stats):
        print(f"  {key:<23} {stats[key]:>6}")
    print("-" * 33)
    print(f"  {'TOTAL':<23} {total:>6}")


if __name__ == "__main__":
    main()
