"""Consensus builder — merges label sets into soft targets and agreement scores.

Usage:
    python -m project_ender.consensus
    python -m project_ender.consensus --db corpus.db
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import Any

from project_ender.logger import DecisionLogger
from project_ender.skirmish.adapter import SkirmishAdapter

log = logging.getLogger(__name__)

_ACTION_SPACE_SIZE = len(SkirmishAdapter().action_space)


def build_consensus(db_path: str) -> int:
    """
    Compute soft targets and agreement scores from all labels in db_path.

    For each state, accumulates confidence mass per action across all teachers,
    normalises to a probability distribution, and scores agreement as the
    fraction of mass on the plurality action.

    Returns the number of consensus rows written.
    """
    logger = DecisionLogger(db_path)
    all_labels = logger.all_labels()

    by_state: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for label in all_labels:
        by_state[label["state_id"]].append(label)

    written = 0
    for state_id, labels in by_state.items():
        votes: list[float] = [0.0] * _ACTION_SPACE_SIZE
        for label in labels:
            aid = int(label["action_id"])
            conf = float(label["confidence"])
            if 0 <= aid < _ACTION_SPACE_SIZE:
                votes[aid] += conf

        total_mass = sum(votes)
        if total_mass == 0.0:
            continue

        soft_targets = [v / total_mass for v in votes]
        # Agreement score: fraction of mass on the plurality action.
        # 1.0 = all teachers agreed, lower = more disagreement.
        agreement_score = max(soft_targets)

        logger.write_consensus(
            state_id=state_id,
            soft_targets=soft_targets,
            agreement_score=agreement_score,
            teacher_count=len(labels),
        )
        written += 1

    log.info("Consensus built for %d states", written)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build consensus from label sets")
    parser.add_argument("--db", default="corpus.db", help="SQLite database path")
    args = parser.parse_args()
    written = build_consensus(args.db)
    print(f"Done. Consensus computed for {written} states.")


if __name__ == "__main__":
    main()
