"""Labelling runner — queries an oracle for each unlabelled state in the corpus.

Usage:
    python -m project_ender.labeller
    python -m project_ender.labeller --db corpus.db --backend claude
"""

from __future__ import annotations

import argparse
import json
import logging

from project_ender.logger import DecisionLogger
from project_ender.oracle import ModelService
from project_ender.skirmish.adapter import SkirmishAdapter

log = logging.getLogger(__name__)
_adapter = SkirmishAdapter()


def run_labelling_pass(db_path: str, service: ModelService, limit: int = 0) -> int:
    """
    Label states in db_path not yet labelled by service.teacher_id.

    Args:
        limit: Maximum number of states to label in this pass (0 = no limit).

    Returns the number of labels written.
    """
    logger = DecisionLogger(db_path)
    unlabelled = logger.unlabelled_states(service.teacher_id)
    if limit > 0:
        unlabelled = unlabelled[:limit]
    log.info(
        "Found %d unlabelled states for teacher %r", len(unlabelled), service.teacher_id
    )

    written = 0
    for row in unlabelled:
        state_id: int = row["id"]
        state_summary: str = row["state_summary"]
        valid_actions: list[int] = json.loads(row["valid_action_ids"])

        label = service.query(
            state_summary=state_summary,
            action_space=_adapter.action_space,
            valid_actions=valid_actions,
        )
        logger.write_label(
            state_id=state_id,
            teacher=label.teacher,
            action_id=label.action_id,
            confidence=label.confidence,
            reasoning=label.reasoning,
        )
        written += 1
        if written % 100 == 0:
            log.info("Labelled %d/%d states", written, len(unlabelled))

    log.info("Labelling pass complete. Labels written: %d", written)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run a labelling pass over a corpus")
    parser.add_argument("--db", default="corpus.db", help="SQLite database path")
    parser.add_argument(
        "--backend",
        default="claude",
        help="Model backend spec (default: 'claude')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max states to label in this pass (default: 0 = no limit)",
    )
    args = parser.parse_args()
    service = ModelService(backend=args.backend)
    written = run_labelling_pass(args.db, service, limit=args.limit)
    print(f"Done. {written} labels written to {args.db}")


if __name__ == "__main__":
    main()
