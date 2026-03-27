"""Labelling runner — queries an oracle for each unlabelled state in the corpus.

Usage:
    python -m project_ender.labeller
    python -m project_ender.labeller --db corpus.db --backend claude
    python -m project_ender.labeller --db corpus.db --backend ollama:qwen2.5:7b --retries 3
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time

from project_ender.logger import DecisionLogger
from project_ender.oracle import ModelService
from project_ender.skirmish.adapter import SkirmishAdapter

log = logging.getLogger(__name__)
_adapter = SkirmishAdapter()

# Strings that indicate a Claude usage limit has been hit.
_USAGE_LIMIT_SIGNALS = [
    "usage limit",
    "rate limit",
    "quota exceeded",
    "too many requests",
    "resource_exhausted",
    "overloaded",
]


class UsageLimitError(Exception):
    """Raised when the oracle backend reports a usage/rate limit."""


def _is_usage_limit(exc: Exception) -> bool:
    """Check if an exception looks like a usage limit error."""
    msg = str(exc).lower()
    return any(signal in msg for signal in _USAGE_LIMIT_SIGNALS)


def run_labelling_pass(
    db_path: str,
    service: ModelService,
    limit: int = 0,
    retries: int = 2,
    stop_on_usage_limit: bool = False,
) -> tuple[int, int, bool]:
    """
    Label states in db_path not yet labelled by service.teacher_id.

    Args:
        limit:               Max states to label (0 = no limit).
        retries:             Number of retries per state on transient errors.
        stop_on_usage_limit: If True, exit cleanly when a usage limit is hit.

    Returns:
        (labels_written, errors_skipped, hit_usage_limit)
    """
    logger = DecisionLogger(db_path)
    unlabelled = logger.unlabelled_states(service.teacher_id)
    if limit > 0:
        unlabelled = unlabelled[:limit]
    total = len(unlabelled)
    log.info("Found %d unlabelled states for teacher %r", total, service.teacher_id)

    written = 0
    skipped = 0
    t0 = time.time()

    for idx, row in enumerate(unlabelled):
        state_id: int = row["id"]
        state_summary: str = row["state_summary"]
        valid_actions: list[int] = json.loads(row["valid_action_ids"])

        success = False
        for attempt in range(1, retries + 1):
            try:
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
                success = True
                break
            except (subprocess.CalledProcessError, ValueError, Exception) as exc:
                if stop_on_usage_limit and _is_usage_limit(exc):
                    log.warning(
                        "Usage limit reached after %d labels. Stopping gracefully.",
                        written,
                    )
                    return written, skipped, True

                if attempt < retries:
                    wait = 2**attempt
                    log.warning(
                        "State %d attempt %d/%d failed: %s — retrying in %ds",
                        state_id,
                        attempt,
                        retries,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    log.error(
                        "State %d failed after %d attempts: %s — skipping",
                        state_id,
                        retries,
                        exc,
                    )

        if not success:
            skipped += 1

        # Progress reporting
        done = written + skipped
        if done % 100 == 0 and done > 0:
            elapsed = time.time() - t0
            rate = written / elapsed if elapsed > 0 else 0
            eta_s = (total - done) / rate if rate > 0 else 0
            eta_m = eta_s / 60
            log.info(
                "Progress: %d/%d labelled, %d skipped (%.1f labels/min, ETA %.0fm)",
                written,
                total,
                skipped,
                rate * 60,
                eta_m,
            )

    elapsed = time.time() - t0
    log.info(
        "Labelling pass complete. %d written, %d skipped in %.1fm",
        written,
        skipped,
        elapsed / 60,
    )
    return written, skipped, False


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Run a labelling pass over a corpus")
    parser.add_argument("--db", default="corpus.db", help="SQLite database path")
    parser.add_argument(
        "--backend",
        default="claude",
        help=(
            "Model backend spec. Examples: 'claude', 'ollama:qwen2.5:7b', "
            "'lmstudio:gpt-oss-20b@100.x.x.x:1234'"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max states to label in this pass (default: 0 = no limit)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry attempts per state on failure (default: 2)",
    )
    parser.add_argument(
        "--stop-on-usage-limit",
        action="store_true",
        default=False,
        help="Exit cleanly when a usage/rate limit is hit (useful for Claude)",
    )
    args = parser.parse_args()

    service = ModelService(backend=args.backend)
    written, skipped, hit_limit = run_labelling_pass(
        args.db,
        service,
        limit=args.limit,
        retries=args.retries,
        stop_on_usage_limit=args.stop_on_usage_limit,
    )

    status = "USAGE LIMIT" if hit_limit else "DONE"
    print(f"\n[{status}] {written} labels written, {skipped} skipped → {args.db}")


if __name__ == "__main__":
    main()
