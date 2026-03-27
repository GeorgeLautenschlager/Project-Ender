"""DecisionLogger — SQLite writer for corpus states and oracle labels."""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CREATE_STATES = """
CREATE TABLE IF NOT EXISTS states (
    id               INTEGER PRIMARY KEY,
    game_id          TEXT,
    turn             INTEGER,
    state_vector     BLOB,
    state_summary    TEXT,
    valid_action_ids TEXT,
    created_at       TIMESTAMP
);
"""

_CREATE_LABELS = """
CREATE TABLE IF NOT EXISTS labels (
    state_id    INTEGER REFERENCES states(id),
    teacher     TEXT,
    action_id   INTEGER,
    confidence  FLOAT,
    reasoning   TEXT,
    labelled_at TIMESTAMP,
    PRIMARY KEY (state_id, teacher)
);
"""

_CREATE_CONSENSUS = """
CREATE TABLE IF NOT EXISTS consensus (
    state_id        INTEGER REFERENCES states(id),
    soft_targets    BLOB,
    agreement_score FLOAT,
    teacher_count   INTEGER,
    PRIMARY KEY (state_id)
);
"""


class DecisionLogger:
    """
    Writes corpus states and oracle labels to SQLite.

    The schema supports multiple teachers from the start so local-model
    labelling passes can be added in M2 without a migration.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_STATES)
            conn.execute(_CREATE_LABELS)
            conn.execute(_CREATE_CONSENSUS)

    def write_state(
        self,
        game_id: str,
        turn: int,
        state_vector: list[float],
        state_summary: str,
        valid_action_ids: list[int],
    ) -> int:
        """Insert a state row and return its id."""
        blob = struct.pack(f"{len(state_vector)}f", *state_vector)
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO states (game_id, turn, state_vector,"
                " state_summary, valid_action_ids, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    game_id,
                    turn,
                    blob,
                    state_summary,
                    json.dumps(valid_action_ids),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def write_label(
        self,
        state_id: int,
        teacher: str,
        action_id: int,
        confidence: float,
        reasoning: str,
    ) -> None:
        """Upsert a label row (state_id, teacher) is the primary key."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO labels"
                " (state_id, teacher, action_id, confidence, reasoning, labelled_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    state_id,
                    teacher,
                    action_id,
                    confidence,
                    reasoning,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def write_consensus(
        self,
        state_id: int,
        soft_targets: list[float],
        agreement_score: float,
        teacher_count: int,
    ) -> None:
        """Upsert a consensus row."""
        blob = struct.pack(f"{len(soft_targets)}f", *soft_targets)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO consensus"
                " (state_id, soft_targets, agreement_score, teacher_count)"
                " VALUES (?, ?, ?, ?)",
                (state_id, blob, agreement_score, teacher_count),
            )

    def unlabelled_states(self, teacher: str) -> list[dict[str, Any]]:
        """Return all states not yet labelled by the given teacher."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.id, s.state_summary, s.valid_action_ids"
                " FROM states s"
                " WHERE NOT EXISTS ("
                "   SELECT 1 FROM labels l"
                "   WHERE l.state_id = s.id AND l.teacher = ?"
                " )",
                (teacher,),
            ).fetchall()
        return [dict(r) for r in rows]

    def all_labels(self) -> list[dict[str, Any]]:
        """Return all label rows (used by consensus builder)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT state_id, teacher, action_id, confidence FROM labels"
            ).fetchall()
        return [dict(r) for r in rows]

    def state_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM states").fetchone()[0])

    def label_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0])

    def consensus_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM consensus").fetchone()[0])
