"""Tests for the consensus builder."""

from __future__ import annotations

from pathlib import Path

from project_ender.consensus import build_consensus
from project_ender.logger import DecisionLogger


def _setup_db(tmp_path: Path) -> tuple[str, int]:
    db = str(tmp_path / "corpus.db")
    logger = DecisionLogger(db)
    sid = logger.write_state("g1", 0, [0.5] * 20, "summary", list(range(18)))
    return db, sid


def test_consensus_single_teacher(tmp_path: Path) -> None:
    db, sid = _setup_db(tmp_path)
    logger = DecisionLogger(db)
    logger.write_label(sid, "claude", 7, 0.9, "attack")
    written = build_consensus(db)
    assert written == 1
    assert logger.consensus_count() == 1


def test_consensus_unanimous_agreement(tmp_path: Path) -> None:
    db, sid = _setup_db(tmp_path)
    logger = DecisionLogger(db)
    logger.write_label(sid, "claude", 7, 1.0, "r")
    logger.write_label(sid, "qwen3_14b", 7, 1.0, "r")
    build_consensus(db)
    # Both agree on action 7 → agreement_score should be 1.0
    # Read back via raw sqlite
    import sqlite3

    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT agreement_score, teacher_count FROM consensus WHERE state_id=?", (sid,)
    ).fetchone()
    conn.close()
    assert row[1] == 2
    assert abs(row[0] - 1.0) < 1e-6


def test_consensus_split_vote(tmp_path: Path) -> None:
    db, sid = _setup_db(tmp_path)
    logger = DecisionLogger(db)
    # Equal confidence split: claude → action 7 (conf 1.0), qwen → action 14 (conf 1.0)
    logger.write_label(sid, "claude", 7, 1.0, "r")
    logger.write_label(sid, "qwen3_14b", 14, 1.0, "r")
    build_consensus(db)
    import sqlite3

    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT agreement_score FROM consensus WHERE state_id=?", (sid,)
    ).fetchone()
    conn.close()
    # 50/50 split → agreement_score ≈ 0.5
    assert abs(row[0] - 0.5) < 1e-6


def test_consensus_no_labels_writes_nothing(tmp_path: Path) -> None:
    db, _ = _setup_db(tmp_path)
    written = build_consensus(db)
    assert written == 0


def test_consensus_multiple_states(tmp_path: Path) -> None:
    db = str(tmp_path / "corpus.db")
    logger = DecisionLogger(db)
    for i in range(5):
        sid = logger.write_state("g1", i, [0.5] * 20, f"state {i}", [14])
        logger.write_label(sid, "claude", 14, 1.0, "hold")
    written = build_consensus(db)
    assert written == 5
    assert logger.consensus_count() == 5
