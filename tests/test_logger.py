"""Tests for DecisionLogger — SQLite schema creation and read/write operations."""

from __future__ import annotations

from pathlib import Path

import pytest
from project_ender.logger import DecisionLogger


@pytest.fixture
def db(tmp_path: Path) -> DecisionLogger:
    return DecisionLogger(tmp_path / "test.db")


def test_schema_created(db: DecisionLogger) -> None:
    assert db.state_count() == 0
    assert db.label_count() == 0
    assert db.consensus_count() == 0


def test_write_state_returns_id(db: DecisionLogger) -> None:
    state_id = db.write_state(
        game_id="g1",
        turn=0,
        state_vector=[0.1, 0.2, 0.3],
        state_summary="summary",
        valid_action_ids=[0, 7, 14],
    )
    assert isinstance(state_id, int)
    assert state_id > 0
    assert db.state_count() == 1


def test_write_multiple_states(db: DecisionLogger) -> None:
    for i in range(5):
        db.write_state("g1", i, [float(i)], f"state {i}", [14])
    assert db.state_count() == 5


def test_write_label(db: DecisionLogger) -> None:
    sid = db.write_state("g1", 0, [0.5], "summary", [7, 14])
    db.write_label(sid, "claude", 7, 0.9, "attack flank")
    assert db.label_count() == 1


def test_write_label_upserts(db: DecisionLogger) -> None:
    sid = db.write_state("g1", 0, [0.5], "summary", [7, 14])
    db.write_label(sid, "claude", 7, 0.9, "first")
    db.write_label(sid, "claude", 14, 0.6, "revised")
    # upsert: still one row
    assert db.label_count() == 1


def test_multiple_teachers(db: DecisionLogger) -> None:
    sid = db.write_state("g1", 0, [0.5], "summary", [7, 14])
    db.write_label(sid, "claude", 7, 0.9, "r1")
    db.write_label(sid, "qwen3_14b", 14, 0.7, "r2")
    assert db.label_count() == 2


def test_unlabelled_states_empty_when_all_labelled(db: DecisionLogger) -> None:
    sid = db.write_state("g1", 0, [0.5], "summary", [14])
    db.write_label(sid, "claude", 14, 1.0, "hold")
    assert db.unlabelled_states("claude") == []


def test_unlabelled_states_returns_unprocessed(db: DecisionLogger) -> None:
    sid = db.write_state("g1", 0, [0.5], "summary", [14])
    db.write_label(sid, "claude", 14, 1.0, "hold")
    # qwen3_14b hasn't labelled this state yet
    unlabelled = db.unlabelled_states("qwen3_14b")
    assert len(unlabelled) == 1
    assert unlabelled[0]["id"] == sid


def test_all_labels(db: DecisionLogger) -> None:
    sid = db.write_state("g1", 0, [0.5], "summary", [7, 14])
    db.write_label(sid, "claude", 7, 0.9, "r")
    db.write_label(sid, "qwen3_14b", 14, 0.7, "r")
    labels = db.all_labels()
    assert len(labels) == 2
    teachers = {row["teacher"] for row in labels}
    assert teachers == {"claude", "qwen3_14b"}


def test_write_consensus(db: DecisionLogger) -> None:
    sid = db.write_state("g1", 0, [0.5], "summary", [7])
    soft = [0.0] * 18
    soft[7] = 1.0
    db.write_consensus(sid, soft, 1.0, 1)
    assert db.consensus_count() == 1
