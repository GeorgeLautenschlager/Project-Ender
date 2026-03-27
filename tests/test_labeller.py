"""Tests for the labelling runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from project_ender.corpus import generate_corpus
from project_ender.labeller import run_labelling_pass
from project_ender.logger import DecisionLogger
from project_ender.oracle import OracleLabel


def _mock_service(teacher: str = "claude", action_id: int = 14) -> MagicMock:
    svc = MagicMock()
    svc.teacher_id = teacher
    svc.query.return_value = OracleLabel(
        action_id=action_id, confidence=0.9, reasoning="hold", teacher=teacher
    )
    return svc


def test_labelling_pass_writes_labels(tmp_path: Path) -> None:
    db = str(tmp_path / "corpus.db")
    generate_corpus(db, n_games=2)
    svc = _mock_service()
    written, skipped, hit_limit = run_labelling_pass(db, svc)
    assert written > 0
    assert skipped == 0
    assert not hit_limit
    logger = DecisionLogger(db)
    assert logger.label_count() == written


def test_labelling_pass_skips_already_labelled(tmp_path: Path) -> None:
    db = str(tmp_path / "corpus.db")
    generate_corpus(db, n_games=2)
    svc = _mock_service()
    first, _, _ = run_labelling_pass(db, svc)
    # Second pass: all states already labelled, nothing to do
    second, _, _ = run_labelling_pass(db, svc)
    assert second == 0
    assert DecisionLogger(db).label_count() == first


def test_labelling_pass_independent_per_teacher(tmp_path: Path) -> None:
    db = str(tmp_path / "corpus.db")
    generate_corpus(db, n_games=2)
    claude_svc = _mock_service("claude", 14)
    qwen_svc = _mock_service("qwen3_14b", 7)
    n1, _, _ = run_labelling_pass(db, claude_svc)
    n2, _, _ = run_labelling_pass(db, qwen_svc)
    assert n1 == n2
    assert DecisionLogger(db).label_count() == n1 + n2
