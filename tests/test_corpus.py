"""Tests for corpus generation."""

from __future__ import annotations

from pathlib import Path

from project_ender.corpus import generate_corpus
from project_ender.logger import DecisionLogger


def test_generate_corpus_populates_states(tmp_path: Path) -> None:
    db_path = str(tmp_path / "corpus.db")
    total = generate_corpus(db_path, n_games=3)
    assert total > 0
    logger = DecisionLogger(db_path)
    assert logger.state_count() == total


def test_generate_corpus_states_have_content(tmp_path: Path) -> None:
    db_path = str(tmp_path / "corpus.db")
    generate_corpus(db_path, n_games=2)
    logger = DecisionLogger(db_path)
    unlabelled = logger.unlabelled_states("claude")
    assert len(unlabelled) > 0
    row = unlabelled[0]
    assert row["state_summary"]
    assert row["valid_action_ids"]
