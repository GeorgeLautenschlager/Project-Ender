"""Tests for the stratified corpus generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from project_ender.logger import DecisionLogger
from project_ender.skirmish.game import Commander, MAX_TURNS
from project_ender.stratified_corpus import (
    BUCKET_KEYS,
    PERSPECTIVES,
    PHASES,
    _phase,
    _perspective,
    generate_stratified_corpus,
)


# ---------------------------------------------------------------------------
# Unit tests for classification helpers
# ---------------------------------------------------------------------------


class TestPhase:
    def test_early_start(self):
        assert _phase(0) == "early"

    def test_early_end(self):
        assert _phase(MAX_TURNS // 3) == "early"  # turn 66

    def test_mid_start(self):
        assert _phase(MAX_TURNS // 3 + 1) == "mid"  # turn 67

    def test_mid_end(self):
        assert _phase(2 * MAX_TURNS // 3) == "mid"  # turn 133

    def test_late_start(self):
        assert _phase(2 * MAX_TURNS // 3 + 1) == "late"  # turn 134

    def test_late_end(self):
        assert _phase(MAX_TURNS) == "late"  # turn 200


class TestPerspective:
    def test_red_wins_red_active(self):
        assert _perspective(Commander.RED, Commander.RED) == "winning"

    def test_red_wins_blue_active(self):
        assert _perspective(Commander.BLUE, Commander.RED) == "losing"

    def test_blue_wins_blue_active(self):
        assert _perspective(Commander.BLUE, Commander.BLUE) == "winning"

    def test_blue_wins_red_active(self):
        assert _perspective(Commander.RED, Commander.BLUE) == "losing"

    def test_draw_red_active(self):
        assert _perspective(Commander.RED, None) == "draw"

    def test_draw_blue_active(self):
        assert _perspective(Commander.BLUE, None) == "draw"


class TestBucketKeys:
    def test_exactly_nine_buckets(self):
        assert len(BUCKET_KEYS) == 9

    def test_all_phases_present(self):
        phases_in_keys = {phase for phase, _ in BUCKET_KEYS}
        assert phases_in_keys == set(PHASES)

    def test_all_perspectives_present(self):
        perspectives_in_keys = {perspective for _, perspective in BUCKET_KEYS}
        assert perspectives_in_keys == set(PERSPECTIVES)

    def test_all_combinations_present(self):
        expected = {(p, v) for p in PHASES for v in PERSPECTIVES}
        assert set(BUCKET_KEYS) == expected


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> str:
    return str(tmp_path / "strat.db")


class TestGenerateStratifiedCorpus:
    def test_states_written_to_db(self, db: str):
        stats = generate_stratified_corpus(db, states_per_bucket=5, max_games=300)
        logger = DecisionLogger(db)
        assert logger.state_count() == sum(stats.values())

    def test_returns_nine_bucket_keys(self, db: str):
        stats = generate_stratified_corpus(db, states_per_bucket=5, max_games=300)
        assert set(stats.keys()) == {f"{p}_{v}" for p in PHASES for v in PERSPECTIVES}

    def test_no_bucket_exceeds_quota(self, db: str):
        quota = 5
        stats = generate_stratified_corpus(db, states_per_bucket=quota, max_games=300)
        for key, count in stats.items():
            assert count <= quota, f"Bucket '{key}' has {count} > {quota} states"

    def test_states_have_content(self, db: str):
        generate_stratified_corpus(db, states_per_bucket=3, max_games=200)
        logger = DecisionLogger(db)
        # unlabelled_states returns all states (no labels yet)
        rows = logger.unlabelled_states("dummy_teacher")
        assert len(rows) > 0
        for row in rows:
            assert row["state_summary"]
            assert row["valid_action_ids"]

    def test_seed_reproducibility(self, tmp_path: Path):
        db1 = str(tmp_path / "a.db")
        db2 = str(tmp_path / "b.db")
        stats1 = generate_stratified_corpus(db1, states_per_bucket=5, max_games=200, seed=42)
        stats2 = generate_stratified_corpus(db2, states_per_bucket=5, max_games=200, seed=42)
        assert stats1 == stats2
