"""Tests for Skirmish game state, adapter, and rules."""

import pytest
from project_ender.skirmish.adapter import SkirmishAdapter
from project_ender.skirmish.game import (
    NUM_ZONES,
    Commander,
    GameResult,
    SkirmishState,
    step,
)

# ── State initialisation ───────────────────────────────────────────────────────


def test_initial_state_unit_counts() -> None:
    s = SkirmishState.initial()
    assert sum(s.units[Commander.RED]) == 6
    assert sum(s.units[Commander.BLUE]) == 6


def test_initial_state_no_overlap() -> None:
    s = SkirmishState.initial()
    for z in range(NUM_ZONES):
        assert not (s.units[Commander.RED][z] > 0 and s.units[Commander.BLUE][z] > 0)


def test_initial_active_commander_is_red() -> None:
    s = SkirmishState.initial()
    assert s.active_commander() == Commander.RED


def test_zones_held_initial() -> None:
    s = SkirmishState.initial()
    assert s.zones_held(Commander.RED) == 3
    assert s.zones_held(Commander.BLUE) == 3


# ── Game step ─────────────────────────────────────────────────────────────────


def test_step_increments_turn() -> None:
    s = SkirmishState.initial()
    new_s, _, _ = step(s, 14)  # Hold
    assert new_s.turn == 1


def test_step_alternates_commander() -> None:
    s = SkirmishState.initial()
    assert s.active_commander() == Commander.RED
    s, _, _ = step(s, 14)
    assert s.active_commander() == Commander.BLUE
    s, _, _ = step(s, 14)
    assert s.active_commander() == Commander.RED


def test_step_on_terminal_raises() -> None:
    s = SkirmishState.initial()
    s.result = GameResult(winner=Commander.RED, turn=1, reason="test")
    with pytest.raises(ValueError, match="already over"):
        step(s, 14)


def test_hold_does_not_change_units() -> None:
    s = SkirmishState.initial()
    before_red = list(s.units[Commander.RED])
    before_blue = list(s.units[Commander.BLUE])
    new_s, _, _ = step(s, 14)  # Red holds
    assert new_s.units[Commander.RED] == before_red
    assert new_s.units[Commander.BLUE] == before_blue


def test_reinforce_moves_unit() -> None:
    s = SkirmishState.initial()
    # Red zones: 0,1,2 each have 2 units. Reinforce zone 3 from zone 2.
    before_z2 = s.units[Commander.RED][2]
    before_z3 = s.units[Commander.RED][3]
    new_s, _, _ = step(s, 3)  # Reinforce zone 3
    assert new_s.units[Commander.RED][2] == before_z2 - 1
    assert new_s.units[Commander.RED][3] == before_z3 + 1


# ── Win conditions ────────────────────────────────────────────────────────────


def test_elimination_win() -> None:
    """Red wipes out Blue in one step."""
    s = SkirmishState.initial()
    # Give Red all units in zone 0; Blue has 1 unit in zone 1 only
    s.units[Commander.RED] = [12, 0, 0, 0, 0, 0, 0]
    s.units[Commander.BLUE] = [0, 1, 0, 0, 0, 0, 0]
    # Red attacks zone 1 (action 8) from zone 0 with 2 units
    new_s, reward_red, _ = step(s, 8)
    # After attack: Red moves 2 into zone 1, Blue has 1 -> combat -> both lose 1, Blue=0
    # Blue total = 0 → Red wins
    assert new_s.result is not None
    assert new_s.result.winner == Commander.RED
    assert new_s.result.reason == "elimination"
    assert reward_red > 0


def test_zone_control_win_accumulates() -> None:
    """Consecutive hold counter increments and triggers win."""
    s = SkirmishState.initial()
    # Give Red 4+ zones immediately
    s.units[Commander.RED] = [2, 2, 2, 2, 0, 0, 0]
    s.units[Commander.BLUE] = [0, 0, 0, 0, 2, 2, 2]

    # Run 6 holds (3 red, 3 blue); Red should win by zone control
    for _ in range(6):
        if s.is_terminal():
            break
        s, _, _ = step(s, 14)

    assert s.result is not None
    assert s.result.winner == Commander.RED
    assert s.result.reason == "zone control"


# ── SkirmishAdapter ───────────────────────────────────────────────────────────


def test_adapter_state_vector_length() -> None:
    adapter = SkirmishAdapter()
    s = SkirmishState.initial()
    vec = adapter.encode_state(s)
    # friendly(7) + enemy(7) + zones_held_f + zones_held_e + turns + hold_streak = 18
    assert len(vec) == 18


def test_adapter_state_vector_normalised() -> None:
    adapter = SkirmishAdapter()
    s = SkirmishState.initial()
    vec = adapter.encode_state(s)
    assert all(0.0 <= v <= 1.0 for v in vec)


def test_adapter_action_space_size() -> None:
    adapter = SkirmishAdapter()
    assert len(adapter.action_space) == 18


def test_adapter_decode_action() -> None:
    adapter = SkirmishAdapter()
    assert "Hold" in adapter.decode_action(14)
    assert "Reinforce zone 3" in adapter.decode_action(3)
    assert "Attack zone 0" in adapter.decode_action(7)


def test_adapter_valid_actions_includes_hold() -> None:
    adapter = SkirmishAdapter()
    s = SkirmishState.initial()
    valid = adapter.valid_action_ids(s)
    assert 14 in valid


def test_adapter_encode_state_summary_is_string() -> None:
    adapter = SkirmishAdapter()
    s = SkirmishState.initial()
    summary = adapter.encode_state_summary(s)
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_adapter_compute_reward_from_dict() -> None:
    adapter = SkirmishAdapter()
    assert adapter.compute_reward({"reward": 0.3}) == pytest.approx(0.3)
    assert adapter.compute_reward({"reward": -1.0}) == pytest.approx(-1.0)
