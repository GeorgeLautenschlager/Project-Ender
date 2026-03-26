"""Tests for Ender protocol serialisation."""

from project_ender.protocol import ActionResponse, RewardEvent, StateRequest


def test_state_request_round_trip() -> None:
    req = StateRequest(
        state_vector=[0.1, 0.2, 0.3],
        state_summary="test summary",
        valid_actions=[0, 3, 7],
    )
    assert StateRequest.from_json(req.to_json()) == req


def test_action_response_round_trip() -> None:
    resp = ActionResponse(action_id=7, confidence=0.84, source="oracle")
    assert ActionResponse.from_json(resp.to_json()) == resp


def test_reward_event_round_trip() -> None:
    event = RewardEvent(action_id=7, reward=0.4, terminal=False)
    assert RewardEvent.from_json(event.to_json()) == event


def test_reward_event_terminal() -> None:
    event = RewardEvent(action_id=14, reward=1.0, terminal=True)
    restored = RewardEvent.from_json(event.to_json())
    assert restored.terminal is True
    assert restored.reward == 1.0
