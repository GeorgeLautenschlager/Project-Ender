"""Integration test: client <-> server round-trip."""

from __future__ import annotations

import time
from collections.abc import Generator

import pytest
from project_ender.client import EnderClient
from project_ender.protocol import RewardEvent, StateRequest
from project_ender.server import EnderServer, _find_free_port


@pytest.fixture()
def running_server() -> Generator[EnderServer, None, None]:
    port = _find_free_port()
    server = EnderServer(host="127.0.0.1", port=port)
    server.start_background()
    time.sleep(0.05)  # give the thread time to bind
    yield server
    server.stop()


def test_state_request_returns_valid_action(running_server: EnderServer) -> None:
    req = StateRequest(
        state_vector=[0.5] * 18,
        state_summary="test",
        valid_actions=[0, 7, 14],
    )
    with EnderClient(host=running_server.host, port=running_server.port) as client:
        resp = client.request(req)

    assert resp.action_id in [0, 7, 14]
    assert 0.0 <= resp.confidence <= 1.0
    assert resp.source in ("policy", "oracle")


def test_reward_event_does_not_crash(running_server: EnderServer) -> None:
    req = StateRequest(
        state_vector=[0.1] * 18,
        state_summary="test",
        valid_actions=[14],
    )
    event = RewardEvent(action_id=14, reward=0.3, terminal=False)
    with EnderClient(host=running_server.host, port=running_server.port) as client:
        resp = client.request(req)
        client.report(event)  # should not raise

    assert resp.action_id == 14


def test_multiple_requests_same_connection(running_server: EnderServer) -> None:
    with EnderClient(host=running_server.host, port=running_server.port) as client:
        for _ in range(5):
            req = StateRequest(
                state_vector=[0.0] * 18,
                state_summary="multi",
                valid_actions=[1, 2, 3],
            )
            resp = client.request(req)
            assert resp.action_id in [1, 2, 3]
