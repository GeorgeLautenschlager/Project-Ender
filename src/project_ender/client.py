"""Thin Python client for the Ender socket protocol.

Intended for use by game adapters written in Python (e.g. Skirmish).
Other languages implement their own thin socket wrapper.

Usage:
    with EnderClient() as client:
        response = client.request(state_request)
        client.report(reward_event)
"""

from __future__ import annotations

import socket
from typing import IO

from project_ender.protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ActionResponse,
    RewardEvent,
    StateRequest,
)


class EnderClient:
    """
    Synchronous, single-connection Ender client.

    One client instance corresponds to one game session.
    The connection is kept alive for the duration of the game so that
    the server can maintain per-session context in future milestones.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._file: IO[bytes] | None = None

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self._host, self._port))
        self._file = self._sock.makefile("rb")

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> EnderClient:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def request(self, state: StateRequest) -> ActionResponse:
        """Send a StateRequest, block until ActionResponse arrives."""
        self._send(state.to_json())
        raw = self._recv()
        return ActionResponse.from_json(raw)

    def report(self, event: RewardEvent) -> None:
        """Send a RewardEvent; no response expected."""
        self._send(event.to_json())

    def _send(self, line: str) -> None:
        assert self._sock is not None, "Client not connected"
        self._sock.sendall((line + "\n").encode())

    def _recv(self) -> str:
        assert self._file is not None, "Client not connected"
        line = self._file.readline()
        if not line:
            raise ConnectionError("Server closed the connection")
        return line.decode().strip()
