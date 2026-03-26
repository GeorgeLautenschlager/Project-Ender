"""Ender socket server stub.

M0: accepts StateRequests, returns a random valid action.
M1+: replaced by oracle/policy commander blend.

Usage:
    python -m project_ender.server
    python -m project_ender.server --host 127.0.0.1 --port 7373
"""

from __future__ import annotations

import argparse
import logging
import random
import socket
import socketserver
import threading

from project_ender.protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ActionResponse,
    RewardEvent,
    StateRequest,
)

logger = logging.getLogger(__name__)


class _EnderHandler(socketserver.StreamRequestHandler):
    """Handle one client connection."""

    def handle(self) -> None:
        peer = self.client_address
        logger.info("Connection from %s", peer)
        try:
            for raw_line in self.rfile:
                line = raw_line.decode().strip()
                if not line:
                    continue
                self._dispatch(line)
        except ConnectionResetError:
            pass
        logger.info("Disconnected %s", peer)

    def _dispatch(self, line: str) -> None:
        # Detect message type by key presence
        try:
            import json

            data = json.loads(line)
        except ValueError:
            logger.warning("Invalid JSON: %r", line)
            return

        if "state_vector" in data:
            req = StateRequest.from_json(line)
            resp = self._handle_state(req)
            self.wfile.write((resp.to_json() + "\n").encode())
            self.wfile.flush()
        elif "terminal" in data:
            _event = RewardEvent.from_json(line)
            # M0 stub: log and discard
            logger.debug("RewardEvent: action=%d reward=%.3f terminal=%s",
                         _event.action_id, _event.reward, _event.terminal)

    def _handle_state(self, req: StateRequest) -> ActionResponse:
        """M0 stub: pick a random valid action with uniform confidence."""
        action_id = random.choice(req.valid_actions) if req.valid_actions else 0
        return ActionResponse(action_id=action_id, confidence=0.5, source="policy")


class EnderServer:
    """Threaded TCP server wrapping the Ender protocol stub."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port
        self._server: socketserver.ThreadingTCPServer | None = None

    def start(self) -> None:
        self._server = socketserver.ThreadingTCPServer(
            (self.host, self.port), _EnderHandler
        )
        self._server.daemon_threads = True
        logger.info("Ender server listening on %s:%d", self.host, self.port)
        self._server.serve_forever()

    def start_background(self) -> threading.Thread:
        """Start serving in a daemon thread; returns the thread."""
        t = threading.Thread(target=self.start, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None

    @property
    def address(self) -> tuple[str, int]:
        return (self.host, self.port)


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ender socket server (M0 stub)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    EnderServer(host=args.host, port=args.port).start()


if __name__ == "__main__":
    main()
