"""Ender socket server.

M0: accepts StateRequests, returns a random valid action.
M1: optionally uses the oracle (--oracle flag) to answer with a real model decision.

Usage:
    python -m project_ender.server
    python -m project_ender.server --host 127.0.0.1 --port 7373
    python -m project_ender.server --oracle
    python -m project_ender.server --oracle --oracle-backend claude
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
            logger.debug(
                "RewardEvent: action=%d reward=%.3f terminal=%s",
                _event.action_id,
                _event.reward,
                _event.terminal,
            )

    def _handle_state(self, req: StateRequest) -> ActionResponse:
        oracle_service = getattr(self.server, "_oracle_service", None)
        if oracle_service is not None:
            return self._oracle_decision(req, oracle_service)
        return self._random_decision(req)

    def _random_decision(self, req: StateRequest) -> ActionResponse:
        action_id = random.choice(req.valid_actions) if req.valid_actions else 0
        return ActionResponse(action_id=action_id, confidence=0.5, source="policy")

    def _oracle_decision(
        self, req: StateRequest, oracle_service: object
    ) -> ActionResponse:
        from project_ender.oracle import ModelService
        from project_ender.skirmish.adapter import SkirmishAdapter

        assert isinstance(oracle_service, ModelService)
        adapter = SkirmishAdapter()
        try:
            label = oracle_service.query(
                state_summary=req.state_summary,
                action_space=adapter.action_space,
                valid_actions=req.valid_actions,
            )
            return ActionResponse(
                action_id=label.action_id,
                confidence=label.confidence,
                source="oracle",
            )
        except Exception as exc:
            logger.warning("Oracle query failed (%s), falling back to random", exc)
            return self._random_decision(req)


class EnderServer:
    """Threaded TCP server wrapping the Ender protocol."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        oracle_backend: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._oracle_backend = oracle_backend
        self._server: socketserver.ThreadingTCPServer | None = None

    def start(self) -> None:
        self._server = socketserver.ThreadingTCPServer(
            (self.host, self.port), _EnderHandler
        )
        self._server.daemon_threads = True

        if self._oracle_backend is not None:
            from project_ender.oracle import ModelService

            self._server._oracle_service = ModelService(self._oracle_backend)  # type: ignore[attr-defined]
            logger.info("Oracle mode: backend=%r", self._oracle_backend)
        else:
            logger.info("Random mode (no oracle configured)")

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
    parser = argparse.ArgumentParser(description="Ender socket server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Use oracle model for decisions instead of random",
    )
    parser.add_argument(
        "--oracle-backend",
        default="claude",
        help="Oracle backend spec (default: 'claude')",
    )
    args = parser.parse_args()
    oracle_backend = args.oracle_backend if args.oracle else None
    EnderServer(host=args.host, port=args.port, oracle_backend=oracle_backend).start()


if __name__ == "__main__":
    main()
