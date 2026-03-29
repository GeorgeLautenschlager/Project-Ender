"""ClaudeCliBackend — calls the Claude CLI via subprocess."""

from __future__ import annotations

import subprocess

from .base import Backend


class ClaudeCliBackend(Backend):
    """
    Invokes the Claude Code CLI in non-interactive (print) mode.

    Requires the `claude` CLI to be installed and authenticated.
    Uses: claude -p "<prompt>"
    """

    def __init__(self) -> None:
        """Initialize Claude CLI backend."""
        super().__init__()

    @property
    def teacher_id(self) -> str:
        return "claude"

    def call(self, prompt: str, valid_actions: list[int] | None = None) -> str:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        return result.stdout.strip()
