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

    @property
    def teacher_id(self) -> str:
        return "claude"

    def call(self, prompt: str) -> str:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        return result.stdout.strip()
