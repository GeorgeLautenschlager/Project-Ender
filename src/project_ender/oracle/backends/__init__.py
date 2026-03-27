"""Model backends — each knows how to call one type of model via bash."""

from .base import Backend
from .claude_cli import ClaudeCliBackend

__all__ = ["Backend", "ClaudeCliBackend"]
