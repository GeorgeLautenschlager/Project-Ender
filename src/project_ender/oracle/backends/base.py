"""Abstract base class for model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from project_ender.oracle.prompts import DEFAULT_STYLE_HINT


class Backend(ABC):
    """Knows how to send a prompt to one model and return the raw text response."""

    def __init__(self) -> None:
        """Initialize backend with default style hint."""
        self.style_hint: str = DEFAULT_STYLE_HINT

    @property
    @abstractmethod
    def teacher_id(self) -> str:
        """Stable identifier written to the labels table (e.g. 'claude', 'qwen3_14b')."""  # noqa: E501
        ...

    @abstractmethod
    def call(self, prompt: str, valid_actions: list[int] | None = None) -> str:
        """Send prompt, block until response, return raw text.

        Backends that support constrained decoding may use *valid_actions*
        to restrict the model's output to only legal action IDs.
        """
        ...
