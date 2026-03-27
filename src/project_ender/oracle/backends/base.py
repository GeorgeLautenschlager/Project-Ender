"""Abstract base class for model backends."""

from abc import ABC, abstractmethod


class Backend(ABC):
    """Knows how to send a prompt to one model and return the raw text response."""

    @property
    @abstractmethod
    def teacher_id(self) -> str:
        """Stable identifier written to the labels table (e.g. 'claude', 'qwen3_14b')."""  # noqa: E501
        ...

    @abstractmethod
    def call(self, prompt: str) -> str:
        """Send prompt, block until response, return raw text."""
        ...
