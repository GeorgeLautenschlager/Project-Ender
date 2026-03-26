"""Tests for DomainAdapter ABC."""

import pytest
from project_ender.adapter import Action, DomainAdapter


class _ConcreteAdapter(DomainAdapter):
    """Minimal concrete implementation for testing."""

    @property
    def action_space(self) -> list[Action]:
        return [Action(id=0, name="noop", description="do nothing")]

    def encode_state(self, world: object) -> list[float]:
        return [0.0]

    def encode_state_summary(self, world: object) -> str:
        return "state"

    def decode_action(self, action_id: int) -> object:
        return action_id

    def compute_reward(self, event: object) -> float:
        return 0.0


def test_domain_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        DomainAdapter()  # type: ignore[abstract]


def test_concrete_adapter_instantiates() -> None:
    adapter = _ConcreteAdapter()
    assert adapter.action_space == [Action(id=0, name="noop", description="do nothing")]


def test_valid_action_ids_default_returns_all() -> None:
    adapter = _ConcreteAdapter()
    assert adapter.valid_action_ids(None) == [0]


def test_action_is_hashable() -> None:
    a = Action(id=1, name="foo", description="bar")
    assert hash(a) is not None
    assert a == Action(id=1, name="foo", description="bar")
