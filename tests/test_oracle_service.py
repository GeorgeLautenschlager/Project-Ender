"""Tests for ModelService: prompt building, JSON parsing, error handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from project_ender.oracle import ModelService
from project_ender.oracle.backends.claude_cli import ClaudeCliBackend
from project_ender.oracle.prompts import build_prompt
from project_ender.oracle.service import _build_backend
from project_ender.skirmish.adapter import SkirmishAdapter

_adapter = SkirmishAdapter()
_action_space = _adapter.action_space
_valid = [0, 7, 14]


def _make_service(raw_response: str) -> ModelService:
    """Return a ModelService whose backend always returns raw_response."""
    svc = ModelService.__new__(ModelService)
    mock_backend = MagicMock()
    mock_backend.teacher_id = "claude"
    mock_backend.call.return_value = raw_response
    svc._backend = mock_backend
    return svc


# --- build_backend ---


def test_build_backend_claude() -> None:
    backend = _build_backend("claude")
    assert isinstance(backend, ClaudeCliBackend)
    assert backend.teacher_id == "claude"


def test_build_backend_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend spec"):
        _build_backend("unknown:model")


# --- build_prompt ---


def test_build_prompt_contains_state_summary() -> None:
    prompt = build_prompt("State: zone 3 contested", _action_space, _valid)
    assert "State: zone 3 contested" in prompt


def test_build_prompt_lists_valid_actions_only() -> None:
    prompt = build_prompt("summary", _action_space, [14])
    assert "14: Hold" in prompt
    # Action 7 not in valid list
    assert "7:" not in prompt


def test_build_prompt_contains_json_schema() -> None:
    prompt = build_prompt("summary", _action_space, _valid)
    assert "action_id" in prompt
    assert "confidence" in prompt
    assert "reasoning" in prompt


# --- ModelService.query / _parse ---


def test_query_clean_json() -> None:
    raw = json.dumps({"action_id": 7, "confidence": 0.9, "reasoning": "attack"})
    svc = _make_service(raw)
    label = svc.query("state", _action_space, _valid)
    assert label.action_id == 7
    assert label.confidence == 0.9
    assert label.reasoning == "attack"
    assert label.teacher == "claude"


def test_query_defaults_confidence_to_1() -> None:
    raw = json.dumps({"action_id": 14, "reasoning": "hold"})
    svc = _make_service(raw)
    label = svc.query("state", _action_space, _valid)
    assert label.confidence == 1.0


def test_query_strips_markdown_fences() -> None:
    raw = (
        "```json\n"
        + json.dumps({"action_id": 0, "confidence": 0.5, "reasoning": "r"})
        + "\n```"
    )
    svc = _make_service(raw)
    label = svc.query("state", _action_space, _valid)
    assert label.action_id == 0


def test_query_raises_on_non_json() -> None:
    svc = _make_service("Sorry, I cannot decide.")
    with pytest.raises(ValueError, match="non-JSON"):
        svc.query("state", _action_space, _valid)


def test_query_raises_on_invalid_action_id() -> None:
    # action_id 5 is not in _valid = [0, 7, 14]
    raw = json.dumps({"action_id": 5, "confidence": 0.8, "reasoning": "r"})
    svc = _make_service(raw)
    with pytest.raises(ValueError, match="not in valid_actions"):
        svc.query("state", _action_space, _valid)


def test_query_raises_on_missing_action_id() -> None:
    raw = json.dumps({"confidence": 0.8, "reasoning": "r"})
    svc = _make_service(raw)
    with pytest.raises(ValueError, match="missing valid 'action_id'"):
        svc.query("state", _action_space, _valid)


# --- ClaudeCliBackend subprocess call ---


def test_claude_cli_backend_calls_subprocess() -> None:
    backend = ClaudeCliBackend()
    mock_result = MagicMock()
    mock_result.stdout = '{"action_id": 14, "confidence": 1.0, "reasoning": "hold"}\n'
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = backend.call("some prompt")
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "claude"
    assert args[1] == "-p"
    assert "some prompt" in args[2]
    assert result == '{"action_id": 14, "confidence": 1.0, "reasoning": "hold"}'
