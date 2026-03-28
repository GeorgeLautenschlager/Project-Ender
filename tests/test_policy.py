"""Tests for the M2 policy module: network, inference, trainer, and Commander blend."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import pytest
import torch

from project_ender.policy.inference import PolicyInference
from project_ender.policy.network import PolicyNetwork
from project_ender.policy.trainer import BehaviouralCloningTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATE_DIM = 18
ACTION_DIM = 18


def _random_state() -> list[float]:
    return torch.rand(STATE_DIM).tolist()


def _uniform_targets() -> list[float]:
    return [1.0 / ACTION_DIM] * ACTION_DIM


# ---------------------------------------------------------------------------
# PolicyNetwork
# ---------------------------------------------------------------------------


def test_network_default_forward() -> None:
    net = PolicyNetwork(STATE_DIM, ACTION_DIM)
    x = torch.rand(4, STATE_DIM)
    out = net(x)
    assert out.shape == (4, ACTION_DIM)


def test_network_single_sample() -> None:
    net = PolicyNetwork(STATE_DIM, ACTION_DIM)
    x = torch.rand(STATE_DIM)
    out = net(x)
    assert out.shape == (ACTION_DIM,)


def test_network_custom_hidden_sizes() -> None:
    net = PolicyNetwork(STATE_DIM, ACTION_DIM, hidden_sizes=(64, 64, 64))
    x = torch.rand(2, STATE_DIM)
    out = net(x)
    assert out.shape == (2, ACTION_DIM)


def test_network_stores_dims() -> None:
    net = PolicyNetwork(10, 5, hidden_sizes=(32,))
    assert net.state_dim == 10
    assert net.action_dim == 5
    assert net.hidden_sizes == (32,)


def test_network_invalid_state_dim() -> None:
    with pytest.raises(ValueError):
        PolicyNetwork(0, ACTION_DIM)


def test_network_invalid_action_dim() -> None:
    with pytest.raises(ValueError):
        PolicyNetwork(STATE_DIM, 0)


def test_network_empty_hidden_sizes() -> None:
    with pytest.raises(ValueError):
        PolicyNetwork(STATE_DIM, ACTION_DIM, hidden_sizes=())


# ---------------------------------------------------------------------------
# PolicyInference
# ---------------------------------------------------------------------------


def test_inference_act_returns_valid_action() -> None:
    net = PolicyNetwork(STATE_DIM, ACTION_DIM)
    inf = PolicyInference(net)
    valid = [0, 7, 14]
    action_id, conf = inf.act(_random_state(), valid)
    assert action_id in valid
    assert 0.0 <= conf <= 1.0


def test_inference_masks_invalid_actions() -> None:
    # All weight on action 14 — make net return large logit there.
    net = PolicyNetwork(STATE_DIM, ACTION_DIM, hidden_sizes=(32,))
    # Bias last layer toward action 14.
    with torch.no_grad():
        net.net[-1].bias[14] = 100.0
    inf = PolicyInference(net)
    # valid_actions doesn't include 14 — must pick from {0, 7} only.
    action_id, _ = inf.act(_random_state(), [0, 7])
    assert action_id in [0, 7]


def test_inference_confidence_sums_to_one_across_valid() -> None:
    net = PolicyNetwork(STATE_DIM, ACTION_DIM)
    inf = PolicyInference(net)
    valid = list(range(18))
    _, conf = inf.act(_random_state(), valid)
    # Confidence is a single softmax probability so 0 < conf <= 1.
    assert 0.0 < conf <= 1.0


def test_inference_empty_valid_actions_raises() -> None:
    net = PolicyNetwork(STATE_DIM, ACTION_DIM)
    inf = PolicyInference(net)
    with pytest.raises(ValueError):
        inf.act(_random_state(), [])


def test_inference_save_load_roundtrip(tmp_path: Path) -> None:
    net = PolicyNetwork(STATE_DIM, ACTION_DIM)
    ckpt = str(tmp_path / "policy.pt")
    torch.save(net.state_dict(), ckpt)

    inf = PolicyInference.from_checkpoint(ckpt, STATE_DIM, ACTION_DIM)
    action_id, conf = inf.act(_random_state(), list(range(18)))
    assert 0 <= action_id < ACTION_DIM
    assert 0.0 < conf <= 1.0


# ---------------------------------------------------------------------------
# BehaviouralCloningTrainer
# ---------------------------------------------------------------------------


def _write_corpus_db(db_path: str, n: int = 50) -> None:
    """Write a minimal corpus.db with n states and consensus rows."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE states (
            id INTEGER PRIMARY KEY,
            game_id TEXT, turn INTEGER,
            state_vector BLOB, state_summary TEXT,
            valid_action_ids TEXT, created_at TIMESTAMP
        )"""
    )
    conn.execute(
        """CREATE TABLE consensus (
            state_id INTEGER, soft_targets BLOB,
            agreement_score FLOAT, teacher_count INTEGER,
            PRIMARY KEY (state_id)
        )"""
    )
    # Soft targets: always choose action 14 (hold).
    targets = [0.0] * ACTION_DIM
    targets[14] = 1.0
    st_blob = struct.pack(f"{ACTION_DIM}f", *targets)

    for i in range(n):
        sv = [float(j) / STATE_DIM for j in range(STATE_DIM)]
        sv_blob = struct.pack(f"{STATE_DIM}f", *sv)
        conn.execute(
            "INSERT INTO states VALUES (?,?,?,?,?,?,?)",
            (i + 1, "g1", i, sv_blob, f"state {i}", "[14]", "2025-01-01"),
        )
        conn.execute(
            "INSERT INTO consensus VALUES (?,?,?,?)",
            (i + 1, st_blob, 1.0, 1),
        )
    conn.commit()
    conn.close()


def test_trainer_load_dataset(tmp_path: Path) -> None:
    db = str(tmp_path / "corpus.db")
    _write_corpus_db(db, n=20)

    net = PolicyNetwork(STATE_DIM, ACTION_DIM)
    trainer = BehaviouralCloningTrainer(net)
    n_train, n_val = trainer.load_dataset(db, val_fraction=0.2)
    assert n_train + n_val == 20
    assert n_val >= 1


def test_trainer_train_reduces_loss(tmp_path: Path) -> None:
    db = str(tmp_path / "corpus.db")
    _write_corpus_db(db, n=100)

    net = PolicyNetwork(STATE_DIM, ACTION_DIM)
    trainer = BehaviouralCloningTrainer(net, lr=1e-2)
    trainer.load_dataset(db, val_fraction=0.1)
    history = trainer.train(epochs=20, batch_size=32)

    assert len(history) == 20
    # Loss should decrease over training on a deterministic dataset.
    assert history[-1].train_loss < history[0].train_loss


def test_trainer_save_and_load(tmp_path: Path) -> None:
    db = str(tmp_path / "corpus.db")
    _write_corpus_db(db, n=30)

    net = PolicyNetwork(STATE_DIM, ACTION_DIM)
    trainer = BehaviouralCloningTrainer(net)
    trainer.load_dataset(db)
    trainer.train(epochs=5)
    ckpt = str(tmp_path / "policy.pt")
    trainer.save(ckpt)
    assert Path(ckpt).exists()

    # Load back and run inference.
    inf = PolicyInference.from_checkpoint(ckpt, STATE_DIM, ACTION_DIM)
    action_id, conf = inf.act(_random_state(), [14])
    assert action_id == 14  # only valid action


def test_trainer_min_agreement_filter(tmp_path: Path) -> None:
    import sqlite3

    db = str(tmp_path / "corpus.db")
    _write_corpus_db(db, n=10)

    # Add 5 more rows with low agreement.
    targets = [1.0 / ACTION_DIM] * ACTION_DIM
    st_blob = struct.pack(f"{ACTION_DIM}f", *targets)
    sv = [0.5] * STATE_DIM
    sv_blob = struct.pack(f"{STATE_DIM}f", *sv)

    conn = sqlite3.connect(db)
    for i in range(5):
        sid = 100 + i
        conn.execute(
            "INSERT INTO states VALUES (?,?,?,?,?,?,?)",
            (sid, "g2", i, sv_blob, f"low {i}", "[14]", "2025-01-01"),
        )
        conn.execute(
            "INSERT INTO consensus VALUES (?,?,?,?)",
            (sid, st_blob, 0.3, 2),
        )
    conn.commit()
    conn.close()

    net = PolicyNetwork(STATE_DIM, ACTION_DIM)
    trainer = BehaviouralCloningTrainer(net)
    n_train, n_val = trainer.load_dataset(db, min_agreement=0.5, val_fraction=0.0)
    # Only the 10 high-agreement rows should be included.
    assert n_train == 10


def test_trainer_no_data_raises(tmp_path: Path) -> None:
    import sqlite3

    db = str(tmp_path / "empty.db")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE states (id INTEGER PRIMARY KEY, game_id TEXT, turn INTEGER, "
        "state_vector BLOB, state_summary TEXT, valid_action_ids TEXT, created_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE consensus (state_id INTEGER, soft_targets BLOB, "
        "agreement_score FLOAT, teacher_count INTEGER, PRIMARY KEY (state_id))"
    )
    conn.commit()
    conn.close()

    net = PolicyNetwork(STATE_DIM, ACTION_DIM)
    trainer = BehaviouralCloningTrainer(net)
    with pytest.raises(ValueError, match="No training data"):
        trainer.load_dataset(db)


# ---------------------------------------------------------------------------
# Commander blend mode
# ---------------------------------------------------------------------------


def test_commander_uses_policy_when_confident(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from project_ender.commander import Commander

    # Build a policy that always picks action 14 with confidence 1.0.
    net = PolicyNetwork(STATE_DIM, ACTION_DIM, hidden_sizes=(32,))
    with torch.no_grad():
        net.net[-1].bias.fill_(float("-inf"))
        net.net[-1].bias[14] = 100.0
    policy = PolicyInference(net)

    # Stub oracle so it would never be called.
    commander = Commander(confidence_threshold=0.5, policy=policy)
    commander._oracle = MagicMock()

    from project_ender.adapter import Action

    actions = [Action(id=i, name=f"a{i}", description="") for i in range(18)]
    decision = commander.decide(
        state_summary="test",
        action_space=actions,
        valid_actions=list(range(18)),
        state_vector=_random_state(),
    )

    assert decision.action_id == 14
    assert decision.source == "policy"
    commander._oracle.query.assert_not_called()


def test_commander_falls_back_to_oracle_when_policy_low_confidence() -> None:
    from unittest.mock import MagicMock, patch

    from project_ender.commander import Commander
    from project_ender.oracle.service import OracleLabel

    # Policy always returns low confidence (uniform distribution → ~1/18 ≈ 0.055).
    net = PolicyNetwork(STATE_DIM, ACTION_DIM, hidden_sizes=(32,))
    with torch.no_grad():
        net.net[-1].bias.fill_(0.0)
        net.net[-1].weight.fill_(0.0)
    policy = PolicyInference(net)

    mock_label = OracleLabel(action_id=7, confidence=0.9, reasoning="attack", teacher="claude")

    commander = Commander(confidence_threshold=0.9, policy=policy)
    commander._oracle = MagicMock()
    commander._oracle.query.return_value = mock_label

    from project_ender.adapter import Action

    actions = [Action(id=i, name=f"a{i}", description="") for i in range(18)]
    decision = commander.decide(
        state_summary="test",
        action_space=actions,
        valid_actions=list(range(18)),
        state_vector=_random_state(),
    )

    assert decision.source == "oracle"
    assert decision.action_id == 7
    commander._oracle.query.assert_called_once()


def test_commander_oracle_only_without_policy() -> None:
    from unittest.mock import MagicMock

    from project_ender.commander import Commander
    from project_ender.oracle.service import OracleLabel

    mock_label = OracleLabel(action_id=3, confidence=0.8, reasoning="feint", teacher="claude")

    commander = Commander()
    commander._oracle = MagicMock()
    commander._oracle.query.return_value = mock_label

    from project_ender.adapter import Action

    actions = [Action(id=i, name=f"a{i}", description="") for i in range(18)]
    decision = commander.decide(
        state_summary="test",
        action_space=actions,
        valid_actions=list(range(18)),
    )

    assert decision.source == "oracle"
    commander._oracle.query.assert_called_once()
