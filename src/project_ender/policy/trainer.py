"""BehaviouralCloningTrainer — trains a PolicyNetwork on soft-target consensus labels.

Loss function: KL divergence between the network's output distribution and
the consensus soft targets.  This is equivalent to soft-target cross-entropy
and carries more information than hard-label training — the network learns
not just what the teachers chose but how confident they were and which
alternatives they considered plausible.

Usage (CLI):
    python -m project_ender.policy.trainer \\
        --db corpus.db \\
        --state-dim 18 \\
        --action-dim 18 \\
        --epochs 50 \\
        --output policy.pt

Usage (library):
    trainer = BehaviouralCloningTrainer(network)
    n_train, n_val = trainer.load_dataset("corpus.db", min_agreement=0.0)
    history = trainer.train(epochs=50)
    trainer.save("policy.pt")
"""

from __future__ import annotations

import argparse
import logging
import random
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from project_ender.logger import DecisionLogger
from project_ender.policy.network import PolicyNetwork

log = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float | None


def _decode_blob(blob: bytes, n: int) -> list[float]:
    return list(struct.unpack(f"{n}f", blob))


class BehaviouralCloningTrainer:
    """
    Trains a PolicyNetwork on (state_vector, soft_target) pairs from a corpus DB.

    Args:
        network:      PolicyNetwork instance to train (modified in-place).
        lr:           Learning rate for Adam optimiser.
        device:       Torch device string ("cpu" or "cuda").
    """

    def __init__(
        self,
        network: PolicyNetwork,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self._net = network.to(device)
        self._device = device
        self._optimiser = torch.optim.Adam(self._net.parameters(), lr=lr)
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None

    def load_dataset(
        self,
        db_path: str,
        min_agreement: float = 0.0,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> tuple[int, int]:
        """
        Load (state_vector, soft_target) pairs from the corpus DB.

        Only states with consensus agreement_score >= min_agreement are
        included.  High-agreement states produce cleaner training signal;
        setting min_agreement=0.5 discards the noisiest half of the corpus.

        Args:
            db_path:       Path to the SQLite corpus DB.
            min_agreement: Minimum agreement score to include (0.0 = all).
            val_fraction:  Fraction of data held out for validation.
            seed:          Random seed for reproducible train/val split.

        Returns:
            (n_train, n_val)
        """
        logger = DecisionLogger(db_path)
        rows = logger.training_data(min_agreement=min_agreement)

        if not rows:
            raise ValueError(
                f"No training data found in {db_path!r} "
                f"(min_agreement={min_agreement}). "
                "Run corpus generation + labelling + consensus first."
            )

        state_dim = self._net.state_dim
        action_dim = self._net.action_dim

        states: list[list[float]] = []
        targets: list[list[float]] = []

        for row in rows:
            sv_blob = row["state_vector"]
            st_blob = row["soft_targets"]
            n_sv = len(sv_blob) // 4
            n_st = len(st_blob) // 4

            sv = _decode_blob(sv_blob, n_sv)
            st = _decode_blob(st_blob, n_st)

            # Pad or truncate to expected dimensions.
            if len(sv) < state_dim:
                sv += [0.0] * (state_dim - len(sv))
            sv = sv[:state_dim]

            if len(st) < action_dim:
                st += [0.0] * (action_dim - len(st))
            st = st[:action_dim]

            # Renormalise soft_targets in case of float rounding drift.
            total = sum(st)
            if total > 0:
                st = [v / total for v in st]

            states.append(sv)
            targets.append(st)

        # Shuffle then split.
        rng = random.Random(seed)
        indices = list(range(len(states)))
        rng.shuffle(indices)

        n_val = max(1, int(len(indices) * val_fraction)) if val_fraction > 0 else 0
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        def _make_loader(idx: list[int], shuffle: bool) -> DataLoader:
            sv_t = torch.tensor([states[i] for i in idx], dtype=torch.float32)
            st_t = torch.tensor([targets[i] for i in idx], dtype=torch.float32)
            ds = TensorDataset(sv_t, st_t)
            return DataLoader(ds, batch_size=64, shuffle=shuffle)

        self._train_loader = _make_loader(train_idx, shuffle=True)
        self._val_loader = _make_loader(val_idx, shuffle=False) if n_val > 0 else None

        log.info(
            "Dataset loaded: %d train, %d val (min_agreement=%.2f)",
            len(train_idx),
            n_val,
            min_agreement,
        )
        return len(train_idx), n_val

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 64,
    ) -> list[EpochMetrics]:
        """
        Run the training loop.

        Loss: KL divergence between log_softmax(logits) and soft_targets.
        This is equivalent to soft-target cross-entropy and is the standard
        loss for knowledge distillation.

        Args:
            epochs:     Number of full passes over the training set.
            batch_size: Mini-batch size (used if load_dataset hasn't been called
                        with a DataLoader that already sets its own batch size).

        Returns:
            List of EpochMetrics (one per epoch), useful for plotting loss curves.
        """
        if self._train_loader is None:
            raise RuntimeError("Call load_dataset() before train().")

        # Rebuild loaders with the requested batch_size if it differs.
        if self._train_loader.batch_size != batch_size:
            train_ds = self._train_loader.dataset
            val_ds = self._val_loader.dataset if self._val_loader else None
            self._train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            if val_ds is not None:
                self._val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        history: list[EpochMetrics] = []

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._eval_epoch() if self._val_loader else None

            metrics = EpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            history.append(metrics)

            if val_loss is not None:
                log.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f",
                    epoch,
                    epochs,
                    train_loss,
                    val_loss,
                )
            else:
                log.info("Epoch %3d/%d  train_loss=%.4f", epoch, epochs, train_loss)

        return history

    def _train_epoch(self) -> float:
        assert self._train_loader is not None
        self._net.train()
        total_loss = 0.0
        n_batches = 0

        for sv, st in self._train_loader:
            sv = sv.to(self._device)
            st = st.to(self._device)

            self._optimiser.zero_grad()
            logits = self._net(sv)
            # KL(soft_targets || policy) — standard soft-label distillation loss.
            loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                st,
                reduction="batchmean",
            )
            loss.backward()
            self._optimiser.step()

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    def _eval_epoch(self) -> float:
        assert self._val_loader is not None
        self._net.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for sv, st in self._val_loader:
                sv = sv.to(self._device)
                st = st.to(self._device)
                logits = self._net(sv)
                loss = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    st,
                    reduction="batchmean",
                )
                total_loss += float(loss.item())
                n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    def save(self, path: str | Path) -> None:
        """Save the network state dict to a file."""
        torch.save(self._net.state_dict(), path)
        log.info("Policy checkpoint saved to %s", path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Train a policy network via behavioural cloning"
    )
    parser.add_argument("--db", default="corpus.db", help="SQLite corpus database")
    parser.add_argument(
        "--state-dim", type=int, default=18, help="State vector length (default: 18)"
    )
    parser.add_argument(
        "--action-dim", type=int, default=18, help="Action space size (default: 18)"
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Hidden layer widths (default: 128 128)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of data held out for validation (default: 0.1)",
    )
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=0.0,
        help=(
            "Minimum consensus agreement score to include in training "
            "(0.0 = all states; 0.5 = drop bottom half)"
        ),
    )
    parser.add_argument(
        "--output", default="policy.pt", help="Checkpoint output path (default: policy.pt)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    network = PolicyNetwork(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_sizes=tuple(args.hidden_sizes),
    )
    trainer = BehaviouralCloningTrainer(network, lr=args.lr, device=device)

    n_train, n_val = trainer.load_dataset(
        args.db,
        min_agreement=args.min_agreement,
        val_fraction=args.val_fraction,
    )
    print(f"Training on {n_train} examples, validating on {n_val}.")

    history = trainer.train(epochs=args.epochs, batch_size=args.batch_size)

    # Print final loss curve summary.
    print("\nLoss curve (epoch / train / val):")
    for m in history:
        val_str = f"{m.val_loss:.4f}" if m.val_loss is not None else "  —  "
        print(f"  {m.epoch:4d}  {m.train_loss:.4f}  {val_str}")

    trainer.save(args.output)
    print(f"\nCheckpoint saved to {args.output}")


if __name__ == "__main__":
    main()
