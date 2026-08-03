"""Wall-time + epoch training budgets with fractional patience.

Epoch-only budgets bias comparisons: a fast epoch burns stall patience in
little wall time, while a slow epoch gets more compute per patience count.
This module stops on **either** wall budget or epoch cap, and measures
patience as a **fraction** of both budgets.

Design rationale and default choices: ``docs/TRAIN_BUDGET.md``
(protocol id ``wall_epoch_budget_v1``).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional


# Defaults for MNIST-scale experiment loops (fair atlas comparisons).
DEFAULT_MAX_EPOCHS = 80
DEFAULT_MAX_WALL_SEC = 1200.0  # 20 minutes wall clock per run
DEFAULT_PATIENCE_FRAC = 0.125  # 12.5% of epochs and of wall budget
DEFAULT_MIN_DELTA = 0.0


@dataclass
class TrainBudget:
    """Resolved train limits for one run."""

    max_epochs: int = DEFAULT_MAX_EPOCHS
    max_wall_sec: Optional[float] = DEFAULT_MAX_WALL_SEC
    patience_frac: float = DEFAULT_PATIENCE_FRAC
    min_delta: float = DEFAULT_MIN_DELTA
    # Resolved absolute patience (derived from frac, or CLI override).
    patience_epochs: int = 0
    patience_wall_sec: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")
        if self.patience_frac < 0 or self.patience_frac > 1:
            raise ValueError(f"patience_frac must be in [0, 1], got {self.patience_frac}")
        if self.patience_epochs <= 0:
            self.patience_epochs = max(1, int(round(self.patience_frac * self.max_epochs)))
        if self.max_wall_sec is not None and self.max_wall_sec <= 0:
            self.max_wall_sec = None
        if self.patience_wall_sec is None and self.max_wall_sec is not None:
            self.patience_wall_sec = max(
                1.0, float(self.patience_frac) * float(self.max_wall_sec)
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_budget(
    *,
    epochs: int = DEFAULT_MAX_EPOCHS,
    max_wall_sec: Optional[float] = DEFAULT_MAX_WALL_SEC,
    patience_frac: float = DEFAULT_PATIENCE_FRAC,
    patience: Optional[int] = None,
    min_delta: float = DEFAULT_MIN_DELTA,
) -> TrainBudget:
    """Build a budget. ``patience`` if set (>=1) overrides epoch patience only."""
    pe = int(patience) if patience is not None and patience >= 1 else 0
    return TrainBudget(
        max_epochs=int(epochs),
        max_wall_sec=max_wall_sec,
        patience_frac=float(patience_frac),
        min_delta=float(min_delta),
        patience_epochs=pe,
        patience_wall_sec=None,  # filled in __post_init__ from frac * wall
    )


def add_budget_args(
    parser: argparse.ArgumentParser,
    *,
    epochs: int = DEFAULT_MAX_EPOCHS,
    max_wall_sec: float = DEFAULT_MAX_WALL_SEC,
    patience_frac: float = DEFAULT_PATIENCE_FRAC,
    patience: Optional[int] = None,
    min_delta: float = DEFAULT_MIN_DELTA,
) -> argparse.ArgumentParser:
    """Add standard budget CLI flags (idempotent names)."""
    parser.add_argument(
        "--epochs",
        type=int,
        default=epochs,
        help=f"Hard max epochs (default {epochs})",
    )
    parser.add_argument(
        "--max-wall-sec",
        type=float,
        default=max_wall_sec,
        help=(
            f"Wall-clock budget in seconds per run (default {max_wall_sec}; "
            "<=0 disables wall limit)"
        ),
    )
    parser.add_argument(
        "--patience-frac",
        type=float,
        default=patience_frac,
        help=(
            f"Patience as fraction of max epochs and of wall budget "
            f"(default {patience_frac} → "
            f"~{max(1, int(round(patience_frac * epochs)))} epochs / "
            f"{patience_frac * max_wall_sec:.0f}s wall)"
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=patience if patience is not None else -1,
        help=(
            "Absolute epoch patience override; <0 uses patience-frac * epochs "
            "(default -1)"
        ),
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=min_delta,
        help="Min test metric gain over best to count as improvement (default 0)",
    )
    return parser


def budget_from_args(args: argparse.Namespace) -> TrainBudget:
    wall = float(getattr(args, "max_wall_sec", DEFAULT_MAX_WALL_SEC))
    if wall <= 0:
        wall_opt: Optional[float] = None
    else:
        wall_opt = wall
    pat = int(getattr(args, "patience", -1))
    return resolve_budget(
        epochs=int(args.epochs),
        max_wall_sec=wall_opt,
        patience_frac=float(getattr(args, "patience_frac", DEFAULT_PATIENCE_FRAC)),
        patience=pat if pat >= 1 else None,
        min_delta=float(getattr(args, "min_delta", DEFAULT_MIN_DELTA)),
    )


@dataclass
class StopDecision:
    stop: bool
    reason: str
    improved: bool


class EarlyStopTracker:
    """Track best metric; stop on epoch patience, wall patience, or wall budget."""

    def __init__(self, budget: TrainBudget):
        self.budget = budget
        self.best = float("-inf")
        self.best_epoch = 0
        self.stall_epochs = 0
        self.t0 = time.time()
        self.t_at_best = self.t0

    @property
    def wall_sec(self) -> float:
        return time.time() - self.t0

    @property
    def stall_wall_sec(self) -> float:
        return time.time() - self.t_at_best

    def observe(self, epoch: int, metric: float) -> StopDecision:
        """Call once per epoch after evaluation. Returns whether to stop."""
        b = self.budget
        improved = metric > self.best + b.min_delta
        if improved:
            self.best = float(metric)
            self.best_epoch = int(epoch)
            self.stall_epochs = 0
            self.t_at_best = time.time()
        else:
            self.stall_epochs += 1

        wall = self.wall_sec
        if b.max_wall_sec is not None and wall >= b.max_wall_sec:
            return StopDecision(True, "max_wall_sec", improved)
        if (
            b.patience_wall_sec is not None
            and self.stall_wall_sec >= b.patience_wall_sec
            and not improved
        ):
            return StopDecision(True, "patience_wall", improved)
        if self.stall_epochs >= b.patience_epochs and not improved:
            return StopDecision(True, "patience_epochs", improved)
        if epoch >= b.max_epochs:
            return StopDecision(True, "max_epochs", improved)
        return StopDecision(False, "", improved)

    def status_str(self) -> str:
        b = self.budget
        wall = self.wall_sec
        parts = [
            f"best={self.best:.4f}@{self.best_epoch}",
            f"stall_ep={self.stall_epochs}/{b.patience_epochs}",
        ]
        if b.patience_wall_sec is not None:
            parts.append(
                f"stall_wall={self.stall_wall_sec:.0f}/{b.patience_wall_sec:.0f}s"
            )
        if b.max_wall_sec is not None:
            parts.append(f"wall={wall:.0f}/{b.max_wall_sec:.0f}s")
        else:
            parts.append(f"wall={wall:.0f}s")
        return " ".join(parts)

    def meta_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget.to_dict(),
            "best_metric": self.best if self.best > float("-inf") else None,
            "best_epoch": self.best_epoch,
            "stall_epochs": self.stall_epochs,
            "wall_sec": self.wall_sec,
            "stall_wall_sec": self.stall_wall_sec,
        }
