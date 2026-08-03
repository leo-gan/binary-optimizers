"""Pure wall-clock training budgets (protocol ``pure_wall_budget_v1``).

Epoch length varies drastically across representations (thin register ~10 s/ep,
unary S=256 ~120 s/ep, S=1024 ~400 s/ep). Stopping on epoch count or epoch
patience therefore allocates **different wall time** to different cells and
biases representation ranking.

**Default policy:** only wall-clock limits stop training:

- ``max_wall_sec`` — hard wall budget per run  
- ``patience_wall_sec`` — no improvement for this long (fraction of max wall)  
- ``min_delta`` — usually 0 (any strict metric gain resets stall)

``max_epochs`` is an optional **safety fuse** only (default off / huge). It does
**not** participate in early-stop ranking under pure-wall mode.

Design note: ``docs/TRAIN_BUDGET.md``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional


# Pure wall defaults for MNIST-scale atlas / comparison runs.
DEFAULT_MAX_WALL_SEC = 1200.0  # 20 minutes wall per run
DEFAULT_PATIENCE_FRAC = 0.125  # 12.5% of wall → 150 s stall
DEFAULT_MIN_DELTA = 0.0
# Safety only: do not stop on epochs in pure-wall mode unless explicitly set low.
DEFAULT_MAX_EPOCHS_SAFETY = 10_000
DEFAULT_USE_EPOCH_STOPS = False


@dataclass
class TrainBudget:
    """Resolved train limits for one run (pure wall by default)."""

    max_wall_sec: Optional[float] = DEFAULT_MAX_WALL_SEC
    patience_frac: float = DEFAULT_PATIENCE_FRAC
    min_delta: float = DEFAULT_MIN_DELTA
    patience_wall_sec: Optional[float] = None
    # Optional / legacy epoch controls (off in pure-wall default).
    max_epochs: int = DEFAULT_MAX_EPOCHS_SAFETY
    patience_epochs: Optional[int] = None  # None = do not stop on epoch stall
    use_epoch_stops: bool = DEFAULT_USE_EPOCH_STOPS

    def __post_init__(self) -> None:
        if self.patience_frac < 0 or self.patience_frac > 1:
            raise ValueError(f"patience_frac must be in [0, 1], got {self.patience_frac}")
        if self.max_wall_sec is not None and self.max_wall_sec <= 0:
            self.max_wall_sec = None
        if self.max_wall_sec is None and not self.use_epoch_stops:
            raise ValueError(
                "Pure wall mode needs max_wall_sec > 0 "
                "(or set use_epoch_stops=True for legacy epoch budgets)"
            )
        if self.patience_wall_sec is None and self.max_wall_sec is not None:
            self.patience_wall_sec = max(
                1.0, float(self.patience_frac) * float(self.max_wall_sec)
            )
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")
        if self.use_epoch_stops and self.patience_epochs is None:
            self.patience_epochs = max(
                1, int(round(self.patience_frac * self.max_epochs))
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_budget(
    *,
    max_wall_sec: Optional[float] = DEFAULT_MAX_WALL_SEC,
    patience_frac: float = DEFAULT_PATIENCE_FRAC,
    min_delta: float = DEFAULT_MIN_DELTA,
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    use_epoch_stops: bool = DEFAULT_USE_EPOCH_STOPS,
) -> TrainBudget:
    """Build a budget. Pure wall by default; epoch stops only if requested."""
    max_ep = (
        int(epochs)
        if epochs is not None and epochs >= 1
        else DEFAULT_MAX_EPOCHS_SAFETY
    )
    pe: Optional[int]
    if use_epoch_stops:
        pe = int(patience) if patience is not None and patience >= 1 else None
    else:
        pe = None  # ignore --patience in pure wall mode
    return TrainBudget(
        max_wall_sec=max_wall_sec,
        patience_frac=float(patience_frac),
        min_delta=float(min_delta),
        max_epochs=max_ep,
        patience_epochs=pe,
        use_epoch_stops=bool(use_epoch_stops),
    )


def add_budget_args(
    parser: argparse.ArgumentParser,
    *,
    max_wall_sec: float = DEFAULT_MAX_WALL_SEC,
    patience_frac: float = DEFAULT_PATIENCE_FRAC,
    min_delta: float = DEFAULT_MIN_DELTA,
    epochs: int = DEFAULT_MAX_EPOCHS_SAFETY,
    patience: Optional[int] = None,
) -> argparse.ArgumentParser:
    """Add pure-wall budget CLI flags."""
    parser.add_argument(
        "--max-wall-sec",
        type=float,
        default=max_wall_sec,
        help=(
            f"Wall-clock budget seconds per run (default {max_wall_sec}). "
            "Primary stop. Use <=0 only with --use-epoch-stops."
        ),
    )
    parser.add_argument(
        "--patience-frac",
        type=float,
        default=patience_frac,
        help=(
            f"Wall patience as fraction of max-wall-sec "
            f"(default {patience_frac} → {patience_frac * max_wall_sec:.0f}s without gain)"
        ),
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=min_delta,
        help="Min metric gain over best to count as improvement (default 0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=epochs,
        help=(
            f"Safety epoch fuse only in pure-wall mode (default {epochs}); "
            "does not early-stop unless --use-epoch-stops"
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=patience if patience is not None else -1,
        help="Epoch patience; only used with --use-epoch-stops (default unused)",
    )
    parser.add_argument(
        "--use-epoch-stops",
        action="store_true",
        help="Legacy: also stop on max_epochs / epoch patience (not pure wall)",
    )
    return parser


def budget_from_args(args: argparse.Namespace) -> TrainBudget:
    wall = float(getattr(args, "max_wall_sec", DEFAULT_MAX_WALL_SEC))
    wall_opt: Optional[float] = None if wall <= 0 else wall
    use_ep = bool(getattr(args, "use_epoch_stops", False))
    pat = int(getattr(args, "patience", -1))
    return resolve_budget(
        max_wall_sec=wall_opt,
        patience_frac=float(getattr(args, "patience_frac", DEFAULT_PATIENCE_FRAC)),
        min_delta=float(getattr(args, "min_delta", DEFAULT_MIN_DELTA)),
        epochs=int(getattr(args, "epochs", DEFAULT_MAX_EPOCHS_SAFETY)),
        patience=pat if pat >= 1 else None,
        use_epoch_stops=use_ep,
    )


@dataclass
class StopDecision:
    stop: bool
    reason: str
    improved: bool


class EarlyStopTracker:
    """Track best metric; pure wall stop by default."""

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
        """Call once per epoch after evaluation."""
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
        # Pure wall stops (always when configured)
        if b.max_wall_sec is not None and wall >= b.max_wall_sec:
            return StopDecision(True, "max_wall_sec", improved)
        if (
            b.patience_wall_sec is not None
            and self.stall_wall_sec >= b.patience_wall_sec
            and not improved
        ):
            return StopDecision(True, "patience_wall", improved)

        # Optional legacy epoch stops
        if b.use_epoch_stops:
            if (
                b.patience_epochs is not None
                and self.stall_epochs >= b.patience_epochs
                and not improved
            ):
                return StopDecision(True, "patience_epochs", improved)
            if epoch >= b.max_epochs:
                return StopDecision(True, "max_epochs", improved)
        elif epoch >= b.max_epochs:
            # Safety fuse only (should almost never hit under pure wall)
            return StopDecision(True, "max_epochs_safety", improved)

        return StopDecision(False, "", improved)

    def status_str(self) -> str:
        b = self.budget
        wall = self.wall_sec
        parts = [f"best={self.best:.4f}@{self.best_epoch}"]
        if b.patience_wall_sec is not None:
            parts.append(
                f"stall_wall={self.stall_wall_sec:.0f}/{b.patience_wall_sec:.0f}s"
            )
        if b.max_wall_sec is not None:
            parts.append(f"wall={wall:.0f}/{b.max_wall_sec:.0f}s")
        else:
            parts.append(f"wall={wall:.0f}s")
        parts.append(f"ep={self.stall_epochs}stall/{self.best_epoch}best")
        if b.use_epoch_stops and b.patience_epochs is not None:
            parts.append(f"stall_ep={self.stall_epochs}/{b.patience_epochs}")
        return " ".join(parts)

    def meta_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget.to_dict(),
            "protocol": "pure_wall_budget_v1",
            "best_metric": self.best if self.best > float("-inf") else None,
            "best_epoch": self.best_epoch,
            "stall_epochs": self.stall_epochs,
            "wall_sec": self.wall_sec,
            "stall_wall_sec": self.stall_wall_sec,
        }
