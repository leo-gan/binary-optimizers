"""
Train/eval checkpoint cache.

Trained nets are keyed by a fingerprint of (model, optimizer, hyperparams,
epochs, seed, dataset, batch size, code-level version). Reuse the checkpoint
unless the net or optimizer config changes — then retrain and overwrite.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


# Bump when optimizer step logic or architecture factories change incompatibly.
CHECKPOINT_SCHEMA_VERSION = 3

DEFAULT_CHECKPOINT_ROOT = Path("checkpoints")


@dataclass(frozen=True)
class TrainSpec:
    """Everything that must match for a checkpoint to be reused."""

    model: str
    optimizer: str
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    dataset: str = "mnist"
    epochs: int = 15
    seed: int = 42
    batch_size_train: int = 128
    hidden_dim: int | None = None  # optional override encoded in model name
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    tag: str = "fit"  # free-form experiment tag

    def fingerprint(self) -> str:
        payload = {
            "model": self.model,
            "optimizer": self.optimizer,
            "optimizer_kwargs": _stable(self.optimizer_kwargs),
            "dataset": self.dataset,
            "epochs": self.epochs,
            "seed": self.seed,
            "batch_size_train": self.batch_size_train,
            "hidden_dim": self.hidden_dim,
            "schema_version": self.schema_version,
            "tag": self.tag,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def slug(self) -> str:
        return f"{self.tag}_{self.model}_{self.optimizer}_e{self.epochs}_s{self.seed}_{self.fingerprint()}"


def _stable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _stable(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_stable(x) for x in obj]
    if isinstance(obj, float):
        return round(obj, 12)
    return obj


@dataclass
class CheckpointPaths:
    root: Path
    run_dir: Path
    weights: Path
    meta: Path

    @classmethod
    def for_spec(cls, spec: TrainSpec, root: str | Path = DEFAULT_CHECKPOINT_ROOT) -> "CheckpointPaths":
        root_p = Path(root)
        run_dir = root_p / spec.slug()
        return cls(
            root=root_p,
            run_dir=run_dir,
            weights=run_dir / "model.pt",
            meta=run_dir / "meta.json",
        )


def checkpoint_exists(spec: TrainSpec, root: str | Path = DEFAULT_CHECKPOINT_ROOT) -> bool:
    paths = CheckpointPaths.for_spec(spec, root)
    return paths.weights.is_file() and paths.meta.is_file()


def load_meta(spec: TrainSpec, root: str | Path = DEFAULT_CHECKPOINT_ROOT) -> dict[str, Any] | None:
    paths = CheckpointPaths.for_spec(spec, root)
    if not paths.meta.is_file():
        return None
    return json.loads(paths.meta.read_text())


def save_checkpoint(
    spec: TrainSpec,
    model: nn.Module,
    metrics: dict[str, Any],
    *,
    root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    extra: dict[str, Any] | None = None,
) -> CheckpointPaths:
    """Persist weights + JSON meta. Overwrites same fingerprint dir."""
    paths = CheckpointPaths.for_spec(spec, root)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "fingerprint": spec.fingerprint()}, paths.weights)
    meta = {
        "spec": {
            "model": spec.model,
            "optimizer": spec.optimizer,
            "optimizer_kwargs": dict(spec.optimizer_kwargs),
            "dataset": spec.dataset,
            "epochs": spec.epochs,
            "seed": spec.seed,
            "batch_size_train": spec.batch_size_train,
            "hidden_dim": spec.hidden_dim,
            "schema_version": spec.schema_version,
            "tag": spec.tag,
            "fingerprint": spec.fingerprint(),
            "slug": spec.slug(),
        },
        "metrics": metrics,
        "extra": extra or {},
    }
    paths.meta.write_text(json.dumps(meta, indent=2))
    # Update root index
    _update_index(paths.root, meta)
    return paths


def load_checkpoint(
    spec: TrainSpec,
    model: nn.Module,
    *,
    root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load weights into ``model``. Returns meta dict. Raises if missing/mismatch."""
    paths = CheckpointPaths.for_spec(spec, root)
    if not paths.weights.is_file():
        raise FileNotFoundError(f"No checkpoint for {spec.slug()}: {paths.weights}")
    blob = torch.load(paths.weights, map_location=map_location, weights_only=False)
    fp = blob.get("fingerprint")
    if fp is not None and fp != spec.fingerprint():
        raise RuntimeError(
            f"Fingerprint mismatch: file={fp} expected={spec.fingerprint()}"
        )
    model.load_state_dict(blob["state_dict"])
    meta = json.loads(paths.meta.read_text()) if paths.meta.is_file() else {}
    return meta


def _update_index(root: Path, meta: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    index: dict[str, Any] = {"runs": {}}
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            index = {"runs": {}}
    runs = index.setdefault("runs", {})
    slug = meta.get("spec", {}).get("slug")
    if slug:
        runs[slug] = {
            "fingerprint": meta.get("spec", {}).get("fingerprint"),
            "model": meta.get("spec", {}).get("model"),
            "optimizer": meta.get("spec", {}).get("optimizer"),
            "epochs": meta.get("spec", {}).get("epochs"),
            "seed": meta.get("spec", {}).get("seed"),
            "best_test_acc": (meta.get("metrics") or {}).get("best_test_acc"),
            "final_test_acc": (meta.get("metrics") or {}).get("final_test_acc"),
            "schema_version": meta.get("spec", {}).get("schema_version"),
        }
    index_path.write_text(json.dumps(index, indent=2))


def list_checkpoints(root: str | Path = DEFAULT_CHECKPOINT_ROOT) -> list[dict[str, Any]]:
    root_p = Path(root)
    index_path = root_p / "index.json"
    if not index_path.is_file():
        return []
    data = json.loads(index_path.read_text())
    return list((data.get("runs") or {}).values())
