"""Experiment ID registry — protocol revisions for re-runs.

Code stays under ``experiments/v0_N/`` (or ``v0_N_*``); the **run id** written to
results/ and DuckDB bumps when the train *protocol* changes so new numbers are
not mixed with legacy epoch-only runs.

Naming: ``v0_N`` → ``v0_N_1`` (patch = protocol rev). Compound atlas ids use
``v0_5_1_width_register`` style (patch after the minor).
"""

from __future__ import annotations

from typing import Any, Mapping

# Shared train-loop protocol introduced with wall-clock budgets.
TRAIN_BUDGET_PROTOCOL = "wall_epoch_budget_v1"

# experiment_id -> metadata
REGISTRY: dict[str, dict[str, Any]] = {
    "v0_1_1": {
        "parent": "v0_1",
        "code_dir": "experiments/v0_1",
        "protocol": TRAIN_BUDGET_PROTOCOL,
        "changelog": (
            "Train budget: max_wall_sec + max_epochs; patience_frac of both; "
            "min_delta default 0 (any strict test gain)."
        ),
    },
    "v0_2_1": {
        "parent": "v0_2",
        "code_dir": "experiments/v0_2",
        "protocol": TRAIN_BUDGET_PROTOCOL,
        "changelog": (
            "Train budget: max_wall_sec + max_epochs; patience_frac of both; "
            "min_delta default 0."
        ),
    },
    "v0_3_1": {
        "parent": "v0_3",
        "code_dir": "experiments/v0_3",
        "protocol": TRAIN_BUDGET_PROTOCOL,
        "changelog": (
            "Train budget: max_wall_sec + max_epochs; patience_frac of both; "
            "min_delta default 0."
        ),
    },
    "v0_4_1": {
        "parent": "v0_4",
        "code_dir": "experiments/v0_4",
        "protocol": TRAIN_BUDGET_PROTOCOL,
        "changelog": (
            "Train budget: max_wall_sec + max_epochs; patience_frac of both; "
            "min_delta default 0."
        ),
    },
    "v0_5_1_width_register": {
        "parent": "v0_5_width_register",
        "code_dir": "experiments/v0_5_width_register",
        "protocol": TRAIN_BUDGET_PROTOCOL,
        "changelog": (
            "Width atlas re-run protocol: wall+epoch budget, patience_frac, "
            "min_delta=0 (legacy parent used epoch-only patience)."
        ),
    },
    "v0_5_1_width_unary": {
        "parent": "v0_5_width_unary",
        "code_dir": "experiments/v0_5_width_unary",
        "protocol": TRAIN_BUDGET_PROTOCOL,
        "changelog": (
            "Width atlas re-run protocol: wall+epoch budget, patience_frac, "
            "min_delta=0."
        ),
    },
    "v0_6_1_encoding": {
        "parent": "v0_6_encoding",
        "code_dir": "experiments/v0_6_encoding",
        "protocol": TRAIN_BUDGET_PROTOCOL,
        "changelog": (
            "Encoding atlas re-run protocol: wall+epoch budget, patience_frac, "
            "min_delta=0 (parent primary grid used epoch-only early stop)."
        ),
    },
    "ste_vs_swarm_1": {
        "parent": "ste_vs_swarm",
        "code_dir": "experiments/ste_vs_swarm",
        "protocol": TRAIN_BUDGET_PROTOCOL,
        "changelog": (
            "Shared-protocol comparison re-run: wall+epoch budget, patience_frac, "
            "min_delta=0."
        ),
    },
}


def get_meta(experiment_id: str) -> dict[str, Any]:
    if experiment_id not in REGISTRY:
        raise KeyError(
            f"Unknown experiment_id {experiment_id!r}; "
            f"known: {sorted(REGISTRY)}"
        )
    return dict(REGISTRY[experiment_id])


def db_notes(experiment_id: str) -> str:
    """Short notes string stored on the DuckDB ``runs.notes`` column."""
    m = get_meta(experiment_id)
    return (
        f"experiment_id={experiment_id}; parent={m['parent']}; "
        f"protocol={m['protocol']}; {m['changelog']}"
    )


def config_version_fields(experiment_id: str) -> dict[str, Any]:
    """Fields to merge into the run ``config`` JSON for DB/filtering."""
    m = get_meta(experiment_id)
    return {
        "experiment_id": experiment_id,
        "experiment_parent": m["parent"],
        "train_protocol": m["protocol"],
        "protocol_changelog": m["changelog"],
        "code_dir": m["code_dir"],
    }


def enrich_config(
    experiment_id: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return config copy with version/protocol fields."""
    out = dict(config or {})
    out.update(config_version_fields(experiment_id))
    return out
