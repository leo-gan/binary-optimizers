"""Registry and DB notes for protocol-bumped experiment ids."""

from __future__ import annotations

from binary_optimizers.store.versions import (
    REGISTRY,
    TRAIN_BUDGET_PROTOCOL,
    config_version_fields,
    db_notes,
    enrich_config,
    get_meta,
)


def test_registry_has_budget_revs():
    for eid in (
        "v0_1_1",
        "v0_2_1",
        "v0_3_1",
        "v0_4_1",
        "v0_5_1_width_register",
        "v0_5_1_width_unary",
        "v0_6_1_encoding",
        "ste_vs_swarm_1",
    ):
        m = get_meta(eid)
        assert m["protocol"] == TRAIN_BUDGET_PROTOCOL
        assert m["parent"]
        assert "wall" in m["changelog"].lower() or "budget" in m["changelog"].lower()


def test_db_notes_mentions_parent_and_protocol():
    notes = db_notes("v0_2_1")
    assert "v0_2_1" in notes
    assert "parent=v0_2" in notes
    assert TRAIN_BUDGET_PROTOCOL in notes


def test_enrich_config():
    cfg = enrich_config("v0_3_1", {"hidden": 128})
    assert cfg["experiment_id"] == "v0_3_1"
    assert cfg["experiment_parent"] == "v0_3"
    assert cfg["train_protocol"] == TRAIN_BUDGET_PROTOCOL
    assert cfg["hidden"] == 128


def test_config_version_fields_keys():
    f = config_version_fields("v0_1_1")
    assert set(f) >= {
        "experiment_id",
        "experiment_parent",
        "train_protocol",
        "protocol_changelog",
        "code_dir",
    }


def test_registry_parents_unique_from_ids():
    for eid, m in REGISTRY.items():
        assert m["parent"] != eid
