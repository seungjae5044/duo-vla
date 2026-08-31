from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from duo_vla.data.calvin import CalvinAnchor

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify_calvin_fixed_anchor_overfit.py"
SPEC = importlib.util.spec_from_file_location("calvin_fixed_anchor_overfit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


def _sample(anchor: CalvinAnchor, *, ordinal: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        action_chunk=SimpleNamespace(
            actions=torch.arange(56, dtype=torch.float32).reshape(8, 7) / 56 + ordinal,
            valid_mask=torch.tensor([True] * 7 + [False]),
        ),
        annotation_index=anchor.annotation_index,
        episode_index=ordinal + 3,
        global_index=anchor.global_index,
        instruction=f"instruction {ordinal}",
        observation=SimpleNamespace(
            state=torch.arange(8, dtype=torch.float32) + ordinal,
            third_person=np.full((4, 5, 3), 10 + ordinal, dtype=np.uint8),
            wrist=np.full((3, 2, 3), 20 + ordinal, dtype=np.uint8),
        ),
        task=anchor.task,
    )


@pytest.mark.parametrize("count", (32, 40, 64, 128))
def test_settings_accept_only_b8_aligned_g3_inventory(count: int) -> None:
    settings = GATE.QualificationSettings(anchor_count=count)

    assert GATE.validate_settings(settings) is settings
    assert settings.microbatches_per_update == count // 8


@pytest.mark.parametrize("count", (0, 31, 33, 129))
def test_settings_reject_out_of_range_or_unaligned_inventory(count: int) -> None:
    with pytest.raises(ValueError, match=r"chunks|divisible"):
        GATE.validate_settings(GATE.QualificationSettings(anchor_count=count))


def test_settings_cannot_weaken_twenty_x_gate_or_use_invalid_schedule() -> None:
    with pytest.raises(ValueError, match="cannot be below"):
        GATE.validate_settings(GATE.QualificationSettings(minimum_reduction=19.999))
    with pytest.raises(ValueError, match="optimizer schedule"):
        GATE.validate_settings(GATE.QualificationSettings(updates=10, warmup_updates=9))


def test_deterministic_anchor_draw_deduplicates_physical_frames() -> None:
    class _Sampler:
        population_size = 8

        def __init__(self) -> None:
            self.values = iter(
                (
                    CalvinAnchor(0, 100, "task_a"),
                    CalvinAnchor(1, 100, "task_b"),
                    CalvinAnchor(2, 101, "task_b"),
                    CalvinAnchor(3, 102, "task_c"),
                )
            )

        def draw(self, _generator: torch.Generator) -> CalvinAnchor:
            return next(self.values)

    anchors = GATE.draw_fixed_action_anchors(_Sampler(), count=3, seed=7)

    assert [(anchor.annotation_index, anchor.global_index) for anchor in anchors] == [
        (0, 100),
        (2, 101),
        (3, 102),
    ]


def test_anchor_inventory_hash_binds_order_identity_and_full_sample_content() -> None:
    anchors = (CalvinAnchor(0, 100, "task_a"), CalvinAnchor(1, 200, "task_b"))
    samples = (_sample(anchors[0], ordinal=0), _sample(anchors[1], ordinal=1))

    records, digest = GATE.build_anchor_inventory(anchors, samples)
    repeated_records, repeated_digest = GATE.build_anchor_inventory(anchors, samples)

    assert records == repeated_records
    assert digest == repeated_digest == GATE.canonical_object_sha256(records)
    assert [record["ordinal"] for record in records] == [0, 1]
    changed = _sample(anchors[0], ordinal=0)
    changed.observation.third_person[0, 0, 0] ^= 1
    _, changed_digest = GATE.build_anchor_inventory(anchors, (changed, samples[1]))
    assert changed_digest != digest
    _, reordered_digest = GATE.build_anchor_inventory(anchors[::-1], samples[::-1])
    assert reordered_digest != digest


def test_fixed_update_hash_is_order_sensitive_and_stable() -> None:
    first = SimpleNamespace(anchor_sha256="a" * 64, static_input_sha256="1" * 64)
    second = SimpleNamespace(anchor_sha256="b" * 64, static_input_sha256="2" * 64)

    expected = GATE.fixed_update_inventory_sha256((first, second))

    assert GATE.fixed_update_inventory_sha256((first, second)) == expected
    assert GATE.fixed_update_inventory_sha256((second, first)) != expected


def test_fixed_input_tensor_guard_detects_in_place_example_mutation() -> None:
    def fixed() -> SimpleNamespace:
        return SimpleNamespace(
            batch=SimpleNamespace(
                clean_actions=torch.zeros(1, 8, 7),
                states=torch.zeros(1, 8),
                action_valid_mask=torch.ones(1, 8, dtype=torch.bool),
            ),
            clean=torch.zeros(1, 8, 7),
            state=torch.zeros(1, 8),
            valid=torch.ones(1, 8, dtype=torch.bool),
            pair=SimpleNamespace(
                input_actions=torch.zeros(1, 8, 7),
                target=torch.ones(1, 8, 7),
                timesteps=torch.zeros(1),
            ),
            prefix=SimpleNamespace(attention_mask=torch.ones(1, 12, dtype=torch.bool)),
        )

    batch = fixed()
    before = GATE.fixed_input_tensor_structure((batch,))

    assert GATE.fixed_input_tensor_structure((batch,)) == before
    batch.pair.input_actions.add_(1)
    assert GATE.fixed_input_tensor_structure((batch,)) != before


def test_frozen_registration_allows_declared_updates_and_rejects_frozen_or_buffer_mutation() -> None:
    class _Guarded(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trainable = nn.Parameter(torch.ones(2))
            self.frozen = nn.Parameter(torch.ones(3), requires_grad=False)
            self.register_buffer("constant", torch.ones(1))

    model = _Guarded()
    modules = (("model", model),)
    expected = GATE.frozen_registration_snapshot(modules)
    with torch.no_grad():
        model.trainable.add_(1)
    GATE.assert_only_declared_tensors_may_change(expected, modules)

    with torch.no_grad():
        model.frozen.add_(1)
    with pytest.raises(RuntimeError, match="frozen parameter"):
        GATE.assert_only_declared_tensors_may_change(expected, modules)

    model = _Guarded()
    modules = (("model", model),)
    expected = GATE.frozen_registration_snapshot(modules)
    model.constant.add_(1)
    with pytest.raises(RuntimeError, match="buffer"):
        GATE.assert_only_declared_tensors_may_change(expected, modules)


def test_pass_criteria_require_exact_repetition_full_trainable_changes_and_no_d_access() -> None:
    settings = GATE.QualificationSettings(anchor_count=32, updates=200)
    values = {
        "settings": settings,
        "observed_anchor_count": 32,
        "distinct_global_indices": 32,
        "update_inventory_verifications": 200,
        "update_inventory_hash_count": 1,
        "initial_loss": 1.0,
        "final_loss": 0.04,
        "reduction": 25.0,
        "changed_lora_tensors": 230,
        "lora_tensors": 230,
        "changed_interface_tensors": 14,
        "interface_tensors": 14,
        "optimizer_inventory_exact": True,
        "frozen_registration_unchanged": True,
        "prefix_cache_unchanged": True,
        "benchmark_environment_accessed": False,
    }

    checks = GATE.qualification_checks(**values)

    assert all(checks.values())
    GATE.require_all_checks(checks)
    weakened = GATE.qualification_checks(**{**values, "update_inventory_hash_count": 2, "reduction": 19.0})
    assert not weakened["fixed_inventory_identical_every_update"]
    assert not weakened["minimum_fixed_loss_reduction_met"]
    with pytest.raises(RuntimeError, match="qualification failed"):
        GATE.require_all_checks(weakened)


def test_pass_report_has_self_hash_and_exclusive_publication(tmp_path: Path) -> None:
    payload = {
        "kind": GATE.QUALIFICATION_KIND,
        "pass_criteria": {"all_contracts_passed": True},
        "protocol": GATE.QUALIFICATION_PROTOCOL,
        "schema": GATE.REPORT_SCHEMA,
        "status": "passed",
        "scope": {
            "benchmark_environment_accessed": False,
            "checkpoint_emitted": False,
            "official_benchmark_claim": False,
            "qualification_only": True,
        },
    }
    report = GATE.finalized_report(payload)

    assert report["report_sha256"] == hashlib.sha256(GATE.canonical_json_bytes(payload)).hexdigest()
    output = tmp_path / "qualification.json"
    assert GATE.write_canonical_json_exclusive(output, report) == output
    assert output.read_bytes() == GATE.canonical_json_bytes(report)
    with pytest.raises(FileExistsError):
        GATE.write_canonical_json_exclusive(output, report)
    with pytest.raises(RuntimeError, match="only a passed"):
        GATE.finalized_report({**payload, "status": "failed"})
    with pytest.raises(RuntimeError, match="failed checks"):
        GATE.finalized_report({**payload, "pass_criteria": {"gate": False}})
    with pytest.raises(RuntimeError, match="self-hash"):
        GATE.write_canonical_json_exclusive(tmp_path / "tampered.json", {**report, "status": "tampered"})


def test_g3_storage_report_requires_and_binds_archive_direct_v4() -> None:
    identity = {"mode": "archive-direct", "manifest": {"schema": "v4"}}
    storage = SimpleNamespace(
        mode=GATE.CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
        content_sha256="a" * 64,
        to_dict=lambda: dict(identity),
    )
    generation = SimpleNamespace(storage=storage)

    assert GATE.calvin_storage_report(generation) == {  # type: ignore[arg-type]
        "identity": identity,
        "identity_sha256": "a" * 64,
    }
    storage.mode = "verified-extraction"
    with pytest.raises(ValueError, match="archive-direct v4"):
        GATE.calvin_storage_report(generation)  # type: ignore[arg-type]


def test_qualification_is_separate_from_production_source_and_never_samples_inside_update_loop() -> None:
    production_hash_source = (ROOT / "scripts/train_calvin.py").read_text(encoding="utf-8")
    qualification_source = SCRIPT.read_text(encoding="utf-8")
    update_loop = qualification_source.split("for update in range(settings.updates):", maxsplit=1)[1].split(
        "elapsed_seconds =", maxsplit=1
    )[0]

    assert "qualify_calvin_fixed_anchor_overfit.py" not in production_hash_source
    assert "--fixed-anchor" not in production_hash_source
    assert "sampler" not in update_loop
    assert "make_microbatch_plan" not in update_loop
    assert "fixed_batches" in update_loop
    assert "checkpoint" not in update_loop


def test_qualification_files_do_not_enter_production_source_identity(tmp_path: Path) -> None:
    tracked = (
        "src/duo_vla/data/calvin.py",
        "configs/base.toml",
        "configs/calvin_abc_to_d.toml",
        "configs/calvin_abc_to_d_direct.toml",
        "scripts/run_calvin_train.sh",
        "scripts/train_calvin.py",
        "scripts/calvin/calvin_bridge.py",
        "scripts/calvin/prepare_archive_direct.py",
        "scripts/calvin/revisions.env",
        "scripts/calvin/run_policy_server.sh",
        "scripts/calvin/serve_policy.py",
        "pyproject.toml",
        "uv.lock",
    )
    for relative in tracked:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    baseline = GATE.TRAIN._source_tree_sha256(tmp_path)
    for relative in (
        "scripts/qualify_calvin_fixed_anchor_overfit.py",
        "scripts/calvin/run_fixed_anchor_overfit.sh",
        "tests/test_calvin_fixed_anchor_overfit.py",
        "docs/calvin_fixed_anchor_overfit.md",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("qualification-only", encoding="utf-8")

    assert GATE.TRAIN._source_tree_sha256(tmp_path) == baseline
    (tmp_path / "scripts/train_calvin.py").write_text("production-changed", encoding="utf-8")
    assert GATE.TRAIN._source_tree_sha256(tmp_path) != baseline


def test_qualification_source_must_match_at_import_start_and_report_boundary() -> None:
    expected = {
        "production_source_tree_sha256": "1" * 64,
        "qualification_launcher_sha256": "2" * 64,
        "qualification_script_sha256": "3" * 64,
        "qualification_source_sha256": "4" * 64,
    }

    assert GATE.require_qualification_source_unchanged(expected, dict(expected), context="test boundary") == expected
    for name in expected:
        changed = dict(expected)
        changed[name] = "f" * 64
        with pytest.raises(RuntimeError, match="source identity changed"):
            GATE.require_qualification_source_unchanged(expected, changed, context="test boundary")
    missing = dict(expected)
    missing.pop("qualification_launcher_sha256")
    with pytest.raises(RuntimeError, match="fields differ"):
        GATE.require_qualification_source_unchanged(expected, missing, context="test boundary")


def test_launcher_pins_tp2_and_canonical_runtime_without_calling_production_trainer() -> None:
    source = (ROOT / "scripts/calvin/run_fixed_anchor_overfit.sh").read_text(encoding="utf-8")
    qualification_source = SCRIPT.read_text(encoding="utf-8")

    assert "--nproc-per-node=2" in source
    assert "export PYTHONHASHSEED=0" in source
    assert 'export PYTHONPATH="${project_dir}/src"' in source
    assert "unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONINSPECT PYTHONSTARTUP" in source
    assert '"${name}" == NCCL_*' in source
    assert "qualify_calvin_fixed_anchor_overfit.py" in source
    assert "train_calvin.py" not in source
    assert "TRAIN._authenticate_calvin_dataset_generation_distributed(training_root)" in qualification_source
    assert "    authenticate_calvin_dataset_generation,\n" not in qualification_source
