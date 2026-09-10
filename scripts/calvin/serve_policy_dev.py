#!/usr/bin/env python3
"""Serve Duo-VLA only for authenticated held-out CALVIN A/B/C development.

This endpoint is deliberately not an official CALVIN ABC-to-D policy server.
It accepts only ``calvin_dev_bridge`` v1 messages, binds every prediction to a
validated reset-bank record, and never imports an official sequence or metric.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ``-P`` intentionally omits the executable's directory from ``sys.path``.
# Re-add only this resolved, repository-owned sibling directory before loading
# the two standalone development-server dependencies below.
_SCRIPT_ROOT = Path(__file__).resolve().parent
_SCRIPT_ROOT_TEXT = str(_SCRIPT_ROOT)
_SCRIPT_ROOT_ADDED = _SCRIPT_ROOT_TEXT not in sys.path
if _SCRIPT_ROOT_ADDED:
    sys.path.insert(0, _SCRIPT_ROOT_TEXT)

import serve_policy as official_policy  # noqa: E402
from calvin_dev_bridge import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    PROTOCOL,
    make_error_response,
    make_health_response,
    make_predict_response,
    make_success_response,
    serve_unix_policy,
)

if _SCRIPT_ROOT_ADDED:
    sys.path.remove(_SCRIPT_ROOT_TEXT)

from duo_vla.data.calvin_dev_states import (  # noqa: E402
    ABC_SCENES,
    AuthenticatedCalvinDevInputs,
    assert_bank_matches_inputs,
    authenticate_dev_inputs,
    canonical_sha256,
    load_bank,
)

DEVELOPMENT_STATUS = "heldout_abc_development_only_not_official_calvin_abc_to_d"
DEVELOPMENT_RUNTIME_SCHEMA = "duo-vla-calvin-heldout-abc-serving-runtime-v4"
TRAINING_PROTOCOL = official_policy.PROTOCOL
MAX_CANONICAL_UPDATES = 30_000

_NORMALIZATION_DATASET_FIELDS = {
    "archive_bytes",
    "archive_sha256",
    "central_directory_sha256",
    "dataset_manifest_file_sha256",
    "dataset_manifest_schema",
    "dataset_manifest_sha256",
    "member_index",
    "member_inventory_sha256",
    "metadata_files",
    "metadata_sha256",
    "name",
    "reader_schema",
    "split",
    "storage_identity_sha256",
    "storage_mode",
}
_CALVIN_STORAGE_IDENTITY_FIELDS = _NORMALIZATION_DATASET_FIELDS - {"name", "split"}


def _require_checkpoint_storage_identity(
    manifest: Mapping[str, Any],
    calvin_identity: Mapping[str, Any],
    normalization_dataset: Mapping[str, Any],
) -> None:
    """Bind the checkpoint's nested and flattened identities to v4 storage."""

    require(
        set(normalization_dataset) == _NORMALIZATION_DATASET_FIELDS,
        "normalization dataset storage identity fields differ",
    )
    for name in _CALVIN_STORAGE_IDENTITY_FIELDS:
        require(
            calvin_identity.get(name) == normalization_dataset[name],
            f"CALVIN identity storage field mismatch: {name}",
        )
    member_index = normalization_dataset["member_index"]
    require(
        isinstance(member_index, dict) and set(member_index) == {"bytes", "path", "schema", "sha256"},
        "normalization member-index identity differs",
    )
    flattened = {
        "archive_bytes": normalization_dataset["archive_bytes"],
        "archive_sha256": normalization_dataset["archive_sha256"],
        "central_directory_sha256": normalization_dataset["central_directory_sha256"],
        "dataset_manifest_file_sha256": normalization_dataset["dataset_manifest_file_sha256"],
        "dataset_manifest_schema": normalization_dataset["dataset_manifest_schema"],
        "dataset_manifest_sha256": normalization_dataset["dataset_manifest_sha256"],
        "member_index_bytes": member_index["bytes"],
        "member_index_path": member_index["path"],
        "member_index_schema": member_index["schema"],
        "member_index_sha256": member_index["sha256"],
        "member_inventory_sha256": normalization_dataset["member_inventory_sha256"],
        "metadata_sha256": normalization_dataset["metadata_sha256"],
        "reader_schema": normalization_dataset["reader_schema"],
        "storage_identity_sha256": normalization_dataset["storage_identity_sha256"],
        "storage_mode": normalization_dataset["storage_mode"],
    }
    for name, expected in flattened.items():
        require(manifest.get(name) == expected, f"checkpoint flattened storage field mismatch: {name}")


# Keep inference behavior identical to the authenticated official-policy
# implementation.  Only checkpoint admission, wire protocol, and reporting are
# development-specific in this module.
FakePolicy = official_policy.FakePolicy
RealPolicy = official_policy.RealPolicy


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@dataclass(frozen=True)
class DevelopmentBankBinding:
    """Immutable identities and records accepted by one development endpoint."""

    calvin_identity: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    replay_bundle_sha256: str
    reset_bank_sha256: str
    split_sha256: str

    @property
    def reset_count(self) -> int:
        return len(self.records)


def bind_development_bank(
    manifest: Mapping[str, Any],
    inputs: AuthenticatedCalvinDevInputs,
) -> DevelopmentBankBinding:
    """Bind a validated bank to the exact authenticated data/split capability."""

    assert_bank_matches_inputs(manifest, inputs)
    records = manifest.get("records")
    require(isinstance(records, list) and bool(records), "development reset bank has no records")
    copied_records = tuple(dict(record) for record in records if isinstance(record, dict))
    require(len(copied_records) == len(records), "development reset bank contains an invalid record")
    require(
        {record.get("scene") for record in copied_records} == set(ABC_SCENES),
        "development reset bank does not cover exactly A/B/C",
    )
    dataset_identity = inputs.stats.get("dataset")
    require(
        isinstance(dataset_identity, dict) and set(dataset_identity) == _NORMALIZATION_DATASET_FIELDS,
        "development inputs lack an exact CALVIN storage identity",
    )
    canonical_dataset_identity = {
        **dataset_identity,
        "member_index": dict(dataset_identity["member_index"]),
        "metadata_files": list(dataset_identity["metadata_files"]),
    }
    return DevelopmentBankBinding(
        calvin_identity=canonical_dataset_identity,
        records=copied_records,
        replay_bundle_sha256=manifest["replay_bundle"]["root_sha256"],
        reset_bank_sha256=manifest["root_sha256"],
        split_sha256=canonical_sha256(inputs.split),
    )


def _validate_predict_binding(request: Mapping[str, Any], binding: DevelopmentBankBinding) -> None:
    """Reject a syntactically valid request not represented by the bound bank."""

    episode = request["development_episode"]
    require(
        episode["reset_bank_sha256"] == binding.reset_bank_sha256,
        "prediction refers to a different development reset bank",
    )
    reset_index = episode["reset_index"]
    require(reset_index < binding.reset_count, "prediction reset_index is outside the development reset bank")
    record = binding.records[reset_index]
    for name in (
        "reset_id_sha256",
        "scene",
        "episode_index",
        "annotation_index",
        "global_start",
        "task",
    ):
        require(episode[name] == record[name], f"prediction {name} differs from the bound reset record")
    require(request["instruction"] == record["instruction"], "prediction instruction differs from the bound reset")


def _health_identity(
    *,
    mode: str,
    train_seed: int,
    binding: DevelopmentBankBinding,
    policy_contract: Mapping[str, Any],
    artifact_identities: Mapping[str, Any] | None,
) -> dict[str, Any]:
    common = {
        "calvin_identity": binding.calvin_identity,
        "replay_bundle_sha256": binding.replay_bundle_sha256,
        "reset_bank_sha256": binding.reset_bank_sha256,
        "reset_count": binding.reset_count,
        "split_sha256": binding.split_sha256,
        "train_seed": train_seed,
    }
    if mode == "fake":
        return {
            **common,
            "checkpoint_manifest_sha256": None,
            "execution_geometry": None,
            "mode": "fake",
            "model_revision": None,
            "nfe": 0,
            "normalization_content_sha256": None,
            "normalization_metadata_sha256": None,
            "objective": "test_fake",
            "policy_contract_sha256": None,
            "sampler": "seeded_test_normal",
        }
    require(mode == "real" and artifact_identities is not None, "real development health lacks artifacts")
    execution_geometry = official_policy._validate_execution_geometry(artifact_identities.get("execution_geometry"))
    return {
        **common,
        **dict(artifact_identities),
        "execution_geometry": execution_geometry,
        "mode": "real",
        "model_revision": official_policy.MODEL_REVISION,
        "nfe": policy_contract["nfe"],
        "objective": policy_contract["objective"],
        "policy_contract_sha256": official_policy._canonical_sha256(dict(policy_contract)),
        "sampler": policy_contract["sampler"],
    }


def _dispatch_local(
    request: dict[str, Any],
    *,
    policy: Any,
    health_identity: dict[str, Any],
    binding: DevelopmentBankBinding,
) -> dict[str, Any]:
    operation = request["operation"]
    if operation == "health":
        return make_health_response(request, **health_identity)
    if operation == "shutdown":
        return make_success_response(request, stopped=True)
    require(request["train_seed"] == health_identity["train_seed"], "request train_seed differs from checkpoint")
    _validate_predict_binding(request, binding)
    actions, seconds = policy.predict(request)
    return make_predict_response(request, actions, policy_seconds=seconds)


def _canonical_prediction(value: Mapping[str, Any]) -> str:
    comparable = {key: item for key, item in value.items() if key != "policy_seconds"}
    return json.dumps(comparable, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _validate_development_progress(
    manifest: Mapping[str, Any],
    optimization: Mapping[str, Any],
    checkpoint: Path,
    *,
    committed_update: int,
    committed_manifest_sha256: str,
) -> dict[str, Any]:
    """Accept a durable pilot/tip while retaining every progress invariant."""

    total_updates = optimization.get("total_updates")
    global_batch_size = optimization.get("global_batch_size")
    require(
        type(total_updates) is int and 0 < total_updates <= MAX_CANONICAL_UPDATES,
        "development total_updates must be in [1,30000]",
    )
    require(type(global_batch_size) is int and global_batch_size > 0, "global batch size is invalid")
    require(
        manifest.get("configured_total_updates") == total_updates,
        "checkpoint and resolved total-update contracts differ",
    )
    trainer_state = manifest.get("trainer_state")
    require(isinstance(trainer_state, dict), "development checkpoint has no trainer state")
    next_update = trainer_state.get("next_update")
    require(
        trainer_state.get("schema") == "duo-vla-trainer-state-v1"
        and type(next_update) is int
        and 0 < next_update <= total_updates,
        "development checkpoint trainer progress is invalid",
    )
    expected_examples = next_update * global_batch_size
    require(
        trainer_state.get("examples_seen") == expected_examples,
        "development checkpoint examples_seen differs from update progress",
    )
    last_metrics = manifest.get("last_metrics")
    require(
        isinstance(last_metrics, dict)
        and last_metrics.get("update") == next_update
        and last_metrics.get("examples_seen") == expected_examples,
        "development checkpoint metrics differ from update progress",
    )
    expected_complete = next_update == total_updates
    require(
        manifest.get("complete") is expected_complete,
        "development checkpoint complete flag differs from update progress",
    )
    require(checkpoint.name == f"update-{next_update:06d}", "development checkpoint directory/update mismatch")
    require(committed_update == next_update, "run journal update differs from development checkpoint")
    require(
        committed_manifest_sha256 == official_policy.sha256_file(checkpoint / "manifest.json"),
        "run journal manifest SHA-256 differs from development checkpoint",
    )
    return {
        "complete": expected_complete,
        "configured_total_updates": total_updates,
        "examples_seen": expected_examples,
        "selected_update": next_update,
    }


def _committed_checkpoint_record(checkpoint: Path, config_sha256: str) -> Any:
    """Require the selected directory to be the latest atomically journaled tip."""

    from duo_vla.run_journal import load_run_journal, validate_resume_checkpoint

    require(checkpoint.parent.name == "checkpoints", "development checkpoint must live below a checkpoints directory")
    output_dir = checkpoint.parent.parent
    journal = load_run_journal(output_dir, expected_config_sha256=config_sha256)
    require(journal.latest_checkpoint is not None, "training run has no committed checkpoint")
    return validate_resume_checkpoint(output_dir, checkpoint)


def _canonical_recipe_mismatches(
    config: Mapping[str, Any],
    canonical_recipe: Mapping[str, Any],
) -> list[str]:
    """Allow only total_updates and its exact trainer-derived warmup."""

    sections = (
        "protocol",
        "model",
        "policy",
        "action",
        "lora",
        "optimization",
        "training",
        "sampling",
        "benchmark",
        "reproducibility",
        "distributed",
    )
    mismatches: list[str] = []
    for section in sections:
        observed = config.get(section)
        expected = canonical_recipe.get(section)
        if section == "optimization" and isinstance(observed, dict) and isinstance(expected, dict):
            expected = dict(expected)
            observed_total = observed.get("total_updates")
            expected["total_updates"] = observed_total
            if type(observed_total) is int and observed_total > 0:
                # train_calvin.py derives this when --total-updates is used.
                # It is not a second development-tuning degree of freedom.
                expected["warmup_updates"] = min(1_000, observed_total // 10)
        if observed != expected:
            mismatches.append(section)
    return mismatches


def _canonical_development_recipe_name(objective: str, tensor_parallel_size: int) -> str:
    require(objective in {"rectified_flow", "direct_regression"}, "development objective is invalid")
    require(tensor_parallel_size in {1, 2}, "development tensor-parallel size is invalid")
    stem = "calvin_abc_to_d" if objective == "rectified_flow" else "calvin_abc_to_d_direct"
    suffix = "_single_gpu" if tensor_parallel_size == 1 else ""
    return f"{stem}{suffix}.toml"


def resolve_development_checkpoint(
    checkpoint_dir: Path,
    *,
    training_root: Path,
    project_root: Path,
    authenticated_inputs: AuthenticatedCalvinDevInputs,
    train_seed_override: int | None,
    model_snapshot_report: dict[str, Any],
) -> tuple[dict[str, Any], Path, int, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate a committed CALVIN checkpoint with development-only progress relaxations."""

    from duo_vla.backbones.loading import (
        validate_decoder_attention_lora_adapter_config,
        validate_decoder_attention_lora_weights,
    )
    from duo_vla.checkpointing import load_checkpoint_manifest
    from duo_vla.data.calvin_stats import CALVIN_CRITICAL_TRAIN_METADATA, load_calvin_state_normalizer
    from duo_vla.policy_contract import validate_manifest_policy_contract
    from duo_vla.run_config import canonical_config_sha256, load_resolved_toml, load_verified_resolved_config

    checkpoint = checkpoint_dir.resolve()
    manifest = load_checkpoint_manifest(checkpoint, verify_hashes=True)
    require(manifest.get("kind") == "resumable-calvin-abc-to-d-training", "checkpoint kind is not CALVIN ABC-to-D")
    require(manifest.get("model_id") == official_policy.MODEL_ID, "checkpoint model identity mismatch")
    require(manifest.get("model_revision") == official_policy.MODEL_REVISION, "checkpoint model revision mismatch")
    require(manifest.get("protocol") == TRAINING_PROTOCOL, "checkpoint training protocol mismatch")
    require(
        manifest.get("dataset") == "task_ABC_D" and manifest.get("dataset_split") == "training",
        "checkpoint dataset split mismatch",
    )
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "checkpoint artifact table is missing")
    expected_artifact_paths = {
        "interface": "interface.safetensors",
        "lora_config": "lora/adapter_config.json",
        "lora_weights": "lora/adapter_model.safetensors",
    }
    mismatched_artifacts = [
        name
        for name, expected_path in expected_artifact_paths.items()
        if not isinstance(artifacts.get(name), dict) or artifacts[name].get("path") != expected_path
    ]
    require(not mismatched_artifacts, f"checkpoint artifact paths are not canonical: {mismatched_artifacts}")

    resolved_artifact = artifacts.get("resolved_config")
    require(
        isinstance(resolved_artifact, dict) and isinstance(resolved_artifact.get("path"), str),
        "checkpoint has no resolved configuration",
    )
    resolved_path = checkpoint / resolved_artifact["path"]
    require(resolved_path.resolve().is_relative_to(checkpoint), "resolved configuration escapes checkpoint")
    config_sha256 = manifest.get("config_sha256")
    require(isinstance(config_sha256, str) and len(config_sha256) == 64, "checkpoint configuration hash is invalid")
    config, _ = load_verified_resolved_config(resolved_path, expected_sha256=config_sha256)
    committed = _committed_checkpoint_record(checkpoint, config_sha256)
    contract = validate_manifest_policy_contract(manifest, config)
    contract_dict = contract.to_dict()
    contract_sha256 = canonical_config_sha256(contract_dict)
    require(manifest.get("policy_contract_sha256") == contract_sha256, "checkpoint policy contract hash mismatch")

    model = config.get("model")
    benchmark = config.get("benchmark")
    action_config = config.get("action")
    lora = config.get("lora")
    optimization = config.get("optimization")
    training = config.get("training")
    run = config.get("run")
    require(
        all(isinstance(value, dict) for value in (model, benchmark, action_config, lora, optimization, training, run)),
        "resolved config is incomplete",
    )
    assert isinstance(model, dict) and isinstance(benchmark, dict) and isinstance(action_config, dict)
    assert isinstance(lora, dict) and isinstance(optimization, dict) and isinstance(training, dict)
    assert isinstance(run, dict)
    tensor_parallel_size = model.get("tensor_parallel_size")
    require(tensor_parallel_size in {1, 2}, "resolved tensor-parallel size is invalid")
    expected_execution_profile = "duovla-single-gpu-tp1-v1" if tensor_parallel_size == 1 else None
    optimized_execution = (
        official_policy.execution_from_config(config)
        if model.get("expert_batch_isolation") == official_policy.FUSED_BACKEND
        else None
    )
    if optimized_execution is not None:
        expected_execution_profile = optimized_execution.profile
    training_batch = optimized_execution.physical_batch if optimized_execution else official_policy.PHYSICAL_BATCH_SIZE
    training_accumulation = optimized_execution.accumulation if optimized_execution else 8
    expected_config = {
        "protocol": (config.get("protocol"), TRAINING_PROTOCOL),
        "model.id": (model.get("id"), official_policy.MODEL_ID),
        "model.revision": (model.get("revision"), official_policy.MODEL_REVISION),
        "model.dtype": (model.get("dtype"), "bfloat16"),
        "model.tensor_parallel_size": (tensor_parallel_size, tensor_parallel_size),
        "execution_profile": (config.get("execution_profile"), expected_execution_profile),
        "model.attention_implementation": (model.get("attention_implementation"), "sdpa"),
        "model.experts_implementation": (
            model.get("experts_implementation"),
            official_policy.EXPERTS_IMPLEMENTATION,
        ),
        "model.expert_batch_isolation": (
            model.get("expert_batch_isolation"),
            official_policy.FUSED_BACKEND if optimized_execution else official_policy.EXPERT_BATCH_ISOLATION,
        ),
        "benchmark.dataset": (benchmark.get("dataset"), "task_ABC_D"),
        "benchmark.train_split": (benchmark.get("train_split"), "training"),
        "benchmark.evaluation_split": (benchmark.get("evaluation_split"), "validation"),
        "benchmark.train_environments": (benchmark.get("train_environments"), ["A", "B", "C"]),
        "benchmark.evaluation_environment": (benchmark.get("evaluation_environment"), "D"),
        "benchmark.state_dimension": (benchmark.get("state_dimension"), official_policy.STATE_DIM),
        "benchmark.camera_order": (benchmark.get("camera_order"), ["rgb_static", "rgb_gripper"]),
        "benchmark.execution_horizons": (benchmark.get("execution_horizons"), [1, 4]),
        "action.horizon": (action_config.get("horizon"), ACTION_HORIZON),
        "action.dimension": (action_config.get("dimension"), ACTION_DIM),
        "action.timestep_embedding_dimension": (action_config.get("timestep_embedding_dimension"), 256),
        "action.timestep_scale": (action_config.get("timestep_scale"), 1000.0),
        "action.timestep_max_period": (action_config.get("timestep_max_period"), 10000.0),
        "action.conditioning_mlp_activation": (action_config.get("conditioning_mlp_activation"), "silu"),
        "action.output_head_initialization_std": (action_config.get("output_head_initialization_std"), 1e-3),
        "lora.rank": (lora.get("rank"), 16),
        "lora.alpha": (lora.get("alpha"), 32),
        "lora.dropout": (lora.get("dropout"), 0.0),
        "lora.projections": (lora.get("projections"), ["q_proj", "k_proj", "v_proj", "o_proj"]),
        "lora.scope": (lora.get("scope"), "decoder_self_attention_only"),
        "optimization.global_batch_size": (optimization.get("global_batch_size"), 64),
        "optimization.microbatch_size": (
            optimization.get("microbatch_size"),
            training_batch,
        ),
        "optimization.gradient_accumulation_steps": (
            optimization.get("gradient_accumulation_steps"),
            training_accumulation,
        ),
        "optimization.physical_batch_size": (
            optimization.get("physical_batch_size"),
            training_batch,
        ),
        "training.seeds": (training.get("seeds"), [0, 1, 2]),
        "run.task": (run.get("task"), None),
    }
    mismatches = [name for name, (observed, expected) in expected_config.items() if observed != expected]
    require(not mismatches, f"resolved CALVIN configuration mismatches: {mismatches}")
    total_updates = optimization.get("total_updates")
    require(
        type(total_updates) is int and 0 < total_updates <= MAX_CANONICAL_UPDATES,
        "development total_updates must be in [1,30000]",
    )
    canonical_config_name = (
        official_policy.canonical_recipe_name(config, contract.objective)
        if optimized_execution
        else _canonical_development_recipe_name(contract.objective, tensor_parallel_size)
    )
    canonical_recipe = load_resolved_toml(project_root / "configs" / canonical_config_name)
    recipe_mismatches = _canonical_recipe_mismatches(config, canonical_recipe)
    require(
        not recipe_mismatches,
        f"resolved config differs from the canonical recipe beyond total_updates: {recipe_mismatches}",
    )
    validate_decoder_attention_lora_adapter_config(
        checkpoint / "lora/adapter_config.json",
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
    )
    validate_decoder_attention_lora_weights(
        checkpoint / "lora/adapter_model.safetensors",
        rank=int(lora["rank"]),
    )
    artifact_trees = config.get("artifact_trees")
    require(isinstance(artifact_trees, dict), "resolved config has no authenticated model tree")
    model_tree_sha256 = artifact_trees.get("model_tree_sha256")
    require(
        isinstance(model_tree_sha256, str)
        and len(model_tree_sha256) == 64
        and manifest.get("model_tree_sha256") == model_tree_sha256,
        "checkpoint/config model tree identity mismatch",
    )
    require(
        model_snapshot_report.get("tree_metadata_sha256") == model_tree_sha256,
        "current model snapshot differs from the development checkpoint",
    )
    prefix_geometry, execution_geometry = official_policy._load_checkpoint_prefix_geometry(
        checkpoint,
        manifest,
        config,
        model_snapshot_report=model_snapshot_report,
    )
    training_instruction_inventory_sha256 = config.get("training_instruction_inventory_sha256")
    require(
        official_policy._valid_sha256(training_instruction_inventory_sha256)
        and manifest.get("training_instruction_inventory_sha256") == training_instruction_inventory_sha256,
        "checkpoint/config authenticated training-instruction inventory mismatch",
    )

    normalization_artifact = artifacts.get("normalization")
    require(
        isinstance(normalization_artifact, dict) and isinstance(normalization_artifact.get("path"), str),
        "checkpoint has no normalization artifact",
    )
    normalization_path = checkpoint / normalization_artifact["path"]
    require(normalization_path.resolve().is_relative_to(checkpoint), "normalization artifact escapes checkpoint")
    _normalizer, stats = load_calvin_state_normalizer(
        normalization_path,
        expected_archive_sha256=official_policy.ARCHIVE_SHA256,
    )
    require(stats == authenticated_inputs.stats, "checkpoint normalization differs from authenticated v4 inputs")
    require(
        authenticated_inputs.training_root == str(training_root.resolve()),
        "checkpoint resolver training root differs from authenticated v4 inputs",
    )
    official_policy._validate_normalization_contract(stats)
    normalization_sha256 = stats["content_sha256"]
    normalization_dataset = stats["dataset"]
    metadata_sha256 = normalization_dataset["metadata_sha256"]
    dataset_manifest_sha256 = normalization_dataset["dataset_manifest_sha256"]
    dataset_manifest_file_sha256 = normalization_dataset["dataset_manifest_file_sha256"]
    member_inventory_sha256 = normalization_dataset["member_inventory_sha256"]
    member_index = normalization_dataset["member_index"]
    storage_identity_sha256 = normalization_dataset["storage_identity_sha256"]
    require(
        stats["dataset"].get("metadata_files") == list(CALVIN_CRITICAL_TRAIN_METADATA),
        "normalization metadata file contract mismatch",
    )
    require(manifest.get("normalization_sha256") == normalization_sha256, "checkpoint normalization hash mismatch")
    require(manifest.get("metadata_sha256") == metadata_sha256, "checkpoint training metadata hash mismatch")
    require(
        manifest.get("dataset_manifest_sha256") == dataset_manifest_sha256,
        "checkpoint dataset-manifest hash mismatch",
    )
    require(
        manifest.get("member_inventory_sha256") == member_inventory_sha256,
        "checkpoint archive-member inventory hash mismatch",
    )
    require(manifest.get("archive_sha256") == official_policy.ARCHIVE_SHA256, "checkpoint archive hash mismatch")

    calvin_identity = manifest.get("calvin_identity")
    config_identity = config.get("calvin_identity")
    require(
        isinstance(calvin_identity, dict) and calvin_identity == config_identity,
        "checkpoint/config CALVIN identity mismatch",
    )
    assert isinstance(calvin_identity, dict)
    _require_checkpoint_storage_identity(manifest, calvin_identity, normalization_dataset)
    require(
        calvin_identity.get("archive_sha256") == official_policy.ARCHIVE_SHA256,
        "CALVIN identity archive mismatch",
    )
    require(calvin_identity.get("metadata_sha256") == metadata_sha256, "CALVIN identity metadata mismatch")
    require(
        calvin_identity.get("dataset_manifest_sha256") == dataset_manifest_sha256,
        "CALVIN identity dataset-manifest mismatch",
    )
    require(
        calvin_identity.get("member_inventory_sha256") == member_inventory_sha256,
        "CALVIN identity member-inventory mismatch",
    )
    require(
        calvin_identity.get("normalization_sha256") == normalization_sha256,
        "CALVIN identity normalization mismatch",
    )
    require(calvin_identity.get("split") == stats["split"] == manifest.get("split"), "CALVIN split identity mismatch")
    split_sha256 = canonical_config_sha256(stats["split"])
    require(
        calvin_identity.get("split_sha256") == split_sha256 == manifest.get("split_sha256"),
        "CALVIN split identity hash mismatch",
    )
    require(
        manifest.get("train_episode_sha256") == stats["split"]["train_episode_sha256"]
        and manifest.get("validation_episode_sha256") == stats["split"]["validation_episode_sha256"],
        "checkpoint split membership hashes mismatch",
    )
    require(
        calvin_identity.get("camera_shapes") == official_policy.CALVIN_CAMERA_SHAPES == manifest.get("camera_shapes"),
        "CALVIN camera identity mismatch",
    )
    require(
        manifest.get("camera_shapes_sha256") == canonical_config_sha256(official_policy.CALVIN_CAMERA_SHAPES),
        "CALVIN camera identity hash mismatch",
    )
    require(
        calvin_identity.get("state_adapter") == official_policy.CALVIN_STATE_ADAPTER == manifest.get("state_adapter"),
        "CALVIN state adapter mismatch",
    )
    require(
        calvin_identity.get("action_adapter")
        == official_policy.CALVIN_ACTION_ADAPTER
        == manifest.get("action_adapter"),
        "CALVIN action adapter mismatch",
    )

    source_revisions = official_policy.load_pinned_source_revisions(project_root)
    source_revision_sha256 = canonical_config_sha256(source_revisions)
    require(manifest.get("calvin_source_revisions") == source_revisions, "checkpoint CALVIN source revisions mismatch")
    require(
        calvin_identity.get("calvin_source_revisions") == source_revisions,
        "CALVIN identity source revisions mismatch",
    )
    require(
        manifest.get("calvin_source_revisions_sha256") == source_revision_sha256,
        "CALVIN source revision hash mismatch",
    )
    require(
        manifest.get("calvin_revision") == source_revisions["calvin"]
        and manifest.get("calvin_env_revision") == source_revisions["calvin_env"]
        and manifest.get("calvin_tacto_revision") == source_revisions["tacto"],
        "checkpoint individual CALVIN source revisions mismatch",
    )
    current_source_sha256 = official_policy._source_tree_sha256(
        project_root,
        single_gpu=tensor_parallel_size == 1,
    )
    require(
        manifest.get("source_tree_sha256") == current_source_sha256 == config.get("source_tree_sha256"),
        "training source tree differs from checkpoint",
    )

    recorded_seed = manifest.get("run_seed")
    require(type(recorded_seed) is int and recorded_seed in (0, 1, 2), "checkpoint run seed is not declared")
    require(run.get("seed") == recorded_seed, "checkpoint and resolved configuration run seeds differ")
    training_execution_environment = official_policy._validate_checkpoint_training_environment(
        manifest,
        config,
        project_root=project_root,
        run_seed=recorded_seed,
    )
    progress = _validate_development_progress(
        manifest,
        optimization,
        checkpoint,
        committed_update=committed.update,
        committed_manifest_sha256=committed.manifest_sha256,
    )
    if train_seed_override is not None:
        require(train_seed_override == recorded_seed, "--train-seed differs from checkpoint run_seed")
    checkpoint_manifest_sha256 = official_policy.sha256_file(checkpoint / "manifest.json")
    report = {
        "archive_bytes": normalization_dataset["archive_bytes"],
        "archive_sha256": normalization_dataset["archive_sha256"],
        "benchmark_status": DEVELOPMENT_STATUS,
        "central_directory_sha256": normalization_dataset["central_directory_sha256"],
        "checkpoint_progress": progress,
        "dataset_manifest_file_sha256": dataset_manifest_file_sha256,
        "dataset_manifest_schema": normalization_dataset["dataset_manifest_schema"],
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "kind": manifest["kind"],
        "manifest_sha256": checkpoint_manifest_sha256,
        "member_inventory_sha256": member_inventory_sha256,
        "member_index_bytes": member_index["bytes"],
        "member_index_path": member_index["path"],
        "member_index_schema": member_index["schema"],
        "member_index_sha256": member_index["sha256"],
        "metadata_sha256": metadata_sha256,
        "model_tree_sha256": model_tree_sha256,
        "execution_geometry": execution_geometry,
        "prefix_geometry_instruction_inventory_sha256": prefix_geometry["instruction_inventory"]["sha256"],
        "training_instruction_inventory_sha256": training_instruction_inventory_sha256,
        "normalization_content_sha256": normalization_sha256,
        "path": str(checkpoint),
        "policy_contract": contract_dict,
        "policy_contract_sha256": contract_sha256,
        "source_tree_sha256": current_source_sha256,
        "split_sha256": split_sha256,
        "storage_identity_sha256": storage_identity_sha256,
        "storage_mode": normalization_dataset["storage_mode"],
        "reader_schema": normalization_dataset["reader_schema"],
        "train_seed": recorded_seed,
        "training_execution_environment": training_execution_environment,
        "training_execution_environment_sha256": manifest["execution_environment_sha256"],
        "train_venv": training_execution_environment["authenticated_runtime"]["train_venv"],
    }
    identities: dict[str, Any] = {
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "execution_geometry": execution_geometry,
        "normalization_content_sha256": normalization_sha256,
        "normalization_metadata_sha256": metadata_sha256,
    }
    return manifest, normalization_path, recorded_seed, report, config, contract_dict, identities


def _development_source_identity(project_root: Path, *, single_gpu: bool) -> dict[str, str]:
    scripts = project_root / "scripts/calvin"
    paths = {
        "calvin_dev_bridge_sha256": scripts / "calvin_dev_bridge.py",
        "development_state_contract_sha256": project_root / "src/duo_vla/data/calvin_dev_states.py",
        "development_launcher_sha256": scripts
        / ("run_policy_server_dev_single_gpu.sh" if single_gpu else "run_policy_server_dev.sh"),
        "development_server_sha256": scripts / "serve_policy_dev.py",
        "official_inference_implementation_sha256": scripts / "serve_policy.py",
    }
    for name, path in paths.items():
        require(path.is_file(), f"development serving source is missing: {name}")
    expected_origins = {
        "calvin_dev_bridge": scripts / "calvin_dev_bridge.py",
        "serve_policy": scripts / "serve_policy.py",
    }
    for module, expected_path in expected_origins.items():
        spec = importlib.util.find_spec(module)
        require(spec is not None and isinstance(spec.origin, str), f"cannot resolve serving module {module}")
        require(
            Path(spec.origin).resolve() == expected_path.resolve(),
            f"serving module {module} resolves outside the project",
        )
    return {name: official_policy.sha256_file(path) for name, path in paths.items()}


def development_runtime_preflight(
    project_root: Path,
    checkpoint_dir: Path,
    *,
    training_root: Path,
    normalization_path: Path,
    source_root: Path,
    revision_file: Path,
    reset_bank: Path,
    train_seed_override: int | None,
) -> tuple[
    dict[str, Any],
    Path,
    int,
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    DevelopmentBankBinding,
]:
    """Authenticate software, model, full data, bank, and committed checkpoint."""

    require(
        platform.python_version() == official_policy.EXPECTED_TRAIN_PYTHON,
        f"development server requires Python {official_policy.EXPECTED_TRAIN_PYTHON}",
    )
    world_size = official_policy.canonical_visible_cuda_world_size(os.environ)
    relative_lock_path, expected_lock_sha256, expected_packages, _expected_cuda_runtime, _ = (
        official_policy._train_runtime_pins(world_size)
    )
    package_versions = {name: importlib.metadata.version(name) for name in expected_packages}
    require(
        package_versions == expected_packages,
        f"train package pin mismatch: {package_versions}",
    )
    lock_path = project_root / relative_lock_path
    require(lock_path.is_file(), f"missing train lockfile: {lock_path}")
    require(
        official_policy.sha256_file(lock_path) == expected_lock_sha256,
        "train lockfile SHA-256 mismatch",
    )
    inputs = authenticate_dev_inputs(training_root, normalization_path, source_root, revision_file)
    bank_manifest, _robot_obs, _scene_obs = load_bank(reset_bank)
    binding = bind_development_bank(bank_manifest, inputs)
    model_report = official_policy.model_snapshot_preflight()
    _, checkpoint_normalization, train_seed, checkpoint_report, config, contract, identities = (
        resolve_development_checkpoint(
            checkpoint_dir,
            training_root=training_root,
            project_root=project_root,
            authenticated_inputs=inputs,
            train_seed_override=train_seed_override,
            model_snapshot_report=model_report,
        )
    )
    for name in (
        "archive_bytes",
        "archive_sha256",
        "central_directory_sha256",
        "dataset_manifest_file_sha256",
        "dataset_manifest_schema",
        "dataset_manifest_sha256",
        "member_index_bytes",
        "member_index_path",
        "member_index_schema",
        "member_index_sha256",
        "member_inventory_sha256",
        "metadata_sha256",
        "reader_schema",
        "storage_identity_sha256",
        "storage_mode",
    ):
        require(
            inputs.identity[name] == checkpoint_report[name],
            f"development inputs/checkpoint {name} mismatch",
        )
    require(
        inputs.identity["normalization_content_sha256"] == checkpoint_report["normalization_content_sha256"],
        "development inputs/checkpoint normalization_content_sha256 mismatch",
    )
    require(
        canonical_sha256(inputs.split) == checkpoint_report["split_sha256"],
        "development split/checkpoint identity mismatch",
    )
    source_revisions = official_policy.load_pinned_source_revisions(project_root)
    require(
        inputs.identity["calvin_revision"] == source_revisions["calvin"]
        and inputs.identity["calvin_env_revision"] == source_revisions["calvin_env"]
        and inputs.identity["calvin_tacto_revision"] == source_revisions["tacto"],
        "development simulator/checkpoint source revisions differ",
    )
    require(
        model_report["tree_metadata_sha256"] == checkpoint_report["model_tree_sha256"],
        "installed model tree differs from checkpoint",
    )
    report = {
        "benchmark_status": DEVELOPMENT_STATUS,
        "checkpoint": checkpoint_report,
        "development_inputs": {
            "identity": inputs.identity,
            "replay_bundle_sha256": binding.replay_bundle_sha256,
            "reset_bank_sha256": binding.reset_bank_sha256,
            "reset_count": binding.reset_count,
            "split_sha256": binding.split_sha256,
        },
        "development_source": _development_source_identity(project_root, single_gpu=world_size == 1),
        "execution_geometry": checkpoint_report["execution_geometry"],
        "lock_path": relative_lock_path.as_posix(),
        "lock_sha256": expected_lock_sha256,
        "model": model_report,
        "official_benchmark_metrics_allowed": False,
        "packages": package_versions,
        "protocol": PROTOCOL,
        "python": sys.version.split()[0],
        "source_revisions": source_revisions,
        "status": "ok",
    }
    require(
        report["checkpoint"]["execution_geometry"].get("tensor_parallel_size", 2) == world_size,
        "checkpoint topology differs from serving launch",
    )
    return report, checkpoint_normalization, train_seed, config, contract, identities, binding


def configure_and_identify_development_runtime(
    torch: Any,
    preflight_report: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Attest the exact official inference core plus the development boundary."""

    official_runtime, official_runtime_sha256 = official_policy.configure_and_identify_serving_runtime(
        torch,
        preflight_report,
    )
    payload = {
        "benchmark_status": DEVELOPMENT_STATUS,
        "development_source": preflight_report["development_source"],
        "execution_geometry": official_policy._validate_execution_geometry(preflight_report.get("execution_geometry")),
        "official_benchmark_metrics_allowed": False,
        "official_inference_runtime": official_runtime,
        "official_inference_runtime_sha256": official_runtime_sha256,
        "protocol": PROTOCOL,
        "schema": DEVELOPMENT_RUNTIME_SCHEMA,
    }
    return payload, official_policy._canonical_sha256(payload)


def run_fake_server(
    socket_path: Path,
    *,
    train_seed: int,
    binding: DevelopmentBankBinding,
) -> None:
    policy = FakePolicy()
    health = _health_identity(
        mode="fake",
        train_seed=train_seed,
        binding=binding,
        policy_contract={},
        artifact_identities=None,
    )

    def dispatch(request: dict[str, Any]) -> dict[str, Any]:
        return _dispatch_local(request, policy=policy, health_identity=health, binding=binding)

    serve_unix_policy(
        socket_path,
        dispatch,
        ready=lambda: print(
            json.dumps(
                {
                    **health,
                    "benchmark_status": DEVELOPMENT_STATUS,
                    "official_benchmark_metrics_allowed": False,
                    "protocol": PROTOCOL,
                    "socket": str(socket_path),
                    "status": "ready",
                },
                sort_keys=True,
            ),
            flush=True,
        ),
    )


def run_distributed_server(
    socket_path: Path,
    *,
    checkpoint_dir: Path,
    normalization_path: Path,
    train_seed: int,
    resolved_config: dict[str, Any],
    policy_contract: dict[str, Any],
    artifact_identities: dict[str, Any],
    preflight_report: dict[str, Any],
    binding: DevelopmentBankBinding,
) -> None:
    import torch
    import torch.distributed as dist

    runtime, runtime_sha256 = configure_and_identify_development_runtime(torch, preflight_report)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    expected_world_size = int(resolved_config["model"]["tensor_parallel_size"])
    require(
        dist.get_world_size() == expected_world_size,
        "real development policy serving world size differs from the checkpoint topology",
    )
    rank = dist.get_rank()
    try:
        runtime_hashes: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(runtime_hashes, runtime_sha256)
        require(
            runtime_hashes == [runtime_sha256] * dist.get_world_size(),
            f"TP ranks disagree on development runtime identity: {runtime_hashes}",
        )
        policy = RealPolicy(
            checkpoint_dir,
            normalization_path,
            device,
            resolved_config=resolved_config,
            policy_contract=policy_contract,
        )
        health = _health_identity(
            mode="real",
            train_seed=train_seed,
            binding=binding,
            policy_contract=policy_contract,
            artifact_identities=artifact_identities,
        )
        dist.barrier()

        def execute(request: dict[str, Any]) -> dict[str, Any]:
            local_result: dict[str, Any] | None = None
            local_error: str | None = None
            try:
                local_result = _dispatch_local(
                    request,
                    policy=policy,
                    health_identity=health,
                    binding=binding,
                )
            except Exception as exc:
                local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
            errors: list[str | None] = [None] * dist.get_world_size()
            dist.all_gather_object(errors, local_error)
            failures = [error for error in errors if error is not None]
            if failures:
                return make_error_response(request, RuntimeError("; ".join(failures)))
            assert local_result is not None
            if request["operation"] == "predict":
                results: list[dict[str, Any] | None] = [None] * dist.get_world_size()
                dist.all_gather_object(results, local_result)
                concrete = [result for result in results if result is not None]
                require(len(concrete) == dist.get_world_size(), "a TP rank returned no prediction")
                reference = _canonical_prediction(concrete[0])
                if not all(_canonical_prediction(result) == reference for result in concrete[1:]):
                    return make_error_response(request, RuntimeError("TP ranks produced different action chunks"))
                local_result["policy_seconds"] = max(float(result["policy_seconds"]) for result in concrete)
            return local_result

        if rank == 0:

            def dispatch(request: dict[str, Any]) -> dict[str, Any]:
                holder = [request]
                dist.broadcast_object_list(holder, src=0, device=device)
                return execute(request)

            serve_unix_policy(
                socket_path,
                dispatch,
                ready=lambda: print(
                    json.dumps(
                        {
                            **health,
                            "benchmark_status": DEVELOPMENT_STATUS,
                            "checkpoint": str(checkpoint_dir),
                            "development_runtime": runtime,
                            "development_runtime_sha256": runtime_sha256,
                            "official_benchmark_metrics_allowed": False,
                            "protocol": PROTOCOL,
                            "socket": str(socket_path),
                            "status": "ready",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                ),
            )
        else:
            while True:
                holder: list[dict[str, Any] | None] = [None]
                dist.broadcast_object_list(holder, src=0, device=device)
                request = holder[0]
                require(request is not None, "rank 0 broadcast an empty development request")
                execute(request)
                if request["operation"] == "shutdown":
                    break
    finally:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=cache_root / "data/calvin/task_ABC_D/training",
    )
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=cache_root / "simulators/calvin")
    parser.add_argument(
        "--revision-file",
        type=Path,
        default=Path(__file__).resolve().with_name("revisions.env"),
    )
    parser.add_argument("--reset-bank", type=Path, required=True)
    parser.add_argument("--socket", type=Path, default=cache_root / "run/calvin-heldout-abc-policy.sock")
    parser.add_argument("--train-seed", type=int)
    parser.add_argument(
        "--flow-steps",
        type=int,
        choices=(1, 5, 10),
        help="Override Euler NFE for rectified flow; forbidden for direct regression.",
    )
    parser.add_argument("--fake-policy", action="store_true", help="serve deterministic IPC-only chunks")
    parser.add_argument("--preflight-only", action="store_true", help="authenticate without loading CUDA")
    return parser.parse_args()


def _authenticate_fake_binding(args: argparse.Namespace) -> DevelopmentBankBinding:
    inputs = authenticate_dev_inputs(
        args.training_root,
        args.normalization,
        args.source_root,
        args.revision_file,
    )
    manifest, _robot_obs, _scene_obs = load_bank(args.reset_bank)
    return bind_development_bank(manifest, inputs)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.fake_policy:
        require(args.checkpoint is None, "fake development policy does not accept a checkpoint")
        require(type(args.train_seed) is int and 0 <= args.train_seed < 2**63, "fake policy requires --train-seed")
        require(args.flow_steps is None, "fake development policy does not accept --flow-steps")
        require(not args.preflight_only, "--preflight-only is only for a real development policy")
        binding = _authenticate_fake_binding(args)
        run_fake_server(args.socket, train_seed=args.train_seed, binding=binding)
        return

    require(args.checkpoint is not None, "real development policy requires a checkpoint directory")
    official_policy._validated_serving_process_environment(
        project_root,
        require_tp_launch=not args.preflight_only,
    )
    report, normalization_path, train_seed, config, checkpoint_contract, identities, binding = (
        development_runtime_preflight(
            project_root,
            args.checkpoint,
            training_root=args.training_root,
            normalization_path=args.normalization,
            source_root=args.source_root,
            revision_file=args.revision_file,
            reset_bank=args.reset_bank,
            train_seed_override=args.train_seed,
        )
    )
    serving_contract = official_policy.select_serving_policy_contract(
        checkpoint_contract,
        flow_steps_override=args.flow_steps,
    )
    report["serving_policy_contract"] = serving_contract
    report["serving_policy_contract_sha256"] = official_policy._canonical_sha256(serving_contract)
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    run_distributed_server(
        args.socket,
        checkpoint_dir=args.checkpoint.resolve(),
        normalization_path=normalization_path,
        train_seed=train_seed,
        resolved_config=config,
        policy_contract=serving_contract,
        artifact_identities=identities,
        preflight_report=report,
        binding=binding,
    )


if __name__ == "__main__":
    main()
