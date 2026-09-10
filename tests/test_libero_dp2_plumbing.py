from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import libero_bridge as BRIDGE  # noqa: E402
import serve_libero_policy as SERVER  # noqa: E402
import train_libero as TRAIN  # noqa: E402

from duo_vla.dp2_fork import (  # noqa: E402
    DP2_EXECUTION_PROFILE,
    EXPECTED_TRAINING_GPU_UUIDS,
    DP2ForkRestoreResult,
    _validate_dp2_config,
    authenticate_dp2_child_environment,
    authenticate_dp2_fork_parent,
    create_dp2_fork_manifest,
    derive_dp2_rank_rng_seeds,
    dp2_fork_run_contract,
    dp2_semantic_recipe_sha256,
    dp2_source_identity,
    load_published_dp2_fork_manifest,
    restore_tp1_fork_training_state,
    validate_dp2_fork_manifest,
    validate_dp2_semantic_recipe_continuity,
    write_dp2_fork_manifest,
)
from duo_vla.run_config import load_resolved_toml, save_resolved_config  # noqa: E402
from duo_vla.run_journal import create_run_journal, record_latest_checkpoint  # noqa: E402
from duo_vla.training import TrainerState  # noqa: E402
from duo_vla.training_checkpoint import (  # noqa: E402
    capture_rng_state,
    optimizer_parameter_inventory,
    optimizer_parameter_inventory_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, checkpoint: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(checkpoint).as_posix(),
        "sha256": _sha256(path),
    }


def _optimizer_and_scheduler() -> tuple[
    torch.nn.Parameter,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.LambdaLR,
]:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([{"name": "trainable", "params": [parameter]}], lr=3e-4)
    optimizer.state[parameter] = {
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.ones_like(parameter),
        "step": torch.tensor(1000.0),
    }
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scheduler.last_epoch = 1000
    scheduler._step_count = 1001
    return parameter, optimizer, scheduler


@pytest.fixture
def authenticated_parent(tmp_path: Path) -> dict[str, Any]:
    run_root = tmp_path / "parent-run"
    checkpoint = run_root / "checkpoints/update-001000"
    checkpoint.mkdir(parents=True)
    interface = checkpoint / "interface.safetensors"
    interface.write_bytes(b"interface")
    lora = checkpoint / "lora"
    lora.mkdir()
    lora_config = lora / "adapter_config.json"
    lora_config.write_bytes(b"{}")
    lora_weights = lora / "adapter_model.safetensors"
    lora_weights.write_bytes(b"weights")

    run_uuid = str(uuid.uuid4())
    parent_config = load_resolved_toml(PROJECT_ROOT / "configs/libero_single_gpu_fused_v2_b64.toml")
    parent_config["run"] = {"max_cached_files": 377, "seed": 0, "task": None}
    resolved_config = checkpoint / "artifacts/resolved_config.json"
    config_sha256 = save_resolved_config(resolved_config, parent_config)
    source_sha256 = "d" * 64
    parameter, optimizer, scheduler = _optimizer_and_scheduler()
    inventory = optimizer_parameter_inventory(optimizer, [("model.weight", parameter)])
    optimizer_schema = optimizer_parameter_inventory_sha256(inventory)
    run_contract = {
        "config_sha256": config_sha256,
        "execution_profile": "duovla-single-gpu-tp1-fused-v2-train-b64-serve-b8-v1",
        "expert_batch_isolation": "sample_isolated_grouped_mm_v2",
        "optimizer_parameter_schema_sha256": optimizer_schema,
        "physical_batch_size": "64",
        "run_uuid": run_uuid,
        "serving_batch_size": "8",
        "source_tree_sha256": source_sha256,
        "tensor_parallel_size": "1",
    }
    rng = capture_rng_state()
    rng.update(torch_cuda=torch.arange(16, dtype=torch.uint8), cuda_device_index=0)
    rank_state = {
        "optimizer": optimizer.state_dict(),
        "optimizer_parameter_inventory": inventory,
        "rank": 0,
        "rng": rng,
        "run_contract": dict(sorted(run_contract.items())),
        "scheduler": scheduler.state_dict(),
        "schema": "duo-vla-training-rank-state-v2",
        "trainer_state": TrainerState(1000, 64_000).to_dict(),
        "world_size": 1,
    }
    rank_path = checkpoint / "artifacts/training_rank_000.pt"
    rank_path.parent.mkdir(exist_ok=True)
    torch.save(rank_state, rank_path)
    venv_identity = {
        "content_inventory_sha256": "1" * 64,
        "root_sha256": "2" * 64,
        "tree_metadata_sha256": "3" * 64,
    }
    artifacts = {
        "interface": _artifact(interface, checkpoint),
        "lora_config": _artifact(lora_config, checkpoint),
        "lora_weights": _artifact(lora_weights, checkpoint),
        "resolved_config": _artifact(resolved_config, checkpoint),
        "training_rank_000": _artifact(rank_path, checkpoint),
    }
    last_metrics = {"train_loss": 0.25, "update": 1000}
    manifest = {
        "artifacts": artifacts,
        "config_sha256": config_sha256,
        "execution_environment": {
            "authenticated_runtime": {
                "environment": {"DUO_VLA_TRAIN_VENV": "/hdd2/hyunbin/vla/cache/venvs/train-single-gpu"},
                "train_venv": venv_identity,
            },
            "cuda_runtime": "12.9",
            "gpu_uuids": ["30424b03-3051-615a-832e-186511378a61"],
            "torch": "2.13.0+cu129",
            "world_size": 1,
        },
        "execution_geometry": {
            "execution_profile": "duovla-single-gpu-tp1-fused-v2-train-b64-serve-b8-v1",
            "expert_batch_isolation": "sample_isolated_grouped_mm_v2",
            "physical_batch_size": 64,
            "serving_batch_size": 8,
            "tensor_parallel_size": 1,
        },
        "execution_profile": "duovla-single-gpu-tp1-fused-v2-train-b64-serve-b8-v1",
        "expert_batch_isolation": "sample_isolated_grouped_mm_v2",
        "kind": "resumable-libero-training",
        "last_metrics": last_metrics,
        "optimizer_parameter_schema_sha256": optimizer_schema,
        "parent_manifest_sha256": None,
        "physical_batch_size": 64,
        "run_seed": 0,
        "run_uuid": run_uuid,
        "schema": "duo-vla-checkpoint-v1",
        "serving_batch_size": 8,
        "source_tree_sha256": source_sha256,
        "tensor_parallel_size": 1,
        "trainer_state": TrainerState(1000, 64_000).to_dict(),
        "training_rank_state_sha256": [artifacts["training_rank_000"]["sha256"]],
    }
    manifest_path = checkpoint / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_sha256 = _sha256(manifest_path)
    create_run_journal(run_root, config_sha256=config_sha256, run_uuid=run_uuid)
    record_latest_checkpoint(
        run_root,
        checkpoint=checkpoint,
        update=1000,
        manifest_sha256=manifest_sha256,
        parent_manifest_sha256=None,
        last_metrics=last_metrics,
    )
    return {
        "checkpoint": checkpoint,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "optimizer_schema": optimizer_schema,
        "run_uuid": run_uuid,
        "source_sha256": source_sha256,
        "venv_identity": venv_identity,
    }


def _create_fork(authenticated_parent: dict[str, Any], tmp_path: Path) -> tuple[dict[str, Any], Path, str]:
    child_output = tmp_path / "child-run"
    child_uuid = str(uuid.uuid4())
    manifest = create_dp2_fork_manifest(
        project_root=PROJECT_ROOT,
        parent_checkpoint=authenticated_parent["checkpoint"],
        expected_parent_manifest_sha256=authenticated_parent["manifest_sha256"],
        expected_parent_source_tree_sha256=authenticated_parent["source_sha256"],
        expected_parent_run_uuid=authenticated_parent["run_uuid"],
        config_path=PROJECT_ROOT / "configs/libero_dp2_fused_v2_b32.toml",
        child_run_uuid=child_uuid,
        child_output_dir=child_output,
    )
    return manifest, child_output, child_uuid


def test_dp2_config_and_source_hash_are_shared_by_trainer_server_and_fork() -> None:
    config = load_resolved_toml(PROJECT_ROOT / "configs/libero_dp2_fused_v2_b32.toml")
    _validate_dp2_config(config)
    assert config["execution_profile"] == DP2_EXECUTION_PROFILE == TRAIN.DP2_EXECUTION_PROFILE
    expected_optimization = {
        "physical_batch_size": 32,
        "microbatch_size": 32,
        "gradient_accumulation_steps": 1,
        "global_batch_size": 64,
        "serving_batch_size": 8,
    }
    assert {name: config["optimization"][name] for name in expected_optimization} == expected_optimization
    assert config["training"]["max_cached_files"] == 377
    source_sha256 = dp2_source_identity(PROJECT_ROOT)["source_tree_sha256"]
    assert source_sha256 == TRAIN._source_tree_sha256(PROJECT_ROOT)
    assert source_sha256 == SERVER._training_source_tree_sha256(PROJECT_ROOT)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("optimization", "lora_learning_rate", 2e-4),
        ("optimization", "adam_beta2", 0.9),
        ("optimization", "gradient_clip_norm", 2.0),
        ("optimization", "warmup_updates", 999),
        ("optimization", "total_updates", 29_999),
        ("policy", "objective", "changed"),
        ("sampling", "flow_path", "changed"),
        ("training", "validation_interval", 999),
    ),
)
def test_dp2_semantic_recipe_rejects_parent_child_drift(
    section: str,
    field: str,
    value: object,
) -> None:
    parent = load_resolved_toml(PROJECT_ROOT / "configs/libero_single_gpu_fused_v2_b64.toml")
    parent["run"] = {"max_cached_files": 377, "seed": 0, "task": None}
    child = load_resolved_toml(PROJECT_ROOT / "configs/libero_dp2_fused_v2_b32.toml")
    expected = validate_dp2_semantic_recipe_continuity(parent, child)
    assert expected == dp2_semantic_recipe_sha256(parent) == dp2_semantic_recipe_sha256(child)

    child[section][field] = value
    with pytest.raises(ValueError, match="semantic recipe differs"):
        validate_dp2_semantic_recipe_continuity(parent, child)


def test_dp2_bridge_and_server_derive_single_gpu_b8_serving() -> None:
    config = load_resolved_toml(PROJECT_ROOT / "configs/libero_dp2_fused_v2_b32.toml")
    kernel_sha256 = SERVER.sha256_file(PROJECT_ROOT / "src/duo_vla/backbones/shared_weight_grouped_mm_triton.py")
    training = {
        "canonical_plan_partition": "contiguous-b8-chunks-by-rank",
        "data_parallel_size": 2,
        "execution_profile": DP2_EXECUTION_PROFILE,
        "expert_batch_isolation": "sample_isolated_grouped_mm_v2",
        "experts_implementation": "grouped_mm",
        "fixed_physical_prefix_width": 545,
        "gradient_reduction": "sum_globally_normalized_sse_gradients",
        "physical_batch_size": 32,
        "prefix_geometry_content_sha256": config["benchmark"]["prefix_geometry_content_sha256"],
        "rank_physical_batch_size": 32,
        "serving_batch_size": 8,
        "shared_weight_kernel_sha256": kernel_sha256,
        "strategy": "data_parallel",
        "tensor_parallel_size": 1,
        "world_size": 2,
    }
    assert BRIDGE.validate_execution_geometry(training) == training
    serving = BRIDGE.serving_execution_geometry(training)
    assert serving["physical_batch_size"] == 8
    assert serving["tensor_parallel_size"] == 1
    assert serving["execution_profile"] == "duovla-single-gpu-tp1-fused-v2-serve-b8-v1"
    config["execution_geometry"] = training
    assert SERVER._execution_geometry_from_config(config) == training
    assert SERVER._training_launcher_name(training) == "scripts/run_libero_train_dp2.sh"
    assert SERVER._training_launcher_name({"tensor_parallel_size": 1}) == "scripts/run_libero_train_single_gpu.sh"
    assert SERVER._training_launcher_name({"tensor_parallel_size": 2}) == "scripts/run_libero_train.sh"

    drifted = dict(training, gradient_reduction="mean")
    with pytest.raises(BRIDGE.BridgeProtocolError, match="data-parallel execution geometry differs"):
        BRIDGE.validate_execution_geometry(drifted)


def test_launcher_seals_dp2_hardware_runtime_config_and_modes(tmp_path: Path) -> None:
    launcher = PROJECT_ROOT / "scripts/run_libero_train_dp2.sh"
    source = launcher.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=0,1" in source
    assert 'environment_path="${cache_root}/venvs/train-single-gpu"' in source
    assert "--nproc-per-node=2" in source
    assert "configs/libero_dp2_fused_v2_b32.toml" in source
    assert "GPU-30424b03-3051-615a-832e-186511378a61" in source
    assert "GPU-84fa4004-92fb-8f86-cc65-01d62a27950e" in source
    assert "--fork-from FROZEN_FORK_MANIFEST.json" in source
    assert "--expected-fork-manifest-sha256" in source
    assert "--query-compute-apps=pid,gpu_uuid" in source
    assert "are not idle" in source
    assert '/venvs/train"' not in source

    environment = {
        **os.environ,
        "DUO_VLA_CACHE_ROOT": "/hdd2/hyunbin/vla/cache",
        "HF_HOME": "/hdd2/hyunbin/vla/huggingface",
    }
    missing_mode = subprocess.run(
        [str(launcher)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_mode.returncode == 2
    assert "requires exactly one" in missing_mode.stderr
    both_modes = subprocess.run(
        [str(launcher), "--fork-from", "/missing", "--resume", "/missing"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert both_modes.returncode == 2
    assert "mutually exclusive" in both_modes.stderr
    missing_digest = subprocess.run(
        [str(launcher), "--fork-from", "/missing"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_digest.returncode == 2
    assert "requires exactly one preregistered" in missing_digest.stderr
    duplicate_digest = subprocess.run(
        [
            str(launcher),
            "--fork-from",
            "/missing",
            "--expected-fork-manifest-sha256",
            "a" * 64,
            "--expected-fork-manifest-sha256",
            "b" * 64,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert duplicate_digest.returncode == 2
    assert "requires exactly one preregistered" in duplicate_digest.stderr
    digest_on_resume = subprocess.run(
        [
            str(launcher),
            "--resume",
            "checkpoints/update-001100",
            "--expected-fork-manifest-sha256",
            "a" * 64,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert digest_on_resume.returncode == 2
    assert "forbidden in resume mode" in digest_on_resume.stderr
    fork_path = (tmp_path / "fork.json").resolve()
    fork_path.write_text("{}\n", encoding="utf-8")
    fork_path.with_name("fork.json.sha256").write_text(f"{'a' * 64}  fork.json\n", encoding="ascii")
    mismatched_publication = subprocess.run(
        [
            str(launcher),
            "--fork-from",
            str(fork_path),
            "--expected-fork-manifest-sha256",
            "a" * 64,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert mismatched_publication.returncode == 2
    assert "JSON or sidecar differs" in mismatched_publication.stderr


def test_fork_manifest_authenticates_journal_parent_child_source_venv_and_hardware(
    authenticated_parent: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest, child_output, child_uuid = _create_fork(authenticated_parent, tmp_path)
    fork_path = (tmp_path / "frozen/fork.json").resolve()
    fork_sha256 = write_dp2_fork_manifest(fork_path, manifest)
    assert validate_dp2_fork_manifest(manifest) == manifest
    assert manifest["parent"]["manifest_sha256"] == authenticated_parent["manifest_sha256"]
    assert manifest["parent"]["run_uuid"] == authenticated_parent["run_uuid"]
    assert manifest["parent"]["update"] == 1000
    assert "resolved_config" in manifest["parent"]["artifacts"]
    assert manifest["parent"]["semantic_recipe_sha256"] == manifest["child"]["config"]["semantic_recipe_sha256"]
    assert manifest["parent"]["train_venv_identity"] == authenticated_parent["venv_identity"]
    assert manifest["child"]["run_uuid"] == child_uuid
    assert manifest["child"]["run_uuid"] != manifest["parent"]["run_uuid"]
    assert manifest["child"]["training_hardware"]["physical_gpu_uuids"] == list(EXPECTED_TRAINING_GPU_UUIDS)
    assert manifest["child"]["train_venv_identity"] == authenticated_parent["venv_identity"]
    authenticate_dp2_child_environment(
        manifest,
        fork_manifest_path=fork_path,
        fork_manifest_sha256=fork_sha256,
        project_root=PROJECT_ROOT,
        config_path=PROJECT_ROOT / "configs/libero_dp2_fused_v2_b32.toml",
        child_output_dir=child_output,
        child_run_uuid=child_uuid,
        live_train_venv_identity=authenticated_parent["venv_identity"],
        live_training_gpu_uuids=EXPECTED_TRAINING_GPU_UUIDS,
    )
    with pytest.raises(ValueError, match="GPU UUID inventory differs"):
        authenticate_dp2_child_environment(
            manifest,
            fork_manifest_path=fork_path,
            fork_manifest_sha256=fork_sha256,
            project_root=PROJECT_ROOT,
            config_path=PROJECT_ROOT / "configs/libero_dp2_fused_v2_b32.toml",
            child_output_dir=child_output,
            child_run_uuid=child_uuid,
            live_train_venv_identity=authenticated_parent["venv_identity"],
            live_training_gpu_uuids=reversed(EXPECTED_TRAINING_GPU_UUIDS),
        )
    wrong_venv = {**authenticated_parent["venv_identity"], "root_sha256": "9" * 64}
    with pytest.raises(ValueError, match="train-venv identity differs"):
        authenticate_dp2_child_environment(
            manifest,
            fork_manifest_path=fork_path,
            fork_manifest_sha256=fork_sha256,
            project_root=PROJECT_ROOT,
            config_path=PROJECT_ROOT / "configs/libero_dp2_fused_v2_b32.toml",
            child_output_dir=child_output,
            child_run_uuid=child_uuid,
            live_train_venv_identity=wrong_venv,
            live_training_gpu_uuids=EXPECTED_TRAINING_GPU_UUIDS,
        )

    copied = tmp_path / "copied-parent/checkpoints/update-001000"
    shutil.copytree(authenticated_parent["checkpoint"], copied)
    with pytest.raises((FileNotFoundError, ValueError), match=r"journal|run_journal"):
        create_dp2_fork_manifest(
            project_root=PROJECT_ROOT,
            parent_checkpoint=copied,
            expected_parent_manifest_sha256=authenticated_parent["manifest_sha256"],
            expected_parent_source_tree_sha256=authenticated_parent["source_sha256"],
            expected_parent_run_uuid=authenticated_parent["run_uuid"],
            config_path=PROJECT_ROOT / "configs/libero_dp2_fused_v2_b32.toml",
            child_run_uuid=str(uuid.uuid4()),
            child_output_dir=tmp_path / "copied-child",
        )


def test_live_fork_and_parent_resolved_config_digests_are_reauthenticated(
    authenticated_parent: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest, child_output, child_uuid = _create_fork(authenticated_parent, tmp_path)
    fork_path = (tmp_path / "frozen/fork.json").resolve()
    fork_sha256 = write_dp2_fork_manifest(fork_path, manifest)
    sidecar = fork_path.with_name("fork.json.sha256")
    sidecar.write_text(f"{'0' * 64}  fork.json\n", encoding="ascii")
    with pytest.raises(ValueError, match=r"raw SHA-256 mismatch|sidecar digest changed"):
        authenticate_dp2_child_environment(
            manifest,
            fork_manifest_path=fork_path,
            fork_manifest_sha256=fork_sha256,
            project_root=PROJECT_ROOT,
            config_path=PROJECT_ROOT / "configs/libero_dp2_fused_v2_b32.toml",
            child_output_dir=child_output,
            child_run_uuid=child_uuid,
            live_train_venv_identity=authenticated_parent["venv_identity"],
            live_training_gpu_uuids=EXPECTED_TRAINING_GPU_UUIDS,
        )

    sidecar.write_text(f"{fork_sha256}  fork.json\n", encoding="ascii")
    resolved_record = manifest["parent"]["artifacts"]["resolved_config"]
    resolved_path = authenticated_parent["checkpoint"] / resolved_record["path"]
    resolved_path.write_bytes(resolved_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match=r"artifact.*(?:size|hash)|resolved-config"):
        authenticate_dp2_fork_parent(
            manifest,
            fork_manifest_path=fork_path,
            fork_manifest_sha256=fork_sha256,
        )


def test_published_fork_uses_json_commit_marker_and_strict_sidecar(
    authenticated_parent: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest, _, _ = _create_fork(authenticated_parent, tmp_path)
    output = (tmp_path / "frozen/fork.json").resolve()
    digest = write_dp2_fork_manifest(output, manifest)
    loaded, observed_digest = load_published_dp2_fork_manifest(output)
    assert loaded == manifest
    assert observed_digest == digest == _sha256(output)
    assert output.with_name("fork.json.sha256").read_text(encoding="ascii") == f"{digest}  fork.json\n"
    with pytest.raises(FileExistsError):
        write_dp2_fork_manifest(output, manifest)

    output.with_name("fork.json.sha256").write_text(f"{'0' * 64}  fork.json\n", encoding="ascii")
    with pytest.raises(ValueError, match="raw SHA-256 mismatch"):
        load_published_dp2_fork_manifest(output)


def test_fork_restore_returns_typed_lineage_and_domain_separates_rank_rng(
    authenticated_parent: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest, _, _ = _create_fork(authenticated_parent, tmp_path)
    output = (tmp_path / "frozen/fork.json").resolve()
    digest = write_dp2_fork_manifest(output, manifest)
    published, _ = load_published_dp2_fork_manifest(output)
    child_contract = dp2_fork_run_contract(published, fork_manifest_sha256=digest)

    rank_samples = []
    rank_results = []
    for rank in range(2):
        parameter, optimizer, scheduler = _optimizer_and_scheduler()
        optimizer.state.clear()
        result = restore_tp1_fork_training_state(
            published,
            fork_manifest_path=output,
            fork_manifest_sha256=digest,
            rank=rank,
            world_size=2,
            optimizer=optimizer,
            named_parameters=[("model.weight", parameter)],
            scheduler=scheduler,
            child_run_contract=child_contract,
        )
        rank_results.append(result)
        rank_samples.append((np.random.randint(0, 2**31), torch.randint(0, 2**31, ()).item()))
        assert optimizer.state[parameter]["step"].item() == 1000
        assert scheduler.last_epoch == 1000

    assert all(isinstance(result, DP2ForkRestoreResult) for result in rank_results)
    assert rank_results[0].trainer_state == TrainerState(1000, 64_000)
    assert rank_results[0].parent_manifest_sha256 == authenticated_parent["manifest_sha256"]
    assert rank_results[0].parent_run_uuid == authenticated_parent["run_uuid"]
    assert rank_results[0].parent_config_sha256 == authenticated_parent["config_sha256"]
    assert rank_results[0].parent_optimizer_parameter_schema_sha256 == authenticated_parent["optimizer_schema"]
    assert rank_samples[0] != rank_samples[1]


def test_rng_derivation_is_deterministic_and_binds_parent_child_and_rank() -> None:
    arguments = {
        "parent_training_state_sha256": "1" * 64,
        "parent_manifest_sha256": "2" * 64,
        "parent_run_uuid": str(uuid.uuid4()),
        "child_run_uuid": str(uuid.uuid4()),
    }
    first = derive_dp2_rank_rng_seeds(**arguments, rank=0)
    assert first == derive_dp2_rank_rng_seeds(**arguments, rank=0)
    assert first != derive_dp2_rank_rng_seeds(**arguments, rank=1)
    changed = copy.deepcopy(arguments)
    changed["child_run_uuid"] = str(uuid.uuid4())
    assert first != derive_dp2_rank_rng_seeds(**changed, rank=0)
