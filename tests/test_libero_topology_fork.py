from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_libero as TRAIN  # noqa: E402

from duo_vla.dp2_fork import DP2_EXECUTION_PROFILE, TP1_PARENT_EXECUTION_PROFILE  # noqa: E402
from duo_vla.run_config import load_resolved_toml, save_resolved_config  # noqa: E402
from duo_vla.run_journal import create_run_journal, record_latest_checkpoint  # noqa: E402
from duo_vla.topology_fork import (  # noqa: E402
    TOPOLOGY_FORK_BOUNDARY,
    TOPOLOGY_FORK_RNG_DERIVATION,
    TOPOLOGY_FORK_SCHEMA,
    create_topology_fork_manifest,
    derive_topology_rank_rng_seeds,
    has_topology_fork_lineage,
    load_published_topology_fork_manifest,
    restore_topology_fork_training_state,
    topology_fork_lineage,
    topology_fork_run_contract,
    validate_topology_checkpoint_lineage,
    write_topology_fork_manifest,
)
from duo_vla.training import TrainerState  # noqa: E402
from duo_vla.training_checkpoint import (  # noqa: E402
    TRAINING_RANK_STATE_SCHEMA,
    capture_rng_state,
    optimizer_parameter_inventory,
    optimizer_parameter_inventory_sha256,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, checkpoint: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(checkpoint).as_posix(),
        "sha256": _sha(path),
    }


def _parent_checkpoint(tmp_path: Path, *, kind: str, update: int = 2500) -> tuple[Path, dict[str, object]]:
    run = tmp_path / f"parent-{kind}"
    checkpoint = run / "checkpoints" / f"update-{update:06d}"
    (checkpoint / "lora").mkdir(parents=True)
    (checkpoint / "artifacts").mkdir()
    interface = checkpoint / "interface.safetensors"
    lora_config = checkpoint / "lora/adapter_config.json"
    lora_weights = checkpoint / "lora/adapter_model.safetensors"
    interface.write_bytes(b"interface")
    lora_config.write_bytes(b"{}")
    lora_weights.write_bytes(b"lora")

    config_name = "libero_dp2_fused_v2_b32.toml" if kind == "dp2" else "libero_single_gpu_fused_v2_b64.toml"
    config = load_resolved_toml(PROJECT_ROOT / "configs" / config_name)
    config["run"] = {"max_cached_files": 377, "seed": 0, "task": None}
    resolved = checkpoint / "artifacts/resolved_config.json"
    config_sha = save_resolved_config(resolved, config)

    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([{"name": "trainable", "params": [parameter]}], lr=1e-4)
    optimizer.state[parameter] = {
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.ones_like(parameter),
        "step": torch.tensor(float(update)),
    }
    inventory = optimizer_parameter_inventory(optimizer, [("model.weight", parameter)])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scheduler.last_epoch = update
    scheduler._step_count = update + 1
    world_size = 2 if kind == "dp2" else 1
    rank_hashes = []
    rank_artifacts: dict[str, object] = {}
    for rank in range(world_size):
        payload = {
            "optimizer": optimizer.state_dict(),
            "optimizer_parameter_inventory": inventory,
            "rank": rank,
            "rng": capture_rng_state(),
            "run_contract": {"shared": "contract"},
            "scheduler": scheduler.state_dict(),
            "schema": TRAINING_RANK_STATE_SCHEMA,
            "trainer_state": TrainerState(update, update * 64).to_dict(),
            "world_size": world_size,
        }
        path = checkpoint / f"artifacts/training_rank_{rank:03d}.pt"
        torch.save(payload, path)
        rank_hashes.append(_sha(path))
        rank_artifacts[f"training_rank_{rank:03d}"] = _artifact(path, checkpoint)

    run_uuid = str(uuid.uuid4())
    source_sha = "d" * 64
    profile = DP2_EXECUTION_PROFILE if kind == "dp2" else TP1_PARENT_EXECUTION_PROFILE
    geometry = {
        "execution_profile": profile,
        "physical_batch_size": 32 if kind == "dp2" else 64,
        "tensor_parallel_size": 1,
    }
    if kind == "dp2":
        geometry.update(strategy="data_parallel", world_size=2, data_parallel_size=2)
    last_metrics = {"train_loss": 0.2, "update": update}
    manifest: dict[str, object] = {
        "artifacts": {
            "interface": _artifact(interface, checkpoint),
            "lora_config": _artifact(lora_config, checkpoint),
            "lora_weights": _artifact(lora_weights, checkpoint),
            "resolved_config": _artifact(resolved, checkpoint),
            **rank_artifacts,
        },
        "config_sha256": config_sha,
        "execution_environment": {"authenticated_runtime": {"train_venv": {"root_sha256": "e" * 64}}},
        "execution_geometry": geometry,
        "execution_profile": profile,
        "last_metrics": last_metrics,
        "optimizer_parameter_schema_sha256": optimizer_parameter_inventory_sha256(inventory),
        "parent_manifest_sha256": None,
        "physical_batch_size": 32 if kind == "dp2" else 64,
        "replicated_optimizer_sha256": "a" * 64,
        "replicated_parameter_sha256": "b" * 64,
        "run_seed": 0,
        "run_uuid": run_uuid,
        "schema": "duo-vla-checkpoint-v1",
        "source_tree_sha256": source_sha,
        "trainer_state": TrainerState(update, update * 64).to_dict(),
        "training_rank_state_sha256": rank_hashes,
    }
    if kind == "dp2":
        manifest.update(data_parallel_size=2, world_size=2)
    manifest_path = checkpoint / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_sha = _sha(manifest_path)
    create_run_journal(run, config_sha256=config_sha, run_uuid=run_uuid)
    record_latest_checkpoint(
        run,
        checkpoint=checkpoint,
        update=update,
        manifest_sha256=manifest_sha,
        parent_manifest_sha256=None,
        last_metrics=last_metrics,
    )
    return checkpoint, {
        "manifest_sha256": manifest_sha,
        "run_uuid": run_uuid,
        "source_sha256": source_sha,
    }


@pytest.mark.parametrize(
    ("parent_kind", "child_config", "gpu_indices", "child_kind"),
    (
        ("dp2", "libero_single_gpu_fused_v2_b64.toml", [1], "tp1"),
        ("tp1", "libero_dp2_fused_v2_b32.toml", [0, 1], "dp2"),
    ),
)
def test_topology_fork_authenticates_both_transition_directions(
    tmp_path: Path,
    parent_kind: str,
    child_config: str,
    gpu_indices: list[int],
    child_kind: str,
) -> None:
    checkpoint, identity = _parent_checkpoint(tmp_path, kind=parent_kind)
    child_output = tmp_path / "child"
    manifest = create_topology_fork_manifest(
        project_root=PROJECT_ROOT,
        parent_checkpoint=checkpoint,
        expected_parent_manifest_sha256=identity["manifest_sha256"],
        expected_parent_source_tree_sha256=identity["source_sha256"],
        expected_parent_run_uuid=identity["run_uuid"],
        expected_parent_update=TOPOLOGY_FORK_BOUNDARY,
        child_config_path=PROJECT_ROOT / "configs" / child_config,
        child_run_uuid=str(uuid.uuid4()),
        child_output_dir=child_output,
        child_physical_gpu_indices=gpu_indices,
    )
    assert manifest["parent"]["topology"]["kind"] == parent_kind
    assert manifest["child"]["topology"]["kind"] == child_kind
    assert manifest["continuity"]["next_update"] == TOPOLOGY_FORK_BOUNDARY
    path = tmp_path / "fork" / "fork.json"
    digest = write_topology_fork_manifest(path, manifest)
    loaded, observed = load_published_topology_fork_manifest(path)
    assert loaded == manifest
    assert observed == digest

    fields = topology_fork_run_contract(manifest, manifest_sha256=digest)
    lineage = topology_fork_lineage(manifest, manifest_sha256=digest)
    checkpoint_manifest = {
        **fields,
        "optimizer_parameter_schema_sha256": manifest["parent"]["optimizer_parameter_schema_sha256"],
        "topology_fork_lineage": lineage,
    }
    assert has_topology_fork_lineage(checkpoint_manifest)
    assert validate_topology_checkpoint_lineage(checkpoint_manifest) == (fields, lineage)


def test_topology_fork_rejects_off_boundary_and_nontransition(tmp_path: Path) -> None:
    checkpoint, identity = _parent_checkpoint(tmp_path, kind="tp1", update=2501)
    with pytest.raises(ValueError, match="2500"):
        create_topology_fork_manifest(
            project_root=PROJECT_ROOT,
            parent_checkpoint=checkpoint,
            expected_parent_manifest_sha256=identity["manifest_sha256"],
            expected_parent_source_tree_sha256=identity["source_sha256"],
            expected_parent_run_uuid=identity["run_uuid"],
            expected_parent_update=2501,
            child_config_path=PROJECT_ROOT / "configs/libero_dp2_fused_v2_b32.toml",
            child_run_uuid=str(uuid.uuid4()),
            child_output_dir=tmp_path / "child",
            child_physical_gpu_indices=[0, 1],
        )


def test_topology_fork_rng_binds_rank_and_child_identity() -> None:
    kwargs = {
        "parent_rank_state_sha256": ["a" * 64, "b" * 64],
        "parent_manifest_sha256": "c" * 64,
        "parent_run_uuid": "00000000-0000-0000-0000-000000000001",
        "child_run_uuid": "00000000-0000-0000-0000-000000000002",
    }
    first = derive_topology_rank_rng_seeds(**kwargs, rank=0)
    assert first == derive_topology_rank_rng_seeds(**kwargs, rank=0)
    assert first != derive_topology_rank_rng_seeds(**kwargs, rank=1)
    assert set(first) == {"python", "numpy", "torch_cpu", "torch_cuda"}


def test_topology_fork_restores_optimizer_scheduler_and_progress(tmp_path: Path) -> None:
    checkpoint, identity = _parent_checkpoint(tmp_path, kind="dp2")
    manifest = create_topology_fork_manifest(
        project_root=PROJECT_ROOT,
        parent_checkpoint=checkpoint,
        expected_parent_manifest_sha256=identity["manifest_sha256"],
        expected_parent_source_tree_sha256=identity["source_sha256"],
        expected_parent_run_uuid=identity["run_uuid"],
        expected_parent_update=2500,
        child_config_path=PROJECT_ROOT / "configs/libero_single_gpu_fused_v2_b64.toml",
        child_run_uuid=str(uuid.uuid4()),
        child_output_dir=tmp_path / "child",
        child_physical_gpu_indices=[1],
    )
    fork_path = tmp_path / "fork.json"
    digest = write_topology_fork_manifest(fork_path, manifest)
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([{"name": "trainable", "params": [parameter]}], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    run_contract = topology_fork_run_contract(manifest, manifest_sha256=digest)
    result = restore_topology_fork_training_state(
        manifest,
        manifest_path=fork_path,
        manifest_sha256=digest,
        rank=0,
        world_size=1,
        optimizer=optimizer,
        named_parameters=[("model.weight", parameter)],
        scheduler=scheduler,
        child_run_contract=run_contract,
    )
    assert result.trainer_state == TrainerState(2500, 160_000)
    assert scheduler.last_epoch == 2500
    assert optimizer.state[parameter]["step"].item() == 2500
    assert result.lineage["parent_rank_state_sha256"] == manifest["parent"]["rank_state_sha256"]


def test_topology_fork_cli_rejects_semantic_overrides_and_allows_sealed_cache() -> None:
    child = {"max_cached_files": 377, "run_seed": 0}
    values = {
        "task": None,
        "seed": None,
        "total_updates": None,
        "warmup_updates": None,
        "microbatch_size": None,
        "gradient_accumulation_steps": None,
        "validation_interval": None,
        "validation_samples": None,
        "checkpoint_interval": None,
        "permanent_checkpoint_interval": None,
        "log_interval": None,
        "max_cached_files": 377,
    }
    TRAIN._validate_topology_fork_cli(SimpleNamespace(**values), {"child": child})
    for name, value in (("seed", 0), ("task", "task"), ("total_updates", 10_000), ("max_cached_files", 128)):
        with pytest.raises(ValueError, match=name):
            TRAIN._validate_topology_fork_cli(
                SimpleNamespace(**{**values, name: value}),
                {"child": child},
            )


def test_dp2_restore_mode_accepts_generic_fork_without_weakening_legacy_rules() -> None:
    TRAIN._validate_dp2_restore_mode(
        is_data_parallel=True,
        has_fork_manifest=False,
        has_resume_checkpoint=False,
        has_topology_fork_manifest=True,
    )
    with pytest.raises(ValueError, match="exactly one"):
        TRAIN._validate_dp2_restore_mode(
            is_data_parallel=True,
            has_fork_manifest=True,
            has_resume_checkpoint=False,
            has_topology_fork_manifest=True,
        )
    assert TOPOLOGY_FORK_SCHEMA == "duo-vla-libero-topology-fork-v1"
    assert TOPOLOGY_FORK_RNG_DERIVATION.endswith("-v1")


def test_launchers_seal_topology_fork_seed_digest_and_restore_mode() -> None:
    environment = {
        **os.environ,
        "DUO_VLA_CACHE_ROOT": "/hdd2/hyunbin/vla/cache",
        "HF_HOME": "/hdd2/hyunbin/vla/huggingface",
    }
    single = subprocess.run(
        [
            str(PROJECT_ROOT / "scripts/run_libero_train_single_gpu.sh"),
            "--topology-fork-from",
            "/missing",
            "--seed",
            "0",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert single.returncode == 2
    assert "seed is sealed" in single.stderr

    dp2 = subprocess.run(
        [str(PROJECT_ROOT / "scripts/run_libero_train_dp2.sh"), "--topology-fork-from", "/missing"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dp2.returncode == 2
    assert "requires exactly one preregistered --expected-topology" in dp2.stderr
