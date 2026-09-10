from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import serve_libero_policy as SERVER
from libero_bridge import LIBERO_EXECUTION_GEOMETRY
from serve_libero_policy import (
    DATASET_CONTENT_INVENTORY_SHA256,
    DATASET_FILES_VERIFIED,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_TOTAL_BYTES,
    DATASET_TREE_SHA256,
    EXPERT_BATCH_ISOLATION,
    EXPERTS_IMPLEMENTATION,
    IMAGE_SHAPE,
    MODEL_ID,
    MODEL_REVISION,
    NORMALIZATION_SHA256,
    PHYSICAL_BATCH_SIZE,
    REQUIRED_SERVING_ENVIRONMENT,
    _authenticated_training_environment,
    _execution_geometry_from_config,
    _expert_backend_functions,
    _health_payload,
    _training_source_tree_sha256,
    _validate_serving_process_environment,
    configure_and_identify_serving_runtime,
    latency_runtime_identity,
    resolve_checkpoint,
)

from duo_vla.policy_contract import policy_contract_from_config
from duo_vla.prefix_geometry import (
    CameraGeometry,
    SnapshotTreeIdentity,
    apply_fixed_prefix_chat_template,
    build_prefix_geometry_contract,
    save_prefix_geometry_contract,
)
from duo_vla.run_config import canonical_config_sha256, load_resolved_toml, save_resolved_config
from duo_vla.runtime_integrity import (
    BASE_PYTHON_RUNTIME_IDENTITY_SCHEMA,
    TRAIN_VENV_IDENTITY_SCHEMA,
    canonical_sha256,
    static_environment_identity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _train_venv_identity(environment_name: str = "train") -> dict[str, object]:
    venv_root = f"/root/.cache/duo-vla/venvs/{environment_name}"
    base_python_runtime: dict[str, object] = {
        "base_prefix": "/usr/local",
        "configured_home": "/usr/local/bin",
        "configured_home_resolved": "/usr/local/bin",
        "content_inventory_sha256": "5" * 64,
        "files_verified": 100,
        "pyvenv_cfg_bytes": 128,
        "pyvenv_cfg_sha256": "6" * 64,
        "resolved_executable": "/usr/local/bin/python3.11",
        "resolved_executable_bytes": 20_000,
        "resolved_executable_sha256": "7" * 64,
        "schema": BASE_PYTHON_RUNTIME_IDENTITY_SCHEMA,
        "startup_hooks": [],
        "startup_hooks_sha256": canonical_sha256([]),
        "symlinks_verified": 4,
        "total_bytes": 1_000_000,
        "tree_metadata_sha256": "8" * 64,
        "venv_python": f"{venv_root}/bin/python",
        "venv_python_link_target": "/usr/local/bin/python3.11",
        "venv_root": venv_root,
    }
    base_python_runtime["root_sha256"] = canonical_sha256(base_python_runtime)
    identity: dict[str, object] = {
        "base_python_runtime": base_python_runtime,
        "content_inventory_sha256": "1" * 64,
        "files_verified": 10,
        "root": venv_root,
        "schema": TRAIN_VENV_IDENTITY_SCHEMA,
        "startup_hooks": ["lib/python3.11/site-packages/known.pth"],
        "startup_hooks_sha256": "3" * 64,
        "symlinks_verified": 3,
        "total_bytes": 100,
        "tree_metadata_sha256": "4" * 64,
    }
    identity["root_sha256"] = canonical_sha256(
        {
            "base_python_runtime_root_sha256": base_python_runtime["root_sha256"],
            "content_inventory_sha256": identity["content_inventory_sha256"],
            "files_verified": identity["files_verified"],
            "schema": identity["schema"],
            "startup_hooks_sha256": identity["startup_hooks_sha256"],
            "symlinks_verified": identity["symlinks_verified"],
            "total_bytes": identity["total_bytes"],
            "tree_metadata_sha256": identity["tree_metadata_sha256"],
        }
    )
    return identity


@pytest.fixture(autouse=True)
def _canonical_paths_and_fake_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")
    monkeypatch.setenv("DUO_VLA_PROJECT_ROOT", str(PROJECT_ROOT))
    monkeypatch.setenv("DUO_VLA_TRAIN_VENV", "/root/.cache/duo-vla/venvs/train")
    monkeypatch.setenv("HF_HOME", "/root/.cache/huggingface")
    monkeypatch.setattr(SERVER.sys, "prefix", "/root/.cache/duo-vla/venvs/train")
    monkeypatch.setattr(SERVER.site, "ENABLE_USER_SITE", False)
    monkeypatch.setattr(SERVER, "content_address_train_venv", lambda _root: _train_venv_identity())


class _Batch(dict[str, torch.Tensor]):
    def to(self, _device: torch.device) -> _Batch:
        return self


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(padding_side="left")
        self.conversations: Any = None

    def apply_chat_template(self, conversations: Any, **kwargs: Any) -> _Batch:
        self.conversations = conversations
        batch = len(conversations)
        width = kwargs["processor_kwargs"]["max_length"]
        attention_mask = torch.zeros((batch, width), dtype=torch.long)
        attention_mask[:, -12:] = 1
        return _Batch(
            input_ids=torch.zeros((batch, width), dtype=torch.long),
            attention_mask=attention_mask,
            mm_token_type_ids=torch.zeros((batch, width), dtype=torch.long),
            pixel_values=torch.zeros((batch * 2, 3, 4, 4)),
            image_position_ids=torch.zeros((batch * 2, 4, 2), dtype=torch.long),
        )


def _checkpoint_fixture(
    tmp_path: Path,
    *,
    config_name: str = "libero.toml",
    expert_batch_isolation: str = EXPERT_BATCH_ISOLATION,
    physical_gpu: int = 0,
    physical_batch_size: int = PHYSICAL_BATCH_SIZE,
) -> tuple[Path, dict[str, object], dict[str, object], dict[str, object]]:
    checkpoint = tmp_path / "checkpoint"
    artifacts = checkpoint / "artifacts"
    artifacts.mkdir(parents=True)
    config = load_resolved_toml(PROJECT_ROOT / "configs" / config_name)
    config["model"]["expert_batch_isolation"] = expert_batch_isolation
    config["optimization"].update(
        physical_batch_size=physical_batch_size,
        microbatch_size=physical_batch_size,
        gradient_accumulation_steps=64 // physical_batch_size,
        global_batch_size=64,
        serving_batch_size=PHYSICAL_BATCH_SIZE,
    )
    if int(config["model"]["tensor_parallel_size"]) == 1 and (
        expert_batch_isolation != EXPERT_BATCH_ISOLATION or physical_batch_size != PHYSICAL_BATCH_SIZE
    ):
        backend_label = "fused-v2" if expert_batch_isolation == "sample_isolated_grouped_mm_v2" else "sequential-v1"
        config["execution_profile"] = (
            f"duovla-single-gpu-tp1-{backend_label}-train-b{physical_batch_size}-serve-b8-v1"
        )
    model_report: dict[str, object] = {
        "content_inventory_sha256": "c" * 64,
        "files_verified": 7,
        "revision": MODEL_REVISION,
        "total_bytes": 1234,
        "tree_metadata_sha256": "b" * 64,
    }
    identity = SnapshotTreeIdentity.from_huggingface_report(MODEL_ID, model_report)
    prefix_geometry = build_prefix_geometry_contract(
        model_identity=identity,
        processor_identity=identity,
        ordered_cameras=(
            CameraGeometry("agentview", *IMAGE_SHAPE[:2]),
            CameraGeometry("eye_in_hand", *IMAGE_SHAPE[:2]),
        ),
        instruction_lengths={"pick the block": 12},
        fixed_physical_prefix_width=int(config["benchmark"]["fixed_physical_prefix_width"]),
        padding_side="left",
    )
    config["benchmark"]["prefix_geometry_content_sha256"] = prefix_geometry["content_sha256"]
    tensor_parallel_size = int(config["model"]["tensor_parallel_size"])
    environment_name = "train-single-gpu" if tensor_parallel_size == 1 else "train"
    execution_geometry: dict[str, object] = {
        "experts_implementation": EXPERTS_IMPLEMENTATION,
        "expert_batch_isolation": expert_batch_isolation,
        "physical_batch_size": physical_batch_size,
        "serving_batch_size": PHYSICAL_BATCH_SIZE,
        "fixed_physical_prefix_width": config["benchmark"]["fixed_physical_prefix_width"],
        "prefix_geometry_content_sha256": prefix_geometry["content_sha256"],
    }
    if config.get("execution_profile") is not None:
        execution_geometry.update(
            execution_profile=config["execution_profile"],
            tensor_parallel_size=tensor_parallel_size,
        )
    if expert_batch_isolation == "sample_isolated_grouped_mm_v2":
        execution_geometry["shared_weight_kernel_sha256"] = SERVER.sha256_file(
            PROJECT_ROOT / "src/duo_vla/backbones/shared_weight_grouped_mm_triton.py"
        )
    config["execution_geometry"] = execution_geometry
    config["artifact_trees"] = {
        "dataset_content_inventory_sha256": DATASET_CONTENT_INVENTORY_SHA256,
        "dataset_files_verified": DATASET_FILES_VERIFIED,
        "dataset_total_bytes": DATASET_TOTAL_BYTES,
        "dataset_tree_sha256": DATASET_TREE_SHA256,
        "model_content_inventory_sha256": model_report["content_inventory_sha256"],
        "model_files_verified": model_report["files_verified"],
        "model_total_bytes": model_report["total_bytes"],
        "model_tree_sha256": model_report["tree_metadata_sha256"],
    }
    train_environment = {
        **REQUIRED_SERVING_ENVIRONMENT,
        "CUDA_VISIBLE_DEVICES": str(physical_gpu) if tensor_parallel_size == 1 else "0,1",
        "DUO_VLA_CACHE_ROOT": "/root/.cache/duo-vla",
        "DUO_VLA_PROJECT_ROOT": str(PROJECT_ROOT),
        "DUO_VLA_TRAIN_VENV": f"/root/.cache/duo-vla/venvs/{environment_name}",
        "HF_HOME": "/root/.cache/huggingface",
        "PYTHONHASHSEED": "1",
    }
    training_environment: dict[str, object] = {
        "authenticated_runtime": {
            "algorithm_override_environment": {},
            "environment": dict(sorted(train_environment.items())),
            "nccl_environment": {},
            "static_environment_sha256": static_environment_identity(train_environment)["sha256"],
            "torchrun": {
                "group_world_size": 1,
                "local_rank_equals_rank": True,
                "local_world_size": tensor_parallel_size,
                "role_world_size": tensor_parallel_size,
                "world_size": tensor_parallel_size,
            },
            "train_venv": _train_venv_identity(environment_name),
        },
        "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "gpu_total_memory_bytes": [96 * 2**30] * tensor_parallel_size,
        "gpu_uuids": (
            [f"GPU-{physical_gpu}"]
            if tensor_parallel_size == 1
            else [f"GPU-{index}" for index in range(tensor_parallel_size)]
        ),
        "matmul_tf32": False,
        "preferred_blas_library": "_BlasBackend.Cublas",
        "preferred_linalg_library": "_LinalgBackend.Default",
        "python_hash_seed": "1",
    }
    config["execution_environment"] = training_environment
    config["source_tree_sha256"] = "d" * 64
    config_sha256 = save_resolved_config(artifacts / "resolved_config.json", config)
    (artifacts / "normalization.json").write_text("{}\n", encoding="utf-8")
    save_prefix_geometry_contract(artifacts / "prefix_geometry.json", prefix_geometry)
    (checkpoint / "manifest.json").write_text("{}\n", encoding="utf-8")
    contract = policy_contract_from_config(config).to_dict()
    manifest: dict[str, object] = {
        "artifacts": {
            "normalization": {"path": "artifacts/normalization.json"},
            "prefix_geometry": {"path": "artifacts/prefix_geometry.json"},
            "resolved_config": {"path": "artifacts/resolved_config.json"},
        },
        "config_sha256": config_sha256,
        "dataset_content_inventory_sha256": DATASET_CONTENT_INVENTORY_SHA256,
        "dataset_files_verified": DATASET_FILES_VERIFIED,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_total_bytes": DATASET_TOTAL_BYTES,
        "dataset_tree_sha256": DATASET_TREE_SHA256,
        "execution_environment": training_environment,
        "execution_environment_sha256": canonical_config_sha256(training_environment),
        "execution_geometry": execution_geometry,
        "kind": "resumable-libero-training",
        "experts_implementation": EXPERTS_IMPLEMENTATION,
        "expert_batch_isolation": expert_batch_isolation,
        "physical_batch_size": physical_batch_size,
        "serving_batch_size": PHYSICAL_BATCH_SIZE,
        "fixed_physical_prefix_width": config["benchmark"]["fixed_physical_prefix_width"],
        "prefix_geometry_content_sha256": prefix_geometry["content_sha256"],
        "model_id": MODEL_ID,
        "model_content_inventory_sha256": model_report["content_inventory_sha256"],
        "model_files_verified": model_report["files_verified"],
        "model_revision": MODEL_REVISION,
        "model_total_bytes": model_report["total_bytes"],
        "model_tree_sha256": model_report["tree_metadata_sha256"],
        "normalization_sha256": NORMALIZATION_SHA256,
        "policy_contract": contract,
        "policy_contract_sha256": canonical_config_sha256(contract),
        "run_seed": 1,
        "source_tree_sha256": "d" * 64,
        **(
            {"shared_weight_kernel_sha256": execution_geometry["shared_weight_kernel_sha256"]}
            if "shared_weight_kernel_sha256" in execution_geometry
            else {}
        ),
        **(
            {
                "execution_profile": config["execution_profile"],
                "tensor_parallel_size": tensor_parallel_size,
            }
            if "execution_profile" in config
            else {}
        ),
    }
    return checkpoint, manifest, config, model_report


@pytest.mark.parametrize(
    ("config_name", "expected_objective", "expected_nfe"),
    [("libero.toml", "rectified_flow", 10), ("libero_direct_regression.toml", "direct_regression", 1)],
)
def test_server_resolves_objective_only_from_verified_config_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    expected_objective: str,
    expected_nfe: int,
) -> None:
    import duo_vla.checkpointing
    import duo_vla.data.libero_stats

    checkpoint, manifest, config, model_report = _checkpoint_fixture(tmp_path, config_name=config_name)
    monkeypatch.setattr(duo_vla.checkpointing, "load_checkpoint_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        duo_vla.data.libero_stats,
        "load_libero_normalizers",
        lambda *args, **kwargs: (object(), object(), {"content_sha256": NORMALIZATION_SHA256}),
    )

    _, _, _, prefix_geometry, train_seed, report, resolved, contract = resolve_checkpoint(
        checkpoint,
        train_seed_override=None,
        model_snapshot_report=model_report,
    )
    assert train_seed == 1
    assert resolved == config
    assert contract["objective"] == expected_objective
    assert contract["nfe"] == expected_nfe
    assert report["policy_contract"] == contract
    assert prefix_geometry["content_sha256"] == config["benchmark"]["prefix_geometry_content_sha256"]
    assert report["execution_geometry"]["physical_batch_size"] == PHYSICAL_BATCH_SIZE
    assert report["dataset_tree_sha256"] == DATASET_TREE_SHA256
    assert report["dataset_content_inventory_sha256"] == DATASET_CONTENT_INVENTORY_SHA256
    assert report["dataset_files_verified"] == DATASET_FILES_VERIFIED
    assert report["dataset_total_bytes"] == DATASET_TOTAL_BYTES
    assert report["model_tree_sha256"] == model_report["tree_metadata_sha256"]
    assert report["model_content_inventory_sha256"] == model_report["content_inventory_sha256"]
    assert report["train_venv"] == _train_venv_identity()


def test_server_resolves_b64_v2_training_checkpoint_for_explicit_b8_serving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.checkpointing
    import duo_vla.data.libero_stats

    checkpoint, manifest, config, model_report = _checkpoint_fixture(
        tmp_path,
        config_name="libero_single_gpu.toml",
        expert_batch_isolation="sample_isolated_grouped_mm_v2",
        physical_gpu=1,
        physical_batch_size=64,
    )
    monkeypatch.setenv("DUO_VLA_TRAIN_VENV", "/root/.cache/duo-vla/venvs/train-single-gpu")
    monkeypatch.setattr(
        SERVER,
        "content_address_train_venv",
        lambda _root: _train_venv_identity("train-single-gpu"),
    )
    monkeypatch.setattr(duo_vla.checkpointing, "load_checkpoint_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        duo_vla.data.libero_stats,
        "load_libero_normalizers",
        lambda *args, **kwargs: (object(), object(), {"content_sha256": NORMALIZATION_SHA256}),
    )

    _, _, _, _, _, report, resolved, _ = resolve_checkpoint(
        checkpoint,
        train_seed_override=None,
        model_snapshot_report=model_report,
    )

    assert resolved == config
    assert report["execution_geometry"] == config["execution_geometry"]
    assert report["execution_geometry"]["physical_batch_size"] == 64
    assert report["execution_geometry"]["serving_batch_size"] == 8
    assert report["execution_geometry"]["expert_batch_isolation"] == "sample_isolated_grouped_mm_v2"


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("dataset_tree_sha256", "0" * 64),
        ("dataset_content_inventory_sha256", "0" * 64),
        ("dataset_files_verified", 381),
        ("dataset_total_bytes", DATASET_TOTAL_BYTES - 1),
    ),
)
def test_server_rejects_dataset_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    wrong_value: object,
) -> None:
    import duo_vla.checkpointing

    checkpoint, manifest, _, model_report = _checkpoint_fixture(tmp_path)
    manifest[field] = wrong_value
    monkeypatch.setattr(duo_vla.checkpointing, "load_checkpoint_manifest", lambda *args, **kwargs: manifest)

    with pytest.raises(RuntimeError, match=field):
        resolve_checkpoint(checkpoint, train_seed_override=None, model_snapshot_report=model_report)


def test_server_rejects_live_train_venv_content_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.checkpointing
    import duo_vla.data.libero_stats

    checkpoint, manifest, _, model_report = _checkpoint_fixture(tmp_path)
    monkeypatch.setattr(duo_vla.checkpointing, "load_checkpoint_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        duo_vla.data.libero_stats,
        "load_libero_normalizers",
        lambda *args, **kwargs: (object(), object(), {"content_sha256": NORMALIZATION_SHA256}),
    )
    changed = _train_venv_identity()
    changed["root_sha256"] = "f" * 64
    monkeypatch.setattr(SERVER, "content_address_train_venv", lambda _root: changed)

    with pytest.raises(RuntimeError, match="live train venv differs"):
        resolve_checkpoint(checkpoint, train_seed_override=None, model_snapshot_report=model_report)


def test_server_rejects_manifest_objective_that_disagrees_with_resolved_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.checkpointing

    checkpoint, manifest, _, model_report = _checkpoint_fixture(tmp_path)
    mismatched = copy.deepcopy(manifest)
    direct_config = load_resolved_toml(PROJECT_ROOT / "configs" / "libero_direct_regression.toml")
    mismatched_contract = policy_contract_from_config(direct_config).to_dict()
    mismatched["policy_contract"] = mismatched_contract
    mismatched["policy_contract_sha256"] = canonical_config_sha256(mismatched_contract)
    monkeypatch.setattr(duo_vla.checkpointing, "load_checkpoint_manifest", lambda *args, **kwargs: mismatched)

    with pytest.raises(ValueError, match="differs"):
        resolve_checkpoint(
            checkpoint,
            train_seed_override=None,
            model_snapshot_report=model_report,
        )


@pytest.mark.parametrize(
    ("mutation", "override", "message"),
    (
        ({"kind": "legacy-libero-training"}, None, "checkpoint kind"),
        ({"run_seed": None}, 1, "run_seed must be one"),
        ({"run_seed": 7}, None, "run_seed must be one"),
        ({"source_tree_sha256": "e" * 64}, None, "source-tree identities disagree"),
    ),
)
def test_server_rejects_legacy_checkpoint_identity_or_seed_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, object],
    override: int | None,
    message: str,
) -> None:
    import duo_vla.checkpointing
    import duo_vla.data.libero_stats

    checkpoint, manifest, _, model_report = _checkpoint_fixture(tmp_path)
    changed = {**manifest, **mutation}
    monkeypatch.setattr(duo_vla.checkpointing, "load_checkpoint_manifest", lambda *args, **kwargs: changed)
    monkeypatch.setattr(
        duo_vla.data.libero_stats,
        "load_libero_normalizers",
        lambda *args, **kwargs: (object(), object(), {"content_sha256": NORMALIZATION_SHA256}),
    )

    with pytest.raises(RuntimeError, match=message):
        resolve_checkpoint(
            checkpoint,
            train_seed_override=override,
            model_snapshot_report=model_report,
        )


def test_server_rejects_checkpoint_from_nondeterministic_training() -> None:
    environment = {
        "authenticated_runtime": {"algorithm_override_environment": {}, "nccl_environment": {}},
        "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "cudnn_tf32": False,
        "deterministic_algorithms": False,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "matmul_tf32": False,
        "preferred_blas_library": "_BlasBackend.Cublas",
        "preferred_linalg_library": "_LinalgBackend.Default",
        "python_hash_seed": "1",
    }
    config = {"execution_environment": environment}
    manifest = {
        "execution_environment": copy.deepcopy(environment),
        "execution_environment_sha256": canonical_config_sha256(environment),
    }

    with pytest.raises(RuntimeError, match="strict deterministic controls"):
        _authenticated_training_environment(config, manifest, project_root=PROJECT_ROOT, train_seed=1)

    environment["cudnn_deterministic"] = True
    environment["deterministic_algorithms"] = True
    environment["preferred_blas_library"] = "_BlasBackend.Cublaslt"
    manifest["execution_environment"] = copy.deepcopy(environment)
    manifest["execution_environment_sha256"] = canonical_config_sha256(environment)
    with pytest.raises(RuntimeError, match="preferred_blas_library"):
        _authenticated_training_environment(config, manifest, project_root=PROJECT_ROOT, train_seed=1)


def test_v5_health_exposes_exact_real_geometry_and_null_fake_identities() -> None:
    real_contract = {
        "objective": "rectified_flow",
        "sampler": "euler_uniform",
        "nfe": 10,
        "inference_seed_behavior": "episode_identity_gaussian_noise",
    }
    checkpoint_report = {"execution_geometry": LIBERO_EXECUTION_GEOMETRY}
    real = _health_payload(
        mode="real",
        train_seed=0,
        checkpoint_report=checkpoint_report,
        policy_contract=real_contract,
        latency_runtime_sha256="b" * 64,
        serving_runtime_sha256="a" * 64,
    )
    assert real["execution_geometry"] == LIBERO_EXECUTION_GEOMETRY
    assert real["training_execution_geometry"] == LIBERO_EXECUTION_GEOMETRY
    assert real["serving_execution_geometry"] == {
        "experts_implementation": "grouped_mm",
        "expert_batch_isolation": EXPERT_BATCH_ISOLATION,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "execution_profile": "duovla-tp2-sequential-v1-serve-b8-v1",
        "tensor_parallel_size": 2,
    }
    assert real["serving_runtime_sha256"] == "a" * 64
    assert real["latency_runtime_sha256"] == "b" * 64

    fake = _health_payload(
        mode="fake",
        train_seed=0,
        checkpoint_report=None,
        policy_contract={
            "objective": "test_fake",
            "sampler": "seeded_test_normal",
            "nfe": 0,
            "inference_seed_behavior": "episode_identity_test_generator",
        },
    )
    for name in (
        "checkpoint",
        "execution_geometry",
        "latency_runtime_sha256",
        "model_revision",
        "normalization_content_sha256",
        "serving_execution_geometry",
        "serving_runtime_sha256",
        "training_execution_geometry",
    ):
        assert fake[name] is None


def test_b64_v2_checkpoint_geometry_dispatches_to_b8_v2_serving() -> None:
    config = load_resolved_toml(PROJECT_ROOT / "configs/libero_single_gpu.toml")
    config["model"]["expert_batch_isolation"] = "sample_isolated_grouped_mm_v2"
    config["optimization"].update(
        physical_batch_size=64,
        microbatch_size=64,
        gradient_accumulation_steps=1,
        global_batch_size=64,
        serving_batch_size=8,
    )
    config["execution_profile"] = "duovla-single-gpu-tp1-fused-v2-train-b64-serve-b8-v1"
    kernel_sha256 = SERVER.sha256_file(
        PROJECT_ROOT / "src/duo_vla/backbones/shared_weight_grouped_mm_triton.py"
    )
    training_geometry = {
        "experts_implementation": "grouped_mm",
        "expert_batch_isolation": "sample_isolated_grouped_mm_v2",
        "physical_batch_size": 64,
        "serving_batch_size": 8,
        "fixed_physical_prefix_width": 545,
        "prefix_geometry_content_sha256": config["benchmark"]["prefix_geometry_content_sha256"],
        "execution_profile": "duovla-single-gpu-tp1-fused-v2-train-b64-serve-b8-v1",
        "tensor_parallel_size": 1,
        "shared_weight_kernel_sha256": kernel_sha256,
    }
    config["execution_geometry"] = training_geometry

    assert _execution_geometry_from_config(config) == training_geometry
    health = _health_payload(
        mode="real",
        train_seed=0,
        checkpoint_report={"execution_geometry": training_geometry},
        policy_contract={
            "objective": "rectified_flow",
            "sampler": "euler_uniform",
            "nfe": 10,
            "inference_seed_behavior": "episode_identity_gaussian_noise",
        },
        latency_runtime_sha256="b" * 64,
        serving_runtime_sha256="a" * 64,
    )
    assert health["execution_geometry"] == training_geometry
    assert health["training_execution_geometry"] == training_geometry
    assert health["serving_execution_geometry"] == {
        "experts_implementation": "grouped_mm",
        "expert_batch_isolation": "sample_isolated_grouped_mm_v2",
        "physical_batch_size": 8,
        "execution_profile": "duovla-single-gpu-tp1-fused-v2-serve-b8-v1",
        "tensor_parallel_size": 1,
        "shared_weight_kernel_sha256": kernel_sha256,
    }

    install, verify = _expert_backend_functions("sample_isolated_grouped_mm_v2")
    assert install.__name__ == "install_sample_isolated_grouped_mm_experts_v2"
    assert verify.__name__ == "verify_sample_isolated_grouped_mm_experts_v2"


def test_checkpoint_geometry_rejects_serving_or_declared_training_drift() -> None:
    config = load_resolved_toml(PROJECT_ROOT / "configs/libero_single_gpu.toml")
    config["optimization"].update(
        physical_batch_size=64,
        microbatch_size=64,
        gradient_accumulation_steps=1,
    )
    with pytest.raises(RuntimeError, match="explicitly configure serving batch eight"):
        _execution_geometry_from_config(config)

    config["optimization"].update(
        physical_batch_size=8,
        microbatch_size=8,
        gradient_accumulation_steps=8,
    )
    config["optimization"]["serving_batch_size"] = 64
    with pytest.raises(RuntimeError, match="serving batch size must be eight"):
        _execution_geometry_from_config(config)

    config["optimization"]["serving_batch_size"] = 8
    config["execution_geometry"] = {
        "experts_implementation": "grouped_mm",
        "expert_batch_isolation": EXPERT_BATCH_ISOLATION,
        "physical_batch_size": 32,
        "serving_batch_size": 8,
        "fixed_physical_prefix_width": 545,
        "prefix_geometry_content_sha256": config["benchmark"]["prefix_geometry_content_sha256"],
        "execution_profile": "duovla-single-gpu-tp1-v1",
        "tensor_parallel_size": 1,
    }
    with pytest.raises(RuntimeError, match="differs from its training tables"):
        _execution_geometry_from_config(config)

    with pytest.raises(RuntimeError, match="unsupported checkpoint expert batch isolation"):
        _expert_backend_functions("sample_isolated_grouped_mm_v3")


def _set_canonical_serving_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith(("CUBLAS_", "CUDA_", "CUDNN_", "NCCL_", "PYTORCH_", "TORCH_")):
            monkeypatch.delenv(name, raising=False)
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONINSPECT", "PYTHONSTARTUP"):
        monkeypatch.delenv(name, raising=False)
    for name, value in REQUIRED_SERVING_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(
        SERVER.sys,
        "flags",
        SimpleNamespace(dont_write_bytecode=1, no_user_site=1, safe_path=1),
    )
    monkeypatch.setattr(SERVER.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(SERVER.sys, "pycache_prefix", "/dev/null")
    version = f"python{SERVER.sys.version_info.major}.{SERVER.sys.version_info.minor}"
    compact_version = f"python{SERVER.sys.version_info.major}{SERVER.sys.version_info.minor}"
    monkeypatch.setattr(
        SERVER.sys,
        "path",
        [
            str(PROJECT_ROOT / "src"),
            str(Path(SERVER.sys.base_prefix) / "lib" / f"{compact_version}.zip"),
            str(Path(SERVER.sys.base_prefix) / "lib" / version),
            str(Path(SERVER.sys.base_exec_prefix) / "lib" / version / "lib-dynload"),
            f"/root/.cache/duo-vla/venvs/train/lib/{version}/site-packages",
        ],
    )


def test_server_process_environment_is_closed_and_rejects_nccl_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_canonical_serving_environment(monkeypatch)

    environment = _validate_serving_process_environment(PROJECT_ROOT)

    assert environment["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["environment"]["PYTHONHASHSEED"] == "0"
    assert environment["static_environment_sha256"] == static_environment_identity(environment["environment"])["sha256"]
    for name in (
        "CUBLAS_UNKNOWN_OVERRIDE",
        "CUDA_UNKNOWN_OVERRIDE",
        "CUDNN_UNKNOWN_OVERRIDE",
        "NCCL_ALGO",
        "PYTORCH_UNKNOWN_OVERRIDE",
        "TORCH_BLAS_PREFER_CUBLASLT",
        "TORCH_LINALG_PREFER_CUSOLVER",
    ):
        monkeypatch.setenv(name, "1")
        with pytest.raises(RuntimeError, match=name):
            _validate_serving_process_environment(PROJECT_ROOT)
        monkeypatch.delenv(name)


def test_serving_runtime_is_strict_deterministic_and_content_addressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_canonical_serving_environment(monkeypatch)
    cuda = SimpleNamespace(
        cudnn_sdp_enabled=lambda: True,
        device_count=lambda: 2,
        flash_sdp_enabled=lambda: True,
        get_device_capability=lambda _index: (8, 6),
        get_device_name=lambda _index: "Test GPU",
        get_device_properties=lambda index: SimpleNamespace(
            total_memory=96 * 2**30,
            uuid=("GPU-0", "GPU-1")[index],
        ),
        math_sdp_enabled=lambda: True,
        mem_efficient_sdp_enabled=lambda: True,
        nccl=SimpleNamespace(version=lambda: (2, 29, 3)),
    )
    fake_torch = SimpleNamespace(
        __version__="2.13.0+cu126",
        backends=SimpleNamespace(cuda=cuda, cudnn=SimpleNamespace(version=lambda: 91002)),
        cuda=cuda,
        version=SimpleNamespace(cuda="12.6"),
    )
    monkeypatch.setattr("serve_libero_policy.configure_strict_cuda_determinism", lambda value: None)
    monkeypatch.setattr(
        "serve_libero_policy.deterministic_torch_runtime",
        lambda value: {
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "preferred_blas_library": "_BlasBackend.Cublas",
            "preferred_linalg_library": "_LinalgBackend.Default",
        },
    )
    monkeypatch.setattr("serve_libero_policy._training_source_tree_sha256", lambda _root: "d" * 64)
    hardware = {
        "binaries": {
            "nvidia_smi_sha256": "1" * 64,
            "python_executable": "/runtime/python",
            "python_executable_sha256": "2" * 64,
            "torch_extension": "/runtime/torch.so",
            "torch_extension_sha256": "3" * 64,
        },
        "logical_cuda_devices": [
            {
                "compute_capability": [8, 6],
                "logical_index": index,
                "name": "Test GPU",
                "physical_index": index,
                "total_memory_bytes": 96 * 2**30,
                "uuid": f"GPU-{index}",
            }
            for index in range(2)
        ],
        "nvidia_smi_devices": [
            {
                "compute_capability": "8.6",
                "driver_version": "test-driver",
                "index": index,
                "name": "Test GPU",
                "uuid": f"GPU-GPU-{index}",
            }
            for index in range(2)
        ],
    }
    monkeypatch.setattr("serve_libero_policy._driver_and_binary_identity", lambda _torch: hardware)

    checkpoint_report = {
        "execution_geometry": LIBERO_EXECUTION_GEOMETRY,
        "source_tree_sha256": "d" * 64,
        "train_venv": _train_venv_identity(),
        "training_execution_environment": {
            "gpu_total_memory_bytes": [96 * 2**30, 96 * 2**30],
            "gpu_uuids": ["GPU-0", "GPU-1"],
        },
        "training_execution_environment_sha256": "e" * 64,
    }
    first, first_sha256 = configure_and_identify_serving_runtime(
        fake_torch,
        project_root=PROJECT_ROOT,
        checkpoint_report=checkpoint_report,
    )
    second, second_sha256 = configure_and_identify_serving_runtime(
        fake_torch,
        project_root=PROJECT_ROOT,
        checkpoint_report=checkpoint_report,
    )

    assert first == second
    assert first_sha256 == second_sha256 == canonical_config_sha256(first)
    assert first["gpu_names"] == ["Test GPU", "Test GPU"]
    assert first["platform"]["cuda_runtime"] == "12.6"
    assert first["gpu_uuids"] == ["GPU-0", "GPU-1"]
    assert first["authenticated_software"]["live_source_tree_sha256"] == "d" * 64
    assert first["authenticated_software"]["checkpoint_source_tree_sha256"] == "d" * 64
    assert first["execution_geometry"] == LIBERO_EXECUTION_GEOMETRY
    assert first["determinism"]["preferred_blas_library"] == "_BlasBackend.Cublas"
    assert first["training_execution_environment_sha256"] == "e" * 64
    assert first["sdpa_backends"] == {"cudnn": True, "flash": True, "math": True, "memory_efficient": True}
    latency_identity, latency_sha256 = latency_runtime_identity(first)
    changed_training = copy.deepcopy(first)
    changed_training["training_execution_environment_sha256"] = "f" * 64
    changed_identity, changed_sha256 = latency_runtime_identity(changed_training)
    assert changed_identity == latency_identity
    assert changed_sha256 == latency_sha256 == canonical_config_sha256(latency_identity)
    changed_hardware = copy.deepcopy(first)
    changed_hardware["gpu_uuids"][0] = "GPU-other"
    _, changed_hardware_sha256 = latency_runtime_identity(changed_hardware)
    assert changed_hardware_sha256 != latency_sha256
    changed_driver = copy.deepcopy(first)
    changed_driver["hardware"]["nvidia_smi_devices"][0]["driver_version"] = "different-driver"
    _, changed_driver_sha256 = latency_runtime_identity(changed_driver)
    assert changed_driver_sha256 != latency_sha256
    changed_binary = copy.deepcopy(first)
    changed_binary["hardware"]["binaries"]["torch_extension_sha256"] = "f" * 64
    _, changed_binary_sha256 = latency_runtime_identity(changed_binary)
    assert changed_binary_sha256 != latency_sha256

    changed_checkpoint = copy.deepcopy(checkpoint_report)
    changed_checkpoint["training_execution_environment"]["gpu_uuids"][0] = "GPU-other"
    with pytest.raises(RuntimeError, match="UUIDs differ"):
        configure_and_identify_serving_runtime(
            fake_torch,
            project_root=PROJECT_ROOT,
            checkpoint_report=changed_checkpoint,
        )


def test_single_gpu_policy_launcher_selects_only_qualified_training_device() -> None:
    source = (PROJECT_ROOT / "scripts/run_libero_policy_server_single_gpu.sh").read_text(encoding="utf-8")

    assert 'requested_physical_gpu="${DUO_VLA_PHYSICAL_GPU:-0}"' in source
    assert '"CUDA_VISIBLE_DEVICES=${requested_physical_gpu}"' in source
    assert "^(0|1)$" in source


def test_canonical_libero_server_launcher_pins_deterministic_runtime() -> None:
    source = (PROJECT_ROOT / "scripts/run_libero_policy_server.sh").read_text(encoding="utf-8")

    assert "exec /usr/bin/env -i" in source
    assert '"CUBLAS_WORKSPACE_CONFIG=:4096:8"' in source
    assert '"CUDA_VISIBLE_DEVICES=0,1"' in source
    assert '"PYTHONHASHSEED=0"' in source
    assert '"PYTHONSAFEPATH=1"' in source
    assert '"PYTHONDONTWRITEBYTECODE=1"' in source
    assert '"TORCH_NCCL_ASYNC_ERROR_HANDLING=1"' in source
    assert "PYTHONPATH" not in source
    assert '"PYTHONPYCACHEPREFIX=/dev/null"' in source
    assert "-P -B -X pycache_prefix=/dev/null" in source
    assert "${PYTHONPATH:+" not in source


def test_policy_server_recomputes_exact_libero_trainer_source_tree_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = str(PROJECT_ROOT / "src")
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path != source_root] + [source_root])
    spec = importlib.util.spec_from_file_location(
        "libero_train_source_identity",
        PROJECT_ROOT / "scripts/train_libero.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert _training_source_tree_sha256(PROJECT_ROOT) == module._source_tree_sha256(PROJECT_ROOT)


@pytest.mark.parametrize(
    ("field", "legacy_value"),
    (("physical_batch_size", 1), ("physical_batch_size", 32), ("expert_batch_isolation", None)),
)
def test_server_rejects_legacy_nonisolated_or_wrong_physical_batch_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    legacy_value: object,
) -> None:
    import duo_vla.checkpointing

    checkpoint, manifest, _, model_report = _checkpoint_fixture(tmp_path)
    legacy = copy.deepcopy(manifest)
    legacy[field] = legacy_value
    monkeypatch.setattr(duo_vla.checkpointing, "load_checkpoint_manifest", lambda *args, **kwargs: legacy)

    with pytest.raises(RuntimeError, match="fixed-batch execution geometry differs"):
        resolve_checkpoint(
            checkpoint,
            train_seed_override=None,
            model_snapshot_report=model_report,
        )


def test_server_rejects_prefix_artifact_substitution_even_with_valid_internal_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duo_vla.checkpointing

    checkpoint, manifest, config, model_report = _checkpoint_fixture(tmp_path)
    identity = SnapshotTreeIdentity.from_huggingface_report(MODEL_ID, model_report)
    substitute = build_prefix_geometry_contract(
        model_identity=identity,
        processor_identity=identity,
        ordered_cameras=(
            CameraGeometry("agentview", *IMAGE_SHAPE[:2]),
            CameraGeometry("eye_in_hand", *IMAGE_SHAPE[:2]),
        ),
        instruction_lengths={"substituted instruction": 13},
        fixed_physical_prefix_width=int(config["benchmark"]["fixed_physical_prefix_width"]),
        padding_side="left",
    )
    prefix_path = checkpoint / manifest["artifacts"]["prefix_geometry"]["path"]
    prefix_path.unlink()
    save_prefix_geometry_contract(prefix_path, substitute)
    monkeypatch.setattr(duo_vla.checkpointing, "load_checkpoint_manifest", lambda *args, **kwargs: manifest)

    with pytest.raises(ValueError, match="externally pinned SHA-256"):
        resolve_checkpoint(
            checkpoint,
            train_seed_override=None,
            model_snapshot_report=model_report,
        )


def test_real_policy_processor_path_replicates_one_request_to_exact_b8() -> None:
    from serve_libero_policy import RealPolicy

    processor = _Processor()
    prefix_geometry = build_prefix_geometry_contract(
        model_identity=SnapshotTreeIdentity.from_huggingface_report(MODEL_ID, _checkpoint_model_report()),
        processor_identity=SnapshotTreeIdentity.from_huggingface_report(MODEL_ID, _checkpoint_model_report()),
        ordered_cameras=(
            CameraGeometry("agentview", *IMAGE_SHAPE[:2]),
            CameraGeometry("eye_in_hand", *IMAGE_SHAPE[:2]),
        ),
        instruction_lengths={"pick the block": 12},
        fixed_physical_prefix_width=16,
        padding_side="left",
    )
    policy = object.__new__(RealPolicy)
    policy.processor = processor
    policy.Image = Image
    policy.device = torch.device("cpu")
    policy.apply_fixed_prefix_chat_template = apply_fixed_prefix_chat_template
    policy.execution_geometry = {"fixed_physical_prefix_width": 16}
    policy.prefix_geometry = prefix_geometry
    policy.prefix_valid_lengths = {"pick the block": 12}
    request = {
        "instruction": "pick the block",
        "observation": {
            "agentview_rgb": np.zeros(IMAGE_SHAPE, dtype=np.uint8),
            "wrist_rgb": np.zeros(IMAGE_SHAPE, dtype=np.uint8),
        },
    }

    result = policy._processor_inputs(request)

    assert result["input_ids"].shape == (PHYSICAL_BATCH_SIZE, 16)
    assert len(processor.conversations) == PHYSICAL_BATCH_SIZE


def _checkpoint_model_report() -> dict[str, object]:
    return {
        "content_inventory_sha256": "c" * 64,
        "files_verified": 7,
        "revision": MODEL_REVISION,
        "total_bytes": 1234,
        "tree_metadata_sha256": "b" * 64,
    }


def test_real_policy_rejects_any_non_bitwise_replica_output() -> None:
    from serve_libero_policy import RealPolicy

    policy = object.__new__(RealPolicy)
    policy.torch = torch
    values = torch.zeros((PHYSICAL_BATCH_SIZE, 8, 7), dtype=torch.float32)
    policy._require_bitwise_replicas(values, name="normalized action")
    values[7, 0, 0] = torch.finfo(torch.float32).eps
    with pytest.raises(RuntimeError, match="replica 7 differs bitwise"):
        policy._require_bitwise_replicas(values, name="normalized action")
    values.zero_()
    values[7, 0, 0] = -0.0
    with pytest.raises(RuntimeError, match="replica 7 differs bitwise"):
        policy._require_bitwise_replicas(values, name="normalized action")
