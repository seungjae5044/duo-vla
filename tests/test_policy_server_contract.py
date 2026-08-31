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

from libero_bridge import LIBERO_EXECUTION_GEOMETRY
from serve_libero_policy import (
    DATASET_ID,
    DATASET_REVISION,
    EXPERT_BATCH_ISOLATION,
    EXPERTS_IMPLEMENTATION,
    IMAGE_SHAPE,
    MODEL_ID,
    MODEL_REVISION,
    NORMALIZATION_SHA256,
    PHYSICAL_BATCH_SIZE,
    REQUIRED_SERVING_ENVIRONMENT,
    _authenticated_training_environment,
    _health_payload,
    _training_source_tree_sha256,
    _validate_serving_process_environment,
    configure_and_identify_serving_runtime,
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
) -> tuple[Path, dict[str, object], dict[str, object], dict[str, object]]:
    checkpoint = tmp_path / "checkpoint"
    artifacts = checkpoint / "artifacts"
    artifacts.mkdir(parents=True)
    config = load_resolved_toml(PROJECT_ROOT / "configs" / config_name)
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
    config["artifact_trees"] = {
        "model_content_inventory_sha256": model_report["content_inventory_sha256"],
        "model_tree_sha256": model_report["tree_metadata_sha256"],
    }
    training_environment: dict[str, object] = {
        "authenticated_runtime": {"algorithm_override_environment": {}, "nccl_environment": {}},
        "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
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
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "execution_environment": training_environment,
        "execution_environment_sha256": canonical_config_sha256(training_environment),
        "kind": "resumable-libero-training",
        "experts_implementation": EXPERTS_IMPLEMENTATION,
        "expert_batch_isolation": EXPERT_BATCH_ISOLATION,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "fixed_physical_prefix_width": config["benchmark"]["fixed_physical_prefix_width"],
        "prefix_geometry_content_sha256": prefix_geometry["content_sha256"],
        "model_id": MODEL_ID,
        "model_content_inventory_sha256": model_report["content_inventory_sha256"],
        "model_revision": MODEL_REVISION,
        "model_tree_sha256": model_report["tree_metadata_sha256"],
        "normalization_sha256": NORMALIZATION_SHA256,
        "policy_contract": contract,
        "policy_contract_sha256": canonical_config_sha256(contract),
        "run_seed": 1,
        "source_tree_sha256": "d" * 64,
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
        _authenticated_training_environment(config, manifest, train_seed=1)

    environment["cudnn_deterministic"] = True
    environment["deterministic_algorithms"] = True
    environment["preferred_blas_library"] = "_BlasBackend.Cublaslt"
    manifest["execution_environment"] = copy.deepcopy(environment)
    manifest["execution_environment_sha256"] = canonical_config_sha256(environment)
    with pytest.raises(RuntimeError, match="preferred_blas_library"):
        _authenticated_training_environment(config, manifest, train_seed=1)


def test_v4_health_exposes_exact_real_geometry_and_null_fake_identities() -> None:
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
        serving_runtime_sha256="a" * 64,
    )
    assert real["execution_geometry"] == LIBERO_EXECUTION_GEOMETRY
    assert real["serving_runtime_sha256"] == "a" * 64

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
        "model_revision",
        "normalization_content_sha256",
        "serving_runtime_sha256",
    ):
        assert fake[name] is None


def _set_canonical_serving_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith(("CUBLAS_", "CUDA_", "CUDNN_", "NCCL_", "PYTORCH_", "TORCH_")):
            monkeypatch.delenv(name, raising=False)
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONINSPECT", "PYTHONSTARTUP"):
        monkeypatch.delenv(name, raising=False)
    for name, value in REQUIRED_SERVING_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PYTHONPATH", f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts'}")


def test_server_process_environment_is_closed_and_rejects_nccl_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_canonical_serving_environment(monkeypatch)

    environment = _validate_serving_process_environment(PROJECT_ROOT)

    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["PYTHONHASHSEED"] == "0"
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
        get_device_properties=lambda index: SimpleNamespace(uuid=("GPU-0", "GPU-1")[index]),
        math_sdp_enabled=lambda: True,
        mem_efficient_sdp_enabled=lambda: True,
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

    checkpoint_report = {
        "execution_geometry": LIBERO_EXECUTION_GEOMETRY,
        "source_tree_sha256": "d" * 64,
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


def test_canonical_libero_server_launcher_pins_deterministic_runtime() -> None:
    source = (PROJECT_ROOT / "scripts/run_libero_policy_server.sh").read_text(encoding="utf-8")

    assert 'export CUBLAS_WORKSPACE_CONFIG=":4096:8"' in source
    assert 'export CUDA_VISIBLE_DEVICES="0,1"' in source
    assert 'export PYTHONHASHSEED="0"' in source
    assert 'export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"' in source
    assert 'export PYTHONPATH="${project_dir}/src:${project_dir}/scripts"' in source
    assert "${PYTHONPATH:+" not in source
    assert "CUBLAS_*|CUDA_*|CUDNN_*|NCCL_*|PYTORCH_*|TORCH_*" in source
    assert "unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONINSPECT PYTHONSTARTUP" in source


def test_policy_server_recomputes_exact_libero_trainer_source_tree_identity() -> None:
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
