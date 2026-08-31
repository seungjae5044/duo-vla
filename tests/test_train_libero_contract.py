from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_libero as TRAIN

from duo_vla.prefix_geometry import (
    SnapshotTreeIdentity,
    build_prefix_geometry_contract,
    save_prefix_geometry_contract,
)
from duo_vla.run_config import load_resolved_toml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _snapshot_report() -> dict[str, object]:
    return {
        "content_inventory_sha256": "c" * 64,
        "files_verified": 7,
        "revision": TRAIN.DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
        "total_bytes": 1234,
        "tree_metadata_sha256": "b" * 64,
    }


def _prefix_contract(*, width: int = 16) -> dict[str, Any]:
    identity = SnapshotTreeIdentity.from_huggingface_report(
        TRAIN.DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
        _snapshot_report(),
    )
    return build_prefix_geometry_contract(
        model_identity=identity,
        processor_identity=identity,
        ordered_cameras=TRAIN.LIBERO_PREFIX_CAMERAS,
        instruction_lengths={"pick the block": 12},
        fixed_physical_prefix_width=width,
        padding_side="left",
    )


def test_libero_configs_pin_exact_fixed_batch_expert_execution() -> None:
    for name in ("libero.toml", "libero_direct_regression.toml"):
        config = load_resolved_toml(PROJECT_ROOT / "configs" / name)
        assert config["model"]["experts_implementation"] == "grouped_mm"
        assert config["model"]["expert_batch_isolation"] == TRAIN.EXPERT_BATCH_ISOLATION
        assert config["optimization"]["physical_batch_size"] == TRAIN.PHYSICAL_BATCH_SIZE
        assert config["optimization"]["microbatch_size"] == TRAIN.PHYSICAL_BATCH_SIZE
        assert config["optimization"]["gradient_accumulation_steps"] == 8
        assert config["optimization"]["global_batch_size"] == 64
        assert config["benchmark"]["fixed_physical_prefix_width"] == 545
        assert config["benchmark"]["prefix_geometry_content_sha256"] == (
            "cc907e22ccd5ae704767edba606233dede39989a119ac544764b47aaa4fbe634"
        )


def _dataset_snapshot_report() -> dict[str, object]:
    return {
        "content_inventory_sha256": TRAIN.LIBERO_DATASET_CONTENT_INVENTORY_SHA256,
        "files_verified": TRAIN.LIBERO_DATASET_FILES_VERIFIED,
        "revision": TRAIN.LIBERO_DATASET_REVISION,
        "snapshot": f"/snapshot/{TRAIN.LIBERO_DATASET_REVISION}",
        "total_bytes": TRAIN.LIBERO_DATASET_TOTAL_BYTES,
        "tree_metadata_sha256": TRAIN.LIBERO_DATASET_TREE_SHA256,
    }


def test_libero_dataset_is_fully_authenticated_on_rank_zero_and_bound_on_every_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, str]] = []
    broadcasts: list[object] = []
    group = object()
    report = _dataset_snapshot_report()
    monkeypatch.setattr(TRAIN.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(TRAIN.dist, "new_group", lambda **kwargs: group)
    monkeypatch.setattr(TRAIN.dist, "destroy_process_group", lambda observed: None)
    monkeypatch.setattr(
        TRAIN,
        "verify_huggingface_snapshot",
        lambda root, *, expected_revision: calls.append((root, expected_revision)) or report,
    )
    monkeypatch.setattr(
        TRAIN.dist,
        "broadcast_object_list",
        lambda holder, *, src, group: broadcasts.append(copy.deepcopy(holder[0])),
    )

    observed = TRAIN._authenticate_dataset_snapshot_distributed(tmp_path)

    assert observed == report
    assert calls == [(tmp_path, TRAIN.LIBERO_DATASET_REVISION)]
    assert broadcasts == [report]


def test_libero_dataset_authentication_fails_closed_on_content_or_broadcast_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corrupted = _dataset_snapshot_report()
    corrupted["content_inventory_sha256"] = "0" * 64
    group = object()
    monkeypatch.setattr(TRAIN.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(TRAIN.dist, "new_group", lambda **kwargs: group)
    monkeypatch.setattr(TRAIN.dist, "destroy_process_group", lambda observed: None)
    monkeypatch.setattr(TRAIN, "verify_huggingface_snapshot", lambda *_args, **_kwargs: corrupted)
    monkeypatch.setattr(TRAIN.dist, "broadcast_object_list", lambda _holder, *, src, group: None)
    with pytest.raises(RuntimeError, match="dataset authentication failed"):
        TRAIN._authenticate_dataset_snapshot_distributed(tmp_path)

    monkeypatch.setattr(TRAIN.dist, "get_rank", lambda: 1)

    def inject_corrupt_broadcast(holder: list[object], *, src: int, group: object) -> None:
        holder[0] = corrupted

    monkeypatch.setattr(TRAIN.dist, "broadcast_object_list", inject_corrupt_broadcast)
    with pytest.raises(RuntimeError, match="broadcast LIBERO dataset identity"):
        TRAIN._authenticate_dataset_snapshot_distributed(tmp_path)


def test_libero_trainer_authenticates_dataset_before_constructing_parquet_reader() -> None:
    source = (PROJECT_ROOT / "scripts/train_libero.py").read_text(encoding="utf-8")
    main_source = source[source.index("def main()") :]
    assert main_source.index("_authenticate_dataset_snapshot_distributed(args.snapshot_root)") < main_source.index(
        "LiberoParquetDataset(args.snapshot_root"
    )


def _set_canonical_train_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(TRAIN.os.environ):
        if name.startswith(TRAIN._ALGORITHM_ENVIRONMENT_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONINSPECT", "PYTHONSTARTUP"):
        monkeypatch.delenv(name, raising=False)
    for name, value in TRAIN.REQUIRED_TRAIN_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")
    monkeypatch.setenv("DUO_VLA_PROJECT_ROOT", str(PROJECT_ROOT))
    monkeypatch.setenv("DUO_VLA_TRAIN_VENV", "/root/.cache/duo-vla/venvs/train")
    monkeypatch.setenv("HF_HOME", "/root/.cache/huggingface")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(TRAIN.sys, "prefix", "/root/.cache/duo-vla/venvs/train")
    monkeypatch.setattr(TRAIN.site, "ENABLE_USER_SITE", False)
    monkeypatch.setattr(
        TRAIN.sys,
        "flags",
        SimpleNamespace(dont_write_bytecode=1, no_user_site=1, safe_path=1),
    )
    monkeypatch.setattr(TRAIN.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(TRAIN.sys, "pycache_prefix", "/dev/null")
    version = f"python{TRAIN.sys.version_info.major}.{TRAIN.sys.version_info.minor}"
    compact_version = f"python{TRAIN.sys.version_info.major}{TRAIN.sys.version_info.minor}"
    monkeypatch.setattr(
        TRAIN.sys,
        "path",
        [
            str(PROJECT_ROOT / "src"),
            str(Path(TRAIN.sys.base_prefix) / "lib" / f"{compact_version}.zip"),
            str(Path(TRAIN.sys.base_prefix) / "lib" / version),
            str(Path(TRAIN.sys.base_exec_prefix) / "lib" / version / "lib-dynload"),
            f"/root/.cache/duo-vla/venvs/train/lib/{version}/site-packages",
        ],
    )
    monkeypatch.setattr(TRAIN, "content_address_train_venv", lambda _root: {"root_sha256": "a" * 64})


def test_libero_training_runtime_enables_strict_determinism_and_rejects_nccl_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_canonical_train_environment(monkeypatch)
    calls: list[object] = []
    monkeypatch.setattr(TRAIN, "configure_strict_cuda_determinism", calls.append)

    report = TRAIN._configure_and_validate_training_runtime(PROJECT_ROOT)

    assert calls == [TRAIN.torch]
    assert report["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "PYTHONPATH" not in report["environment"]
    assert report["static_environment_sha256"] == TRAIN.static_environment_identity(report["environment"])["sha256"]
    assert report["train_venv"] == {"root_sha256": "a" * 64}
    assert report["algorithm_override_environment"] == {}
    assert report["nccl_environment"] == {}

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
            TRAIN._configure_and_validate_training_runtime(PROJECT_ROOT)
        monkeypatch.delenv(name)


def test_canonical_libero_training_launcher_pins_numeric_runtime_and_scrubs_nccl() -> None:
    source = (PROJECT_ROOT / "scripts/run_libero_train.sh").read_text(encoding="utf-8")

    assert "--nproc-per-node=2" in source
    assert "exec /usr/bin/env -i" in source
    assert '"CUBLAS_WORKSPACE_CONFIG=:4096:8"' in source
    assert '"CUDA_DEVICE_ORDER=PCI_BUS_ID"' in source
    assert '"CUDA_VISIBLE_DEVICES=0,1"' in source
    assert '"PYTHONHASHSEED=${train_seed}"' in source
    assert '"PYTHONSAFEPATH=1"' in source
    assert '"PYTHONDONTWRITEBYTECODE=1"' in source
    assert '"TORCH_NCCL_ASYNC_ERROR_HANDLING=1"' in source
    assert "PYTHONPATH" not in source
    assert '"PYTHONPYCACHEPREFIX=/dev/null"' in source
    assert "-P -B -X pycache_prefix=/dev/null" in source
    assert "${PYTHONPATH:+" not in source
    assert '"LANG=C.UTF-8"' in source and '"LC_ALL=C.UTF-8"' in source and '"TZ=UTC"' in source


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (("fixed_physical_prefix_width", 546), ("physical_batch_size", 7)),
)
def test_resume_manifest_run_contract_accepts_typed_execution_integers_only(
    field: str,
    wrong_value: int,
) -> None:
    run_contract = {
        "fixed_physical_prefix_width": "545",
        "physical_batch_size": "8",
        "source_tree_sha256": "a" * 64,
    }
    manifest: dict[str, Any] = {
        **run_contract,
        "fixed_physical_prefix_width": 545,
        "physical_batch_size": 8,
    }
    expected_run_contract = copy.deepcopy(run_contract)

    TRAIN._validate_resume_manifest_run_contract(manifest, run_contract)

    assert run_contract == expected_run_contract
    changed = {**manifest, field: wrong_value}
    with pytest.raises(ValueError, match=field):
        TRAIN._validate_resume_manifest_run_contract(changed, run_contract)
    wrong_type = {**manifest, field: run_contract[field]}
    with pytest.raises(ValueError, match=field):
        TRAIN._validate_resume_manifest_run_contract(wrong_type, run_contract)


@pytest.mark.parametrize("legacy_batch", (1, 32))
def test_trainer_rejects_legacy_physical_training_batches_before_model_work(
    monkeypatch: pytest.MonkeyPatch,
    legacy_batch: int,
) -> None:
    config = load_resolved_toml(PROJECT_ROOT / "configs/libero.toml")
    legacy = copy.deepcopy(config)
    legacy["optimization"]["microbatch_size"] = legacy_batch
    legacy["optimization"]["gradient_accumulation_steps"] = 64 // legacy_batch
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)

    with pytest.raises(ValueError, match="unsupported by this trainer"):
        TRAIN._validate_and_build_interface_config(legacy)


@pytest.mark.parametrize(
    ("section", "field", "wrong_value"),
    (
        ("lora", "rank", 8),
        ("lora", "alpha", 7),
        ("lora", "dropout", 0.25),
        ("action", "timestep_embedding_dimension", 128),
        ("action", "timestep_scale", 1.0),
        ("action", "timestep_max_period", 1000.0),
        ("action", "output_head_initialization_std", 0.01),
    ),
)
def test_libero_trainer_rejects_off_contract_interface_and_lora_hyperparameters(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    wrong_value: object,
) -> None:
    config = load_resolved_toml(PROJECT_ROOT / "configs/libero.toml")
    config[section][field] = wrong_value
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)

    with pytest.raises(ValueError, match=f"{section}.{field}"):
        TRAIN._validate_and_build_interface_config(config)


def test_libero_checkpoint_retention_interval_must_be_a_positive_checkpoint_multiple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_resolved_toml(PROJECT_ROOT / "configs/libero.toml")
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)

    assert config["training"]["checkpoint_interval"] == 1000
    assert config["training"]["permanent_checkpoint_interval"] == 5000
    TRAIN._validate_and_build_interface_config(config)
    config["training"]["permanent_checkpoint_interval"] = 1500
    with pytest.raises(ValueError, match="positive multiple"):
        TRAIN._validate_and_build_interface_config(config)


def test_prefix_geometry_authentication_binds_external_pin_model_cameras_instructions_and_width(
    tmp_path: Path,
) -> None:
    contract = _prefix_contract()
    path = tmp_path / "prefix-geometry.json"
    save_prefix_geometry_contract(path, contract)

    authenticated = TRAIN._authenticate_prefix_geometry(
        path,
        expected_content_sha256=contract["content_sha256"],
        expected_fixed_physical_prefix_width=16,
        model_snapshot_report=_snapshot_report(),
        instructions=("pick the block",),
    )
    assert authenticated == contract

    with pytest.raises(ValueError, match="instruction inventory mismatch"):
        TRAIN._authenticate_prefix_geometry(
            path,
            expected_content_sha256=contract["content_sha256"],
            expected_fixed_physical_prefix_width=16,
            model_snapshot_report=_snapshot_report(),
            instructions=("different instruction",),
        )


@pytest.mark.parametrize("next_update", (1, 500, 999))
def test_stop_boundary_does_not_add_out_of_schedule_libero_validation(next_update: int) -> None:
    assert not TRAIN._validation_is_due(next_update=next_update, total_updates=30_000, interval=1_000)
    assert TRAIN._validation_is_due(next_update=1_000, total_updates=30_000, interval=1_000)
    assert TRAIN._validation_is_due(next_update=30_000, total_updates=30_000, interval=1_000)


class _Batch(dict[str, torch.Tensor]):
    def to(self, _device: torch.device) -> _Batch:
        return self


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(padding_side="left")
        self.last_conversations: Any = None
        self.last_kwargs: dict[str, Any] | None = None

    def apply_chat_template(self, conversations: Any, **kwargs: Any) -> _Batch:
        self.last_conversations = conversations
        self.last_kwargs = kwargs
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


def _sample() -> SimpleNamespace:
    observation = SimpleNamespace(
        third_person=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist=np.zeros((256, 256, 3), dtype=np.uint8),
    )
    return SimpleNamespace(observation=observation, instruction="pick the block")


def test_processor_path_requires_exact_b8_and_fixed_authenticated_width() -> None:
    processor = _Processor()
    contract = _prefix_contract()
    samples = tuple(_sample() for _ in range(TRAIN.PHYSICAL_BATCH_SIZE))

    result = TRAIN._processor_inputs(processor, samples, torch.device("cpu"), contract)

    assert result["input_ids"].shape == (8, 16)
    assert len(processor.last_conversations) == 8
    assert processor.last_kwargs is not None
    assert processor.last_kwargs["processor_kwargs"] == {
        "padding": "max_length",
        "max_length": 16,
        "truncation": False,
    }
    with pytest.raises(ValueError, match="requires physical batch 8"):
        TRAIN._processor_inputs(processor, samples[:-1], torch.device("cpu"), contract)


def test_libero_checkpoint_manifest_binds_parent_and_retention_uses_authenticated_config() -> None:
    source = (PROJECT_ROOT / "scripts/train_libero.py").read_text(encoding="utf-8")
    parent = {
        "relative_path": "checkpoints/update-001000",
        "update": 1000,
    }

    assert '"checkpoint_retention": make_checkpoint_retention_contract(' in source
    assert "retention = apply_checkpoint_retention(output_dir)" in source
    assert "apply_checkpoint_retention(\n" not in source
    assert '"manifest_sha256": parent_manifest_sha256' not in source
    assert (
        TRAIN.make_checkpoint_retention_contract(
            permanent_checkpoint_interval=5000,
            parent_checkpoint=parent,
        )["parent_checkpoint"]
        == parent
    )
