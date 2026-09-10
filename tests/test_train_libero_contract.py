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


def test_libero_single_gpu_configs_change_only_the_execution_topology() -> None:
    for base_name, single_name in (
        ("libero.toml", "libero_single_gpu.toml"),
        ("libero_direct_regression.toml", "libero_direct_regression_single_gpu.toml"),
    ):
        base = load_resolved_toml(PROJECT_ROOT / "configs" / base_name)
        single = load_resolved_toml(PROJECT_ROOT / "configs" / single_name)
        assert single["execution_profile"] == "duovla-single-gpu-tp1-v1"
        assert single["model"]["tensor_parallel_size"] == 1
        single.pop("execution_profile")
        single["model"]["tensor_parallel_size"] = 2
        assert single == base


def test_libero_single_gpu_g3_keeps_physical_b8_and_seals_gpu_zero() -> None:
    launcher = (PROJECT_ROOT / "scripts/run_overfit_real_libero_batch_single_gpu.sh").read_text(encoding="utf-8")
    for fragment in (
        '"CUDA_VISIBLE_DEVICES=0"',
        '"DUO_VLA_TRAIN_VENV=${environment_path}"',
        '"PYTHONHASHSEED=0"',
        '"${cache_root}/venvs/train-single-gpu"',
        "--nproc-per-node=1",
    ):
        assert fragment in launcher

    gate = (PROJECT_ROOT / "scripts/overfit_real_libero_batch.py").read_text(encoding="utf-8")
    for fragment in (
        "args.batch_size % PHYSICAL_BATCH_SIZE",
        "range(0, args.batch_size, PHYSICAL_BATCH_SIZE)",
        "TRAIN._processor_inputs(",
        "install_sample_isolated_grouped_mm_experts(",
        "component.loss_for_total(total_elements).backward()",
    ):
        assert fragment in gate


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
    (("fixed_physical_prefix_width", 546), ("physical_batch_size", 7), ("serving_batch_size", 16)),
)
def test_resume_manifest_run_contract_accepts_typed_execution_integers_only(
    field: str,
    wrong_value: int,
) -> None:
    run_contract = {
        "fixed_physical_prefix_width": "545",
        "physical_batch_size": "8",
        "serving_batch_size": "8",
        "source_tree_sha256": "a" * 64,
    }
    manifest: dict[str, Any] = {
        **run_contract,
        "fixed_physical_prefix_width": 545,
        "physical_batch_size": 8,
        "serving_batch_size": 8,
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


@pytest.mark.parametrize("physical_batch_size", sorted(TRAIN.SUPPORTED_PHYSICAL_BATCH_SIZES))
def test_trainer_accepts_qualified_physical_batches_at_fixed_global_batch(
    monkeypatch: pytest.MonkeyPatch,
    physical_batch_size: int,
) -> None:
    config = load_resolved_toml(PROJECT_ROOT / "configs/libero_single_gpu.toml")
    optimization = config["optimization"]
    optimization["physical_batch_size"] = physical_batch_size
    optimization["microbatch_size"] = physical_batch_size
    optimization["gradient_accumulation_steps"] = TRAIN.GLOBAL_BATCH_SIZE // physical_batch_size
    config["execution_profile"] = TRAIN._expected_execution_profile(
        world_size=1,
        expert_batch_isolation=TRAIN.EXPERT_BATCH_ISOLATION,
        physical_batch_size=physical_batch_size,
        serving_batch_size=TRAIN.DEFAULT_SERVING_BATCH_SIZE,
    )
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 1)

    TRAIN._validate_and_build_interface_config(config)
    geometry = TRAIN._execution_geometry(config)
    assert geometry["physical_batch_size"] == physical_batch_size
    assert geometry["serving_batch_size"] == TRAIN.DEFAULT_SERVING_BATCH_SIZE


@pytest.mark.parametrize("physical_batch_size", (8, 16, 32, 64))
def test_fused_v2_configs_pin_exact_backend_batch_and_profile(
    monkeypatch: pytest.MonkeyPatch,
    physical_batch_size: int,
) -> None:
    path = PROJECT_ROOT / "configs" / f"libero_single_gpu_fused_v2_b{physical_batch_size}.toml"
    config = load_resolved_toml(path)
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 1)

    TRAIN._validate_and_build_interface_config(config)
    assert config["model"]["expert_batch_isolation"] == TRAIN.SAMPLE_ISOLATED_GROUPED_MM_V2
    assert config["optimization"]["physical_batch_size"] == physical_batch_size
    assert config["optimization"]["gradient_accumulation_steps"] == 64 // physical_batch_size
    assert config["optimization"]["serving_batch_size"] == 8
    assert config["execution_profile"] == (f"duovla-single-gpu-tp1-fused-v2-train-b{physical_batch_size}-serve-b8-v1")
    geometry = TRAIN._execution_geometry(config)
    assert geometry["shared_weight_kernel_sha256"] == TRAIN.file_sha256(
        PROJECT_ROOT / "src/duo_vla/backbones/shared_weight_grouped_mm_triton.py"
    )


def test_tp2_rejects_unqualified_coalesced_or_fused_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)
    for path in (
        PROJECT_ROOT / "configs/libero_single_gpu_ab_v1_b64.toml",
        PROJECT_ROOT / "configs/libero_single_gpu_fused_v2_b8.toml",
    ):
        config = load_resolved_toml(path)
        config["model"]["tensor_parallel_size"] = 2
        with pytest.raises(ValueError, match="TP=2 is qualified only"):
            TRAIN._validate_and_build_interface_config(config)


def test_resume_rejects_nested_execution_geometry_drift(tmp_path: Path) -> None:
    config = load_resolved_toml(PROJECT_ROOT / "configs/libero.toml")
    expected = TRAIN._execution_geometry(config)
    manifest = {
        **expected,
        "execution_geometry": {**expected, "serving_batch_size": 16},
    }

    with pytest.raises(ValueError, match="execution_geometry"):
        TRAIN._validate_checkpoint_execution_geometry(tmp_path, manifest, config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("physical_batch_size", 12, "must be one of"),
        ("physical_batch_size", True, "must be an integer"),
        ("microbatch_size", 16, "must equal"),
        ("gradient_accumulation_steps", 4, "must equal"),
        ("global_batch_size", 32, "must equal 64"),
        ("serving_batch_size", 12, "must be one of"),
        ("serving_batch_size", 16, "must remain 8"),
    ),
)
def test_trainer_rejects_invalid_or_inconsistent_batch_contract(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_resolved_toml(PROJECT_ROOT / "configs/libero.toml")
    config["optimization"][field] = value
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)

    with pytest.raises(ValueError, match=message):
        TRAIN._validate_and_build_interface_config(config)


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


@pytest.mark.parametrize("physical_batch_size", sorted(TRAIN.SUPPORTED_PHYSICAL_BATCH_SIZES))
def test_processor_path_requires_selected_physical_batch_and_fixed_authenticated_width(
    physical_batch_size: int,
) -> None:
    processor = _Processor()
    contract = _prefix_contract()
    samples = tuple(_sample() for _ in range(physical_batch_size))

    result = TRAIN._processor_inputs(
        processor,
        samples,
        torch.device("cpu"),
        contract,
        expected_batch_size=physical_batch_size,
    )

    assert result["input_ids"].shape == (physical_batch_size, 16)
    assert len(processor.last_conversations) == physical_batch_size
    assert processor.last_kwargs is not None
    assert processor.last_kwargs["processor_kwargs"] == {
        "padding": "max_length",
        "max_length": 16,
        "truncation": False,
    }
    with pytest.raises(ValueError, match=f"requires physical batch {physical_batch_size}"):
        TRAIN._processor_inputs(
            processor,
            samples[:-1],
            torch.device("cpu"),
            contract,
            expected_batch_size=physical_batch_size,
        )


@pytest.mark.parametrize("physical_batch_size", sorted(TRAIN.SUPPORTED_PHYSICAL_BATCH_SIZES))
def test_physical_plan_groups_preserve_the_exact_eight_step_b8_stream(
    physical_batch_size: int,
) -> None:
    expected = TRAIN.make_update_plan(
        17,
        23,
        gradient_accumulation_steps=TRAIN.CANONICAL_MICROSTEPS_PER_UPDATE,
    )

    groups = TRAIN._canonical_update_plan_groups(
        17,
        23,
        physical_batch_size=physical_batch_size,
    )

    assert tuple(plan for group in groups for plan in group) == expected
    assert len(groups) == TRAIN.GLOBAL_BATCH_SIZE // physical_batch_size
    assert {len(group) for group in groups} == {physical_batch_size // TRAIN.CANONICAL_STREAM_BATCH_SIZE}
    assert [plan.microstep for group in groups for plan in group] == list(range(TRAIN.CANONICAL_MICROSTEPS_PER_UPDATE))


def test_materialization_uses_all_eight_canonical_b8_data_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plans = TRAIN.make_update_plan(
        37,
        41,
        gradient_accumulation_steps=TRAIN.CANONICAL_MICROSTEPS_PER_UPDATE,
    )
    calls: list[tuple[object, object, int, int, object, object]] = []
    results = tuple(object() for _ in plans)

    def materialize(
        dataset: object,
        sampler: object,
        *,
        count: int,
        seed: int,
        state_normalizer: object,
        action_normalizer: object,
    ) -> object:
        calls.append((dataset, sampler, count, seed, state_normalizer, action_normalizer))
        return results[len(calls) - 1]

    monkeypatch.setattr(TRAIN, "_materialize_batch", materialize)
    dataset = object()
    sampler = object()
    state_normalizer = object()
    action_normalizer = object()

    observed = TRAIN._materialize_canonical_batches(
        dataset,
        sampler,
        plans,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
    )

    assert observed == results
    assert calls == [
        (
            dataset,
            sampler,
            TRAIN.CANONICAL_STREAM_BATCH_SIZE,
            plan.data_seed,
            state_normalizer,
            action_normalizer,
        )
        for plan in plans
    ]


def _canonical_batch(chunk_index: int) -> TRAIN.LiberoBatch:
    start = chunk_index * TRAIN.CANONICAL_STREAM_BATCH_SIZE
    samples = tuple(SimpleNamespace(canonical_index=start + index) for index in range(8))
    states = torch.arange(start * 8, (start + 8) * 8, dtype=torch.float32).reshape(8, 8)
    clean_actions = torch.arange(start * 56, (start + 8) * 56, dtype=torch.float32).reshape(8, 8, 7)
    valid = torch.ones((8, 8), dtype=torch.bool)
    return TRAIN.LiberoBatch(
        samples=samples,
        states=states,
        clean_actions=clean_actions,
        action_valid_mask=valid,
    )


@pytest.mark.parametrize("physical_batch_size", sorted(TRAIN.SUPPORTED_PHYSICAL_BATCH_SIZES))
def test_coalescing_preserves_canonical_sample_and_tensor_order(physical_batch_size: int) -> None:
    chunk_count = physical_batch_size // TRAIN.CANONICAL_STREAM_BATCH_SIZE
    canonical = tuple(_canonical_batch(index) for index in range(chunk_count))

    observed = TRAIN._coalesce_canonical_batches(
        canonical,
        physical_batch_size=physical_batch_size,
    )

    assert [sample.canonical_index for sample in observed.samples] == list(range(physical_batch_size))
    assert torch.equal(observed.states, torch.cat([batch.states for batch in canonical]))
    assert torch.equal(observed.clean_actions, torch.cat([batch.clean_actions for batch in canonical]))
    assert torch.equal(observed.action_valid_mask, torch.cat([batch.action_valid_mask for batch in canonical]))


@pytest.mark.parametrize("config_name", ("libero.toml", "libero_direct_regression.toml"))
@pytest.mark.parametrize("physical_batch_size", sorted(TRAIN.SUPPORTED_PHYSICAL_BATCH_SIZES))
def test_coalesced_objective_pair_preserves_each_canonical_b8_seed_stream(
    config_name: str,
    physical_batch_size: int,
) -> None:
    contract = TRAIN.policy_contract_from_config(load_resolved_toml(PROJECT_ROOT / "configs" / config_name))
    plans = TRAIN._canonical_update_plan_groups(
        29,
        31,
        physical_batch_size=physical_batch_size,
    )[0]
    clean = torch.linspace(-1.0, 1.0, physical_batch_size * 8 * 7, dtype=torch.float32).reshape(
        physical_batch_size,
        8,
        7,
    )
    expected_parts = tuple(
        TRAIN.make_seeded_policy_training_pair(
            clean[index * 8 : (index + 1) * 8],
            contract,
            seed=plan.flow_seed,
        )
        for index, plan in enumerate(plans)
    )

    observed = TRAIN._canonical_training_pair(clean, contract, plans)

    assert torch.equal(observed.input_actions, torch.cat([part.input_actions for part in expected_parts]))
    assert torch.equal(observed.timesteps, torch.cat([part.timesteps for part in expected_parts]))
    assert torch.equal(observed.target, torch.cat([part.target for part in expected_parts]))


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
