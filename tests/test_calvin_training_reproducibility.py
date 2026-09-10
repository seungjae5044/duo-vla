from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from duo_vla.checkpointing import SCHEMA_VERSION
from duo_vla.policy_contract import policy_contract_from_config
from duo_vla.run_config import canonical_config_sha256, load_resolved_toml, save_resolved_config
from duo_vla.run_journal import create_run_journal, record_latest_checkpoint
from duo_vla.training import TrainerState
from duo_vla.training_checkpoint import TRAINING_RANK_STATE_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_calvin_training_reproducibility.py"
SPEC = importlib.util.spec_from_file_location("calvin_training_reproducibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
TEST_TRAIN_VENV = Path(sys.prefix).resolve()
TEST_SITE_PACKAGES = TEST_TRAIN_VENV / "lib/python3.11/site-packages"
TEST_PROJECT_SRC = (ROOT / "src").resolve()
GATE.__dict__["__duo_vla_qualification_bootstrap_capability__"] = {
    "comparator_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    "expected_train_venv": str(TEST_TRAIN_VENV),
    "forbidden_modules_preloaded": [],
    "interpreter_flags": {
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
        "safe_path": True,
    },
    "launcher_sha256": hashlib.sha256(
        (ROOT / "scripts/calvin/run_compare_training_reproducibility.sh").read_bytes()
    ).hexdigest(),
    "mode": "launcher-exact-comparator-bytes-isolated-v2",
    "project_src": str(TEST_PROJECT_SRC),
    "pyvenv_cfg_sha256": hashlib.sha256((TEST_TRAIN_VENV / "pyvenv.cfg").read_bytes()).hexdigest(),
    "site_packages": str(TEST_SITE_PACKAGES),
    "sys_path": [*sys.path, str(TEST_PROJECT_SRC), str(TEST_SITE_PACKAGES)],
}
SPEC.loader.exec_module(GATE)

LEFT_UUID = "11111111-1111-4111-8111-111111111111"
RIGHT_UUID = "22222222-2222-4222-8222-222222222222"
PRODUCTION_LORA_PARAMETER_SHAPES = GATE._expected_lora_parameter_shapes(16)


@pytest.fixture(autouse=True)
def canonical_cpu_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(GATE, "_LAUNCHER_BOOTSTRAP_AUTHENTICATED", True)
    monkeypatch.setattr(GATE, "EXPECTED_MODEL_HIDDEN_SIZE", 32)
    monkeypatch.setattr(GATE, "EXPECTED_MODEL_DECODER_LAYERS", 1)
    monkeypatch.setattr(GATE, "EXPECTED_MODEL_SLIDING_ATTENTION_WIDTH", 32)
    monkeypatch.setattr(GATE, "EXPECTED_MODEL_FULL_ATTENTION_WIDTH", 32)
    monkeypatch.setattr(GATE, "EXPECTED_MODEL_SLIDING_KV_WIDTH", 32)
    monkeypatch.setattr(GATE, "EXPECTED_MODEL_FULL_KV_WIDTH", 32)
    monkeypatch.setattr(GATE, "EXPECTED_MODEL_SLIDING_ATTENTION_LAYERS", (0,))
    identity = {
        "bootstrap_forbidden_modules_preloaded": [],
        "cuda_initialized": False,
        "cuda_visible_devices": "",
        "expected_train_venv": str(TEST_TRAIN_VENV),
        "interpreter_flags": copy.deepcopy(GATE._EXPECTED_INTERPRETER_FLAGS),
        "project_src": str(TEST_PROJECT_SRC),
        "pyvenv_cfg_sha256": GATE._QUALIFICATION_BOOTSTRAP_IDENTITY["pyvenv_cfg_sha256"],
        "python_base_prefix": str(Path(sys.base_prefix).resolve()),
        "python_executable": str(TEST_TRAIN_VENV / "bin/python"),
        "python_implementation": "CPython",
        "python_prefix": str(Path(sys.base_prefix).resolve()),
        "python_version": GATE.EXPECTED_PYTHON_VERSION,
        "site_packages": str(TEST_SITE_PACKAGES),
        "sys_path": copy.deepcopy(GATE._QUALIFICATION_BOOTSTRAP_IDENTITY["sys_path"]),
        "torch_version": GATE.EXPECTED_TORCH_VERSION,
    }
    monkeypatch.setattr(GATE, "qualification_runtime_identity", lambda: copy.deepcopy(identity))


@pytest.fixture
def publication_dir(tmp_path: Path):
    """Use the normal pytest directory when it supports the required O_TMPFILE contract."""

    try:
        descriptor = os.open(tmp_path, os.O_RDWR | os.O_TMPFILE, 0o600)
    except OSError:
        with tempfile.TemporaryDirectory(prefix=".publisher-test-", dir=ROOT / "reports") as directory:
            yield Path(directory)
    else:
        os.close(descriptor)
        yield tmp_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict[str, Any]:
    config = load_resolved_toml(ROOT / "configs/calvin_abc_to_d.toml")
    config["training"]["checkpoint_interval"] = 1
    config["training"]["permanent_checkpoint_interval"] = 1
    prefix_sha256 = "6" * 64
    config["benchmark"]["prefix_geometry_content_sha256"] = prefix_sha256
    config["benchmark"]["fixed_physical_prefix_width"] = 600
    split = {
        "algorithm": "scene-grouped stable sha256 whole-episode ordering with all-task coverage assertion",
        "seed": 1729,
        "train_episode_indices": [0, 2],
        "train_episode_sha256": hashlib.sha256(b"0,2").hexdigest(),
        "validation_episode_indices": [1, 3],
        "validation_episode_sha256": hashlib.sha256(b"1,3").hexdigest(),
        "validation_fraction": 0.1,
    }
    camera_shapes = {"rgb_static": [200, 200, 3], "rgb_gripper": [84, 84, 3]}
    revisions = copy.deepcopy(GATE.EXPECTED_CALVIN_SOURCE_REVISIONS)
    config["calvin_identity"] = {
        "action_adapter": GATE.EXPECTED_ACTION_ADAPTER,
        "archive_bytes": GATE.EXPECTED_CALVIN_ARCHIVE_BYTES,
        "archive_sha256": GATE.EXPECTED_CALVIN_ARCHIVE_SHA256,
        "calvin_source_revisions": revisions,
        "calvin_source_revisions_sha256": canonical_config_sha256(revisions),
        "camera_shapes": camera_shapes,
        "camera_shapes_sha256": canonical_config_sha256(camera_shapes),
        "central_directory_sha256": GATE.EXPECTED_CALVIN_CENTRAL_DIRECTORY_SHA256,
        "dataset_manifest_file_sha256": "7" * 64,
        "dataset_manifest_schema": GATE.EXPECTED_CALVIN_MANIFEST_SCHEMA,
        "dataset_manifest_sha256": "3" * 64,
        "member_index": {
            "bytes": 238_215_168,
            "path": GATE.EXPECTED_CALVIN_MEMBER_INDEX_PATH,
            "schema": GATE.EXPECTED_CALVIN_MEMBER_INDEX_SCHEMA,
            "sha256": "9" * 64,
        },
        "member_inventory_sha256": "4" * 64,
        "metadata_files": copy.deepcopy(GATE.EXPECTED_CALVIN_METADATA_FILES),
        "metadata_sha256": "2" * 64,
        "normalization_sha256": "1" * 64,
        "protocol": GATE.CALVIN_PROTOCOL,
        "split": split,
        "split_sha256": canonical_config_sha256(split),
        "state_adapter": GATE.EXPECTED_STATE_ADAPTER,
        "reader_schema": GATE.EXPECTED_CALVIN_READER_SCHEMA,
        "storage_identity_sha256": "8" * 64,
        "storage_mode": GATE.EXPECTED_CALVIN_STORAGE_MODE,
    }
    module_names = {
        "PIL": "PIL",
        "accelerate": "accelerate",
        "duo_vla": "duo_vla",
        "huggingface_hub": "huggingface_hub",
        "numpy": "numpy",
        "peft": "peft",
        "pyarrow": "pyarrow",
        "safetensors": "safetensors",
        "tokenizers": "tokenizers",
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
    }
    origins = {
        name: str(
            (ROOT / "src/duo_vla/__init__.py").resolve()
            if name == "duo_vla"
            else (TEST_SITE_PACKAGES / import_name / "__init__.py").resolve()
        )
        for name, import_name in module_names.items()
    }
    authenticated_runtime = {
        "environment": {**GATE.EXPECTED_TRAIN_ENVIRONMENT, "PYTHONPATH": str(TEST_PROJECT_SRC)},
        "lock_sha256": "0b1fb188747ee99224078b3c40975ca7e6f8e082e22d2860f9b50ee679a67c46",
        "module_origins": origins,
        "packages": copy.deepcopy(GATE.EXPECTED_TRAIN_PACKAGES),
        "python": GATE.EXPECTED_PYTHON_VERSION,
        "sys_path": [str((ROOT / "scripts").resolve()), str(TEST_PROJECT_SRC), str(TEST_SITE_PACKAGES)],
    }
    config["execution_environment"] = {
        "authenticated_runtime": authenticated_runtime,
        "cublas_workspace_config": ":4096:8",
        "cuda_runtime": "12.6",
        "cudnn": 91002,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "gpu_capability": [[8, 6], [8, 6]],
        "gpu_names": ["Test GPU", "Test GPU"],
        "matmul_tf32": False,
        "peft": GATE.EXPECTED_TRAIN_PACKAGES["peft"],
        "python": GATE.EXPECTED_PYTHON_VERSION,
        "python_hash_seed": "0",
        "torch": GATE.EXPECTED_TORCH_VERSION,
        "transformers": GATE.EXPECTED_TRAIN_PACKAGES["transformers"],
        "world_size": 2,
    }
    config["artifact_trees"] = {"model_tree_sha256": "5" * 64}
    config["execution_geometry"] = {
        "expert_batch_isolation": "sample_isolated_grouped_mm_v1",
        "experts_implementation": "grouped_mm",
        "fixed_physical_prefix_width": 600,
        "physical_batch_size": 8,
        "prefix_geometry_content_sha256": prefix_sha256,
    }
    config["training_instruction_inventory_sha256"] = "7" * 64
    config["run"] = {"max_cached_frames": 512, "seed": 0, "task": None}
    config["source_tree_sha256"] = GATE._IMPORTED_QUALIFICATION_SOURCE_IDENTITY["production_source_tree_sha256"]
    return config


def _metric(update: int, *, seconds: float) -> dict[str, Any]:
    return {
        "examples_seen": 64 * update,
        "gradient_norm": 1.5 + update,
        "interface_learning_rate": 0.001,
        "lora_learning_rate": 0.0001,
        "objective": "rectified_flow",
        "train_loss": 2.0 / update,
        "update": update,
        "update_seconds": seconds,
    }


def _optimizer_parameter_inventory(config: dict[str, Any]) -> dict[str, Any]:
    hidden_size = 32
    lora_rank = config["lora"]["rank"]
    lora_parameters = []
    for projection in config["lora"]["projections"]:
        prefix = f"base_model.model.model.decoder.layers.0.self_attn.{projection}"
        lora_parameters.extend(
            (
                {
                    "dtype": "torch.float32",
                    "name": f"{prefix}.lora_A.default.weight",
                    "requires_grad": True,
                    "shape": [lora_rank, hidden_size],
                },
                {
                    "dtype": "torch.float32",
                    "name": f"{prefix}.lora_B.default.weight",
                    "requires_grad": True,
                    "shape": [hidden_size, lora_rank],
                },
            )
        )
    action = config["action"]
    benchmark = config["benchmark"]
    interface_shapes = {
        "action_projector.horizon_embedding": [action["horizon"], hidden_size],
        "action_projector.action_type_embedding": [hidden_size],
        "action_projector.action_projection.weight": [hidden_size, action["dimension"]],
        "action_projector.action_projection.bias": [hidden_size],
        "action_projector.timestep_mlp.net.0.weight": [hidden_size, action["timestep_embedding_dimension"]],
        "action_projector.timestep_mlp.net.0.bias": [hidden_size],
        "action_projector.timestep_mlp.net.2.weight": [hidden_size, hidden_size],
        "action_projector.timestep_mlp.net.2.bias": [hidden_size],
        "action_projector.state_mlp.net.0.weight": [hidden_size, benchmark["state_dimension"]],
        "action_projector.state_mlp.net.0.bias": [hidden_size],
        "action_projector.state_mlp.net.2.weight": [hidden_size, hidden_size],
        "action_projector.state_mlp.net.2.bias": [hidden_size],
        "velocity_head.projection.weight": [action["dimension"], hidden_size],
        "velocity_head.projection.bias": [action["dimension"]],
    }
    interface_parameters = [
        {
            "dtype": "torch.float32",
            "name": name,
            "requires_grad": True,
            "shape": shape,
        }
        for name, shape in interface_shapes.items()
    ]
    return {
        "groups": [
            {"name": "lora", "parameters": lora_parameters},
            {"name": "interface", "parameters": interface_parameters},
        ],
        "schema": GATE.OPTIMIZER_PARAMETER_SCHEMA,
    }


def _run_contract(config: dict[str, Any], config_sha256: str, run_uuid: str, policy_sha256: str) -> dict[str, str]:
    identity = config["calvin_identity"]
    revisions = identity["calvin_source_revisions"]
    optimizer_inventory_sha256 = GATE.optimizer_parameter_inventory_sha256(_optimizer_parameter_inventory(config))
    return dict(
        sorted(
            {
                "action_adapter": identity["action_adapter"],
                "archive_bytes": str(identity["archive_bytes"]),
                "archive_sha256": identity["archive_sha256"],
                "calvin_env_revision": revisions["calvin_env"],
                "calvin_revision": revisions["calvin"],
                "calvin_source_revisions_sha256": identity["calvin_source_revisions_sha256"],
                "calvin_tacto_revision": revisions["tacto"],
                "camera_shapes_sha256": identity["camera_shapes_sha256"],
                "central_directory_sha256": identity["central_directory_sha256"],
                "config_sha256": config_sha256,
                "dataset_manifest_file_sha256": identity["dataset_manifest_file_sha256"],
                "dataset_manifest_schema": identity["dataset_manifest_schema"],
                "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
                "execution_environment_sha256": canonical_config_sha256(config["execution_environment"]),
                "expert_batch_isolation": config["execution_geometry"]["expert_batch_isolation"],
                "experts_implementation": config["execution_geometry"]["experts_implementation"],
                "fixed_physical_prefix_width": "600",
                "member_index_bytes": str(identity["member_index"]["bytes"]),
                "member_index_path": identity["member_index"]["path"],
                "member_index_schema": identity["member_index"]["schema"],
                "member_index_sha256": identity["member_index"]["sha256"],
                "member_inventory_sha256": identity["member_inventory_sha256"],
                "metadata_sha256": identity["metadata_sha256"],
                "model_revision": config["model"]["revision"],
                "model_tree_sha256": config["artifact_trees"]["model_tree_sha256"],
                "normalization_sha256": identity["normalization_sha256"],
                "optimizer_parameter_schema_sha256": optimizer_inventory_sha256,
                "physical_batch_size": "8",
                "policy_contract_sha256": policy_sha256,
                "prefix_geometry_content_sha256": config["execution_geometry"]["prefix_geometry_content_sha256"],
                "protocol": GATE.CALVIN_PROTOCOL,
                "reader_schema": identity["reader_schema"],
                "run_uuid": run_uuid,
                "source_tree_sha256": config["source_tree_sha256"],
                "split_sha256": identity["split_sha256"],
                "state_adapter": identity["state_adapter"],
                "storage_identity_sha256": identity["storage_identity_sha256"],
                "storage_mode": identity["storage_mode"],
                "train_episode_sha256": identity["split"]["train_episode_sha256"],
                "training_instruction_inventory_sha256": config["training_instruction_inventory_sha256"],
                "validation_episode_sha256": identity["split"]["validation_episode_sha256"],
            }.items()
        )
    )


def _rank_payload(
    *,
    rank: int,
    update: int,
    run_contract: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    optimization = config["optimization"]
    scale = GATE._expected_learning_rate_scale(update, config)
    bases = [optimization["lora_learning_rate"], optimization["interface_learning_rate"]]
    parameter_inventory = _optimizer_parameter_inventory(config)
    groups = []
    next_identifier = 0
    for index, (name, inventory_group) in enumerate(
        zip(("lora", "interface"), parameter_inventory["groups"], strict=True)
    ):
        identifiers = list(range(next_identifier, next_identifier + len(inventory_group["parameters"])))
        next_identifier += len(identifiers)
        groups.append(
            {
                "amsgrad": False,
                "betas": (optimization["adam_beta1"], optimization["adam_beta2"]),
                "capturable": False,
                "decoupled_weight_decay": True,
                "differentiable": False,
                "eps": optimization["adam_epsilon"],
                "foreach": None,
                "fused": None,
                "initial_lr": bases[index],
                "lr": bases[index] * scale,
                "maximize": False,
                "name": name,
                "params": identifiers,
                "weight_decay": optimization["weight_decay"],
            }
        )
    states = {}
    parameters = [
        parameter for inventory_group in parameter_inventory["groups"] for parameter in inventory_group["parameters"]
    ]
    for identifier, parameter in enumerate(parameters):
        shape = parameter["shape"]
        states[identifier] = {
            "exp_avg": torch.full(shape, 0.01 * (identifier + update + rank), dtype=torch.float32),
            "exp_avg_sq": torch.full(shape, 0.02 * (identifier + update + rank), dtype=torch.float32),
            "step": torch.tensor(float(update), dtype=torch.float32),
        }
    return {
        "optimizer": {
            "param_groups": groups,
            "state": states,
        },
        "optimizer_parameter_inventory": parameter_inventory,
        "rank": rank,
        "rng": {
            "cuda_device_index": rank,
            "numpy": {
                "bit_generator": "MT19937",
                "cached_gaussian": 0.0,
                "has_gauss": 0,
                "keys": torch.tensor([update, rank + 1], dtype=torch.uint32),
                "position": 7,
            },
            "python": (3, (1, 2, update, rank), None),
            "torch_cpu": torch.tensor([1, 2, update, rank], dtype=torch.uint8),
            "torch_cuda": torch.tensor([4, 5, update, rank], dtype=torch.uint8),
        },
        "run_contract": dict(sorted(run_contract.items())),
        "scheduler": {
            "_get_lr_called_within_step": False,
            "_is_initial": False,
            "_last_lr": [group["lr"] for group in groups],
            "_step_count": update + 1,
            "base_lrs": bases,
            "last_epoch": update,
            "lr_lambdas": [None, None],
        },
        "schema": TRAINING_RANK_STATE_SCHEMA,
        "trainer_state": TrainerState(next_update=update, examples_seen=64 * update).to_dict(),
        "world_size": 2,
    }


def _artifact_entry(path: Path, checkpoint: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(checkpoint).as_posix(),
        "sha256": _sha256(path),
    }


def _write_checkpoint(
    run: Path,
    *,
    update: int,
    run_uuid: str,
    config: dict[str, Any],
    config_sha256: str,
    parent_manifest_sha256: str | None,
    metric: dict[str, Any],
) -> tuple[Path, str]:
    checkpoint = run / "checkpoints" / f"update-{update:06d}"
    (checkpoint / "lora").mkdir(parents=True)
    (checkpoint / "artifacts").mkdir()
    paths = {
        "interface": checkpoint / "interface.safetensors",
        "lora_config": checkpoint / "lora/adapter_config.json",
        "lora_weights": checkpoint / "lora/adapter_model.safetensors",
        "normalization": checkpoint / "artifacts/normalization.json",
        "prefix_geometry": checkpoint / "artifacts/prefix_geometry.json",
        "resolved_config": checkpoint / "artifacts/resolved_config.json",
        "training_rank_000": checkpoint / "artifacts/training_rank_000.pt",
        "training_rank_001": checkpoint / "artifacts/training_rank_001.pt",
    }
    paths["interface"].write_bytes(f"interface-update-{update}".encode())
    paths["lora_config"].write_bytes(b'{"alpha":32,"rank":16}\n')
    paths["lora_weights"].write_bytes(f"adapter-update-{update}".encode())
    paths["normalization"].write_bytes(b'{"normalization":"fixed"}\n')
    paths["prefix_geometry"].write_bytes(b'{"prefix":"fixed"}\n')
    paths["resolved_config"].write_bytes((run / "resolved_config.json").read_bytes())
    policy = policy_contract_from_config(config)
    policy_sha256 = canonical_config_sha256(policy.to_dict())
    contract = _run_contract(config, config_sha256, run_uuid, policy_sha256)
    for rank in range(2):
        torch.save(
            _rank_payload(rank=rank, update=update, run_contract=contract, config=config),
            paths[f"training_rank_{rank:03d}"],
        )
    artifacts = {name: _artifact_entry(path, checkpoint) for name, path in paths.items()}
    manifest_contract: dict[str, Any] = {
        **contract,
        "archive_bytes": int(contract["archive_bytes"]),
        "fixed_physical_prefix_width": 600,
        "member_index_bytes": int(contract["member_index_bytes"]),
        "physical_batch_size": 8,
    }
    manifest = {
        **manifest_contract,
        "artifacts": artifacts,
        "calvin_identity": copy.deepcopy(config["calvin_identity"]),
        "calvin_source_revisions": copy.deepcopy(config["calvin_identity"]["calvin_source_revisions"]),
        "camera_shapes": copy.deepcopy(config["calvin_identity"]["camera_shapes"]),
        "complete": False,
        "configured_total_updates": 30_000,
        "dataset": "task_ABC_D",
        "dataset_split": "training",
        "kind": "resumable-calvin-abc-to-d-training",
        "last_metrics": metric,
        "execution_environment": copy.deepcopy(config["execution_environment"]),
        "execution_geometry": copy.deepcopy(config["execution_geometry"]),
        "model_id": GATE.EXPECTED_MODEL_ID,
        "parent_manifest_sha256": parent_manifest_sha256,
        "platform": "synthetic-test-platform",
        "policy_contract": policy.to_dict(),
        "run_seed": 0,
        "schema": SCHEMA_VERSION,
        "split": copy.deepcopy(config["calvin_identity"]["split"]),
        "task": None,
        "trainer_state": TrainerState(next_update=update, examples_seen=64 * update).to_dict(),
        "training_rank_state_sha256": [
            artifacts["training_rank_000"]["sha256"],
            artifacts["training_rank_001"]["sha256"],
        ],
    }
    manifest_path = checkpoint / "manifest.json"
    manifest_path.write_bytes(GATE._canonical_pretty_json_bytes(manifest))
    return checkpoint, _sha256(manifest_path)


def _make_run(root: Path, *, run_uuid: str, seconds_offset: float) -> dict[str, Any]:
    root.mkdir()
    config = _config()
    config_sha256 = save_resolved_config(root / "resolved_config.json", config)
    create_run_journal(root, config_sha256=config_sha256, run_uuid=run_uuid)
    metrics = [_metric(1, seconds=seconds_offset + 1.0), _metric(2, seconds=seconds_offset + 2.0)]
    checkpoint_one, manifest_one_sha256 = _write_checkpoint(
        root,
        update=1,
        run_uuid=run_uuid,
        config=config,
        config_sha256=config_sha256,
        parent_manifest_sha256=None,
        metric=metrics[0],
    )
    record_latest_checkpoint(
        root,
        checkpoint=checkpoint_one,
        update=1,
        manifest_sha256=manifest_one_sha256,
        parent_manifest_sha256=None,
        last_metrics=metrics[0],
    )
    checkpoint_two, manifest_two_sha256 = _write_checkpoint(
        root,
        update=2,
        run_uuid=run_uuid,
        config=config,
        config_sha256=config_sha256,
        parent_manifest_sha256=manifest_one_sha256,
        metric=metrics[1],
    )
    record_latest_checkpoint(
        root,
        checkpoint=checkpoint_two,
        update=2,
        manifest_sha256=manifest_two_sha256,
        parent_manifest_sha256=manifest_one_sha256,
        last_metrics=metrics[1],
    )
    (root / "metrics.jsonl").write_text(
        "".join(json.dumps(metric, allow_nan=False, sort_keys=True) + "\n" for metric in metrics),
        encoding="utf-8",
    )
    return {
        "config_sha256": config_sha256,
        "journal_sha256": _sha256(root / "run_journal.json"),
        "root": root,
        "run_uuid": run_uuid,
    }


@pytest.fixture
def run_pair(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        _make_run(tmp_path / "resumed", run_uuid=LEFT_UUID, seconds_offset=10.0),
        _make_run(tmp_path / "uninterrupted", run_uuid=RIGHT_UUID, seconds_offset=20.0),
    )


def _compare(pair: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    left, right = pair
    return GATE.compare_training_runs(
        left["root"],
        right["root"],
        interrupted_resumed_journal_sha256=left["journal_sha256"],
        uninterrupted_journal_sha256=right["journal_sha256"],
        interrupted_resumed_run_uuid=left["run_uuid"],
        uninterrupted_run_uuid=right["run_uuid"],
        expected_config_sha256=left["config_sha256"],
    )


def _rewrite_tip_journal(run: dict[str, Any]) -> None:
    root = run["root"]
    manifest_path = root / "checkpoints/update-000002/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_bytes(GATE._canonical_pretty_json_bytes(manifest))
    journal_path = root / "run_journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["latest_checkpoint"]["manifest_sha256"] = _sha256(manifest_path)
    journal["latest_checkpoint"]["parent_manifest_sha256"] = manifest["parent_manifest_sha256"]
    journal["latest_checkpoint"]["last_metrics"] = manifest["last_metrics"]
    journal_path.write_bytes(GATE.canonical_json_bytes(journal))
    run["journal_sha256"] = _sha256(journal_path)


def _reauthenticate_tip_artifact(run: dict[str, Any], name: str) -> None:
    checkpoint = run["root"] / "checkpoints/update-000002"
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = checkpoint / manifest["artifacts"][name]["path"]
    manifest["artifacts"][name] = _artifact_entry(path, checkpoint)
    if name.startswith("training_rank_"):
        rank = int(name.rsplit("_", maxsplit=1)[1])
        manifest["training_rank_state_sha256"][rank] = manifest["artifacts"][name]["sha256"]
    manifest_path.write_bytes(GATE._canonical_pretty_json_bytes(manifest))
    _rewrite_tip_journal(run)


def _reauthenticate_update_one_artifact(run: dict[str, Any], name: str) -> None:
    checkpoint_one = run["root"] / "checkpoints/update-000001"
    manifest_one_path = checkpoint_one / "manifest.json"
    manifest_one = json.loads(manifest_one_path.read_text(encoding="utf-8"))
    path = checkpoint_one / manifest_one["artifacts"][name]["path"]
    manifest_one["artifacts"][name] = _artifact_entry(path, checkpoint_one)
    if name.startswith("training_rank_"):
        rank = int(name.rsplit("_", maxsplit=1)[1])
        manifest_one["training_rank_state_sha256"][rank] = manifest_one["artifacts"][name]["sha256"]
    manifest_one_path.write_bytes(GATE._canonical_pretty_json_bytes(manifest_one))

    manifest_two_path = run["root"] / "checkpoints/update-000002/manifest.json"
    manifest_two = json.loads(manifest_two_path.read_text(encoding="utf-8"))
    manifest_two["parent_manifest_sha256"] = _sha256(manifest_one_path)
    manifest_two_path.write_bytes(GATE._canonical_pretty_json_bytes(manifest_two))
    _rewrite_tip_journal(run)


def test_pass_report_is_self_hashed_and_exclusively_published(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    publication_dir: Path,
) -> None:
    report = _compare(run_pair)

    assert report["status"] == "passed"
    assert report["execution"]["launcher_exact_comparator_bytes_bootstrap"] is True
    assert report["runtime_identity"]["interpreter_flags"] == GATE._EXPECTED_INTERPRETER_FLAGS
    assert report["runtime_identity"]["bootstrap_forbidden_modules_preloaded"] == []
    assert report["comparison"]["metrics"]["ignored_fields"] == ["update_seconds"]
    assert report["comparison"]["logical_rank_state"]["ignored_fields"] == ["run_contract.run_uuid"]
    assert all(
        record["differences"] == 0
        for update in report["comparison"]["logical_rank_state"]["updates"].values()
        for record in update.values()
    )
    assert set(report["comparison"]["logical_rank_state"]["updates"]) == {"update_1", "update_2"}
    assert report["pass_criteria"]["logical_update_1_and_2_optimizer_scheduler_rng_state_exact"] is True
    assert report["pass_criteria"]["optimizer_inventory_hash_group_names_shapes_and_state_coverage_validated"] is True
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    assert report["report_sha256"] == hashlib.sha256(GATE.canonical_json_bytes(unsigned)).hexdigest()
    output = publication_dir / "qualification.json"
    entries_before_publication = set(publication_dir.iterdir())
    assert GATE.write_canonical_json_exclusive(output, report) == output
    committed = output.read_bytes()
    assert committed == GATE.canonical_json_bytes(report)
    with pytest.raises(FileExistsError):
        GATE.write_canonical_json_exclusive(output, report)
    assert output.read_bytes() == committed
    assert set(publication_dir.iterdir()) == entries_before_publication | {output}


def test_report_final_name_appears_only_after_complete_unnamed_inode_link(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    publication_dir: Path,
) -> None:
    report = _compare(run_pair)
    output = publication_dir / "qualification.json"
    expected = GATE.canonical_json_bytes(report)
    real_link = GATE.os.link
    observed_link = False

    def assert_atomic_visibility(source: Any, destination: Any, **kwargs: Any) -> None:
        nonlocal observed_link
        observed_link = True
        assert not output.exists() and not output.is_symlink()
        descriptor_stat = os.stat(source, follow_symlinks=True)
        assert descriptor_stat.st_nlink == 0
        assert Path(source).read_bytes() == expected
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(GATE.os, "link", assert_atomic_visibility)
    assert GATE.write_canonical_json_exclusive(output, report) == output
    assert observed_link
    assert output.read_bytes() == expected


def test_externally_pinned_journal_and_uuid_are_mandatory(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    left, _ = run_pair
    left["journal_sha256"] = "f" * 64
    with pytest.raises(GATE.QualificationError, match="external SHA-256 mismatch"):
        _compare(run_pair)


def test_unauthenticated_artifact_mutation_is_rejected(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    right = run_pair[1]
    path = right["root"] / "checkpoints/update-000002/lora/adapter_model.safetensors"
    path.write_bytes(b"tampered-adapter")
    with pytest.raises(ValueError, match="hash mismatch"):
        _compare(run_pair)


@pytest.mark.parametrize("section", ("optimizer", "scheduler", "rng"))
def test_authenticated_rank_state_mutations_are_rejected(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    section: str,
) -> None:
    right = run_pair[1]
    path = right["root"] / "checkpoints/update-000002/artifacts/training_rank_000.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if section == "optimizer":
        payload["optimizer"]["state"][0]["exp_avg"][0] += 1
    elif section == "scheduler":
        payload["scheduler"]["last_epoch"] += 1
    else:
        payload["rng"]["torch_cpu"][0] ^= 1
    torch.save(payload, path)
    _reauthenticate_tip_artifact(right, "training_rank_000")

    expected_message = "scheduler epoch" if section == "scheduler" else "logical update 2 rank 0 training state"
    with pytest.raises(GATE.QualificationError, match=expected_message):
        _compare(run_pair)


def test_authenticated_manifest_or_metric_drift_is_rejected(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    right = run_pair[1]
    manifest_path = right["root"] / "checkpoints/update-000002/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["platform"] = "unauthorized-drift"
    manifest_path.write_bytes(GATE._canonical_pretty_json_bytes(manifest))
    _rewrite_tip_journal(right)

    with pytest.raises(GATE.QualificationError, match="normalized update 2 manifest"):
        _compare(run_pair)


def test_authenticated_non_timing_metric_drift_is_rejected(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    right = run_pair[1]
    metrics_path = right["root"] / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    metrics[-1]["train_loss"] += 0.25
    metrics_path.write_text(
        "".join(json.dumps(metric, allow_nan=False, sort_keys=True) + "\n" for metric in metrics),
        encoding="utf-8",
    )
    manifest_path = right["root"] / "checkpoints/update-000002/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["last_metrics"] = metrics[-1]
    manifest_path.write_bytes(GATE._canonical_pretty_json_bytes(manifest))
    _rewrite_tip_journal(right)

    with pytest.raises(GATE.QualificationError, match="normalized update 2 manifest"):
        _compare(run_pair)


def test_authenticated_lora_byte_drift_is_rejected(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    right = run_pair[1]
    path = right["root"] / "checkpoints/update-000002/lora/adapter_config.json"
    path.write_bytes(b'{"alpha":31,"rank":16}\n')
    _reauthenticate_tip_artifact(right, "lora_config")

    with pytest.raises(GATE.QualificationError, match="lora_config bytes differ"):
        _compare(run_pair)


def test_parent_lineage_cannot_be_normalized_without_authentication(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    right = run_pair[1]
    manifest_path = right["root"] / "checkpoints/update-000002/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parent_manifest_sha256"] = "e" * 64
    manifest_path.write_bytes(GATE._canonical_pretty_json_bytes(manifest))
    _rewrite_tip_journal(right)

    with pytest.raises(GATE.QualificationError, match="parent manifest lineage"):
        _compare(run_pair)


def test_duplicate_json_is_rejected_even_when_externally_hashed(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    left = run_pair[0]
    path = left["root"] / "run_journal.json"
    raw = path.read_text(encoding="utf-8").replace(
        '"schema":"duo-vla-run-journal-v1"',
        '"schema":"duo-vla-run-journal-v1","schema":"duo-vla-run-journal-v1"',
    )
    path.write_text(raw, encoding="utf-8")
    left["journal_sha256"] = _sha256(path)

    with pytest.raises(GATE.QualificationError, match="duplicate JSON key"):
        _compare(run_pair)


@pytest.mark.parametrize("section", ("optimizer", "rng"))
def test_authenticated_update_one_rank_state_mutations_are_rejected(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    section: str,
) -> None:
    right = run_pair[1]
    path = right["root"] / "checkpoints/update-000001/artifacts/training_rank_000.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if section == "optimizer":
        payload["optimizer"]["state"][0]["exp_avg"][0] += 1
    else:
        payload["rng"]["torch_cpu"][0] ^= 1
    torch.save(payload, path)
    _reauthenticate_update_one_artifact(right, "training_rank_000")

    with pytest.raises(GATE.QualificationError, match="logical update 1 rank 0 training state"):
        _compare(run_pair)


def test_rank_state_hardlinks_are_rejected(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    path = run_pair[0]["root"] / "checkpoints/update-000001/artifacts/training_rank_000.pt"
    (tmp_path / "rank-state-alias.pt").hardlink_to(path)

    with pytest.raises(GATE.QualificationError, match="exactly one link"):
        _compare(run_pair)


def test_rank_state_symlink_substitution_is_rejected(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    path = run_pair[0]["root"] / "checkpoints/update-000001/artifacts/training_rank_000.pt"
    original = path.with_name("original-rank-state.pt")
    path.rename(original)
    path.symlink_to(original.name)

    with pytest.raises(GATE.QualificationError, match="real file"):
        _compare(run_pair)


def test_rank_state_load_is_bound_to_authenticated_bytes_during_path_swap(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = run_pair[0]["root"] / "checkpoints/update-000001/artifacts/training_rank_000.pt"
    displaced = path.with_name("displaced-rank-state.pt")
    real_torch_load = GATE.torch.load
    swapped = False

    def swap_path_after_byte_authentication(source: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal swapped
        assert isinstance(source, GATE.io.BytesIO)
        if not swapped:
            swapped = True
            path.rename(displaced)
            path.write_bytes(source.getvalue())
        return real_torch_load(source, *args, **kwargs)

    monkeypatch.setattr(GATE.torch, "load", swap_path_after_byte_authentication)
    with pytest.raises(
        GATE.QualificationError,
        match="artifact training_rank_000 file identity changed",
    ):
        _compare(run_pair)
    assert swapped


def test_logical_rank_comparison_canonicalizes_loaded_dtensor_local_bytes(tmp_path: Path) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, distribute_tensor

    assert not dist.is_initialized()
    buffer = GATE.io.BytesIO()
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{tmp_path / 'process-group'}",
            rank=0,
            world_size=1,
        )
        mesh = init_device_mesh("cpu", (1,))
        distributed = distribute_tensor(torch.arange(6, dtype=torch.float32).reshape(2, 3), mesh, [Replicate()])
        torch.save({"moment": distributed}, buffer)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    buffer.seek(0)
    left = torch.load(buffer, map_location="cpu", weights_only=True)
    buffer.seek(0)
    right = torch.load(buffer, map_location="cpu", weights_only=True)
    GATE._assert_equal(left, right, context="loaded DTensor rank state")
    assert GATE._logical_digest(left) == GATE._logical_digest(right)
    assert GATE._tensor_stats(left) == (1, 6 * torch.tensor([], dtype=torch.float32).element_size())


@pytest.mark.parametrize(
    ("table", "field", "value", "message"),
    (
        ("optimization", "total_updates", 2, "30,000-update"),
        ("optimization", "warmup_updates", 0, "1,000-update warmup"),
        ("optimization", "physical_batch_size", 4, "physical batch"),
        ("optimization", "microbatch_size", 4, "microbatch"),
        ("optimization", "gradient_accumulation_steps", 4, "gradient accumulation"),
        ("optimization", "global_batch_size", 32, "global batch"),
        ("training", "checkpoint_interval", 2, "intervals of one"),
        ("training", "permanent_checkpoint_interval", 2, "intervals of one"),
        ("model", "tensor_parallel_size", 1, "tensor-parallel"),
        ("run", "task", "open_drawer", "full task inventory"),
        ("run", "max_cached_frames", 256, "max-cached-frames"),
    ),
)
def test_noncanonical_smoke_config_drift_is_rejected(
    table: str,
    field: str,
    value: Any,
    message: str,
) -> None:
    config = _config()
    config[table][field] = value
    config_sha256 = canonical_config_sha256(config)

    with pytest.raises(GATE.QualificationError, match=message):
        GATE._validate_config(
            config,
            config_sha256=config_sha256,
            source_identity=GATE._IMPORTED_QUALIFICATION_SOURCE_IDENTITY,
        )


@pytest.mark.parametrize(
    ("table", "field"),
    (
        ("lora", "rank"),
        ("model", "id"),
        ("optimization", "optimizer"),
        ("training", "validation_interval"),
    ),
)
def test_canonical_config_rejects_truncated_static_tables(table: str, field: str) -> None:
    config = _config()
    del config[table][field]

    with pytest.raises(GATE.QualificationError, match="exact canonical CALVIN smoke resolved config"):
        GATE._validate_config(
            config,
            config_sha256=canonical_config_sha256(config),
            source_identity=GATE._IMPORTED_QUALIFICATION_SOURCE_IDENTITY,
        )


@pytest.mark.parametrize("table", ("optimization", "run"))
def test_canonical_config_rejects_extra_fields(table: str) -> None:
    config = _config()
    config[table]["unregistered_smoke_override"] = 1

    with pytest.raises(GATE.QualificationError, match="exact canonical CALVIN smoke resolved config"):
        GATE._validate_config(
            config,
            config_sha256=canonical_config_sha256(config),
            source_identity=GATE._IMPORTED_QUALIFICATION_SOURCE_IDENTITY,
        )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_rank_run_contract_requires_exact_production_inventory(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    mutation: str,
) -> None:
    for run in run_pair:
        path = run["root"] / "checkpoints/update-000002/artifacts/training_rank_000.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if mutation == "missing":
            del payload["run_contract"]["archive_sha256"]
        else:
            payload["run_contract"]["unsupported"] = "value"
        torch.save(payload, path)
        _reauthenticate_tip_artifact(run, "training_rank_000")

    with pytest.raises(GATE.QualificationError, match="run contract inventory"):
        _compare(run_pair)


@pytest.mark.parametrize("section", ("optimizer", "scheduler"))
def test_rank_optimizer_scheduler_reject_extra_schema_fields(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    section: str,
) -> None:
    for run in run_pair:
        path = run["root"] / "checkpoints/update-000002/artifacts/training_rank_000.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload[section]["unsupported"] = True
        torch.save(payload, path)
        _reauthenticate_tip_artifact(run, "training_rank_000")

    with pytest.raises(GATE.QualificationError, match=f"{section} schema"):
        _compare(run_pair)


def test_rank_optimizer_inventory_rejects_arbitrary_contract_digest(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    run = run_pair[0]
    path = run["root"] / "checkpoints/update-000002/artifacts/training_rank_000.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["run_contract"]["optimizer_parameter_schema_sha256"] = "f" * 64
    torch.save(payload, path)
    _reauthenticate_tip_artifact(run, "training_rank_000")

    with pytest.raises(GATE.QualificationError, match="optimizer inventory hash differs"):
        _compare(run_pair)


def test_production_optimizer_inventory_uses_the_full_pinned_lora_topology() -> None:
    from duo_vla.backbones.loading import expected_decoder_attention_lora_weight_schema

    assert len(PRODUCTION_LORA_PARAMETER_SHAPES) == 230
    prefix = "base_model.model.model.decoder.layers"
    assert PRODUCTION_LORA_PARAMETER_SHAPES[f"{prefix}.0.self_attn.q_proj.lora_A.default.weight"] == [16, 2816]
    assert PRODUCTION_LORA_PARAMETER_SHAPES[f"{prefix}.0.self_attn.q_proj.lora_B.default.weight"] == [4096, 16]
    assert PRODUCTION_LORA_PARAMETER_SHAPES[f"{prefix}.5.self_attn.q_proj.lora_B.default.weight"] == [8192, 16]
    assert PRODUCTION_LORA_PARAMETER_SHAPES[f"{prefix}.5.self_attn.k_proj.lora_B.default.weight"] == [1024, 16]
    assert f"{prefix}.5.self_attn.v_proj.lora_A.default.weight" not in PRODUCTION_LORA_PARAMETER_SHAPES
    production_schema = {
        name.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        ): list(shape)
        for name, shape in expected_decoder_attention_lora_weight_schema(rank=16).items()
    }
    assert production_schema == PRODUCTION_LORA_PARAMETER_SHAPES


def test_rank_optimizer_inventory_rejects_abbreviated_two_state_fixture(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    for run in run_pair:
        path = run["root"] / "checkpoints/update-000002/artifacts/training_rank_000.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        inventory = payload["optimizer_parameter_inventory"]
        inventory["groups"][0]["parameters"] = inventory["groups"][0]["parameters"][:1]
        inventory["groups"][1]["parameters"] = inventory["groups"][1]["parameters"][:1]
        payload["optimizer"]["param_groups"][0]["params"] = [0]
        payload["optimizer"]["param_groups"][1]["params"] = [1]
        payload["optimizer"]["state"] = {
            0: payload["optimizer"]["state"][0],
            1: payload["optimizer"]["state"][8],
        }
        payload["run_contract"]["optimizer_parameter_schema_sha256"] = GATE.optimizer_parameter_inventory_sha256(
            inventory
        )
        torch.save(payload, path)
        _reauthenticate_tip_artifact(run, "training_rank_000")

    with pytest.raises(GATE.QualificationError, match=r"complete pinned topology|interface ordering"):
        _compare(run_pair)


def test_rank_optimizer_moment_shape_must_match_named_inventory(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    for run in run_pair:
        path = run["root"] / "checkpoints/update-000002/artifacts/training_rank_000.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1, dtype=torch.float32)
        torch.save(payload, path)
        _reauthenticate_tip_artifact(run, "training_rank_000")

    with pytest.raises(GATE.QualificationError, match="moment tensors differ"):
        _compare(run_pair)


def _rehashed_source_identity(field: str) -> dict[str, str]:
    identity = dict(GATE._IMPORTED_QUALIFICATION_SOURCE_IDENTITY)
    identity[field] = "0" * 64
    if field != "qualification_source_sha256":
        payload = {key: value for key, value in identity.items() if key != "qualification_source_sha256"}
        identity["qualification_source_sha256"] = hashlib.sha256(
            json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
    return identity


@pytest.mark.parametrize("field", tuple(sorted(GATE._IMPORTED_QUALIFICATION_SOURCE_IDENTITY)))
def test_every_qualification_source_identity_field_is_fail_closed(field: str) -> None:
    with pytest.raises(GATE.QualificationError, match="source"):
        GATE.require_qualification_source_unchanged(
            GATE._IMPORTED_QUALIFICATION_SOURCE_IDENTITY,
            _rehashed_source_identity(field),
            context="mutation test",
        )


def test_source_mutation_during_comparison_is_rejected(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = dict(GATE._IMPORTED_QUALIFICATION_SOURCE_IDENTITY)
    drifted = _rehashed_source_identity("production_run_journal_sha256")
    snapshots = iter((baseline, drifted))
    monkeypatch.setattr(GATE, "_bootstrap_source_identity", lambda: next(snapshots))

    with pytest.raises(GATE.QualificationError, match="during training reproducibility comparison"):
        _compare(run_pair)


def test_source_mutation_before_publication_leaves_no_report(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    publication_dir: Path,
) -> None:
    report = _compare(run_pair)
    monkeypatch.setattr(
        GATE,
        "_bootstrap_source_identity",
        lambda: _rehashed_source_identity("production_training_checkpoint_sha256"),
    )
    output = publication_dir / "must-not-exist.json"
    with pytest.raises(GATE.QualificationError, match="publication"):
        GATE.write_canonical_json_exclusive(output, report)
    assert not output.exists()


def test_source_mutation_after_durable_staging_prevents_atomic_commit(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    publication_dir: Path,
) -> None:
    report = _compare(run_pair)
    baseline = dict(GATE._IMPORTED_QUALIFICATION_SOURCE_IDENTITY)
    drifted = _rehashed_source_identity("production_training_checkpoint_sha256")
    snapshots = iter((baseline, drifted))
    output = publication_dir / "must-not-be-committed.json"
    entries_before_publication = set(publication_dir.iterdir())

    source_checks = 0

    def source_identity_sequence() -> dict[str, str]:
        nonlocal source_checks
        source_checks += 1
        if source_checks == 2:
            assert not output.exists() and not output.is_symlink()
            assert set(publication_dir.iterdir()) == entries_before_publication
        return next(snapshots)

    monkeypatch.setattr(GATE, "_bootstrap_source_identity", source_identity_sequence)
    with pytest.raises(GATE.QualificationError, match="after durable qualification report staging"):
        GATE.write_canonical_json_exclusive(output, report)
    assert source_checks == 2
    assert set(publication_dir.iterdir()) == entries_before_publication
    assert not output.exists()


def test_runtime_mutation_after_durable_staging_prevents_atomic_commit(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    publication_dir: Path,
) -> None:
    report = _compare(run_pair)
    baseline = GATE.qualification_runtime_identity()
    drifted = {**baseline, "torch_version": "2.12.0"}
    snapshots = iter((baseline, drifted))
    monkeypatch.setattr(GATE, "qualification_runtime_identity", lambda: next(snapshots))

    output = publication_dir / "must-not-be-committed.json"
    entries_before_publication = set(publication_dir.iterdir())
    with pytest.raises(GATE.QualificationError, match="requires torch"):
        GATE.write_canonical_json_exclusive(output, report)
    assert set(publication_dir.iterdir()) == entries_before_publication
    assert not output.exists()


def test_exclusive_create_collision_preserves_foreign_target(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    publication_dir: Path,
) -> None:
    report = _compare(run_pair)
    output = publication_dir / "qualification.json"
    entries_before_publication = set(publication_dir.iterdir())
    collision_bytes = b"independently-created collision\n"

    real_link = GATE.os.link

    def collide_at_link(*args: Any, **kwargs: Any) -> None:
        output.write_bytes(collision_bytes)
        real_link(*args, **kwargs)

    monkeypatch.setattr(GATE.os, "link", collide_at_link)
    with pytest.raises(FileExistsError):
        GATE.write_canonical_json_exclusive(output, report)
    assert output.read_bytes() == collision_bytes
    assert set(publication_dir.iterdir()) == entries_before_publication | {output}


def test_exclusive_create_collision_preserves_foreign_symlink(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    publication_dir: Path,
) -> None:
    report = _compare(run_pair)
    output = publication_dir / "qualification.json"
    foreign = publication_dir / "foreign.txt"
    foreign.write_bytes(b"foreign\n")
    real_link = GATE.os.link

    def collide_with_symlink(*args: Any, **kwargs: Any) -> None:
        output.symlink_to(foreign.name)
        real_link(*args, **kwargs)

    monkeypatch.setattr(GATE.os, "link", collide_with_symlink)
    with pytest.raises(FileExistsError):
        GATE.write_canonical_json_exclusive(output, report)
    assert output.is_symlink() and output.readlink() == Path(foreign.name)
    assert foreign.read_bytes() == b"foreign\n"


def test_post_link_hardlink_race_invalidates_only_the_owned_inode(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    publication_dir: Path,
) -> None:
    report = _compare(run_pair)
    output = publication_dir / "qualification.json"
    alias = publication_dir / "attacker-alias"
    real_link = GATE.os.link

    def add_alias_after_atomic_link(*args: Any, **kwargs: Any) -> None:
        real_link(*args, **kwargs)
        real_link(output, alias)

    monkeypatch.setattr(GATE.os, "link", add_alias_after_atomic_link)
    with pytest.raises(GATE.QualificationError, match="link count drifted"):
        GATE.write_canonical_json_exclusive(output, report)
    assert output.read_bytes() == alias.read_bytes() == GATE._FAILED_PUBLICATION_BYTES
    assert output.stat().st_ino == alias.stat().st_ino


def test_target_replacement_after_atomic_link_is_rejected_and_preserved(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    publication_dir: Path,
) -> None:
    report = _compare(run_pair)
    output = publication_dir / "qualification.json"
    entries_before_publication = set(publication_dir.iterdir())
    foreign_bytes = b"foreign replacement target\n"
    real_verify = GATE._verify_published_entry
    verify_calls = 0

    def replace_before_final_verification(*args: Any, **kwargs: Any) -> None:
        nonlocal verify_calls
        verify_calls += 1
        output.unlink()
        output.write_bytes(foreign_bytes)
        real_verify(*args, **kwargs)

    monkeypatch.setattr(GATE, "_verify_published_entry", replace_before_final_verification)
    with pytest.raises(GATE.QualificationError, match=r"link count drifted|path inode identity drifted"):
        GATE.write_canonical_json_exclusive(output, report)

    assert verify_calls == 1
    assert output.read_bytes() == foreign_bytes
    assert set(publication_dir.iterdir()) == entries_before_publication | {output}


def test_parent_directory_replacement_prevents_false_success(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    publication_dir: Path,
) -> None:
    report = _compare(run_pair)
    output = publication_dir / "qualification.json"
    moved_parent = publication_dir.with_name(f"{publication_dir.name}-moved")
    real_verify = GATE._verify_published_entry
    verify_calls = 0

    def replace_parent_before_final_verification(*args: Any, **kwargs: Any) -> None:
        nonlocal verify_calls
        verify_calls += 1
        publication_dir.rename(moved_parent)
        publication_dir.mkdir()
        real_verify(*args, **kwargs)

    monkeypatch.setattr(GATE, "_verify_published_entry", replace_parent_before_final_verification)
    with pytest.raises(GATE.QualificationError, match="parent directory identity changed"):
        GATE.write_canonical_json_exclusive(output, report)

    assert not output.exists()
    residue = moved_parent / output.name
    assert residue.read_bytes() == GATE._FAILED_PUBLICATION_BYTES
    assert verify_calls == 1
    publication_dir.rmdir()
    moved_parent.rename(publication_dir)


def test_failed_publication_never_unlinks_a_potentially_swapped_name(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    publication_dir: Path,
) -> None:
    report = _compare(run_pair)

    def forbidden_unlink(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("publisher must not use stat-then-unlink rollback")

    def fail_after_atomic_link(*_args: Any, **_kwargs: Any) -> None:
        raise GATE.QualificationError("injected post-link verification failure")

    real_unlink = GATE.os.unlink
    monkeypatch.setattr(GATE.os, "unlink", forbidden_unlink)
    monkeypatch.setattr(GATE, "_verify_published_entry", fail_after_atomic_link)
    output = publication_dir / "failed-residue.json"
    try:
        with pytest.raises(GATE.QualificationError, match="injected post-link verification failure"):
            GATE.write_canonical_json_exclusive(output, report)
    finally:
        monkeypatch.setattr(GATE.os, "unlink", real_unlink)
    assert output.read_bytes() == GATE._FAILED_PUBLICATION_BYTES


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("python_version", "3.11.14"),
        ("python_implementation", "PyPy"),
        ("torch_version", "2.12.0"),
        ("cuda_initialized", True),
        ("cuda_visible_devices", "0"),
        ("python_prefix", "/tmp/not-the-train-venv"),
        ("python_base_prefix", "/tmp/not-the-base-prefix"),
        ("python_executable", "/tmp/python"),
        ("expected_train_venv", "/tmp/other-venv"),
        ("interpreter_flags", {**GATE._EXPECTED_INTERPRETER_FLAGS, "isolated": 0}),
        ("bootstrap_forbidden_modules_preloaded", ["_virtualenv"]),
        ("project_src", "/tmp/src"),
        ("site_packages", "/tmp/site-packages"),
        ("sys_path", ["/tmp"]),
        ("pyvenv_cfg_sha256", "0" * 64),
    ),
)
def test_every_runtime_identity_field_drift_is_rejected(field: str, value: Any) -> None:
    identity = GATE.qualification_runtime_identity()
    identity[field] = value
    with pytest.raises(GATE.QualificationError):
        GATE.require_canonical_cpu_runtime(identity, context="runtime mutation test")


def test_runtime_mutation_during_comparison_is_rejected(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = GATE.qualification_runtime_identity()
    drifted = {**baseline, "torch_version": "2.12.0"}
    snapshots = iter((baseline, drifted))
    monkeypatch.setattr(GATE, "qualification_runtime_identity", lambda: next(snapshots))

    with pytest.raises(GATE.QualificationError, match="requires torch"):
        _compare(run_pair)


def test_qualification_sources_are_not_part_of_the_training_source_tree() -> None:
    assert "scripts/compare_calvin_training_reproducibility.py" not in GATE._TRAINING_SOURCE_EXPLICIT_RELATIVE_PATHS
    assert "scripts/calvin/run_compare_training_reproducibility.sh" not in GATE._TRAINING_SOURCE_EXPLICIT_RELATIVE_PATHS


def test_production_source_tree_identity_exactly_matches_the_trainer() -> None:
    train_script = ROOT / "scripts/train_calvin.py"
    spec = importlib.util.spec_from_file_location("calvin_train_source_identity_reference", train_script)
    assert spec is not None and spec.loader is not None
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    assert GATE._TRAINING_SOURCE_EXPLICIT_RELATIVE_PATHS == train._CALVIN_SOURCE_EXPLICIT_RELATIVE_PATHS
    assert GATE.EXPECTED_RUN_CONTRACT_FIELDS == train._CALVIN_RUN_CONTRACT_FIELDS
    assert GATE._bootstrap_production_source_tree_sha256(ROOT) == train._source_tree_sha256(ROOT)


def _write_minimal_comparator_source_tree(root: Path) -> None:
    for relative in GATE._TRAINING_SOURCE_EXPLICIT_RELATIVE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"explicit:{relative}", encoding="utf-8")
    source = root / "src/duo_vla/module.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("source", encoding="utf-8")


@pytest.mark.parametrize("ancestor", ("src", "configs"))
def test_comparator_source_hash_rejects_symlinked_ancestors(tmp_path: Path, ancestor: str) -> None:
    _write_minimal_comparator_source_tree(tmp_path)
    original = tmp_path / ancestor
    real = tmp_path / f"real-{ancestor}"
    original.rename(real)
    original.symlink_to(real.name, target_is_directory=True)

    with pytest.raises(RuntimeError, match="ancestor must be a real directory"):
        GATE._bootstrap_production_source_tree_sha256(tmp_path)


def test_comparator_source_hash_rejects_symlink_above_project_root(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    project = real_parent / "project"
    _write_minimal_comparator_source_tree(project)
    alias = tmp_path / "alias-parent"
    alias.symlink_to(real_parent.name, target_is_directory=True)

    with pytest.raises(RuntimeError, match="ancestor must be a real directory"):
        GATE._bootstrap_production_source_tree_sha256(alias / "project")


def test_comparator_source_hash_rejects_ancestor_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_minimal_comparator_source_tree(tmp_path)
    real_open = GATE.os.open
    replaced = False

    def replace_configs_on_file_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal replaced
        if path == "base.toml" and not flags & os.O_DIRECTORY and not replaced:
            replaced = True
            (tmp_path / "configs").rename(tmp_path / "configs-original")
            (tmp_path / "configs").mkdir()
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(GATE.os, "open", replace_configs_on_file_open)
    with pytest.raises(RuntimeError, match="ancestor changed while reading"):
        GATE._bootstrap_production_source_tree_sha256(tmp_path)
    assert replaced


def test_launcher_declares_exact_byte_and_runtime_bootstrap() -> None:
    source = (ROOT / "scripts/calvin/run_compare_training_reproducibility.sh").read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/env -S -i DUO_VLA_CLOSED_LAUNCHER_ENTRY=1 /bin/bash --noprofile --norc\n")
    assert 'cache_root="/hdd2/hyunbin/vla/cache"' in source
    assert "${DUO_VLA_CACHE_ROOT" not in source and "${CALVIN_TRAIN_VENV" not in source
    assert "$(/usr/bin/dirname" in source
    assert "launcher_start_sha256" in source
    assert "launcher_pre_exec_sha256" in source
    assert "compile(_comparator_raw" in source
    assert "_sys.version_info[:3] != (3, 11, 15)" in source
    assert '-I -S -B -c "${python_bootstrap}"' in source
    assert "__duo_vla_qualification_bootstrap_capability__" in source
    assert "_virtualenv" in source and "_cuda_bindings_redirector" in source and "_distutils_hack" in source
    assert "DUO_VLA_QUALIFICATION_EXECUTED_COMPARATOR_SHA256" not in source
    assert "CUDA_VISIBLE_DEVICES=" in source


def test_direct_launcher_entry_blocks_bash_env_and_path_hooks(tmp_path: Path) -> None:
    launcher = ROOT / "scripts/calvin/run_compare_training_reproducibility.sh"
    marker = tmp_path / "shell-hook-marker"
    bash_env = tmp_path / "malicious-bash-env"
    bash_env.write_text(f"/usr/bin/touch {marker}\n", encoding="utf-8")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    for name in ("dirname", "readlink", "sha256sum"):
        fake = fake_bin / name
        fake.write_text(f"#!/bin/sh\n/usr/bin/touch {marker}\nexit 97\n", encoding="utf-8")
        fake.chmod(0o755)

    result = subprocess.run(
        [str(launcher), "--help"],
        cwd=ROOT,
        env={"BASH_ENV": str(bash_env), "PATH": str(fake_bin)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout
    assert not marker.exists()


def test_bash_interpreter_invocation_fails_closed_before_launcher_body() -> None:
    launcher = ROOT / "scripts/calvin/run_compare_training_reproducibility.sh"
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", str(launcher), "--help"],
        cwd=ROOT,
        env={},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Invoke this qualification launcher directly" in result.stderr
    assert "usage:" not in result.stdout + result.stderr


def test_direct_python_invocation_cannot_forge_the_old_environment_proof() -> None:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "DUO_VLA_QUALIFICATION_BOOTSTRAP_MODE": "launcher-exact-comparator-bytes-v1",
        "DUO_VLA_QUALIFICATION_EXECUTED_COMPARATOR_SHA256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "DUO_VLA_QUALIFICATION_EXECUTED_LAUNCHER_SHA256": hashlib.sha256(
            (ROOT / "scripts/calvin/run_compare_training_reproducibility.sh").read_bytes()
        ).hexdigest(),
        "DUO_VLA_QUALIFICATION_EXPECTED_TRAIN_VENV": str(TEST_TRAIN_VENV),
    }
    result = subprocess.run(
        [str(TEST_TRAIN_VENV / "bin/python"), str(SCRIPT), "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "closed isolated qualification launcher" in combined
    assert "usage:" not in combined


def test_comparison_rejects_an_unbound_direct_python_invocation(
    run_pair: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(GATE, "_LAUNCHER_BOOTSTRAP_AUTHENTICATED", False)
    with pytest.raises(GATE.QualificationError, match="exact-byte qualification launcher"):
        _compare(run_pair)
