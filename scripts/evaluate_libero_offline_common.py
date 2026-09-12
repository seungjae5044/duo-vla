#!/usr/bin/env python3
"""Read-only TP=2 offline action-space comparison for strict LIBERO checkpoints.

The ``evaluate`` command evaluates one checkpoint. Run it once for the flow
checkpoint and once for the direct-regression checkpoint, then use ``compare``
to fail closed unless both reports used bit-identical canonical inputs.

This is deliberately not a simulator evaluator.  The primary metric compares
final clipped normalized action predictions with normalized clean A1 chunks.
The checkpoint-native training loss is reported separately because its target
has different semantics for rectified flow and direct regression.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "duovla-libero-offline-common-metric-v1"
COMPARISON_SCHEMA = "duovla-libero-offline-common-comparison-v1"
DATASET_ID = "HuggingFaceVLA/libero"
DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
MODEL_ID = "google/diffusiongemma-26B-A4B-it"
MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
PROTOCOL = "duovla-libero-v1"
PHYSICAL_BATCH_SIZE = 8
FIXED_PHYSICAL_PREFIX_WIDTH = 545
ACTION_HORIZON = 8
ACTION_DIM = 7
STATE_DIM = 8
FLOW_NFE = (1, 5, 10)
VALIDATION_SEED_XOR = 0x5A17
DEFAULT_COMMON_SAMPLING_SEED = 20_260_830
EXPECTED_LOCK_SHA256 = "0b1fb188747ee99224078b3c40975ca7e6f8e082e22d2860f9b50ee679a67c46"
EXPECTED_PACKAGES = {
    "accelerate": "1.14.0",
    "huggingface-hub": "1.29.0",
    "numpy": "2.4.6",
    "peft": "0.20.0",
    "pillow": "12.3.0",
    "pyarrow": "20.0.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.13.0+cu126",
    "torchvision": "0.28.0+cu126",
    "transformers": "5.15.0",
}
REQUIRED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "HF_HUB_OFFLINE": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
_ALGORITHM_ENVIRONMENT_PREFIXES = ("CUBLAS_", "CUDA_", "CUDNN_", "NCCL_", "PYTORCH_", "TORCH_")
_ALLOWED_ALGORITHM_ENVIRONMENT = frozenset(
    {"CUBLAS_WORKSPACE_CONFIG", "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "TORCH_NCCL_ASYNC_ERROR_HANDLING"}
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value!r} in {path}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=reject_constant,
    )
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    require(path.parent.is_dir(), f"output parent does not exist: {path.parent}")
    payload = json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _training_source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    for relative in ("src/duo_vla", "configs"):
        paths.extend(
            path for path in (root / relative).rglob("*") if path.is_file() and "__pycache__" not in path.parts
        )
    paths.extend(
        path
        for path in (
            root / "scripts/run_libero_train.sh",
            root / "scripts/train_libero.py",
            root / "pyproject.toml",
            root / "uv.lock",
        )
        if path.is_file()
    )
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _stat_inventory(paths: list[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for root in paths:
        require(root.exists(), f"protected input does not exist: {root}")
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if path.is_dir():
                continue
            lexical = path.lstat()
            target = path.stat()
            records.append(
                {
                    "lexical_mtime_ns": lexical.st_mtime_ns,
                    "lexical_size": lexical.st_size,
                    "path": f"{root}:{path.relative_to(root).as_posix() if path != root else '.'}",
                    "symlink_target": os.readlink(path) if path.is_symlink() else None,
                    "target_mtime_ns": target.st_mtime_ns,
                    "target_size": target.st_size,
                }
            )
    return {"entry_count": len(records), "sha256": _canonical_sha256(records)}


def _ensure_output_is_separate(output: Path, protected: list[Path]) -> None:
    destination = output.absolute()
    for path in protected:
        root = path.resolve()
        require(
            not destination.is_relative_to(root),
            f"output must not be created under protected checkpoint/data/model input: {root}",
        )


def _tensor_record(tensor: Any) -> dict[str, Any]:
    import torch

    require(isinstance(tensor, torch.Tensor), "tensor record input must be a torch.Tensor")
    local = tensor.detach().cpu().contiguous()
    raw = local.view(torch.uint8).numpy().tobytes(order="C")
    metadata = {"dtype": str(local.dtype), "shape": list(local.shape)}
    return {
        **metadata,
        "bytes": len(raw),
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "tensor_sha256": hashlib.sha256(_canonical_bytes(metadata) + b"\0" + raw).hexdigest(),
    }


def _processor_tensor_records(values: Any) -> dict[str, dict[str, Any]]:
    import torch

    require(isinstance(values, dict) or hasattr(values, "items"), "processor output is not a mapping")
    records: dict[str, dict[str, Any]] = {}
    for name, value in sorted(values.items()):
        require(isinstance(value, torch.Tensor), f"processor output {name!r} is not a tensor")
        records[name] = _tensor_record(value)
    return records


def _fixed_distinct_anchors(sampler: Any, *, count: int, seed: int) -> tuple[Any, ...]:
    import torch

    require(count > 0, "anchor count must be positive")
    require(count <= sampler.population_size, "anchor count exceeds validation population")
    generator = torch.Generator().manual_seed(seed)
    anchors: list[Any] = []
    seen: set[tuple[int, int]] = set()
    while len(anchors) < count:
        anchor = sampler.draw(generator)
        identity = (anchor.episode_index, anchor.frame_index)
        if identity not in seen:
            anchors.append(anchor)
            seen.add(identity)
    return tuple(anchors)


def _processor_inputs(processor: Any, samples: tuple[Any, ...], prefix_geometry: dict[str, Any]) -> Any:
    import torch
    from PIL import Image

    from duo_vla.prefix_geometry import apply_fixed_prefix_chat_template

    require(len(samples) == PHYSICAL_BATCH_SIZE, "processor input must use physical B8")
    conversations = []
    for index, sample in enumerate(samples):
        third_person = np.asarray(sample.observation.third_person)
        wrist = np.asarray(sample.observation.wrist)
        for camera, values in (("agentview", third_person), ("eye_in_hand", wrist)):
            require(
                values.shape == (256, 256, 3) and values.dtype == np.uint8,
                f"sample {index} camera {camera} is not uint8[256,256,3]",
            )
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.fromarray(third_person)},
                        {"type": "image", "image": Image.fromarray(wrist)},
                        {"type": "text", "text": sample.instruction},
                    ],
                }
            ]
        )
    values = apply_fixed_prefix_chat_template(
        processor,
        conversations,
        fixed_physical_prefix_width=FIXED_PHYSICAL_PREFIX_WIDTH,
        padding_side=str(prefix_geometry["tokenization"]["padding_side"]),
        expected_batch_size=PHYSICAL_BATCH_SIZE,
        images_per_prefix=2,
    )
    require(
        tuple(values["input_ids"].shape) == (PHYSICAL_BATCH_SIZE, FIXED_PHYSICAL_PREFIX_WIDTH), "bad input_ids geometry"
    )
    require(
        tuple(values["attention_mask"].shape) == (PHYSICAL_BATCH_SIZE, FIXED_PHYSICAL_PREFIX_WIDTH),
        "bad attention_mask geometry",
    )
    expected_by_instruction = {
        record["instruction"]: int(record["valid_prefix_length"])
        for record in prefix_geometry["instruction_inventory"]["records"]
    }
    expected = tuple(expected_by_instruction[sample.instruction] for sample in samples)
    observed = tuple(int(value) for value in values["attention_mask"].bool().sum(dim=1).tolist())
    require(observed == expected, f"processor prefix lengths differ: expected={expected}, observed={observed}")
    require(values["attention_mask"].dtype in (torch.int64, torch.int32, torch.bool), "attention mask dtype is invalid")
    return values


def _metric_region(prediction: Any, target: Any) -> dict[str, Any]:
    import torch

    predicted = prediction.detach().cpu().to(torch.float64).reshape(-1)
    expected = target.detach().cpu().to(torch.float64).reshape(-1)
    require(predicted.numel() == expected.numel() and predicted.numel() > 0, "metric region is empty or mismatched")
    error = predicted - expected
    squared_error_sum = float(error.square().sum().item())
    absolute_error_sum = float(error.abs().sum().item())
    count = int(error.numel())
    return {
        "absolute_error_sum": absolute_error_sum,
        "count": count,
        "mae": absolute_error_sum / count,
        "mse": squared_error_sum / count,
        "squared_error_sum": squared_error_sum,
    }


def _common_action_metrics(prediction: Any, target: Any, valid_mask: Any) -> dict[str, Any]:
    final = prediction.detach().cpu().float()
    clean = target.detach().cpu().float()
    valid = valid_mask.detach().cpu().bool()
    require(final.shape == clean.shape == (PHYSICAL_BATCH_SIZE, ACTION_HORIZON, ACTION_DIM), "bad action metric shape")
    require(valid.shape == final.shape[:2] and bool(valid.any()), "bad action metric mask")
    overall = _metric_region(final[valid], clean[valid])
    continuous = _metric_region(final[..., :6][valid], clean[..., :6][valid])
    predicted_gripper = final[..., 6][valid]
    target_gripper = clean[..., 6][valid]
    require(
        bool(((target_gripper == -1) | (target_gripper == 1)).all()), "clean normalized gripper is not exactly {-1,+1}"
    )
    gripper = _metric_region(predicted_gripper, target_gripper)
    ties = predicted_gripper == 0
    correct = ((predicted_gripper > 0) & (target_gripper == 1)) | ((predicted_gripper < 0) & (target_gripper == -1))
    gripper.update(
        {
            "sign_accuracy": float(correct.float().mean().item()),
            "sign_correct": int(correct.sum().item()),
            "sign_rule": "pred>0 => +1; pred<0 => -1; exact-zero tie is always incorrect",
            "zero_tie_count": int(ties.sum().item()),
        }
    )
    per_sample = []
    for index in range(PHYSICAL_BATCH_SIZE):
        row_valid = valid[index]
        per_sample.append(
            {
                "continuous": _metric_region(final[index, :, :6][row_valid], clean[index, :, :6][row_valid]),
                "gripper": _metric_region(final[index, :, 6][row_valid], clean[index, :, 6][row_valid]),
                "overall": _metric_region(final[index][row_valid], clean[index][row_valid]),
                "sample_index": index,
                "valid_action_positions": int(row_valid.sum().item()),
            }
        )
    return {
        "accumulation": "CPU float64 sums over selected float32 prediction/target values",
        "continuous_dimensions": [0, 1, 2, 3, 4, 5],
        "continuous": continuous,
        "gripper_dimension": 6,
        "gripper": gripper,
        "mask_broadcast": "bool[B,H] expanded across selected action dimensions",
        "overall": overall,
        "per_sample": per_sample,
    }


def _cuda_timed(device: Any, function: Any) -> tuple[Any, float]:
    import torch

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - started


def _cuda_memory(device: Any) -> dict[str, int]:
    import torch

    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _process_memory() -> dict[str, int]:
    status: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, value, unit = line.split()
            require(unit == "kB", "unexpected /proc memory unit")
            status[name.removesuffix(":").lower() + "_bytes"] = int(value) * 1024
    status["ru_maxrss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return status


def _runtime_environment(project_root: Path) -> dict[str, Any]:
    forbidden = ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONINSPECT", "PYTHONSTARTUP")
    present_forbidden = [name for name in forbidden if os.environ.get(name)]
    present_algorithm_overrides = sorted(
        name
        for name in os.environ
        if name.startswith(_ALGORITHM_ENVIRONMENT_PREFIXES) and name not in _ALLOWED_ALGORITHM_ENVIRONMENT
    )
    require(not present_forbidden, f"forbidden environment values are set: {present_forbidden}")
    require(
        not present_algorithm_overrides, f"unpinned algorithm environment values are set: {present_algorithm_overrides}"
    )
    observed = {name: os.environ.get(name) for name in REQUIRED_ENVIRONMENT}
    require(observed == REQUIRED_ENVIRONMENT, f"offline evaluator environment differs: {observed}")
    expected_pythonpath = str((project_root / "src").resolve())
    require(os.environ.get("PYTHONPATH") == expected_pythonpath, f"PYTHONPATH must be exactly {expected_pythonpath}")
    packages = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    require(packages == EXPECTED_PACKAGES, f"package versions differ: {packages}")
    require(sys.version_info[:2] == (3, 11), f"Python 3.11 is required, found {sys.version.split()[0]}")
    lock_sha256 = _file_sha256(project_root / "uv.lock")
    require(lock_sha256 == EXPECTED_LOCK_SHA256, f"uv.lock hash differs: {lock_sha256}")
    return {
        "environment": {**observed, "PYTHONPATH": expected_pythonpath},
        "lock_sha256": lock_sha256,
        "packages": packages,
        "python": sys.version.split()[0],
    }


def _snapshot_roots(dataset_snapshot: Path) -> tuple[Path, Path, Path, Path]:
    hf_home = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    model_snapshot = hf_home / "hub/models--google--diffusiongemma-26B-A4B-it/snapshots" / MODEL_REVISION
    dataset_tree = dataset_snapshot.parents[1] / "trees" / f"{DATASET_REVISION}.json"
    model_tree = model_snapshot.parents[1] / "trees" / f"{MODEL_REVISION}.json"
    return model_snapshot, dataset_tree, model_tree, hf_home


def _authenticate_snapshots_distributed(
    dataset_snapshot: Path,
    model_snapshot: Path,
    *,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    import torch.distributed as dist

    holder: list[dict[str, Any] | None] = [None]
    if rank == 0:
        started = time.perf_counter()
        try:
            from duo_vla.hf_snapshot import verify_huggingface_snapshot

            dataset_report = verify_huggingface_snapshot(dataset_snapshot, expected_revision=DATASET_REVISION)
            model_report = verify_huggingface_snapshot(model_snapshot, expected_revision=MODEL_REVISION)
            holder[0] = {
                "dataset": dataset_report,
                "model": model_report,
                "seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            holder[0] = {"error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(holder, src=0)
    payload = holder[0]
    require(isinstance(payload, dict), "rank 0 did not broadcast snapshot authentication")
    require("error" not in payload, f"snapshot authentication failed: {payload.get('error')}")
    return payload["dataset"], payload["model"], float(payload["seconds"])


def _checkpoint_inputs(
    checkpoint: Path,
    *,
    project_root: Path,
    dataset_report: dict[str, Any],
    model_report: dict[str, Any],
    expected_objective: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    from duo_vla.checkpointing import load_checkpoint_manifest
    from duo_vla.policy_contract import validate_manifest_policy_contract
    from duo_vla.run_config import canonical_config_sha256, load_verified_resolved_config

    manifest = load_checkpoint_manifest(checkpoint, verify_hashes=True)
    require(manifest.get("kind") == "resumable-libero-training", "checkpoint is not strict resumable LIBERO")
    require(
        manifest.get("model_id") == MODEL_ID and manifest.get("model_revision") == MODEL_REVISION, "model pin mismatch"
    )
    require(
        manifest.get("dataset_id") == DATASET_ID and manifest.get("dataset_revision") == DATASET_REVISION,
        "dataset pin mismatch",
    )
    require(manifest.get("run_seed") == 0, "this initial comparison is pinned to the completed seed-0 checkpoints")
    trainer_state = manifest.get("trainer_state")
    require(
        isinstance(trainer_state, dict)
        and trainer_state.get("next_update") == 10
        and trainer_state.get("examples_seen") == 640,
        "checkpoint is not the completed update-10/global-64 state",
    )
    require(manifest.get("physical_batch_size") == PHYSICAL_BATCH_SIZE, "checkpoint is not physical B8")
    require(
        manifest.get("fixed_physical_prefix_width") == FIXED_PHYSICAL_PREFIX_WIDTH,
        "checkpoint is not fixed P545",
    )
    require(manifest.get("dataset_tree_sha256") == dataset_report["tree_metadata_sha256"], "dataset tree pin mismatch")
    require(manifest.get("model_tree_sha256") == model_report["tree_metadata_sha256"], "model tree pin mismatch")
    require(
        manifest.get("model_content_inventory_sha256") == model_report["content_inventory_sha256"],
        "model content inventory pin mismatch",
    )
    resolved_entry = manifest["artifacts"]["resolved_config"]
    resolved_path = (checkpoint / resolved_entry["path"]).resolve()
    require(resolved_path.is_relative_to(checkpoint), "resolved config escapes checkpoint")
    config, config_sha256 = load_verified_resolved_config(
        resolved_path,
        expected_sha256=str(manifest["config_sha256"]),
    )
    require(config.get("protocol") == PROTOCOL, "checkpoint protocol mismatch")
    require(config.get("source_tree_sha256") == manifest.get("source_tree_sha256"), "source pins disagree")
    live_source_sha256 = _training_source_tree_sha256(project_root)
    require(live_source_sha256 == manifest.get("source_tree_sha256"), "live training source differs from checkpoint")
    require(
        config.get("artifact_trees", {}).get("dataset_tree_sha256") == dataset_report["tree_metadata_sha256"],
        "config dataset tree mismatch",
    )
    require(
        config.get("artifact_trees", {}).get("model_tree_sha256") == model_report["tree_metadata_sha256"],
        "config model tree mismatch",
    )
    require(
        config.get("artifact_trees", {}).get("model_content_inventory_sha256")
        == model_report["content_inventory_sha256"],
        "config model inventory mismatch",
    )
    require(
        config.get("optimization", {}).get("physical_batch_size") == PHYSICAL_BATCH_SIZE,
        "config physical batch mismatch",
    )
    require(config.get("optimization", {}).get("microbatch_size") == PHYSICAL_BATCH_SIZE, "config microbatch mismatch")
    require(
        config.get("benchmark", {}).get("fixed_physical_prefix_width") == FIXED_PHYSICAL_PREFIX_WIDTH,
        "config P mismatch",
    )
    require(canonical_config_sha256(config) == config_sha256, "config semantic hash changed")
    contract = validate_manifest_policy_contract(manifest, config)
    require(contract.objective == expected_objective, f"expected {expected_objective}, found {contract.objective}")
    require(
        contract.action_horizon == ACTION_HORIZON and contract.action_dim == ACTION_DIM, "policy action shape mismatch"
    )
    normalization_entry = manifest["artifacts"]["normalization"]
    normalization_path = (checkpoint / normalization_entry["path"]).resolve()
    prefix_entry = manifest["artifacts"]["prefix_geometry"]
    prefix_path = (checkpoint / prefix_entry["path"]).resolve()
    require(normalization_path.is_relative_to(checkpoint), "normalization artifact escapes checkpoint")
    require(prefix_path.is_relative_to(checkpoint), "prefix artifact escapes checkpoint")
    checkpoint_report = {
        "artifacts": manifest["artifacts"],
        "config_sha256": config_sha256,
        "manifest_sha256": _file_sha256(checkpoint / "manifest.json"),
        "model_content_inventory_sha256": manifest["model_content_inventory_sha256"],
        "model_tree_sha256": manifest["model_tree_sha256"],
        "normalization_sha256": manifest["normalization_sha256"],
        "path": str(checkpoint),
        "policy_contract": contract.to_dict(),
        "policy_contract_sha256": manifest["policy_contract_sha256"],
        "prefix_geometry_content_sha256": manifest["prefix_geometry_content_sha256"],
        "run_seed": manifest["run_seed"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "trainer_state": trainer_state,
    }
    return manifest, config, checkpoint_report, normalization_path, prefix_path


def _materialize_canonical_batch(
    dataset_snapshot: Path,
    normalization_path: Path,
    prefix_path: Path,
    checkpoint_manifest: dict[str, Any],
    model_report: dict[str, Any],
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    import torch

    from duo_vla.data.batching import collate_libero_samples
    from duo_vla.data.libero import LiberoParquetDataset
    from duo_vla.data.libero_stats import load_libero_normalizers
    from duo_vla.data.sampling import TaskUniformAnchorSampler
    from duo_vla.prefix_geometry import CameraGeometry, SnapshotTreeIdentity, load_prefix_geometry_contract
    from duo_vla.training import make_microbatch_plan

    state_normalizer, action_normalizer, normalization_manifest = load_libero_normalizers(
        normalization_path,
        expected_revision=DATASET_REVISION,
    )
    require(
        normalization_manifest["content_sha256"] == checkpoint_manifest["normalization_sha256"],
        "checkpoint normalization semantic hash mismatch",
    )
    dataset = LiberoParquetDataset(dataset_snapshot, max_cached_files=PHYSICAL_BATCH_SIZE)
    validation_indices = tuple(int(index) for index in normalization_manifest["split"]["validation_episode_indices"])
    require(len(validation_indices) == 168, "validation episode count changed")
    sampler = TaskUniformAnchorSampler(dataset.episodes, validation_indices)
    validation_seed = int(checkpoint_manifest["run_seed"]) ^ VALIDATION_SEED_XOR
    plan = make_microbatch_plan(validation_seed, 0, 0)
    anchors = _fixed_distinct_anchors(sampler, count=PHYSICAL_BATCH_SIZE, seed=plan.data_seed)
    samples = dataset.sample_many(anchors, horizon=ACTION_HORIZON)
    batch = collate_libero_samples(
        samples,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
    )
    raw_states = torch.stack([sample.observation.state for sample in samples]).float()
    raw_actions = torch.stack([sample.action_chunk.actions for sample in samples]).float()
    agentview = torch.from_numpy(np.stack([sample.observation.third_person for sample in samples]))
    wrist = torch.from_numpy(np.stack([sample.observation.wrist for sample in samples]))
    sample_records = []
    for position, (anchor, sample) in enumerate(zip(anchors, samples, strict=True)):
        episode = dataset.episodes[anchor.episode_index]
        sample_records.append(
            {
                "episode_index": anchor.episode_index,
                "frame_index": anchor.frame_index,
                "global_frame_index": episode.global_start + anchor.frame_index,
                "instruction": sample.instruction,
                "instruction_sha256": hashlib.sha256(sample.instruction.encode()).hexdigest(),
                "mask": sample.action_chunk.valid_mask.tolist(),
                "position": position,
                "task_index": sample.task_index,
                "valid_action_positions": int(sample.action_chunk.valid_mask.sum().item()),
            }
        )
    model_identity = SnapshotTreeIdentity.from_huggingface_report(MODEL_ID, model_report)
    prefix_geometry = load_prefix_geometry_contract(
        prefix_path,
        expected_content_sha256=str(checkpoint_manifest["prefix_geometry_content_sha256"]),
        expected_model_identity=model_identity,
        expected_processor_identity=model_identity,
        expected_ordered_cameras=(
            CameraGeometry("agentview", 256, 256),
            CameraGeometry("eye_in_hand", 256, 256),
        ),
        expected_instructions=tuple(dataset.task_by_index.values()),
        expected_fixed_physical_prefix_width=FIXED_PHYSICAL_PREFIX_WIDTH,
    )
    batch_inputs = {
        "action_mask": _tensor_record(batch.action_valid_mask),
        "agentview_uint8": _tensor_record(agentview),
        "clean_normalized_A1": _tensor_record(batch.clean_actions),
        "clean_physical_actions": _tensor_record(raw_actions),
        "normalized_states": _tensor_record(batch.states),
        "raw_states": _tensor_record(raw_states),
        "wrist_uint8": _tensor_record(wrist),
    }
    identity = {
        "anchor_algorithm": "trainer TaskUniformAnchorSampler + fixed-distinct rejection in draw order",
        "batch_inputs": batch_inputs,
        "data_seed": plan.data_seed,
        "flow_training_seed": plan.flow_seed,
        "normalization_artifact_file_sha256": _file_sha256(normalization_path),
        "normalization_content_sha256": normalization_manifest["content_sha256"],
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "prefix_geometry_artifact_file_sha256": _file_sha256(prefix_path),
        "prefix_geometry_content_sha256": prefix_geometry["content_sha256"],
        "sample_count": PHYSICAL_BATCH_SIZE,
        "sample_identity_sha256": _canonical_sha256(sample_records),
        "sample_records": sample_records,
        "validation_episode_sha256": normalization_manifest["split"]["validation_episode_sha256"],
        "validation_seed": validation_seed,
        "validation_seed_construction": "run_seed XOR 0x5A17",
    }
    return batch, action_normalizer, plan, prefix_geometry, identity


def _load_real_model(
    checkpoint: Path,
    config: dict[str, Any],
    *,
    device: Any,
) -> tuple[Any, Any, Any]:
    from transformers import AutoProcessor

    from duo_vla.action_interface import ActionInputProjector, VelocityHead
    from duo_vla.backbones.diffusion_gemma import DiffusionGemmaActionDecoder
    from duo_vla.backbones.loading import DEFAULT_DIFFUSION_GEMMA_SPEC, load_diffusion_gemma_bf16_tp
    from duo_vla.backbones.sample_isolated_experts import (
        install_sample_isolated_grouped_mm_experts,
        verify_sample_isolated_grouped_mm_experts,
    )
    from duo_vla.checkpointing import load_interface_state_dict, load_lora_checkpoint
    from duo_vla.config import ActionInterfaceConfig
    from duo_vla.modeling import DuoVLADenoiser

    processor = AutoProcessor.from_pretrained(
        DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
        revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
        local_files_only=True,
    )
    model = load_diffusion_gemma_bf16_tp(local_files_only=True, tp_size=2)
    install_sample_isolated_grouped_mm_experts(model, physical_batch_size=PHYSICAL_BATCH_SIZE)
    verify_sample_isolated_grouped_mm_experts(model, physical_batch_size=PHYSICAL_BATCH_SIZE)
    backend = DiffusionGemmaActionDecoder.from_block_diffusion_model(model)
    adapted, _ = load_lora_checkpoint(
        checkpoint,
        model,
        is_trainable=False,
        validate_decoder_contract=True,
        expected_rank=int(config["lora"]["rank"]),
    )
    verify_sample_isolated_grouped_mm_experts(model, physical_batch_size=PHYSICAL_BATCH_SIZE)
    action = config["action"]
    benchmark = config["benchmark"]
    interface_config = ActionInterfaceConfig(
        hidden_size=DEFAULT_DIFFUSION_GEMMA_SPEC.expected_hidden_size,
        state_dim=int(benchmark["state_dimension"]),
        action_horizon=int(action["horizon"]),
        action_dim=int(action["dimension"]),
        timestep_embedding_dim=int(action["timestep_embedding_dimension"]),
        timestep_scale=float(action["timestep_scale"]),
        timestep_max_period=float(action["timestep_max_period"]),
        output_init_std=float(action["output_head_initialization_std"]),
    )
    projector = ActionInputProjector(interface_config).to(device)
    head = VelocityHead(interface_config.hidden_size, interface_config.action_dim).to(device)
    load_interface_state_dict(
        checkpoint / "interface.safetensors",
        {"action_projector": projector, "velocity_head": head},
    )
    denoiser = DuoVLADenoiser(projector, backend, head).eval()
    adapted.eval()
    model.model.encoder.eval()
    return processor, model, denoiser


def _evaluate_real(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    from duo_vla.backbones.diffusion_gemma import encode_diffusion_gemma_prefix
    from duo_vla.flow import euler_sample
    from duo_vla.objectives import make_seeded_policy_training_pair
    from duo_vla.runtime_determinism import configure_strict_cuda_determinism, deterministic_torch_runtime
    from duo_vla.self_conditioning import enabled, sample_action_flow, training_velocity
    from duo_vla.training import masked_sse

    project_root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve()
    dataset_snapshot = args.dataset_snapshot.resolve()
    output = args.output.absolute()
    require(checkpoint.is_dir(), f"checkpoint does not exist: {checkpoint}")
    require(
        dataset_snapshot.name == DATASET_REVISION and dataset_snapshot.is_dir(), "dataset snapshot revision mismatch"
    )
    model_snapshot, dataset_tree, model_tree, _ = _snapshot_roots(dataset_snapshot)
    protected = [checkpoint, dataset_snapshot, model_snapshot, dataset_tree, model_tree]
    _ensure_output_is_separate(output, protected)
    runtime_preflight = _runtime_environment(project_root)
    configure_strict_cuda_determinism(torch)
    require(int(os.environ.get("WORLD_SIZE", "0")) == 2, "evaluate requires torchrun --nproc-per-node=2")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    require(dist.get_world_size() == 2, "TP world size must be 2")
    rank = dist.get_rank()
    started_total = time.perf_counter()
    before_inventory: dict[str, Any] | None = _stat_inventory(protected) if rank == 0 else None
    try:
        dataset_report, model_report, authentication_seconds = _authenticate_snapshots_distributed(
            dataset_snapshot,
            model_snapshot,
            rank=rank,
        )
        manifest, config, checkpoint_report, normalization_path, prefix_path = _checkpoint_inputs(
            checkpoint,
            project_root=project_root,
            dataset_report=dataset_report,
            model_report=model_report,
            expected_objective=args.expected_objective,
        )
        batch_started = time.perf_counter()
        batch, _, plan, prefix_geometry, input_identity = _materialize_canonical_batch(
            dataset_snapshot,
            normalization_path,
            prefix_path,
            manifest,
            model_report,
        )
        batch_seconds = time.perf_counter() - batch_started

        torch.manual_seed(2026 + int(manifest["run_seed"]))
        torch.cuda.reset_peak_memory_stats(device)
        model_started = time.perf_counter()
        processor, model, denoiser = _load_real_model(checkpoint, config, device=device)
        torch.cuda.synchronize(device)
        model_load_seconds = time.perf_counter() - model_started
        model_load_memory = _cuda_memory(device)

        processor_started = time.perf_counter()
        prefix_inputs_cpu = _processor_inputs(processor, batch.samples, prefix_geometry)
        processor_seconds = time.perf_counter() - processor_started
        processor_records = _processor_tensor_records(prefix_inputs_cpu)
        prefix_valid_lengths = [int(value) for value in prefix_inputs_cpu["attention_mask"].bool().sum(dim=1).tolist()]
        prefix_inputs = dict(prefix_inputs_cpu.to(device))
        states = batch.states.to(device)
        clean = batch.clean_actions.to(device)
        valid = batch.action_valid_mask.to(device)

        common_generator = torch.Generator(device=device).manual_seed(args.common_sampling_seed)
        initial_epsilon = torch.randn(
            (PHYSICAL_BATCH_SIZE, ACTION_HORIZON, ACTION_DIM),
            device=device,
            dtype=states.dtype,
            generator=common_generator,
        )
        initial_epsilon_record = _tensor_record(initial_epsilon)
        input_identity.update(
            {
                "common_sampling_seed": args.common_sampling_seed,
                "common_sampling_seed_scope": (
                    "one B8 epsilon tensor, cloned unchanged for flow NFE 1/5/10; unused by direct"
                ),
                "fixed_physical_prefix_width": FIXED_PHYSICAL_PREFIX_WIDTH,
                "initial_epsilon": initial_epsilon_record,
                "processor_outputs": processor_records,
                "processor_valid_prefix_lengths": prefix_valid_lengths,
            }
        )
        input_contract_sha256 = _canonical_sha256(input_identity)

        torch.cuda.reset_peak_memory_stats(device)
        prefix, prefix_seconds = _cuda_timed(
            device,
            lambda: encode_diffusion_gemma_prefix(model, prefix_inputs),
        )
        prefix_memory = _cuda_memory(device)

        contract_dict = checkpoint_report["policy_contract"]
        objective = contract_dict["objective"]
        pair = make_seeded_policy_training_pair(clean, validate_contract(manifest, config), seed=plan.flow_seed)
        objective_pair = {
            "input_actions": _tensor_record(pair.input_actions),
            "target": _tensor_record(pair.target),
            "timesteps": _tensor_record(pair.timesteps),
        }

        def objective_forward(bootstrap: bool = False) -> Any:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return training_velocity(
                    denoiser,
                    pair.input_actions,
                    pair.timesteps,
                    states,
                    prefix_cache=prefix.past_key_values,
                    prefix_attention_mask=prefix.attention_mask,
                    action_valid_mask=valid,
                    bootstrap=bootstrap,
                )

        torch.cuda.reset_peak_memory_stats(device)
        objective_prediction, objective_seconds = _cuda_timed(device, objective_forward)
        objective_memory = _cuda_memory(device)
        objective_component = masked_sse(objective_prediction, pair.target, valid)
        objective_specific = {
            "comparable_across_objectives": False,
            "definition": (
                "masked raw velocity MSE at seeded training (t,epsilon)"
                if objective == "rectified_flow"
                else "masked raw direct-regression MSE to normalized clean A1"
            ),
            "element_count": objective_component.element_count,
            "flow_training_seed": plan.flow_seed if objective == "rectified_flow" else None,
            "latency_seconds": objective_seconds,
            "masked_mse": float(objective_component.mean.detach().item()),
            "pair": objective_pair,
            "prediction": _tensor_record(objective_prediction),
            "squared_error_sum_fp32": float(objective_component.squared_error_sum.detach().item()),
            "target_semantics": contract_dict["training_target"],
        }
        if enabled(denoiser):
            candidate_prediction, candidate_seconds = _cuda_timed(device, lambda: objective_forward(True))
            candidate_component = masked_sse(candidate_prediction, pair.target, valid)
            objective_specific["bootstrap_candidate"] = {
                "masked_mse": float(candidate_component.mean.detach().item()),
                "latency_seconds": candidate_seconds,
                "definition": "same-pair detached endpoint bootstrap velocity MSE",
            }

        common_results: dict[str, Any] = {}
        common_memory: dict[str, Any] = {}
        if objective == "rectified_flow":
            for nfe in FLOW_NFE:

                def velocity(actions: Any, timesteps: Any) -> Any:
                    return denoiser(
                        actions,
                        timesteps,
                        states,
                        prefix_cache=prefix.past_key_values,
                        prefix_attention_mask=prefix.attention_mask,
                        action_valid_mask=valid,
                    )

                def sample(selected_nfe: int = nfe) -> Any:
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if enabled(denoiser):
                            return sample_action_flow(
                                denoiser,
                                initial_epsilon,
                                states,
                                num_steps=selected_nfe,
                                prefix_cache=prefix.past_key_values,
                                prefix_attention_mask=prefix.attention_mask,
                                action_valid_mask=valid,
                            )
                        return euler_sample(
                            velocity,
                            initial_noise=initial_epsilon.clone(),
                            num_steps=selected_nfe,
                        )

                torch.cuda.reset_peak_memory_stats(device)
                raw_prediction, inference_seconds = _cuda_timed(device, sample)
                common_memory[str(nfe)] = _cuda_memory(device)
                require(
                    _tensor_record(initial_epsilon)["tensor_sha256"] == initial_epsilon_record["tensor_sha256"],
                    "flow integration mutated canonical initial epsilon",
                )
                final_prediction = raw_prediction.clamp(-1.0, 1.0)
                valid_scalars = valid[..., None].expand_as(raw_prediction)
                common_results[str(nfe)] = {
                    "final_prediction": _tensor_record(final_prediction),
                    "inference_seconds": inference_seconds,
                    "initial_epsilon_sha256": initial_epsilon_record["tensor_sha256"],
                    "metrics": _common_action_metrics(final_prediction, clean, valid),
                    "nfe": nfe,
                    "preclip_fraction_valid": float((raw_prediction[valid_scalars].abs() > 1.0).float().mean().item()),
                    "raw_prediction": _tensor_record(raw_prediction),
                }
        elif objective == "direct_regression":
            raw_prediction = objective_prediction
            final_prediction = raw_prediction.clamp(-1.0, 1.0)
            valid_scalars = valid[..., None].expand_as(raw_prediction)
            common_results["direct"] = {
                "final_prediction": _tensor_record(final_prediction),
                "inference_seconds": objective_seconds,
                "initial_epsilon_sha256": None,
                "metrics": _common_action_metrics(final_prediction, clean, valid),
                "nfe": 1,
                "preclip_fraction_valid": float((raw_prediction[valid_scalars].abs() > 1.0).float().mean().item()),
                "raw_prediction": _tensor_record(raw_prediction),
                "reuses_objective_forward": True,
            }
            common_memory["direct"] = objective_memory
        else:
            raise RuntimeError(f"unsupported objective: {objective}")

        deterministic_common_results = {
            name: {key: value for key, value in result.items() if key != "inference_seconds"}
            for name, result in common_results.items()
        }
        deterministic_objective = {key: value for key, value in objective_specific.items() if key != "latency_seconds"}
        deterministic_payload = {
            "common_results": deterministic_common_results,
            "input_contract_sha256": input_contract_sha256,
            "objective_specific": deterministic_objective,
        }
        deterministic_sha256 = _canonical_sha256(deterministic_payload)
        rank_hashes: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(rank_hashes, deterministic_sha256)
        require(rank_hashes == [deterministic_sha256] * dist.get_world_size(), f"TP ranks differ: {rank_hashes}")

        total_seconds = time.perf_counter() - started_total
        rank_runtime = {
            "common_inference_memory": common_memory,
            "cuda": _cuda_memory(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "device_name": torch.cuda.get_device_name(device),
            "device_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "")),
            "local_rank": local_rank,
            "model_load_memory": model_load_memory,
            "objective_forward_memory": objective_memory,
            "prefix_encode_memory": prefix_memory,
            "process": _process_memory(),
            "rank": rank,
            "timing_seconds": {
                "batch_materialization": batch_seconds,
                "common_inference": {name: result["inference_seconds"] for name, result in common_results.items()},
                "model_load": model_load_seconds,
                "objective_forward": objective_seconds,
                "prefix_encode": prefix_seconds,
                "processor": processor_seconds,
                "total": total_seconds,
            },
        }
        rank_runtimes: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(rank_runtimes, rank_runtime)

        if rank == 0:
            after_inventory = _stat_inventory(protected)
            require(
                before_inventory == after_inventory, "protected checkpoint/data/model inputs changed during evaluation"
            )
            assert before_inventory is not None
            concrete_rank_runtimes = [value for value in rank_runtimes if value is not None]
            require(len(concrete_rank_runtimes) == dist.get_world_size(), "missing TP rank runtime report")
            for name, result in common_results.items():
                per_rank_seconds = [
                    float(value["timing_seconds"]["common_inference"][name]) for value in concrete_rank_runtimes
                ]
                result["inference_seconds_by_rank"] = per_rank_seconds
                result["inference_seconds"] = max(per_rank_seconds)
            objective_rank_seconds = [
                float(value["timing_seconds"]["objective_forward"]) for value in concrete_rank_runtimes
            ]
            objective_specific["latency_seconds_by_rank"] = objective_rank_seconds
            objective_specific["latency_seconds"] = max(objective_rank_seconds)
            report = {
                "authenticated_inputs": {
                    "dataset": {"id": DATASET_ID, **dataset_report},
                    "model": {"id": MODEL_ID, **model_report},
                },
                "canonical_input_contract": input_identity,
                "canonical_input_contract_sha256": input_contract_sha256,
                "checkpoint": checkpoint_report,
                "common_action_space": {
                    "finalization": "one final clamp to [-1,+1]; no intermediate clipping",
                    "metric": "micro-averaged masked MSE/MAE in normalized clean-A1 action space",
                    "objective_independent": True,
                    "results": common_results,
                    "target": input_identity["batch_inputs"]["clean_normalized_A1"],
                },
                "deterministic_payload_sha256": deterministic_sha256,
                "evaluator": {
                    "argv": sys.argv,
                    "command": shlex.join([sys.executable, *sys.argv]),
                    "cwd": str(Path.cwd()),
                    "file_sha256": _file_sha256(Path(__file__)),
                    "path": str(Path(__file__).resolve()),
                },
                "limitations": [
                    "one training seed (seed 0)",
                    "only update 10 (640 training examples)",
                    "one deterministic validation batch (8 chunks)",
                    "offline action reconstruction only; not a LIBERO simulator rollout or task-success estimate",
                    "latencies are one observed TP2 pass per setting, not a warmed multi-trial benchmark",
                ],
                "objective_specific_validation": objective_specific,
                "read_only_guard": {
                    "after": after_inventory,
                    "before": before_inventory,
                    "protected_paths": [str(path) for path in protected],
                    "status": "unchanged size/mtime/symlink-target inventory",
                },
                "runtime": {
                    "authentication_seconds_rank0": authentication_seconds,
                    "batch_materialization_seconds": batch_seconds,
                    "determinism": deterministic_torch_runtime(torch),
                    "model_load_seconds": model_load_seconds,
                    "platform": platform.platform(),
                    "prefix_encode_seconds": prefix_seconds,
                    "processor_seconds": processor_seconds,
                    "rank_runtime": concrete_rank_runtimes,
                    "runtime_preflight": runtime_preflight,
                    "total_seconds": total_seconds,
                    "world_size": dist.get_world_size(),
                },
                "schema": SCHEMA,
                "status": "ok",
            }
            _write_new_json(output, report)
            print(
                json.dumps(
                    {
                        "input_sha256": input_contract_sha256,
                        "objective": objective,
                        "output": str(output),
                        "status": "ok",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def validate_contract(manifest: dict[str, Any], config: dict[str, Any]) -> Any:
    from duo_vla.policy_contract import validate_manifest_policy_contract

    return validate_manifest_policy_contract(manifest, config)


def _comparison(args: argparse.Namespace) -> None:
    flow_path = args.flow_report.resolve()
    direct_path = args.direct_report.resolve()
    output = args.output.absolute()
    require(output not in (flow_path, direct_path), "comparison output must differ from source reports")
    flow = _strict_json(flow_path)
    direct = _strict_json(direct_path)
    for name, report in (("flow", flow), ("direct", direct)):
        require(report.get("schema") == SCHEMA and report.get("status") == "ok", f"{name} report is not successful")
    require(flow["checkpoint"]["policy_contract"]["objective"] == "rectified_flow", "flow report objective mismatch")
    require(
        direct["checkpoint"]["policy_contract"]["objective"] == "direct_regression", "direct report objective mismatch"
    )
    flow_input_sha = flow["canonical_input_contract_sha256"]
    direct_input_sha = direct["canonical_input_contract_sha256"]
    require(flow_input_sha == direct_input_sha, "flow/direct canonical input hashes differ")
    require(
        flow["canonical_input_contract"] == direct["canonical_input_contract"],
        "flow/direct canonical input payloads differ despite their recorded hashes",
    )
    common_checkpoint_fields = (
        "model_content_inventory_sha256",
        "model_tree_sha256",
        "normalization_sha256",
        "prefix_geometry_content_sha256",
        "run_seed",
        "source_tree_sha256",
        "trainer_state",
    )
    pins = {}
    for name in common_checkpoint_fields:
        observed_flow = flow["checkpoint"][name]
        observed_direct = direct["checkpoint"][name]
        require(observed_flow == observed_direct, f"flow/direct checkpoint pin differs: {name}")
        pins[name] = observed_flow
    flow_results = flow["common_action_space"]["results"]
    require(set(flow_results) == {"1", "5", "10"}, "flow report does not contain NFE 1/5/10")
    epsilon_hashes = {flow_results[name]["initial_epsilon_sha256"] for name in ("1", "5", "10")}
    require(len(epsilon_hashes) == 1 and None not in epsilon_hashes, "flow NFE settings did not share one epsilon")
    direct_result = direct["common_action_space"]["results"]
    require(set(direct_result) == {"direct"}, "direct report has an invalid result set")
    rows = []
    for name in ("1", "5", "10"):
        result = flow_results[name]
        rows.append(
            {
                "checkpoint_manifest_sha256": flow["checkpoint"]["manifest_sha256"],
                "continuous_mae": result["metrics"]["continuous"]["mae"],
                "continuous_mse": result["metrics"]["continuous"]["mse"],
                "gripper_mae": result["metrics"]["gripper"]["mae"],
                "gripper_mse": result["metrics"]["gripper"]["mse"],
                "gripper_sign_accuracy": result["metrics"]["gripper"]["sign_accuracy"],
                "gripper_zero_tie_count": result["metrics"]["gripper"]["zero_tie_count"],
                "inference_seconds": result["inference_seconds"],
                "label": f"flow_nfe{name}",
                "nfe": int(name),
                "overall_mae": result["metrics"]["overall"]["mae"],
                "overall_mse": result["metrics"]["overall"]["mse"],
                "prediction_sha256": result["final_prediction"]["tensor_sha256"],
                "preclip_fraction_valid": result["preclip_fraction_valid"],
            }
        )
    result = direct_result["direct"]
    rows.append(
        {
            "checkpoint_manifest_sha256": direct["checkpoint"]["manifest_sha256"],
            "continuous_mae": result["metrics"]["continuous"]["mae"],
            "continuous_mse": result["metrics"]["continuous"]["mse"],
            "gripper_mae": result["metrics"]["gripper"]["mae"],
            "gripper_mse": result["metrics"]["gripper"]["mse"],
            "gripper_sign_accuracy": result["metrics"]["gripper"]["sign_accuracy"],
            "gripper_zero_tie_count": result["metrics"]["gripper"]["zero_tie_count"],
            "inference_seconds": result["inference_seconds"],
            "label": "direct",
            "nfe": 1,
            "overall_mae": result["metrics"]["overall"]["mae"],
            "overall_mse": result["metrics"]["overall"]["mse"],
            "prediction_sha256": result["final_prediction"]["tensor_sha256"],
            "preclip_fraction_valid": result["preclip_fraction_valid"],
        }
    )
    comparison = {
        "canonical_input_contract": flow["canonical_input_contract"],
        "canonical_input_contract_sha256": flow_input_sha,
        "checkpoint_common_pins": pins,
        "checkpoint_specific": {
            "direct": direct["checkpoint"],
            "flow": flow["checkpoint"],
        },
        "common_metric": {
            "definition": flow["common_action_space"]["metric"],
            "finalization": flow["common_action_space"]["finalization"],
            "gripper_sign_rule": "pred>0 => +1; pred<0 => -1; exact-zero tie is always incorrect",
            "rows": rows,
        },
        "flow_shared_initial_epsilon": {
            "nfe": list(FLOW_NFE),
            "same_tensor_for_every_nfe": True,
            "tensor_sha256": next(iter(epsilon_hashes)),
        },
        "limitations": flow["limitations"],
        "objective_specific_validation_not_comparable": {
            "direct": direct["objective_specific_validation"],
            "explanation": (
                "Flow predicts clean-minus-noise velocity at sampled (t,epsilon); direct regression predicts clean A1. "
                "These losses are logged for checkpoint-native validation only and are excluded from the common "
                "ranking."
            ),
            "flow": flow["objective_specific_validation"],
        },
        "read_only": {
            "direct": direct["read_only_guard"],
            "flow": flow["read_only_guard"],
        },
        "schema": COMPARISON_SCHEMA,
        "source_reports": {
            "direct": {"path": str(direct_path), "sha256": _file_sha256(direct_path)},
            "flow": {"path": str(flow_path), "sha256": _file_sha256(flow_path)},
        },
        "status": "ok",
    }
    comparison["comparison_payload_sha256"] = _canonical_sha256(comparison)
    _write_new_json(output, comparison)
    print(json.dumps({"output": str(output), "status": "ok"}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="evaluate one strict checkpoint under TP2")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--dataset-snapshot", type=Path, required=True)
    evaluate.add_argument("--expected-objective", choices=("rectified_flow", "direct_regression"), required=True)
    evaluate.add_argument("--common-sampling-seed", type=int, default=DEFAULT_COMMON_SAMPLING_SEED)
    evaluate.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare", help="verify identical inputs and combine two reports")
    compare.add_argument("--flow-report", type=Path, required=True)
    compare.add_argument("--direct-report", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "evaluate":
        require(0 <= args.common_sampling_seed < 2**63, "common sampling seed must be in [0,2^63)")
        _evaluate_real(args)
    elif args.command == "compare":
        _comparison(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
