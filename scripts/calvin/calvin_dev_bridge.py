#!/usr/bin/env python3
"""Strict IPC for held-out CALVIN A/B/C development rollouts.

This protocol is intentionally disjoint from the official ABC-to-D protocol.
It contains no official sequence identity and accepts only scenes A, B, and C.
The module uses only the Python 3.8 standard library and NumPy.
"""

from __future__ import annotations

# ruff: noqa: UP006, UP007, UP035, UP041, UP045
import base64
import binascii
import hashlib
import json
import math
import os
import socket
import stat
import struct
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

import numpy as np

SCHEMA = "duovla-calvin-dev-policy-ipc-v4"
PROTOCOL = "duovla-calvin-heldout-abc-v1"
DATASET_MANIFEST_SCHEMA = "duo-vla-calvin-dataset-manifest-v4"
MEMBER_INDEX_SCHEMA = "duo-vla-calvin-member-index-v2"
ARCHIVE_READER_SCHEMA = "duo-vla-calvin-archive-reader-v1"
STORAGE_MODE = "archive-direct"
ARCHIVE_BYTES = 555_309_812_705
ARCHIVE_SHA256 = "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74"
CENTRAL_DIRECTORY_SHA256 = "b4f79bda7f6b966b51aa419badd0f7db7a8972a7b58d6d342af60aceff0ea31b"
MEMBER_INDEX_PATH = "task_ABC_D.members-v2.sqlite3"
MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
METADATA_FILES = [
    "ep_start_end_ids.npy",
    "lang_annotations/auto_lang_ann.npy",
    "scene_info.npy",
    ".hydra/merged_config.yaml",
]
INFERENCE_SEED_DOMAIN = "duo-vla-calvin-heldout-abc-inference-seed-v1"
ALLOWED_SCENES = ("calvin_scene_A", "calvin_scene_B", "calvin_scene_C")

ACTION_HORIZON = 8
ACTION_DIM = 7
STATE_DIM = 8
STATIC_IMAGE_SHAPE = (200, 200, 3)
GRIPPER_IMAGE_SHAPE = (84, 84, 3)
SUPPORTED_EXECUTION_HORIZONS = (1, 4)
MAX_FRAME_BYTES = 2 * 1024 * 1024
EXPERTS_IMPLEMENTATION = "grouped_mm"
EXPERT_BATCH_ISOLATION = "sample_isolated_grouped_mm_v1"
PHYSICAL_BATCH_SIZE = 8

_HEADER = struct.Struct(">Q")
_RGB_FIELDS = {"data", "dtype", "encoding", "sha256", "shape"}
_CONTROL_REQUEST_FIELDS = {"operation", "request_id", "schema"}
_PREDICT_REQUEST_FIELDS = {
    "development_episode",
    "evaluation_seed",
    "inference_seed",
    "instruction",
    "observation",
    "operation",
    "request_id",
    "schema",
    "train_seed",
}
_EPISODE_FIELDS = {
    "annotation_index",
    "episode_index",
    "execution_horizon",
    "global_start",
    "replan_idx",
    "reset_bank_sha256",
    "reset_id_sha256",
    "reset_index",
    "scene",
    "task",
}
_OBSERVATION_FIELDS = {"rgb_gripper", "rgb_static", "state"}
_ENVELOPE_FIELDS = {"operation", "request_id", "schema", "status"}
_ERROR_RESPONSE_FIELDS = _ENVELOPE_FIELDS | {"error"}
_ERROR_FIELDS = {"message", "type"}
_ECHO_FIELDS = {
    "annotation_index",
    "episode_index",
    "evaluation_seed",
    "execution_horizon",
    "global_start",
    "inference_seed",
    "replan_idx",
    "reset_bank_sha256",
    "reset_id_sha256",
    "reset_index",
    "scene",
    "task",
    "train_seed",
}
_PREDICT_RESPONSE_FIELDS = _ENVELOPE_FIELDS | _ECHO_FIELDS | {"actions", "policy_seconds"}
_HEALTH_RESPONSE_FIELDS = _ENVELOPE_FIELDS | {
    "action_dim",
    "action_horizon",
    "allowed_scenes",
    "calvin_identity",
    "checkpoint_manifest_sha256",
    "execution_horizons",
    "execution_geometry",
    "gripper_image_shape",
    "mode",
    "model_revision",
    "nfe",
    "normalization_content_sha256",
    "normalization_metadata_sha256",
    "objective",
    "policy_contract_sha256",
    "protocol",
    "replay_bundle_sha256",
    "reset_bank_sha256",
    "reset_count",
    "sampler",
    "split_sha256",
    "state_dim",
    "static_image_shape",
    "train_seed",
}
_CALVIN_IDENTITY_FIELDS = {
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
_CALVIN_MEMBER_INDEX_FIELDS = {"bytes", "path", "schema", "sha256"}
_EXECUTION_GEOMETRY_FIELDS = {
    "expert_batch_isolation",
    "experts_implementation",
    "fixed_physical_prefix_width",
    "physical_batch_size",
    "prefix_geometry_content_sha256",
}
_EXECUTION_TOPOLOGY_FIELDS = {"execution_profile", "tensor_parallel_size"}


class DevBridgeError(RuntimeError):
    """Base error for held-out A/B/C policy IPC."""


class DevBridgeProtocolError(DevBridgeError):
    """A peer violated the exact development wire schema."""


class DevBridgeTimeoutError(DevBridgeError):
    """A development policy exchange exceeded its deadline."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DevBridgeProtocolError(message)


def _exact_keys(value: Mapping[str, Any], expected: Set[str], name: str) -> None:
    observed = set(value)
    require(
        observed == expected,
        f"{name} fields differ: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}",
    )


def _valid_integer(value: Any, minimum: int = 0, maximum: Optional[int] = None) -> bool:
    if type(value) is not int or value < minimum:
        return False
    return maximum is None or value < maximum


def _valid_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_sha256(value: Any, name: str) -> None:
    require(
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{name} must be 64 lowercase hexadecimal characters",
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise DevBridgeProtocolError("value is not finite canonical JSON") from exc


def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result = {}  # type: Dict[str, Any]
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise DevBridgeProtocolError(f"non-finite JSON constant: {value}")


def dev_replan_seed(
    evaluation_seed: int,
    reset_bank_sha256: str,
    reset_id_sha256: str,
    task: str,
    replan_idx: int,
) -> int:
    """Derive common development noise without train seed, objective, NFE, or K."""

    require(_valid_integer(evaluation_seed, 0, 2**63), "evaluation_seed must be in [0,2^63)")
    _validate_sha256(reset_bank_sha256, "reset_bank_sha256")
    _validate_sha256(reset_id_sha256, "reset_id_sha256")
    require(isinstance(task, str) and 0 < len(task) <= 200, "task must be non-empty text")
    require(_valid_integer(replan_idx), "replan_idx must be nonnegative")
    identity = [
        INFERENCE_SEED_DOMAIN,
        evaluation_seed,
        reset_bank_sha256,
        reset_id_sha256,
        task,
        replan_idx,
    ]
    digest = hashlib.blake2b(_canonical_json(identity).encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _encode_rgb(image: np.ndarray, shape: Tuple[int, int, int], name: str) -> Dict[str, Any]:
    value = np.asarray(image)
    require(value.shape == shape, f"{name} must have shape {shape}")
    require(value.dtype == np.uint8, f"{name} must have dtype uint8")
    raw = np.ascontiguousarray(value).tobytes(order="C")
    return {
        "data": base64.b64encode(raw).decode("ascii"),
        "dtype": "uint8",
        "encoding": "base64",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(shape),
    }


def _decode_rgb(value: Any, shape: Tuple[int, int, int], name: str) -> np.ndarray:
    require(isinstance(value, dict), f"{name} must be an encoded RGB object")
    _exact_keys(value, _RGB_FIELDS, name)
    require(value["shape"] == list(shape), f"{name} shape mismatch")
    require(value["dtype"] == "uint8" and value["encoding"] == "base64", f"{name} encoding mismatch")
    require(isinstance(value["data"], str), f"{name} data must be text")
    _validate_sha256(value["sha256"], f"{name} sha256")
    try:
        raw = base64.b64decode(value["data"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DevBridgeProtocolError(f"{name} contains invalid base64") from exc
    require(len(raw) == int(np.prod(shape)), f"{name} byte length mismatch")
    require(hashlib.sha256(raw).hexdigest() == value["sha256"], f"{name} SHA-256 mismatch")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy(order="C")


def encode_static_rgb(image: np.ndarray) -> Dict[str, Any]:
    return _encode_rgb(image, STATIC_IMAGE_SHAPE, "rgb_static")


def encode_gripper_rgb(image: np.ndarray) -> Dict[str, Any]:
    return _encode_rgb(image, GRIPPER_IMAGE_SHAPE, "rgb_gripper")


def decode_static_rgb(value: Any) -> np.ndarray:
    return _decode_rgb(value, STATIC_IMAGE_SHAPE, "rgb_static")


def decode_gripper_rgb(value: Any) -> np.ndarray:
    return _decode_rgb(value, GRIPPER_IMAGE_SHAPE, "rgb_gripper")


def _state_array(value: Any, wire: bool) -> np.ndarray:
    if wire:
        require(isinstance(value, list) and len(value) == STATE_DIM, "state must contain 8 values")
        require(all(_valid_number(item) for item in value), "state values must be finite")
    else:
        require(isinstance(value, np.ndarray), "state must be a numpy array")
        require(value.shape == (STATE_DIM,) and value.dtype == np.float32, "state must be float32[8]")
    try:
        result = np.asarray(value, dtype=np.float32)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DevBridgeProtocolError("state cannot be represented as float32") from exc
    require(result.shape == (STATE_DIM,) and bool(np.isfinite(result).all()), "state is invalid")
    require(result[7] in (-1.0, 1.0), "state gripper must be exactly {-1,+1}")
    return result.copy(order="C")


def encode_state(value: np.ndarray) -> List[float]:
    return _state_array(value, wire=False).tolist()


def decode_state(value: Any) -> np.ndarray:
    return _state_array(value, wire=True)


def _actions_array(value: Any, wire: bool) -> np.ndarray:
    if wire:
        require(isinstance(value, list) and len(value) == ACTION_HORIZON, "actions must contain 8 rows")
        require(
            all(
                isinstance(row, list) and len(row) == ACTION_DIM and all(_valid_number(item) for item in row)
                for row in value
            ),
            "actions must be a finite 8x7 matrix",
        )
    else:
        require(isinstance(value, np.ndarray), "actions must be a numpy array")
        require(
            value.shape == (ACTION_HORIZON, ACTION_DIM) and value.dtype == np.float32, "actions must be float32[8,7]"
        )
    try:
        result = np.asarray(value, dtype=np.float32)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DevBridgeProtocolError("actions cannot be represented as float32") from exc
    require(result.shape == (ACTION_HORIZON, ACTION_DIM) and bool(np.isfinite(result).all()), "actions are invalid")
    require(bool(np.isin(result[:, 6], (-1.0, 1.0)).all()), "action gripper must be exactly {-1,+1}")
    return result.copy(order="C")


def encode_actions(value: np.ndarray) -> List[List[float]]:
    return _actions_array(value, wire=False).tolist()


def decode_actions(value: Any) -> np.ndarray:
    return _actions_array(value, wire=True)


def _validate_request_id(value: Any) -> None:
    require(isinstance(value, str) and 0 < len(value) <= 160, "request_id is invalid")


def make_control_request(operation: str, request_id: str) -> Dict[str, Any]:
    require(operation in ("health", "shutdown"), "invalid control operation")
    _validate_request_id(request_id)
    return {"operation": operation, "request_id": request_id, "schema": SCHEMA}


def _validate_episode(episode: Any) -> Dict[str, Any]:
    require(isinstance(episode, dict), "development_episode must be an object")
    _exact_keys(episode, _EPISODE_FIELDS, "development_episode")
    require(episode["scene"] in ALLOWED_SCENES, "development scene must be one of A/B/C")
    for name in ("annotation_index", "episode_index", "global_start", "replan_idx", "reset_index"):
        require(_valid_integer(episode[name]), f"{name} must be nonnegative")
    require(
        _valid_integer(episode["execution_horizon"], 1)
        and episode["execution_horizon"] in SUPPORTED_EXECUTION_HORIZONS,
        "execution_horizon must be one of {1,4}",
    )
    for name in ("reset_bank_sha256", "reset_id_sha256"):
        _validate_sha256(episode[name], name)
    require(isinstance(episode["task"], str) and 0 < len(episode["task"]) <= 200, "task is invalid")
    return dict(episode)


def make_predict_request(
    request_id: str,
    evaluation_seed: int,
    train_seed: int,
    reset_bank_sha256: str,
    reset_id_sha256: str,
    reset_index: int,
    scene: str,
    episode_index: int,
    annotation_index: int,
    global_start: int,
    task: str,
    replan_idx: int,
    execution_horizon: int,
    instruction: str,
    rgb_static: np.ndarray,
    rgb_gripper: np.ndarray,
    state: np.ndarray,
) -> Dict[str, Any]:
    _validate_request_id(request_id)
    episode = _validate_episode(
        {
            "annotation_index": annotation_index,
            "episode_index": episode_index,
            "execution_horizon": execution_horizon,
            "global_start": global_start,
            "replan_idx": replan_idx,
            "reset_bank_sha256": reset_bank_sha256,
            "reset_id_sha256": reset_id_sha256,
            "reset_index": reset_index,
            "scene": scene,
            "task": task,
        }
    )
    require(_valid_integer(train_seed, 0, 2**63), "train_seed must be in [0,2^63)")
    require(isinstance(instruction, str) and 0 < len(instruction) <= 1000, "instruction is invalid")
    inference_seed = dev_replan_seed(evaluation_seed, reset_bank_sha256, reset_id_sha256, task, replan_idx)
    return {
        "development_episode": episode,
        "evaluation_seed": evaluation_seed,
        "inference_seed": inference_seed,
        "instruction": instruction,
        "observation": {
            "rgb_gripper": encode_gripper_rgb(rgb_gripper),
            "rgb_static": encode_static_rgb(rgb_static),
            "state": encode_state(state),
        },
        "operation": "predict",
        "request_id": request_id,
        "schema": SCHEMA,
        "train_seed": train_seed,
    }


def validate_request(value: Any) -> Dict[str, Any]:
    require(isinstance(value, dict), "request must be an object")
    require(value.get("schema") == SCHEMA, "development request schema mismatch")
    operation = value.get("operation")
    require(operation in ("health", "predict", "shutdown"), "unknown operation")
    _validate_request_id(value.get("request_id"))
    if operation != "predict":
        _exact_keys(value, _CONTROL_REQUEST_FIELDS, "control request")
        return dict(value)
    _exact_keys(value, _PREDICT_REQUEST_FIELDS, "predict request")
    episode = _validate_episode(value["development_episode"])
    observation = value["observation"]
    require(isinstance(observation, dict), "observation must be an object")
    # Exact keys make it structurally impossible to forward reset-only scene_obs.
    _exact_keys(observation, _OBSERVATION_FIELDS, "observation")
    expected_seed = dev_replan_seed(
        value["evaluation_seed"],
        episode["reset_bank_sha256"],
        episode["reset_id_sha256"],
        episode["task"],
        episode["replan_idx"],
    )
    require(
        type(value["inference_seed"]) is int and value["inference_seed"] == expected_seed, "inference seed mismatch"
    )
    require(_valid_integer(value["train_seed"], 0, 2**63), "train_seed is invalid")
    require(isinstance(value["instruction"], str) and 0 < len(value["instruction"]) <= 1000, "instruction is invalid")
    result = dict(value)
    result["development_episode"] = episode
    result["observation"] = {
        "rgb_gripper": decode_gripper_rgb(observation["rgb_gripper"]),
        "rgb_static": decode_static_rgb(observation["rgb_static"]),
        "state": decode_state(observation["state"]),
    }
    return result


def _prediction_echo(request: Mapping[str, Any]) -> Dict[str, Any]:
    episode = request["development_episode"]
    return {
        "annotation_index": episode["annotation_index"],
        "episode_index": episode["episode_index"],
        "evaluation_seed": request["evaluation_seed"],
        "execution_horizon": episode["execution_horizon"],
        "global_start": episode["global_start"],
        "inference_seed": request["inference_seed"],
        "replan_idx": episode["replan_idx"],
        "reset_bank_sha256": episode["reset_bank_sha256"],
        "reset_id_sha256": episode["reset_id_sha256"],
        "reset_index": episode["reset_index"],
        "scene": episode["scene"],
        "task": episode["task"],
        "train_seed": request["train_seed"],
    }


def make_success_response(request: Mapping[str, Any], **payload: Any) -> Dict[str, Any]:
    require(not _ENVELOPE_FIELDS.intersection(payload), "payload overrides response envelope")
    response = {
        "operation": request["operation"],
        "request_id": request["request_id"],
        "schema": SCHEMA,
        "status": "ok",
    }
    if request["operation"] == "predict":
        echo = _prediction_echo(request)
        for name, expected in echo.items():
            require(name not in payload or payload[name] == expected, f"payload changed echoed {name}")
        response.update(echo)
    response.update(payload)
    return response


def _validate_execution_geometry(value: Any, *, allow_none: bool) -> None:
    if value is None and allow_none:
        return
    require(isinstance(value, Mapping), "execution geometry must be an object")
    if value.get("expert_batch_isolation") == "sample_isolated_grouped_mm_v2":
        expected_fields = (
            _EXECUTION_GEOMETRY_FIELDS
            | _EXECUTION_TOPOLOGY_FIELDS
            | {
                "serving_batch_size",
                "data_parallel_size",
                "global_batch_size",
            }
        )
        require(set(value) == expected_fields, "optimized execution geometry fields differ")
        require(value["experts_implementation"] == "grouped_mm", "optimized expert backend mismatch")
        for key in (
            "physical_batch_size",
            "serving_batch_size",
            "data_parallel_size",
            "global_batch_size",
            "tensor_parallel_size",
            "fixed_physical_prefix_width",
        ):
            require(type(value[key]) is int, "optimized geometry sizes must be plain integers")
        batch, dp = value["physical_batch_size"], value["data_parallel_size"]
        require(batch in (8, 16, 32, 64) and dp in (1, 2), "optimized training batch/DP size differs")
        require(
            value["serving_batch_size"] == 8
            and value["global_batch_size"] == 64
            and value["tensor_parallel_size"] == 1,
            "optimized serving requires TP1/B8, training global B64",
        )
        require(dp != 2 or batch == 32, "optimized DP2 requires rank B32")
        expected_profile = (
            "duovla-calvin-dp2-tp1-fused-v2-train-b32-serve-b8-v1"
            if dp == 2
            else f"duovla-calvin-tp1-fused-v2-train-b{batch}-serve-b8-v1"
        )
        require(value["execution_profile"] == expected_profile, "optimized execution profile differs")
        require(0 < value["fixed_physical_prefix_width"] <= 1024 - ACTION_HORIZON, "invalid prefix width")
        _validate_sha256(value["prefix_geometry_content_sha256"], "prefix_geometry_content_sha256")
        return
    fields = set(value)
    require(
        fields in (_EXECUTION_GEOMETRY_FIELDS, _EXECUTION_GEOMETRY_FIELDS | _EXECUTION_TOPOLOGY_FIELDS),
        "execution geometry fields differ",
    )
    require(value["experts_implementation"] == EXPERTS_IMPLEMENTATION, "expert backend mismatch")
    require(value["expert_batch_isolation"] == EXPERT_BATCH_ISOLATION, "expert isolation mismatch")
    require(
        type(value["physical_batch_size"]) is int and value["physical_batch_size"] == PHYSICAL_BATCH_SIZE,
        "physical batch size must equal eight",
    )
    require(
        type(value["fixed_physical_prefix_width"]) is int
        and 0 < value["fixed_physical_prefix_width"] <= 1024 - ACTION_HORIZON,
        "fixed physical prefix width is invalid",
    )
    _validate_sha256(value["prefix_geometry_content_sha256"], "prefix_geometry_content_sha256")
    if fields != _EXECUTION_GEOMETRY_FIELDS:
        require(value["tensor_parallel_size"] in {1, 2}, "tensor parallel size must be one or two")
        expected_profile = "duovla-single-gpu-tp1-v1" if value["tensor_parallel_size"] == 1 else "duovla-tp2-v1"
        require(value["execution_profile"] == expected_profile, "execution profile differs from tensor topology")


def _validate_calvin_identity(value: Any) -> Dict[str, Any]:
    require(isinstance(value, Mapping), "CALVIN health identity must be an object")
    _exact_keys(value, _CALVIN_IDENTITY_FIELDS, "CALVIN health identity")
    require(value["name"] == "task_ABC_D" and value["split"] == "training", "CALVIN dataset/split differs")
    require(value["archive_bytes"] == ARCHIVE_BYTES, "CALVIN archive byte count differs")
    require(value["archive_sha256"] == ARCHIVE_SHA256, "CALVIN archive SHA-256 differs")
    require(
        value["central_directory_sha256"] == CENTRAL_DIRECTORY_SHA256,
        "CALVIN central-directory SHA-256 differs",
    )
    for name in (
        "dataset_manifest_file_sha256",
        "dataset_manifest_sha256",
        "member_inventory_sha256",
        "metadata_sha256",
        "storage_identity_sha256",
    ):
        _validate_sha256(value[name], "CALVIN health " + name)
    require(value["dataset_manifest_schema"] == DATASET_MANIFEST_SCHEMA, "dataset manifest schema is not v4")
    require(value["reader_schema"] == ARCHIVE_READER_SCHEMA, "archive reader schema differs")
    require(value["storage_mode"] == STORAGE_MODE, "development storage mode is not archive-direct")
    require(value["metadata_files"] == METADATA_FILES, "CALVIN metadata inventory differs")
    member_index = value["member_index"]
    require(isinstance(member_index, Mapping), "CALVIN member-index identity must be an object")
    _exact_keys(member_index, _CALVIN_MEMBER_INDEX_FIELDS, "CALVIN member-index identity")
    require(
        type(member_index["bytes"]) is int and member_index["bytes"] > 0,
        "CALVIN member-index byte count is invalid",
    )
    require(member_index["path"] == MEMBER_INDEX_PATH, "CALVIN member-index path differs")
    require(member_index["schema"] == MEMBER_INDEX_SCHEMA, "CALVIN member-index schema is not v2")
    _validate_sha256(member_index["sha256"], "CALVIN member-index SHA-256")
    result = dict(value)
    result["member_index"] = dict(member_index)
    result["metadata_files"] = list(value["metadata_files"])
    return result


def _validate_policy_identity(
    mode: str,
    objective: str,
    sampler: str,
    nfe: int,
    checkpoint_manifest_sha256: Optional[str],
    policy_contract_sha256: Optional[str],
    normalization_content_sha256: Optional[str],
    normalization_metadata_sha256: Optional[str],
    model_revision: Optional[str],
    execution_geometry: Optional[Mapping[str, Any]],
) -> None:
    require(mode in ("fake", "real"), "health mode must be fake or real")
    if mode == "fake":
        require(
            objective == "test_fake" and sampler == "seeded_test_normal" and nfe == 0, "fake policy identity mismatch"
        )
        require(
            checkpoint_manifest_sha256 is None
            and policy_contract_sha256 is None
            and normalization_content_sha256 is None
            and normalization_metadata_sha256 is None
            and model_revision is None
            and execution_geometry is None,
            "fake policy cannot claim real artifact identity",
        )
        return
    require(objective in ("rectified_flow", "direct_regression"), "real policy objective is invalid")
    if objective == "rectified_flow":
        require(sampler == "euler_uniform" and nfe in (1, 5, 10), "flow sampler identity mismatch")
    else:
        require(sampler == "single_forward" and nfe == 1, "direct sampler identity mismatch")
    for name, value in (
        ("checkpoint_manifest_sha256", checkpoint_manifest_sha256),
        ("policy_contract_sha256", policy_contract_sha256),
        ("normalization_content_sha256", normalization_content_sha256),
        ("normalization_metadata_sha256", normalization_metadata_sha256),
    ):
        _validate_sha256(value, name)
    require(model_revision == MODEL_REVISION, "policy model revision mismatch")
    _validate_execution_geometry(execution_geometry, allow_none=False)


def make_health_response(
    request: Mapping[str, Any],
    train_seed: int,
    calvin_identity: Mapping[str, Any],
    reset_bank_sha256: str,
    replay_bundle_sha256: str,
    split_sha256: str,
    reset_count: int,
    *,
    mode: str,
    objective: str,
    sampler: str,
    nfe: int,
    checkpoint_manifest_sha256: Optional[str],
    policy_contract_sha256: Optional[str],
    normalization_content_sha256: Optional[str],
    normalization_metadata_sha256: Optional[str],
    model_revision: Optional[str],
    execution_geometry: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    require(request.get("operation") == "health", "health response requires a health request")
    require(_valid_integer(train_seed, 0, 2**63), "train_seed is invalid")
    canonical_calvin_identity = _validate_calvin_identity(calvin_identity)
    for name, value in (
        ("reset_bank_sha256", reset_bank_sha256),
        ("replay_bundle_sha256", replay_bundle_sha256),
        ("split_sha256", split_sha256),
    ):
        _validate_sha256(value, name)
    require(_valid_integer(reset_count, 1), "reset_count must be positive")
    _validate_policy_identity(
        mode,
        objective,
        sampler,
        nfe,
        checkpoint_manifest_sha256,
        policy_contract_sha256,
        normalization_content_sha256,
        normalization_metadata_sha256,
        model_revision,
        execution_geometry,
    )
    if mode == "real":
        require(
            normalization_metadata_sha256 == canonical_calvin_identity["metadata_sha256"],
            "normalization metadata differs from the CALVIN dataset identity",
        )
    return make_success_response(
        request,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        allowed_scenes=list(ALLOWED_SCENES),
        calvin_identity=canonical_calvin_identity,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        execution_horizons=list(SUPPORTED_EXECUTION_HORIZONS),
        execution_geometry=None if execution_geometry is None else dict(execution_geometry),
        gripper_image_shape=list(GRIPPER_IMAGE_SHAPE),
        mode=mode,
        model_revision=model_revision,
        nfe=nfe,
        normalization_content_sha256=normalization_content_sha256,
        normalization_metadata_sha256=normalization_metadata_sha256,
        objective=objective,
        policy_contract_sha256=policy_contract_sha256,
        protocol=PROTOCOL,
        replay_bundle_sha256=replay_bundle_sha256,
        reset_bank_sha256=reset_bank_sha256,
        reset_count=reset_count,
        sampler=sampler,
        split_sha256=split_sha256,
        state_dim=STATE_DIM,
        static_image_shape=list(STATIC_IMAGE_SHAPE),
        train_seed=train_seed,
    )


def make_predict_response(request: Mapping[str, Any], actions: np.ndarray, policy_seconds: float) -> Dict[str, Any]:
    require(request.get("operation") == "predict", "prediction response requires predict request")
    require(_valid_number(policy_seconds) and policy_seconds >= 0, "policy_seconds must be finite and nonnegative")
    return make_success_response(request, actions=encode_actions(actions), policy_seconds=float(policy_seconds))


def make_error_response(request: Optional[Mapping[str, Any]], error: Exception) -> Dict[str, Any]:
    request_id = request.get("request_id", "invalid-request") if isinstance(request, Mapping) else "invalid-request"
    operation = request.get("operation", "error") if isinstance(request, Mapping) else "error"
    return {
        "error": {"message": str(error)[:1000], "type": type(error).__name__},
        "operation": operation,
        "request_id": request_id,
        "schema": SCHEMA,
        "status": "error",
    }


def _validate_envelope(response: Any, request_id: str, operation: str) -> Dict[str, Any]:
    require(isinstance(response, dict), "response must be an object")
    require(response.get("schema") == SCHEMA, "response schema mismatch")
    require(
        response.get("request_id") == request_id and response.get("operation") == operation,
        "response identity mismatch",
    )
    if response.get("status") == "error":
        _exact_keys(response, _ERROR_RESPONSE_FIELDS, "error response")
        error = response.get("error")
        require(isinstance(error, dict), "error response payload must be an object")
        _exact_keys(error, _ERROR_FIELDS, "error response payload")
        require(isinstance(error["message"], str) and bool(error["message"]), "error message must be text")
        require(isinstance(error["type"], str) and bool(error["type"]), "error type must be text")
        raise DevBridgeError("development policy error: {}".format(error["message"]))
    require(response.get("status") == "ok", "response status is invalid")
    return response


def validate_health_response(response: Any, request_id: str) -> Dict[str, Any]:
    value = _validate_envelope(response, request_id, "health")
    _exact_keys(value, _HEALTH_RESPONSE_FIELDS, "health response")
    require(value["protocol"] == PROTOCOL, "development protocol mismatch")
    require(value["allowed_scenes"] == list(ALLOWED_SCENES), "development allowed-scene set mismatch")
    require(value["action_horizon"] == ACTION_HORIZON and value["action_dim"] == ACTION_DIM, "action shape mismatch")
    require(value["state_dim"] == STATE_DIM, "state dimension mismatch")
    require(value["static_image_shape"] == list(STATIC_IMAGE_SHAPE), "static image shape mismatch")
    require(value["gripper_image_shape"] == list(GRIPPER_IMAGE_SHAPE), "gripper image shape mismatch")
    require(value["execution_horizons"] == list(SUPPORTED_EXECUTION_HORIZONS), "execution horizon set mismatch")
    require(_valid_integer(value["train_seed"], 0, 2**63), "health train_seed is invalid")
    require(_valid_integer(value["reset_count"], 1), "health reset_count is invalid")
    canonical_calvin_identity = _validate_calvin_identity(value["calvin_identity"])
    for name in ("replay_bundle_sha256", "reset_bank_sha256", "split_sha256"):
        _validate_sha256(value[name], f"health {name}")
    _validate_policy_identity(
        value["mode"],
        value["objective"],
        value["sampler"],
        value["nfe"],
        value["checkpoint_manifest_sha256"],
        value["policy_contract_sha256"],
        value["normalization_content_sha256"],
        value["normalization_metadata_sha256"],
        value["model_revision"],
        value["execution_geometry"],
    )
    if value["mode"] == "real":
        require(
            value["normalization_metadata_sha256"] == canonical_calvin_identity["metadata_sha256"],
            "normalization metadata differs from the CALVIN dataset identity",
        )
    return dict(value)


def validate_predict_response(response: Any, request: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    value = _validate_envelope(response, request["request_id"], "predict")
    _exact_keys(value, _PREDICT_RESPONSE_FIELDS, "prediction response")
    expected = _prediction_echo(request)
    for name, expected_value in expected.items():
        require(
            value[name] == expected_value and type(value[name]) is type(expected_value),
            f"prediction echo changed: {name}",
        )
    require(_valid_number(value["policy_seconds"]) and value["policy_seconds"] >= 0, "policy_seconds is invalid")
    return decode_actions(value["actions"]), dict(value)


def validate_shutdown_response(response: Any, request_id: str) -> Dict[str, Any]:
    value = _validate_envelope(response, request_id, "shutdown")
    _exact_keys(value, _ENVELOPE_FIELDS | {"stopped"}, "shutdown response")
    require(value["stopped"] is True, "shutdown response did not stop")
    return dict(value)


def send_frame(peer: socket.socket, value: Mapping[str, Any]) -> None:
    try:
        payload = _canonical_json(dict(value)).encode("utf-8")
    except DevBridgeProtocolError:
        raise
    require(0 < len(payload) <= MAX_FRAME_BYTES, "wire frame size is invalid")
    try:
        peer.sendall(_HEADER.pack(len(payload)) + payload)
    except socket.timeout as exc:
        raise DevBridgeTimeoutError("timed out sending development policy frame") from exc


def _receive_exact(peer: socket.socket, length: int) -> Optional[bytes]:
    chunks = []  # type: List[bytes]
    remaining = length
    while remaining:
        try:
            chunk = peer.recv(remaining)
        except socket.timeout as exc:
            raise DevBridgeTimeoutError("timed out receiving development policy frame") from exc
        if not chunk:
            if remaining == length:
                return None
            raise DevBridgeProtocolError("peer closed during a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_frame(peer: socket.socket) -> Optional[Dict[str, Any]]:
    header = _receive_exact(peer, _HEADER.size)
    if header is None:
        return None
    length = _HEADER.unpack(header)[0]
    require(0 < length <= MAX_FRAME_BYTES, "wire frame length is invalid")
    payload = _receive_exact(peer, length)
    require(payload is not None, "peer closed before frame payload")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise DevBridgeProtocolError("wire frame is not valid JSON") from exc
    require(isinstance(value, dict), "wire frame root must be an object")
    return value


class DevPolicyClient:
    """One private, persistent Unix-socket development policy connection."""

    def __init__(self, socket_path: Union[str, Path], timeout_seconds: float = 300.0) -> None:
        require(_valid_number(timeout_seconds) and timeout_seconds > 0, "timeout_seconds must be positive")
        self.socket_path = Path(socket_path)
        self.timeout_seconds = float(timeout_seconds)
        self._peer = None  # type: Optional[socket.socket]
        self._counter = 0
        self._health = None  # type: Optional[Dict[str, Any]]

    def connect(self) -> DevPolicyClient:
        require(self._peer is None, "development policy client is already connected")
        status_value = self.socket_path.lstat()
        require(stat.S_ISSOCK(status_value.st_mode), "development endpoint is not a Unix socket")
        require(status_value.st_uid == os.getuid(), "development socket is owned by another user")
        require(stat.S_IMODE(status_value.st_mode) & 0o077 == 0, "development socket is not private")
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        peer.settimeout(self.timeout_seconds)
        peer.connect(str(self.socket_path))
        self._peer = peer
        return self

    def close(self) -> None:
        if self._peer is not None:
            self._peer.close()
            self._peer = None
        self._health = None

    def __enter__(self) -> DevPolicyClient:
        return self.connect()

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _request_id(self, operation: str) -> str:
        self._counter += 1
        return f"{os.getpid()}-{self._counter}-{operation}"

    def _exchange(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        require(self._peer is not None, "development policy client is not connected")
        send_frame(self._peer, request)
        response = receive_frame(self._peer)
        require(response is not None, "development policy server closed without response")
        return response

    def health(self) -> Dict[str, Any]:
        request_id = self._request_id("health")
        result = validate_health_response(self._exchange(make_control_request("health", request_id)), request_id)
        self._health = dict(result)
        return result

    def predict(self, **request_fields: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        require(self._health is not None, "call health() before development predict()")
        request = make_predict_request(request_id=self._request_id("predict"), **request_fields)
        episode = request["development_episode"]
        require(request["train_seed"] == self._health["train_seed"], "predict train_seed differs from health")
        require(
            episode["reset_bank_sha256"] == self._health["reset_bank_sha256"],
            "predict reset bank differs from health",
        )
        require(episode["reset_index"] < self._health["reset_count"], "predict reset_index exceeds health reset count")
        return validate_predict_response(self._exchange(request), request)

    def shutdown(self) -> Dict[str, Any]:
        request_id = self._request_id("shutdown")
        return validate_shutdown_response(self._exchange(make_control_request("shutdown", request_id)), request_id)


def _prepare_socket(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        status_value = path.lstat()
    except FileNotFoundError:
        return
    require(stat.S_ISSOCK(status_value.st_mode), "refusing to replace a non-socket development path")
    require(status_value.st_uid == os.getuid(), "refusing to replace another user's socket")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        probe.connect(str(path))
    except OSError:
        path.unlink()
    else:
        raise FileExistsError("development policy socket already has a live listener")
    finally:
        probe.close()


def serve_unix_policy(
    socket_path: Union[str, Path],
    dispatch: Callable[[Dict[str, Any]], Mapping[str, Any]],
    accept_timeout_seconds: float = 0.5,
    ready: Optional[Callable[[], None]] = None,
) -> None:
    """Serve only the held-out A/B/C schema until a validated shutdown."""

    require(_valid_number(accept_timeout_seconds) and accept_timeout_seconds > 0, "accept timeout is invalid")
    path = Path(socket_path)
    _prepare_socket(path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.settimeout(float(accept_timeout_seconds))
    listener.bind(str(path))
    inode = path.lstat().st_ino
    os.chmod(path, 0o600)
    listener.listen(1)
    if ready is not None:
        ready()
    should_stop = False
    try:
        while not should_stop:
            try:
                connection, _address = listener.accept()
            except socket.timeout:
                continue
            with connection:
                connection.settimeout(None)
                while not should_stop:
                    request = None  # type: Optional[Dict[str, Any]]
                    try:
                        wire = receive_frame(connection)
                        if wire is None:
                            break
                        request = validate_request(wire)
                        response = dict(dispatch(request))
                    except Exception as exc:
                        response = make_error_response(request, exc)
                    try:
                        send_frame(connection, response)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    should_stop = request is not None and request.get("operation") == "shutdown"
    finally:
        listener.close()
        try:
            final = path.lstat()
            if final.st_ino == inode and stat.S_ISSOCK(final.st_mode):
                path.unlink()
        except FileNotFoundError:
            pass


def wait_for_socket(socket_path: Union[str, Path], timeout_seconds: float = 10.0) -> None:
    require(_valid_number(timeout_seconds) and timeout_seconds > 0, "timeout_seconds must be positive")
    path = Path(socket_path)
    deadline = time.monotonic() + timeout_seconds
    last_error = None  # type: Optional[OSError]
    while time.monotonic() < deadline:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.1)
        try:
            probe.connect(str(path))
        except OSError as exc:
            last_error = exc
            time.sleep(0.02)
        else:
            return
        finally:
            probe.close()
    raise TimeoutError(f"development policy socket did not become ready: {last_error}")
