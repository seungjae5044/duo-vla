#!/usr/bin/env python3
"""Strict, lossless IPC shared by the CALVIN simulator and policy processes.

This module deliberately depends only on the Python 3.8 standard library and
NumPy.  The simulator can therefore import it without importing any of the
modern model stack.
"""

# The evaluator is pinned to Python 3.8.  Ruff runs with the repository's
# Python 3.11 target, so keep the older typing spellings intentionally.
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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

SCHEMA = "duovla-calvin-policy-ipc-v4"
PROTOCOL = "duovla-calvin-abc-to-d-v1"
CALVIN_DATASET_MANIFEST_SCHEMA = "duo-vla-calvin-dataset-manifest-v4"
CALVIN_MEMBER_INDEX_SCHEMA = "duo-vla-calvin-member-index-v2"
CALVIN_ARCHIVE_READER_SCHEMA = "duo-vla-calvin-archive-reader-v1"
CALVIN_STORAGE_MODE = "archive-direct"
CALVIN_ARCHIVE_BYTES = 555_309_812_705
CALVIN_ARCHIVE_SHA256 = "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74"
CALVIN_CENTRAL_DIRECTORY_SHA256 = "b4f79bda7f6b966b51aa419badd0f7db7a8972a7b58d6d342af60aceff0ea31b"
CALVIN_MEMBER_INDEX_PATH = "task_ABC_D.members-v2.sqlite3"
CALVIN_METADATA_FILES = [
    "ep_start_end_ids.npy",
    "lang_annotations/auto_lang_ann.npy",
    "scene_info.npy",
    ".hydra/merged_config.yaml",
]
INFERENCE_SEED_DOMAIN = "duo-vla-calvin-inference-seed-v1"
SEQUENCE_SHA256 = "90191d9ac76baecb4f292ab766bbbf3ae65dbf43a83bbe5c376d99db10fd6446"
PINNED_SEQUENCE_SHA256 = SEQUENCE_SHA256
MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"

ACTION_HORIZON = 8
ACTION_DIM = 7
STATE_DIM = 8
STATIC_IMAGE_SHAPE = (200, 200, 3)
GRIPPER_IMAGE_SHAPE = (84, 84, 3)
SUPPORTED_EXECUTION_HORIZONS = (1, 4)
NUM_SEQUENCES = 1000
SUBTASKS_PER_SEQUENCE = 5
MAX_FRAME_BYTES = 2 * 1024 * 1024
EXPERTS_IMPLEMENTATION = "grouped_mm"
EXPERT_BATCH_ISOLATION = "sample_isolated_grouped_mm_v1"
PHYSICAL_BATCH_SIZE = 8

_HEADER = struct.Struct(">Q")
_RGB_FIELDS = {"data", "dtype", "encoding", "sha256", "shape"}
_CONTROL_REQUEST_FIELDS = {"operation", "request_id", "schema"}
_PREDICT_REQUEST_FIELDS = {
    "episode",
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
    "execution_horizon",
    "replan_idx",
    "sequence_idx",
    "sequence_sha256",
    "subtask_idx",
    "subtask_name",
}
_OBSERVATION_FIELDS = {"rgb_gripper", "rgb_static", "state"}
_RESPONSE_ENVELOPE_FIELDS = {"operation", "request_id", "schema", "status"}
_PREDICTION_ECHO_FIELDS = {
    "evaluation_seed",
    "execution_horizon",
    "inference_seed",
    "replan_idx",
    "sequence_idx",
    "sequence_sha256",
    "subtask_idx",
    "subtask_name",
    "train_seed",
}
_PREDICTION_RESPONSE_FIELDS = (
    _RESPONSE_ENVELOPE_FIELDS
    | _PREDICTION_ECHO_FIELDS
    | {
        "actions",
        "policy_seconds",
    }
)
_HEALTH_RESPONSE_FIELDS = _RESPONSE_ENVELOPE_FIELDS | {
    "action_dim",
    "action_horizon",
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
    "sampler",
    "sequence_sha256",
    "serving_runtime_sha256",
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


class BridgeError(RuntimeError):
    """Base exception for an invalid or failed policy exchange."""


class BridgeProtocolError(BridgeError):
    """The peer violated the pinned CALVIN wire contract."""


class BridgeTimeoutError(BridgeError):
    """The peer did not complete an exchange before its deadline."""


def require(condition: bool, message: str) -> None:
    """Raise a consistently typed error for a wire-contract violation."""

    if not condition:
        raise BridgeProtocolError(message)


def _exact_keys(value: Mapping[str, Any], expected: Set[str], name: str) -> None:
    observed = set(value.keys())
    require(
        observed == expected,
        f"{name} fields differ: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}",
    )


def _valid_integer(value: Any, minimum: int, maximum: Optional[int] = None) -> bool:
    if type(value) is not int or value < minimum:
        return False
    return maximum is None or value < maximum


def _valid_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_sha256(value: Any, name: str) -> None:
    require(
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{name} must be 64 lowercase hexadecimal characters",
    )


def _validate_sequence_identity(
    sequence_idx: Any,
    subtask_idx: Any,
    subtask_name: Any,
    replan_idx: Any,
) -> None:
    require(
        _valid_integer(sequence_idx, 0, NUM_SEQUENCES),
        f"sequence_idx must be an integer in [0, {NUM_SEQUENCES})",
    )
    require(
        _valid_integer(subtask_idx, 0, SUBTASKS_PER_SEQUENCE),
        f"subtask_idx must be an integer in [0, {SUBTASKS_PER_SEQUENCE})",
    )
    require(
        isinstance(subtask_name, str) and 0 < len(subtask_name) <= 200,
        "subtask_name must be non-empty text of at most 200 characters",
    )
    require(_valid_integer(replan_idx, 0), "replan_idx must be a nonnegative integer")


def calvin_replan_seed(
    evaluation_seed: int,
    sequence_sha256: str,
    sequence_idx: int,
    subtask_idx: int,
    subtask_name: str,
    replan_idx: int,
) -> int:
    """Derive inference randomness only from immutable evaluation identity.

    ``train_seed``, checkpoint identity, policy objective, NFE, and execution
    horizon are intentionally absent.  ``sequence_sha256`` is an argument so
    the derivation commits to the exact generated sequence list; requests
    additionally require it to equal :data:`SEQUENCE_SHA256`.
    """

    require(
        _valid_integer(evaluation_seed, 0, 2**63),
        "evaluation_seed must be an integer in [0, 2^63)",
    )
    _validate_sha256(sequence_sha256, "sequence_sha256")
    _validate_sequence_identity(sequence_idx, subtask_idx, subtask_name, replan_idx)
    parts = (
        INFERENCE_SEED_DOMAIN,
        evaluation_seed,
        sequence_sha256,
        sequence_idx,
        subtask_idx,
        subtask_name,
        replan_idx,
    )
    canonical = json.dumps(parts, allow_nan=False, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.blake2b(canonical.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _normalize_image_shape(shape: Any) -> Tuple[int, int, int]:
    if isinstance(shape, np.ndarray):
        shape = tuple(shape.shape)
    elif isinstance(shape, (list, tuple)):
        shape = tuple(shape)
    require(
        shape in (STATIC_IMAGE_SHAPE, GRIPPER_IMAGE_SHAPE),
        f"RGB shape must be either {STATIC_IMAGE_SHAPE} or {GRIPPER_IMAGE_SHAPE}",
    )
    return shape


def encode_rgb(
    image: np.ndarray,
    expected_shape: Optional[Sequence[int]] = None,
    name: str = "RGB image",
) -> Dict[str, Any]:
    """Encode one of the two pinned CALVIN uint8 HWC images without loss."""

    values = np.asarray(image)
    shape = _normalize_image_shape(values.shape if expected_shape is None else expected_shape)
    require(values.shape == shape, f"{name} must have shape {shape}, got {values.shape}")
    require(values.dtype == np.uint8, f"{name} must have dtype uint8, got {values.dtype}")
    raw = np.ascontiguousarray(values).tobytes(order="C")
    return {
        "data": base64.b64encode(raw).decode("ascii"),
        "dtype": "uint8",
        "encoding": "base64",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(shape),
    }


def decode_rgb(
    value: Any,
    name: str = "RGB image",
    expected_shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Validate and decode a CALVIN image into a C-contiguous owned array."""

    require(isinstance(value, dict), f"{name} must be an encoded RGB object")
    _exact_keys(value, _RGB_FIELDS, name)
    wire_shape = value["shape"]
    require(
        isinstance(wire_shape, list) and all(type(item) is int for item in wire_shape),
        f"{name} shape must be a list of integers",
    )
    shape = _normalize_image_shape(wire_shape if expected_shape is None else expected_shape)
    require(wire_shape == list(shape), f"{name} shape must be {shape}")
    require(value["dtype"] == "uint8", f"{name} dtype must be uint8")
    require(value["encoding"] == "base64", f"{name} encoding must be base64")
    require(isinstance(value["data"], str), f"{name} data must be text")
    _validate_sha256(value["sha256"], f"{name} sha256")
    try:
        raw = base64.b64decode(value["data"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BridgeProtocolError(f"{name} contains invalid base64") from exc
    expected_bytes = shape[0] * shape[1] * shape[2]
    require(len(raw) == expected_bytes, f"{name} byte length is invalid")
    require(hashlib.sha256(raw).hexdigest() == value["sha256"], f"{name} SHA-256 mismatch")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy(order="C")


def encode_static_rgb(image: np.ndarray) -> Dict[str, Any]:
    return encode_rgb(image, STATIC_IMAGE_SHAPE, "rgb_static")


def encode_gripper_rgb(image: np.ndarray) -> Dict[str, Any]:
    return encode_rgb(image, GRIPPER_IMAGE_SHAPE, "rgb_gripper")


def decode_static_rgb(value: Any) -> np.ndarray:
    return decode_rgb(value, "rgb_static", STATIC_IMAGE_SHAPE)


def decode_gripper_rgb(value: Any) -> np.ndarray:
    return decode_rgb(value, "rgb_gripper", GRIPPER_IMAGE_SHAPE)


def _state_array(value: Any, wire: bool) -> np.ndarray:
    if wire:
        require(isinstance(value, list) and len(value) == STATE_DIM, "state must contain 8 values")
        require(all(_valid_number(item) for item in value), "state must contain eight finite numbers")
    else:
        require(isinstance(value, np.ndarray), "state must be a numpy array")
        require(value.shape == (STATE_DIM,), "state must have shape (8,)")
        require(value.dtype == np.float32, "state must have dtype float32")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            state = np.asarray(value, dtype=np.float32)
    except (OverflowError, TypeError, ValueError) as exc:
        raise BridgeProtocolError("state cannot be represented as float32") from exc
    require(state.shape == (STATE_DIM,), "state must have shape (8,)")
    require(bool(np.isfinite(state).all()), "state must contain eight finite values")
    require(state[7] in (-1.0, 1.0), "state gripper value must be exactly {-1, +1}")
    return state.copy(order="C")


def encode_state(state: np.ndarray) -> List[float]:
    """Validate a local float32 state and return its exact JSON representation."""

    return _state_array(state, wire=False).tolist()


def decode_state(value: Any) -> np.ndarray:
    """Decode the wire state into an owned float32[8] array."""

    return _state_array(value, wire=True)


def _actions_array(value: Any, wire: bool) -> np.ndarray:
    if wire:
        require(isinstance(value, list) and len(value) == ACTION_HORIZON, "actions must contain 8 rows")
        rows_ok = all(
            isinstance(row, list) and len(row) == ACTION_DIM and all(_valid_number(item) for item in row)
            for row in value
        )
        require(rows_ok, "actions must contain an 8x7 matrix of finite numbers")
    else:
        require(isinstance(value, np.ndarray), "actions must be a numpy array")
        require(value.shape == (ACTION_HORIZON, ACTION_DIM), "actions must have shape (8, 7)")
        require(value.dtype == np.float32, "actions must have dtype float32")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            actions = np.asarray(value, dtype=np.float32)
    except (OverflowError, TypeError, ValueError) as exc:
        raise BridgeProtocolError("actions cannot be represented as float32") from exc
    require(actions.shape == (ACTION_HORIZON, ACTION_DIM), "policy actions must have shape (8, 7)")
    require(bool(np.isfinite(actions).all()), "policy actions contain non-finite values")
    require(
        bool(np.logical_or(actions[:, 6] == -1.0, actions[:, 6] == 1.0).all()),
        "policy gripper actions must be exactly {-1, +1}",
    )
    return actions.copy(order="C")


def encode_actions(actions: np.ndarray) -> List[List[float]]:
    """Validate a local float32 action chunk and return its JSON representation."""

    return _actions_array(actions, wire=False).tolist()


def decode_actions(value: Any) -> np.ndarray:
    """Decode the wire action chunk into an owned float32[8,7] array."""

    return _actions_array(value, wire=True)


def _validate_request_id(request_id: Any) -> None:
    require(
        isinstance(request_id, str) and 0 < len(request_id) <= 160,
        "request_id must be non-empty text of at most 160 characters",
    )


def make_control_request(operation: str, request_id: str) -> Dict[str, Any]:
    require(operation in ("health", "shutdown"), "invalid control operation")
    _validate_request_id(request_id)
    return {"operation": operation, "request_id": request_id, "schema": SCHEMA}


def make_predict_request(
    request_id: str,
    evaluation_seed: int,
    train_seed: int,
    sequence_sha256: str,
    sequence_idx: int,
    subtask_idx: int,
    subtask_name: str,
    replan_idx: int,
    execution_horizon: int,
    instruction: str,
    rgb_static: np.ndarray,
    rgb_gripper: np.ndarray,
    state: np.ndarray,
) -> Dict[str, Any]:
    """Construct a strict prediction request from simulator-owned arrays."""

    _validate_request_id(request_id)
    require(sequence_sha256 == SEQUENCE_SHA256, "sequence_sha256 does not match the pinned official sequences")
    inference_seed = calvin_replan_seed(
        evaluation_seed,
        sequence_sha256,
        sequence_idx,
        subtask_idx,
        subtask_name,
        replan_idx,
    )
    require(
        _valid_integer(execution_horizon, 1) and execution_horizon in SUPPORTED_EXECUTION_HORIZONS,
        "execution_horizon must be one of {1, 4}",
    )
    require(_valid_integer(train_seed, 0, 2**63), "train_seed must be an integer in [0, 2^63)")
    require(
        isinstance(instruction, str) and 0 < len(instruction) <= 1000,
        "instruction must be non-empty text of at most 1000 characters",
    )
    return {
        "episode": {
            "execution_horizon": execution_horizon,
            "replan_idx": replan_idx,
            "sequence_idx": sequence_idx,
            "sequence_sha256": sequence_sha256,
            "subtask_idx": subtask_idx,
            "subtask_name": subtask_name,
        },
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
    """Validate a request and replace encoded observations with owned arrays."""

    require(isinstance(value, dict), "request must be an object")
    require(value.get("schema") == SCHEMA, "request schema mismatch")
    operation = value.get("operation")
    require(operation in ("health", "predict", "shutdown"), "unknown request operation")
    _validate_request_id(value.get("request_id"))
    if operation != "predict":
        _exact_keys(value, _CONTROL_REQUEST_FIELDS, "control request")
        return dict(value)

    _exact_keys(value, _PREDICT_REQUEST_FIELDS, "predict request")
    episode = value["episode"]
    observation = value["observation"]
    require(isinstance(episode, dict), "episode must be an object")
    require(isinstance(observation, dict), "observation must be an object")
    _exact_keys(episode, _EPISODE_FIELDS, "episode")
    # This exact-key check is the fail-closed guarantee that scene_obs can
    # never cross into the policy process.
    _exact_keys(observation, _OBSERVATION_FIELDS, "observation")

    sequence_sha256 = episode["sequence_sha256"]
    require(sequence_sha256 == SEQUENCE_SHA256, "sequence_sha256 does not match the pinned official sequences")
    expected_seed = calvin_replan_seed(
        value["evaluation_seed"],
        sequence_sha256,
        episode["sequence_idx"],
        episode["subtask_idx"],
        episode["subtask_name"],
        episode["replan_idx"],
    )
    require(
        _valid_integer(episode["execution_horizon"], 1)
        and episode["execution_horizon"] in SUPPORTED_EXECUTION_HORIZONS,
        "execution_horizon must be one of {1, 4}",
    )
    require(
        _valid_integer(value["train_seed"], 0, 2**63),
        "train_seed must be an integer in [0, 2^63)",
    )
    require(
        type(value["inference_seed"]) is int and value["inference_seed"] == expected_seed,
        "inference seed does not match the evaluation identity",
    )
    require(
        isinstance(value["instruction"], str) and 0 < len(value["instruction"]) <= 1000,
        "instruction must be non-empty text of at most 1000 characters",
    )

    validated = dict(value)
    validated["episode"] = dict(episode)
    validated["observation"] = {
        "rgb_gripper": decode_gripper_rgb(observation["rgb_gripper"]),
        "rgb_static": decode_static_rgb(observation["rgb_static"]),
        "state": decode_state(observation["state"]),
    }
    return validated


def _prediction_echo(request: Mapping[str, Any]) -> Dict[str, Any]:
    episode = request["episode"]
    require(isinstance(episode, Mapping), "prediction episode must be an object")
    return {
        "evaluation_seed": request["evaluation_seed"],
        "execution_horizon": episode["execution_horizon"],
        "inference_seed": request["inference_seed"],
        "replan_idx": episode["replan_idx"],
        "sequence_idx": episode["sequence_idx"],
        "sequence_sha256": episode["sequence_sha256"],
        "subtask_idx": episode["subtask_idx"],
        "subtask_name": episode["subtask_name"],
        "train_seed": request["train_seed"],
    }


def make_success_response(request: Mapping[str, Any], **payload: Any) -> Dict[str, Any]:
    """Build an operation-bound success envelope and immutable prediction echo."""

    reserved = _RESPONSE_ENVELOPE_FIELDS
    require(not reserved.intersection(payload), "success response payload overrides its envelope")
    response = {
        "operation": request["operation"],
        "request_id": request["request_id"],
        "schema": SCHEMA,
        "status": "ok",
    }
    if request["operation"] == "predict":
        echoed = _prediction_echo(request)
        for name, expected in echoed.items():
            require(name not in payload or payload[name] == expected, f"prediction payload changed echoed {name}")
        response.update(echoed)
    response.update(payload)
    return response


def _validate_execution_geometry(value: Any, *, allow_none: bool) -> None:
    if value is None and allow_none:
        return
    require(isinstance(value, Mapping), "execution geometry must be an object")
    _exact_keys(value, _EXECUTION_GEOMETRY_FIELDS, "execution geometry")
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


def _validate_calvin_identity(value: Any, *, allow_none: bool) -> Optional[Dict[str, Any]]:
    if value is None and allow_none:
        return None
    require(isinstance(value, Mapping), "CALVIN health identity must be an object")
    _exact_keys(value, _CALVIN_IDENTITY_FIELDS, "CALVIN health identity")
    require(value["name"] == "task_ABC_D" and value["split"] == "training", "CALVIN dataset/split differs")
    require(value["archive_bytes"] == CALVIN_ARCHIVE_BYTES, "CALVIN archive byte count differs")
    require(value["archive_sha256"] == CALVIN_ARCHIVE_SHA256, "CALVIN archive SHA-256 differs")
    require(
        value["central_directory_sha256"] == CALVIN_CENTRAL_DIRECTORY_SHA256,
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
    require(
        value["dataset_manifest_schema"] == CALVIN_DATASET_MANIFEST_SCHEMA,
        "CALVIN dataset manifest schema is not v4",
    )
    require(value["reader_schema"] == CALVIN_ARCHIVE_READER_SCHEMA, "CALVIN archive reader schema differs")
    require(value["storage_mode"] == CALVIN_STORAGE_MODE, "CALVIN storage mode is not archive-direct")
    require(value["metadata_files"] == CALVIN_METADATA_FILES, "CALVIN metadata inventory differs")
    member_index = value["member_index"]
    require(isinstance(member_index, Mapping), "CALVIN member-index identity must be an object")
    _exact_keys(member_index, _CALVIN_MEMBER_INDEX_FIELDS, "CALVIN member-index identity")
    require(
        type(member_index["bytes"]) is int and member_index["bytes"] > 0,
        "CALVIN member-index byte count is invalid",
    )
    require(member_index["path"] == CALVIN_MEMBER_INDEX_PATH, "CALVIN member-index path differs")
    require(member_index["schema"] == CALVIN_MEMBER_INDEX_SCHEMA, "CALVIN member-index schema is not v2")
    _validate_sha256(member_index["sha256"], "CALVIN member-index SHA-256")
    result = dict(value)
    result["member_index"] = dict(member_index)
    result["metadata_files"] = list(value["metadata_files"])
    return result


def make_health_response(
    request: Mapping[str, Any],
    train_seed: int,
    *,
    calvin_identity: Optional[Mapping[str, Any]],
    mode: str,
    objective: str,
    sampler: str,
    nfe: int,
    checkpoint_manifest_sha256: Optional[str],
    policy_contract_sha256: Optional[str],
    normalization_content_sha256: Optional[str],
    normalization_metadata_sha256: Optional[str],
    model_revision: Optional[str],
    serving_runtime_sha256: Optional[str],
    execution_geometry: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build the complete, exact health response understood by PolicyClient."""

    require(request.get("operation") == "health", "health response requires a health request")
    require(_valid_integer(train_seed, 0, 2**63), "train_seed must be an integer in [0, 2^63)")
    canonical_calvin_identity = _validate_calvin_identity(calvin_identity, allow_none=mode == "fake")
    _validate_policy_health_identity(
        mode=mode,
        objective=objective,
        sampler=sampler,
        nfe=nfe,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        policy_contract_sha256=policy_contract_sha256,
        normalization_content_sha256=normalization_content_sha256,
        normalization_metadata_sha256=normalization_metadata_sha256,
        model_revision=model_revision,
        serving_runtime_sha256=serving_runtime_sha256,
        execution_geometry=execution_geometry,
    )
    if mode == "real":
        assert canonical_calvin_identity is not None
        require(
            normalization_metadata_sha256 == canonical_calvin_identity["metadata_sha256"],
            "normalization metadata differs from the CALVIN dataset identity",
        )
    return make_success_response(
        request,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
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
        sampler=sampler,
        sequence_sha256=SEQUENCE_SHA256,
        serving_runtime_sha256=serving_runtime_sha256,
        state_dim=STATE_DIM,
        static_image_shape=list(STATIC_IMAGE_SHAPE),
        train_seed=train_seed,
    )


def _validate_policy_health_identity(
    *,
    mode: Any,
    objective: Any,
    sampler: Any,
    nfe: Any,
    checkpoint_manifest_sha256: Any,
    policy_contract_sha256: Any,
    normalization_content_sha256: Any,
    normalization_metadata_sha256: Any,
    model_revision: Any,
    serving_runtime_sha256: Any,
    execution_geometry: Any,
) -> None:
    """Validate the model-side identity which is bound before any prediction."""

    require(mode in ("real", "fake"), "policy mode must be real or fake")
    if mode == "fake":
        require(objective == "test_fake", "fake policy objective mismatch")
        require(sampler == "seeded_test_normal", "fake policy sampler mismatch")
        require(type(nfe) is int and nfe == 0, "fake policy NFE must equal zero")
        optional = {
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "policy_contract_sha256": policy_contract_sha256,
            "normalization_content_sha256": normalization_content_sha256,
            "normalization_metadata_sha256": normalization_metadata_sha256,
            "model_revision": model_revision,
            "serving_runtime_sha256": serving_runtime_sha256,
            "execution_geometry": execution_geometry,
        }
        require(all(value is None for value in optional.values()), "fake policy must not claim real artifact identity")
        return

    require(objective in ("rectified_flow", "direct_regression"), "real policy objective is unsupported")
    if objective == "rectified_flow":
        require(sampler == "euler_uniform", "rectified-flow sampler mismatch")
        require(type(nfe) is int and nfe in (1, 5, 10), "rectified-flow NFE must be one of {1, 5, 10}")
    else:
        require(sampler == "single_forward", "direct-regression sampler mismatch")
        require(type(nfe) is int and nfe == 1, "direct-regression NFE must equal one")
    for name, value in (
        ("checkpoint_manifest_sha256", checkpoint_manifest_sha256),
        ("policy_contract_sha256", policy_contract_sha256),
        ("normalization_content_sha256", normalization_content_sha256),
        ("normalization_metadata_sha256", normalization_metadata_sha256),
        ("serving_runtime_sha256", serving_runtime_sha256),
    ):
        _validate_sha256(value, name)
    require(model_revision == MODEL_REVISION, "policy model revision mismatch")
    _validate_execution_geometry(execution_geometry, allow_none=False)


def make_predict_response(
    request: Mapping[str, Any],
    actions: np.ndarray,
    policy_seconds: Union[int, float],
) -> Dict[str, Any]:
    """Build the complete prediction response with a validated action chunk."""

    require(request.get("operation") == "predict", "prediction response requires a predict request")
    require(_valid_number(policy_seconds) and policy_seconds >= 0, "invalid policy_seconds")
    return make_success_response(
        request,
        actions=encode_actions(actions),
        policy_seconds=float(policy_seconds),
    )


def make_error_response(request: Optional[Mapping[str, Any]], exc: Exception) -> Dict[str, Any]:
    return {
        "error": {"message": str(exc), "type": type(exc).__name__},
        "operation": request.get("operation", "invalid") if request is not None else "invalid",
        "request_id": request.get("request_id", "unknown") if request is not None else "unknown",
        "schema": SCHEMA,
        "status": "error",
    }


def _validate_response_envelope(value: Any, request_id: str, operation: str) -> None:
    require(isinstance(value, dict), "response must be an object")
    require(value.get("schema") == SCHEMA, "response schema mismatch")
    require(value.get("request_id") == request_id, "response request_id mismatch")
    require(value.get("operation") == operation, "response operation mismatch")
    status_value = value.get("status")
    require(status_value in ("ok", "error"), "response status is invalid")
    if status_value == "error":
        _exact_keys(value, _RESPONSE_ENVELOPE_FIELDS | {"error"}, "error response")
        error = value["error"]
        require(isinstance(error, dict), "error response payload must be an object")
        _exact_keys(error, {"message", "type"}, "error response payload")
        require(isinstance(error["message"], str) and bool(error["message"]), "error message must be text")
        require(isinstance(error["type"], str) and bool(error["type"]), "error type must be text")
        raise BridgeError("policy server rejected {}: {}".format(operation, error["message"]))


def validate_health_response(value: Any, request_id: str) -> Dict[str, Any]:
    _validate_response_envelope(value, request_id, "health")
    require(isinstance(value, dict), "response must be an object")
    _exact_keys(value, _HEALTH_RESPONSE_FIELDS, "health response")
    require(value["protocol"] == PROTOCOL, "policy protocol mismatch")
    require(value["sequence_sha256"] == SEQUENCE_SHA256, "policy sequence digest mismatch")
    require(
        type(value["action_horizon"]) is int and value["action_horizon"] == ACTION_HORIZON,
        "policy action horizon mismatch",
    )
    require(type(value["action_dim"]) is int and value["action_dim"] == ACTION_DIM, "policy action dimension mismatch")
    require(type(value["state_dim"]) is int and value["state_dim"] == STATE_DIM, "policy state dimension mismatch")
    require(
        isinstance(value["static_image_shape"], list)
        and all(type(item) is int for item in value["static_image_shape"])
        and value["static_image_shape"] == list(STATIC_IMAGE_SHAPE),
        "policy static image shape mismatch",
    )
    require(
        isinstance(value["gripper_image_shape"], list)
        and all(type(item) is int for item in value["gripper_image_shape"])
        and value["gripper_image_shape"] == list(GRIPPER_IMAGE_SHAPE),
        "policy gripper image shape mismatch",
    )
    require(
        isinstance(value["execution_horizons"], list)
        and all(type(item) is int for item in value["execution_horizons"])
        and value["execution_horizons"] == list(SUPPORTED_EXECUTION_HORIZONS),
        "policy execution horizons mismatch",
    )
    require(
        _valid_integer(value["train_seed"], 0, 2**63),
        "policy health response has an invalid train_seed",
    )
    canonical_calvin_identity = _validate_calvin_identity(
        value["calvin_identity"],
        allow_none=value["mode"] == "fake",
    )
    _validate_policy_health_identity(
        mode=value["mode"],
        objective=value["objective"],
        sampler=value["sampler"],
        nfe=value["nfe"],
        checkpoint_manifest_sha256=value["checkpoint_manifest_sha256"],
        policy_contract_sha256=value["policy_contract_sha256"],
        normalization_content_sha256=value["normalization_content_sha256"],
        normalization_metadata_sha256=value["normalization_metadata_sha256"],
        model_revision=value["model_revision"],
        serving_runtime_sha256=value["serving_runtime_sha256"],
        execution_geometry=value["execution_geometry"],
    )
    if value["mode"] == "real":
        assert canonical_calvin_identity is not None
        require(
            value["normalization_metadata_sha256"] == canonical_calvin_identity["metadata_sha256"],
            "normalization metadata differs from the CALVIN dataset identity",
        )
    return dict(value)


def _require_echo(value: Mapping[str, Any], name: str, expected: Any) -> None:
    observed = value.get(name)
    matches = type(observed) is int and observed == expected if type(expected) is int else observed == expected
    require(matches, f"prediction {name} does not match the request")


def validate_action_response(
    value: Any,
    request_id: str,
    expected_evaluation_seed: int,
    expected_inference_seed: int,
    expected_sequence_sha256: str,
    expected_sequence_idx: int,
    expected_subtask_idx: int,
    expected_subtask_name: str,
    expected_replan_idx: int,
    expected_execution_horizon: int,
    expected_train_seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reject response drift and return an owned float32 action chunk."""

    require(
        expected_sequence_sha256 == SEQUENCE_SHA256,
        "expected sequence_sha256 does not match the pinned official sequences",
    )
    _validate_response_envelope(value, request_id, "predict")
    require(isinstance(value, dict), "response must be an object")
    _exact_keys(value, _PREDICTION_RESPONSE_FIELDS, "prediction response")
    expected = {
        "evaluation_seed": expected_evaluation_seed,
        "execution_horizon": expected_execution_horizon,
        "inference_seed": expected_inference_seed,
        "replan_idx": expected_replan_idx,
        "sequence_idx": expected_sequence_idx,
        "sequence_sha256": expected_sequence_sha256,
        "subtask_idx": expected_subtask_idx,
        "subtask_name": expected_subtask_name,
        "train_seed": expected_train_seed,
    }
    for name, expected_value in expected.items():
        _require_echo(value, name, expected_value)
    policy_seconds = value["policy_seconds"]
    require(_valid_number(policy_seconds) and policy_seconds >= 0, "invalid policy_seconds")
    actions = decode_actions(value["actions"])
    return actions, dict(value)


def validate_shutdown_response(value: Any, request_id: str) -> Dict[str, Any]:
    _validate_response_envelope(value, request_id, "shutdown")
    require(isinstance(value, dict), "response must be an object")
    _exact_keys(value, _RESPONSE_ENVELOPE_FIELDS | {"stopped"}, "shutdown response")
    require(value["stopped"] is True, "shutdown response did not confirm termination")
    return dict(value)


def send_frame(peer: socket.socket, value: Mapping[str, Any]) -> None:
    """Send one length-prefixed finite-JSON frame."""

    try:
        payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BridgeProtocolError(f"message is not finite JSON: {exc}") from exc
    require(len(payload) <= MAX_FRAME_BYTES, f"message exceeds {MAX_FRAME_BYTES} byte limit")
    try:
        peer.sendall(_HEADER.pack(len(payload)) + payload)
    except socket.timeout as exc:
        raise BridgeTimeoutError("timed out while sending policy message") from exc


def _receive_exact(peer: socket.socket, size: int, allow_clean_eof: bool = False) -> Optional[bytes]:
    chunks = []  # type: List[bytes]
    remaining = size
    while remaining:
        try:
            chunk = peer.recv(remaining)
        except socket.timeout as exc:
            raise BridgeTimeoutError("timed out while receiving policy message") from exc
        if not chunk:
            if allow_clean_eof and not chunks:
                return None
            raise BridgeProtocolError("policy connection closed within a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result = {}  # type: Dict[str, Any]
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field {name}")
        result[name] = value
    return result


def receive_frame(peer: socket.socket) -> Optional[Dict[str, Any]]:
    """Receive one frame, rejecting malformed UTF-8, JSON, and duplicate keys."""

    header = _receive_exact(peer, _HEADER.size, allow_clean_eof=True)
    if header is None:
        return None
    size = _HEADER.unpack(header)[0]
    require(0 < size <= MAX_FRAME_BYTES, f"invalid policy frame length {size}")
    payload = _receive_exact(peer, size)
    assert payload is not None
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise BridgeProtocolError("policy frame is not valid finite UTF-8 JSON") from exc
    require(isinstance(value, dict), "policy frame root must be an object")
    return value


class PolicyClient:
    """One persistent, timeout-bound Unix-socket connection."""

    def __init__(self, socket_path: Union[str, Path], timeout_seconds: float = 300.0) -> None:
        if not _valid_number(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        self.socket_path = Path(socket_path)
        self.timeout_seconds = float(timeout_seconds)
        self._peer = None  # type: Optional[socket.socket]
        self._counter = 0
        self._health = None  # type: Optional[Dict[str, Any]]

    def connect(self) -> "PolicyClient":
        if self._peer is not None:
            raise RuntimeError("policy client is already connected")
        status_value = self.socket_path.lstat()
        require(stat.S_ISSOCK(status_value.st_mode), f"policy endpoint is not a Unix socket: {self.socket_path}")
        require(status_value.st_uid == os.getuid(), f"policy socket is owned by another user: {self.socket_path}")
        require(
            stat.S_IMODE(status_value.st_mode) & 0o077 == 0,
            f"policy socket is not private: {self.socket_path}",
        )
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        peer.settimeout(self.timeout_seconds)
        try:
            peer.connect(str(self.socket_path))
        except socket.timeout as exc:
            peer.close()
            raise BridgeTimeoutError(f"timed out connecting to {self.socket_path}") from exc
        except OSError:
            peer.close()
            raise
        self._peer = peer
        return self

    def close(self) -> None:
        if self._peer is not None:
            self._peer.close()
            self._peer = None
        self._health = None

    def __enter__(self) -> "PolicyClient":
        return self.connect()

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _request_id(self, operation: str) -> str:
        self._counter += 1
        return f"{os.getpid()}-{self._counter}-{operation}"

    def _exchange(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        if self._peer is None:
            raise RuntimeError("policy client is not connected")
        send_frame(self._peer, request)
        response = receive_frame(self._peer)
        if response is None:
            raise BridgeProtocolError("policy server closed without a response")
        return response

    def health(self) -> Dict[str, Any]:
        request_id = self._request_id("health")
        response = self._exchange(make_control_request("health", request_id))
        validated = validate_health_response(response, request_id)
        self._health = dict(validated)
        return validated

    def predict(self, **request_fields: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self._health is None:
            raise RuntimeError("call health() before predict() to bind the CALVIN policy contract")
        request_id = self._request_id("predict")
        request = make_predict_request(request_id=request_id, **request_fields)
        require(request["train_seed"] == self._health["train_seed"], "predict train_seed differs from policy health")
        require(
            request["episode"]["sequence_sha256"] == self._health["sequence_sha256"],
            "predict sequence digest differs from policy health",
        )
        response = self._exchange(request)
        episode = request["episode"]
        return validate_action_response(
            response,
            request_id=request_id,
            expected_evaluation_seed=request["evaluation_seed"],
            expected_inference_seed=request["inference_seed"],
            expected_sequence_sha256=episode["sequence_sha256"],
            expected_sequence_idx=episode["sequence_idx"],
            expected_subtask_idx=episode["subtask_idx"],
            expected_subtask_name=episode["subtask_name"],
            expected_replan_idx=episode["replan_idx"],
            expected_execution_horizon=episode["execution_horizon"],
            expected_train_seed=request["train_seed"],
        )

    def shutdown(self) -> Dict[str, Any]:
        request_id = self._request_id("shutdown")
        response = self._exchange(make_control_request("shutdown", request_id))
        return validate_shutdown_response(response, request_id)


def _prepare_socket(socket_path: Path) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        status_value = socket_path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(status_value.st_mode):
        raise FileExistsError(f"refusing to replace non-socket path: {socket_path}")
    if status_value.st_uid != os.getuid():
        raise FileExistsError(f"refusing to replace socket owned by another user: {socket_path}")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        probe.connect(str(socket_path))
    except OSError:
        socket_path.unlink()
    else:
        raise FileExistsError(f"policy socket already has a live listener: {socket_path}")
    finally:
        probe.close()


def serve_unix_policy(
    socket_path: Union[str, Path],
    dispatch: Callable[[Dict[str, Any]], Mapping[str, Any]],
    accept_timeout_seconds: float = 0.5,
    ready: Optional[Callable[[], None]] = None,
) -> None:
    """Serve sequential persistent connections until a validated shutdown."""

    if not _valid_number(accept_timeout_seconds) or accept_timeout_seconds <= 0:
        raise ValueError("accept_timeout_seconds must be a positive finite number")
    path = Path(socket_path)
    _prepare_socket(path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.settimeout(float(accept_timeout_seconds))
    listener.bind(str(path))
    socket_inode = path.lstat().st_ino
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
                        wire_request = receive_frame(connection)
                        if wire_request is None:
                            break
                        request = validate_request(wire_request)
                        response = dict(dispatch(request))
                    except Exception as exc:  # The error envelope is part of the IPC contract.
                        response = make_error_response(request, exc)
                    try:
                        send_frame(connection, response)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    should_stop = request is not None and request.get("operation") == "shutdown"
    finally:
        listener.close()
        try:
            final_status = path.lstat()
            if final_status.st_ino == socket_inode and stat.S_ISSOCK(final_status.st_mode):
                path.unlink()
        except FileNotFoundError:
            pass


def wait_for_socket(socket_path: Union[str, Path], timeout_seconds: float = 10.0) -> None:
    """Wait until a Unix listener accepts connections or raise TimeoutError."""

    if not _valid_number(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive finite number")
    deadline = time.monotonic() + timeout_seconds
    last_error = None  # type: Optional[OSError]
    while time.monotonic() < deadline:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.1)
        try:
            probe.connect(str(socket_path))
        except OSError as exc:
            last_error = exc
            time.sleep(0.02)
        else:
            probe.close()
            return
        finally:
            probe.close()
    raise TimeoutError(f"policy socket did not become ready: {socket_path}: {last_error}")
