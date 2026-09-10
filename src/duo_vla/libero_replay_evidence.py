"""Shared, deterministic primitives for LIBERO expert-replay evidence.

The simulator-side collector and the parquet-side binder intentionally run in
different pinned environments.  This module defines the byte-level contract
that lets those two stages compare trajectories without copying the 34 GB
source HDF5 corpus or every decoded training image into the evidence bundle.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import stat
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ORIGINAL_HDF5_INVENTORY_SCHEMA = "duo-vla-libero-original-hdf5-inventory-v1"
ORIGINAL_HDF5_REPOSITORY_ID = "yifengzhu-hf/LIBERO-datasets"
ORIGINAL_HDF5_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
ORIGINAL_HDF5_CONTENT_SHA256 = "c4976da211c1895914d3d07c956ef48a6a9175367a22ff03ff43112064ee14e1"
ORIGINAL_HDF5_FILE_COUNT = 40
ORIGINAL_HDF5_TOTAL_BYTES = 33_784_856_577

SIMULATOR_STAGE_SCHEMA = "duo-vla-libero-expert-replay-simulator-stage-v2"
SIMULATOR_TASK_SCHEMA = "duo-vla-libero-expert-replay-simulator-task-v2"
PARQUET_BINDING_SCHEMA = "duo-vla-libero-expert-replay-parquet-binding-v2"
PARQUET_TASK_SCHEMA = "duo-vla-libero-expert-replay-parquet-task-v2"
SOURCE_PARQUET_ALIGNMENT_SCHEMA = "duo-vla-libero-source-parquet-alignment-v1"
SOURCE_PARQUET_ALIGNMENT_CONTENT_SHA256 = "8f554538f1fe3c2aeaf6c0f46d2b526f5bfe27589613f8e856c6fcacfdf18714"
OBSERVATION_ALIGNMENT_FEATURE_SCHEMA = "duo-vla-libero-observation-alignment-features-v1"
OBSERVATION_ALIGNMENT_GRID_SIZE = 8
OBSERVATION_ALIGNMENT_IMAGE_MEAN_ABS_MAX = 45.0
OBSERVATION_ALIGNMENT_IMAGE_MAX_FRAME_MEAN_ABS_MAX = 50.0
OBSERVATION_ALIGNMENT_IMAGE_CORRELATION_MEAN_MIN = 0.70
OBSERVATION_ALIGNMENT_IMAGE_CORRELATION_FRAME_MIN = 0.20
OBSERVATION_ALIGNMENT_STATE_MEAN_ABS_MAX = 0.02
OBSERVATION_ALIGNMENT_STATE_MAX_ABS_MAX = 0.15
OBSERVATION_ALIGNMENT_STATE_TEMPORAL_MARGIN_MIN = 0.001

PROTOCOL = "duovla-libero-v1"
EVIDENCE_SCHEMA = "duo-vla-libero-expert-replay-evidence-v1"
SIMULATOR_ATTESTATION_SCHEMA = "duo-vla-libero-simulator-attestation-v3"
DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
DATASET_TREE_METADATA_SHA256 = "d9c14b4aff28bcc56f341b171c6a5a3b10510d4bd0378662891c5156d245add8"
DATASET_TREE_FILE_COUNT = 383
DATASET_TREE_TOTAL_BYTES = 34_926_157_548
DATASET_CONTENT_INVENTORY_SHA256 = "63fd7a951ebb397a33c43cad4a7c48c7c6911bd8d1481ff99b07da5f7890782c"
DATASET_SNAPSHOT_FILES_VERIFIED = 382
DATASET_SNAPSHOT_TOTAL_BYTES = 34_926_155_087
NORMALIZATION_CONTENT_SHA256 = "a972b5d95a8aaa8ae7582bafcbc071261979cb46c2a3515b4da7a7cf0156ac73"
NORMALIZATION_RAW_SHA256 = "8a0428184e4db8463f3986e9a8c4f912e1027815669f4b4880c4bd3e8d3bcf29"
TASK_INVENTORY_SHA256 = "d00c211a09f34003089ba5a4dbbbb0e11af2543f4bba9cb1901a04a2a25e0117"
ENVIRONMENT_SEED = 0

GATE_NAMES = (
    "schema_revision_counts",
    "exact_gripper_set",
    "controller_impulse_directions",
    "normalization_round_trip",
    "episode_boundary_chunk_fixture",
    "pre_action_observation_alignment",
    "camera_transform_parity",
    "deterministic_fixed_state_reset",
    "pre_dispatch_integrity_controls",
)
PRE_DISPATCH_MUTATIONS = ("zero_action", "mismatched_language", "swapped_cameras", "inverted_gripper")
IMPULSE_DIRECTIONS = ("+x", "-x", "+y", "-y", "+z", "-z", "+rx", "-rx", "+ry", "-ry", "+rz", "-rz")

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
TRAINING_CAMERA_KEYS = ("observation.images.image", "observation.images.image2")
ACTION_DIM = 7
STATE_DIM = 8
IMAGE_SHAPE = (256, 256, 3)
REGENERATION_ENVIRONMENT_SEED = 0
SETTLE_STEPS = 10
NOOP_THRESHOLD = 1e-4

_LENGTH = struct.Struct(">Q")


class ReplayEvidenceError(RuntimeError):
    """An evidence input or generated value violated the pinned contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayEvidenceError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ReplayEvidenceError(f"value is not finite canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"duplicate JSON field {name!r}")
        value[name] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def require_finite_json_numbers(value: Any, *, name: str = "JSON value") -> None:
    """Reject exponent-overflow floats that ``parse_constant`` cannot see."""

    if isinstance(value, float):
        require(math.isfinite(value), f"{name} contains a non-finite number")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            require_finite_json_numbers(item, name=f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_json_numbers(item, name=f"{name}[{index}]")


def load_strict_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReplayEvidenceError(f"{name} is not strict finite UTF-8 JSON: {path}") from exc
    require(isinstance(value, dict), f"{name} root must be an object")
    require_finite_json_numbers(value, name=name)
    return value


def _require_sha256(value: Any, name: str) -> str:
    require(
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{name} must be 64 lowercase hexadecimal characters",
    )
    return value


def validate_original_hdf5_inventory(value: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    expected_fields = {"content_sha256", "file_count", "files", "repository", "schema", "total_bytes"}
    require(set(value) == expected_fields, "original HDF5 inventory fields differ")
    require(value["schema"] == ORIGINAL_HDF5_INVENTORY_SCHEMA, "original HDF5 inventory schema differs")
    repository = value["repository"]
    require(
        repository
        == {
            "id": ORIGINAL_HDF5_REPOSITORY_ID,
            "repo_type": "dataset",
            "revision": ORIGINAL_HDF5_REVISION,
        },
        "original HDF5 repository identity differs",
    )
    unsigned = {name: item for name, item in value.items() if name != "content_sha256"}
    observed_content = canonical_sha256(unsigned)
    require(
        value["content_sha256"] == observed_content == ORIGINAL_HDF5_CONTENT_SHA256,
        "original HDF5 inventory semantic hash differs",
    )
    files = value["files"]
    require(isinstance(files, list) and len(files) == ORIGINAL_HDF5_FILE_COUNT, "original HDF5 file count differs")
    expected_tasks = [(suite, task_id) for suite in SUITES for task_id in range(10)]
    records: list[dict[str, Any]] = []
    paths: list[str] = []
    for index, item in enumerate(files):
        require(isinstance(item, Mapping), f"original HDF5 record {index} is not an object")
        require(
            set(item) == {"bytes", "path", "sha256", "suite", "task_id", "task_name"},
            f"original HDF5 record {index} fields differ",
        )
        suite, task_id = expected_tasks[index]
        task_name = item["task_name"]
        expected_path = f"{suite}/{task_name}_demo.hdf5"
        require(
            item["suite"] == suite
            and item["task_id"] == task_id
            and isinstance(task_name, str)
            and bool(task_name)
            and item["path"] == expected_path,
            f"original HDF5 record {index} task/path identity differs",
        )
        require(type(item["bytes"]) is int and item["bytes"] > 0, f"original HDF5 record {index} size is invalid")
        _require_sha256(item["sha256"], f"original HDF5 record {index} SHA-256")
        paths.append(item["path"])
        records.append(dict(item))
    require(len(paths) == len(set(paths)), "original HDF5 paths are not unique")
    require(value["file_count"] == len(records), "original HDF5 declared file count differs")
    require(
        value["total_bytes"] == sum(record["bytes"] for record in records) == ORIGINAL_HDF5_TOTAL_BYTES,
        "original HDF5 declared byte count differs",
    )
    return tuple(records)


def load_original_hdf5_inventory(path: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], str]:
    raw = path.read_bytes()
    value = load_strict_json(path, name="original HDF5 inventory")
    records = validate_original_hdf5_inventory(value)
    return value, records, hashlib.sha256(raw).hexdigest()


def load_source_parquet_alignment(
    path: Path,
) -> tuple[dict[str, Any], dict[tuple[str, int], frozenset[int]], str]:
    raw = path.read_bytes()
    value = load_strict_json(path, name="source/parquet alignment")
    require(
        set(value) == {"content_sha256", "original_hdf5", "schema", "tasks", "training_dataset"},
        "source/parquet alignment fields differ",
    )
    require(value["schema"] == SOURCE_PARQUET_ALIGNMENT_SCHEMA, "source/parquet alignment schema differs")
    unsigned = {name: item for name, item in value.items() if name != "content_sha256"}
    require(
        value["content_sha256"] == canonical_sha256(unsigned) == SOURCE_PARQUET_ALIGNMENT_CONTENT_SHA256,
        "source/parquet alignment semantic identity differs",
    )
    require(
        value["original_hdf5"]
        == {
            "content_sha256": ORIGINAL_HDF5_CONTENT_SHA256,
            "repository_id": ORIGINAL_HDF5_REPOSITORY_ID,
            "revision": ORIGINAL_HDF5_REVISION,
        },
        "source/parquet alignment original-HDF5 identity differs",
    )
    require(
        value["training_dataset"]
        == {
            "content_inventory_sha256": DATASET_CONTENT_INVENTORY_SHA256,
            "episode_count": 1693,
            "revision": DATASET_REVISION,
        },
        "source/parquet alignment training-dataset identity differs",
    )
    tasks = value["tasks"]
    require(isinstance(tasks, list) and len(tasks) == 40, "source/parquet alignment task count differs")
    expected = sorted((suite, task_id) for suite in SUITES for task_id in range(10))
    observed: list[tuple[str, int]] = []
    mapping: dict[tuple[str, int], frozenset[int]] = {}
    linked_episode_count = 0
    for task in tasks:
        require(
            isinstance(task, dict) and set(task) == {"suite", "task_id", "training_source_episode_indices"},
            "source/parquet alignment task fields differ",
        )
        identity = (task["suite"], task["task_id"])
        indices = task["training_source_episode_indices"]
        require(
            isinstance(indices, list)
            and bool(indices)
            and all(type(index) is int and 0 <= index < 50 for index in indices)
            and indices == sorted(set(indices)),
            f"source/parquet alignment indices differ: {identity}",
        )
        observed.append(identity)
        mapping[identity] = frozenset(indices)
        linked_episode_count += len(indices)
    require(observed == expected, "source/parquet alignment task order differs")
    require(linked_episode_count == 1693, "source/parquet alignment episode count differs")
    return value, mapping, hashlib.sha256(raw).hexdigest()


def stable_regular_file_identity(path: Path, *, expected_bytes: int | None = None) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise ReplayEvidenceError(f"cannot open a source file without following links: {path}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"source is not a regular file: {path}")
        require(before.st_nlink == 1, f"source must have exactly one hard link: {path}")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            while block := source.read(8 * 1024 * 1024):
                size += len(block)
                digest.update(block)
            after = os.fstat(source.fileno())
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        require(
            all(getattr(before, name) == getattr(after, name) for name in stable_fields) and size == after.st_size,
            f"source changed while it was hashed: {path}",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if expected_bytes is not None:
        require(size == expected_bytes, f"source byte count differs: {path}")
    return {"bytes": size, "sha256": digest.hexdigest()}


def is_noop(
    action: np.ndarray,
    previous_retained_action: np.ndarray | None,
    *,
    threshold: float = NOOP_THRESHOLD,
) -> bool:
    """Match OpenVLA's LIBERO regeneration no-op filter exactly."""

    current = np.asarray(action)
    require(current.shape == (ACTION_DIM,), f"LIBERO action must have shape ({ACTION_DIM},)")
    require(bool(np.isfinite(current).all()), "LIBERO action contains a non-finite value")
    spatial_noop = float(np.linalg.norm(current[:-1])) < threshold
    if previous_retained_action is None:
        return spatial_noop
    previous = np.asarray(previous_retained_action)
    require(previous.shape == (ACTION_DIM,), "previous LIBERO action has the wrong shape")
    return spatial_noop and bool(current[-1] == previous[-1])


def retained_action_indices(actions: np.ndarray) -> tuple[int, ...]:
    values = np.asarray(actions)
    require(values.ndim == 2 and values.shape[1] == ACTION_DIM, "LIBERO action matrix must have shape [T, 7]")
    require(len(values) > 0 and bool(np.isfinite(values).all()), "LIBERO action matrix is empty or non-finite")
    retained: list[int] = []
    previous: np.ndarray | None = None
    for index, action in enumerate(values):
        if is_noop(action, previous):
            continue
        retained.append(index)
        previous = action
    require(bool(retained), "LIBERO action sequence contains no retained transition")
    return tuple(retained)


def canonical_array(value: Any, *, dtype: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.dtype(dtype))
    if shape is not None:
        require(array.shape == shape, f"array shape differs: expected {shape}, got {array.shape}")
    require(bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.floating) else True, "array is non-finite")
    return np.ascontiguousarray(array)


def update_array_digest(
    digest: Any,
    name: str,
    value: Any,
    *,
    dtype: str,
    shape: tuple[int, ...] | None = None,
) -> None:
    array = canonical_array(value, dtype=dtype, shape=shape)
    metadata = canonical_json_bytes({"dtype": array.dtype.str, "name": name, "shape": list(array.shape)})
    payload = array.tobytes(order="C")
    digest.update(_LENGTH.pack(len(metadata)))
    digest.update(metadata)
    digest.update(_LENGTH.pack(len(payload)))
    digest.update(payload)


def action_sequence_sha256(actions: Any) -> str:
    values = np.asarray(actions)
    require(values.ndim == 2 and values.shape[1] == ACTION_DIM and len(values) > 0, "actions must have shape [T, 7]")
    digest = hashlib.sha256(b"duo-vla-libero-action-sequence-v1\0")
    update_array_digest(digest, "actions", values, dtype="<f4")
    return digest.hexdigest()


class ObservationSequenceDigester:
    """Streaming digest for exactly ordered pre-action RGB/state observations."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256(b"duo-vla-libero-observation-sequence-v1\0")
        self._count = 0

    def update(self, agentview: Any, wrist: Any, state: Any) -> None:
        prefix = f"frame-{self._count:06d}:"
        update_array_digest(self._digest, prefix + "agentview", agentview, dtype="|u1", shape=IMAGE_SHAPE)
        update_array_digest(self._digest, prefix + "wrist", wrist, dtype="|u1", shape=IMAGE_SHAPE)
        update_array_digest(self._digest, prefix + "state", state, dtype="<f4", shape=(STATE_DIM,))
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def hexdigest(self) -> str:
        require(self._count > 0, "cannot digest an empty observation sequence")
        copy = self._digest.copy()
        copy.update(_LENGTH.pack(self._count))
        return copy.hexdigest()


def rotate_simulator_rgb_for_training(value: Any) -> np.ndarray:
    image = np.asarray(value)
    require(image.shape == IMAGE_SHAPE, f"simulator RGB shape differs: expected {IMAGE_SHAPE}, got {image.shape}")
    require(image.dtype == np.uint8, f"simulator RGB dtype must be native uint8, got {image.dtype}")
    return np.ascontiguousarray(image[::-1, ::-1])


def canonical_proprioceptive_state(observation: Mapping[str, Any]) -> np.ndarray:
    """Return OpenVLA's six-dimensional EEF state plus two gripper coordinates."""

    from robosuite.utils import transform_utils

    position = canonical_array(observation["robot0_eef_pos"], dtype="<f8", shape=(3,))
    quaternion = canonical_array(observation["robot0_eef_quat"], dtype="<f8", shape=(4,))
    gripper = canonical_array(observation["robot0_gripper_qpos"], dtype="<f8", shape=(2,))
    axis_angle = canonical_array(transform_utils.quat2axisangle(quaternion.copy()), dtype="<f8", shape=(3,))
    return canonical_array(np.concatenate((position, axis_angle, gripper)), dtype="<f4", shape=(STATE_DIM,))


def observation_alignment_frame(
    agentview: Any,
    wrist: Any,
    state: Any,
) -> dict[str, Any]:
    def block_means(image: Any) -> list[int]:
        values = canonical_array(image, dtype="|u1", shape=IMAGE_SHAPE)
        grid = OBSERVATION_ALIGNMENT_GRID_SIZE
        block = IMAGE_SHAPE[0] // grid
        require(IMAGE_SHAPE[0] == IMAGE_SHAPE[1] == grid * block, "observation alignment grid is invalid")
        means = values.reshape(grid, block, grid, block, IMAGE_SHAPE[2]).mean(axis=(1, 3))
        return np.rint(means).astype(np.uint8).reshape(-1).tolist()

    return {
        "agentview_block_means": block_means(agentview),
        "state": canonical_array(state, dtype="<f4", shape=(STATE_DIM,)).tolist(),
        "wrist_block_means": block_means(wrist),
    }


def observation_alignment_metrics(
    simulator_features: Mapping[str, Any],
    dataset_frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(
        set(simulator_features) == {"frames", "grid_size", "schema"}
        and simulator_features["schema"] == OBSERVATION_ALIGNMENT_FEATURE_SCHEMA
        and simulator_features["grid_size"] == OBSERVATION_ALIGNMENT_GRID_SIZE,
        "simulator observation-alignment feature header differs",
    )
    simulator_frames = simulator_features["frames"]
    require(
        isinstance(simulator_frames, list) and len(simulator_frames) == len(dataset_frames) and bool(simulator_frames),
        "observation-alignment frame count differs",
    )
    expected_fields = {"agentview_block_means", "state", "wrist_block_means"}
    feature_width = OBSERVATION_ALIGNMENT_GRID_SIZE**2 * IMAGE_SHAPE[2]
    for name, frames in (("simulator", simulator_frames), ("dataset", dataset_frames)):
        require(
            all(
                isinstance(frame, Mapping)
                and set(frame) == expected_fields
                and isinstance(frame["agentview_block_means"], list)
                and len(frame["agentview_block_means"]) == feature_width
                and isinstance(frame["wrist_block_means"], list)
                and len(frame["wrist_block_means"]) == feature_width
                and isinstance(frame["state"], list)
                and len(frame["state"]) == STATE_DIM
                for frame in frames
            ),
            f"{name} observation-alignment frame shape differs",
        )

    def image_metrics(field: str) -> dict[str, float]:
        simulator = np.asarray([frame[field] for frame in simulator_frames], dtype=np.float64)
        dataset = np.asarray([frame[field] for frame in dataset_frames], dtype=np.float64)
        difference = np.abs(simulator - dataset)
        frame_mean_abs = difference.mean(axis=1)
        simulator_centered = simulator - simulator.mean(axis=1, keepdims=True)
        dataset_centered = dataset - dataset.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(simulator_centered**2, axis=1) * np.sum(dataset_centered**2, axis=1))
        require(bool(np.all(denominator > 0.0)), f"{field} observation-alignment correlation is undefined")
        correlations = np.sum(simulator_centered * dataset_centered, axis=1) / denominator
        return {
            "correlation_mean": float(correlations.mean()),
            "correlation_min": float(correlations.min()),
            "max_frame_mean_abs": float(frame_mean_abs.max()),
            "mean_abs": float(difference.mean()),
        }

    agentview = image_metrics("agentview_block_means")
    wrist = image_metrics("wrist_block_means")
    simulator_state = np.asarray([frame["state"] for frame in simulator_frames], dtype=np.float64)
    dataset_state = np.asarray([frame["state"] for frame in dataset_frames], dtype=np.float64)
    state_difference = np.abs(simulator_state - dataset_state)
    state_mean_abs = float(state_difference.mean())
    if len(dataset_frames) > 1:
        previous_dataset_state = np.concatenate((dataset_state[:1], dataset_state[:-1]), axis=0)
        next_dataset_state = np.concatenate((dataset_state[1:], dataset_state[-1:]), axis=0)
        adjacent_shift_mean_abs_min = float(
            min(
                np.abs(simulator_state - previous_dataset_state).mean(),
                np.abs(simulator_state - next_dataset_state).mean(),
            )
        )
        temporal_margin = adjacent_shift_mean_abs_min - state_mean_abs
    else:
        adjacent_shift_mean_abs_min = state_mean_abs
        temporal_margin = 0.0
    state_metrics = {
        "adjacent_shift_mean_abs_min": adjacent_shift_mean_abs_min,
        "max_abs": float(state_difference.max()),
        "mean_abs": state_mean_abs,
        "temporal_margin": temporal_margin,
    }
    thresholds = {
        "image_correlation_frame_min": OBSERVATION_ALIGNMENT_IMAGE_CORRELATION_FRAME_MIN,
        "image_correlation_mean_min": OBSERVATION_ALIGNMENT_IMAGE_CORRELATION_MEAN_MIN,
        "image_max_frame_mean_abs_max": OBSERVATION_ALIGNMENT_IMAGE_MAX_FRAME_MEAN_ABS_MAX,
        "image_mean_abs_max": OBSERVATION_ALIGNMENT_IMAGE_MEAN_ABS_MAX,
        "state_max_abs_max": OBSERVATION_ALIGNMENT_STATE_MAX_ABS_MAX,
        "state_mean_abs_max": OBSERVATION_ALIGNMENT_STATE_MEAN_ABS_MAX,
        "state_temporal_margin_min": OBSERVATION_ALIGNMENT_STATE_TEMPORAL_MARGIN_MIN,
    }
    passed = (
        all(
            camera["mean_abs"] <= thresholds["image_mean_abs_max"]
            and camera["max_frame_mean_abs"] <= thresholds["image_max_frame_mean_abs_max"]
            and camera["correlation_mean"] >= thresholds["image_correlation_mean_min"]
            and camera["correlation_min"] >= thresholds["image_correlation_frame_min"]
            for camera in (agentview, wrist)
        )
        and state_metrics["mean_abs"] <= thresholds["state_mean_abs_max"]
        and state_metrics["max_abs"] <= thresholds["state_max_abs_max"]
        and (len(dataset_frames) == 1 or state_metrics["temporal_margin"] >= thresholds["state_temporal_margin_min"])
    )
    return {
        "agentview": agentview,
        "frames": len(simulator_frames),
        "passed": passed,
        "state": state_metrics,
        "thresholds": thresholds,
        "wrist": wrist,
    }


def validate_observation_alignment_metrics(value: Any) -> dict[str, Any]:
    require(
        isinstance(value, dict) and set(value) == {"agentview", "frames", "passed", "state", "thresholds", "wrist"},
        "observation-alignment metric fields differ",
    )
    expected_thresholds = {
        "image_correlation_frame_min": OBSERVATION_ALIGNMENT_IMAGE_CORRELATION_FRAME_MIN,
        "image_correlation_mean_min": OBSERVATION_ALIGNMENT_IMAGE_CORRELATION_MEAN_MIN,
        "image_max_frame_mean_abs_max": OBSERVATION_ALIGNMENT_IMAGE_MAX_FRAME_MEAN_ABS_MAX,
        "image_mean_abs_max": OBSERVATION_ALIGNMENT_IMAGE_MEAN_ABS_MAX,
        "state_max_abs_max": OBSERVATION_ALIGNMENT_STATE_MAX_ABS_MAX,
        "state_mean_abs_max": OBSERVATION_ALIGNMENT_STATE_MEAN_ABS_MAX,
        "state_temporal_margin_min": OBSERVATION_ALIGNMENT_STATE_TEMPORAL_MARGIN_MIN,
    }
    require(value["thresholds"] == expected_thresholds, "observation-alignment thresholds differ")
    require(type(value["frames"]) is int and value["frames"] > 0, "observation-alignment frame count differs")
    for name in ("agentview", "wrist"):
        camera = value[name]
        require(
            isinstance(camera, dict)
            and set(camera) == {"correlation_mean", "correlation_min", "max_frame_mean_abs", "mean_abs"}
            and all(
                isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
                for item in camera.values()
            ),
            f"{name} observation-alignment metrics differ",
        )
    state = value["state"]
    require(
        isinstance(state, dict)
        and set(state) == {"adjacent_shift_mean_abs_min", "max_abs", "mean_abs", "temporal_margin"}
        and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
            for item in state.values()
        ),
        "state observation-alignment metrics differ",
    )
    passed = (
        all(
            value[name]["mean_abs"] <= expected_thresholds["image_mean_abs_max"]
            and value[name]["max_frame_mean_abs"] <= expected_thresholds["image_max_frame_mean_abs_max"]
            and value[name]["correlation_mean"] >= expected_thresholds["image_correlation_mean_min"]
            and value[name]["correlation_min"] >= expected_thresholds["image_correlation_frame_min"]
            for name in ("agentview", "wrist")
        )
        and state["mean_abs"] <= expected_thresholds["state_mean_abs_max"]
        and state["max_abs"] <= expected_thresholds["state_max_abs_max"]
        and (value["frames"] == 1 or state["temporal_margin"] >= expected_thresholds["state_temporal_margin_min"])
    )
    require(value["passed"] is passed, "observation-alignment pass result differs")
    return value


def initial_state_sha256(value: Any) -> str:
    digest = hashlib.sha256(b"duo-vla-libero-initial-simulator-state-v1\0")
    array = np.asarray(value)
    require(array.ndim == 1 and len(array) > 0, "initial simulator state must be a non-empty vector")
    update_array_digest(digest, "initial_state", array, dtype="<f8")
    return digest.hexdigest()


def trajectory_sha256(
    *,
    suite: str,
    task_id: int,
    source_episode_index: int,
    initial_state_digest: str,
    action_digest: str,
    observation_digest: str,
) -> str:
    for value, name in (
        (initial_state_digest, "initial state"),
        (action_digest, "action sequence"),
        (observation_digest, "observation sequence"),
    ):
        _require_sha256(value, name)
    return canonical_sha256(
        {
            "action_sequence_sha256": action_digest,
            "initial_state_sha256": initial_state_digest,
            "observation_sequence_sha256": observation_digest,
            "source_episode_index": source_episode_index,
            "suite": suite,
            "task_id": task_id,
        }
    )


def task_slug(suite: str, task_id: int) -> str:
    require(suite in SUITES and type(task_id) is int and 0 <= task_id < 10, "invalid LIBERO task identity")
    return f"{suite}-{task_id:02d}"


def sorted_demo_names(names: Iterable[str]) -> tuple[str, ...]:
    values = tuple(names)
    require(bool(values), "HDF5 task contains no demonstrations")
    parsed: list[tuple[int, str]] = []
    for name in values:
        require(isinstance(name, str) and name.startswith("demo_") and name[5:].isdigit(), "invalid HDF5 demo name")
        parsed.append((int(name[5:]), name))
    parsed.sort()
    require([index for index, _ in parsed] == list(range(len(parsed))), "HDF5 demo indices are not contiguous")
    return tuple(name for _, name in parsed)


def raw_evidence_record(root: Path, relative: str, identifier: str) -> dict[str, Any]:
    path = root / relative
    identity = stable_regular_file_identity(path)
    return {"bytes": identity["bytes"], "id": identifier, "path": relative, "sha256": identity["sha256"]}


def digest_records(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256([dict(record) for record in records])


# Shared qualification contract used by the simulator collector and parquet
# binder.  It lives under ``duo_vla`` so neither executable imports from the
# scripts directory, which must never be present on their module search path.


def _read_descriptor_json(descriptor: int, *, name: str) -> tuple[dict[str, Any], bytes]:
    chunks: list[bytes] = []
    before = os.fstat(descriptor)
    require(stat.S_ISREG(before.st_mode), f"{name} is not a regular file")
    with os.fdopen(os.dup(descriptor), "rb") as source:
        while block := source.read(1024 * 1024):
            chunks.append(block)
        after = os.fstat(source.fileno())
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    require(all(getattr(before, field) == getattr(after, field) for field in stable), f"{name} changed while read")
    raw = b"".join(chunks)
    require(len(raw) == after.st_size, f"{name} byte count changed while read")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ReplayEvidenceError(f"{name} is not strict finite UTF-8 JSON") from exc
    require(isinstance(value, dict), f"{name} root must be an object")
    require_finite_json_numbers(value, name=name)
    return value, raw


def read_stable_json(path: Path, *, name: str, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise ReplayEvidenceError(f"cannot open {name}: {path}") from exc
    try:
        value, raw = _read_descriptor_json(descriptor, name=name)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, f"expected {name} SHA-256")
        require(hmac.compare_digest(digest, expected_sha256), f"{name} differs from its externally recorded SHA-256")
    return value, digest


def stable_file_sha256(path: Path) -> str:
    return stable_regular_file_identity(path)["sha256"]


def training_source_tree_sha256(root: Path) -> str:
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
            root / "scripts/run_libero_train_single_gpu.sh",
            root / "scripts/bootstrap_train_single_gpu_env.sh",
            root / "scripts/train_libero.py",
            root / "envs/train-single-gpu/pyproject.toml",
            root / "envs/train-single-gpu/uv.lock",
            root / "pyproject.toml",
            root / "uv.lock",
        )
        if path.is_file()
    )
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("ascii"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_identity(project_root: Path, *, single_gpu: bool = False) -> dict[str, Any]:
    binder_launcher = (
        "run_bind_libero_expert_replay_single_gpu.sh" if single_gpu else "run_bind_libero_expert_replay.sh"
    )
    qualification_launcher = (
        "run_qualify_libero_expert_replay_single_gpu.sh" if single_gpu else "run_qualify_libero_expert_replay.sh"
    )
    training_launcher = "run_libero_train_single_gpu.sh" if single_gpu else "run_libero_train.sh"
    named_paths = {
        "data_reader": project_root / "src/duo_vla/data/libero.py",
        "normalization": project_root / "src/duo_vla/data/libero_stats.py",
        "expert_replay_binder": project_root / "scripts/bind_libero_expert_replay.py",
        "expert_replay_binder_launcher": project_root / "scripts" / binder_launcher,
        "expert_replay_collector": project_root / "scripts/collect_libero_expert_replay.py",
        "expert_replay_collector_launcher": project_root / "scripts/run_collect_libero_expert_replay.sh",
        "expert_replay_contract": project_root / "src/duo_vla/libero_replay_evidence.py",
        "original_hdf5_inventory": project_root / "configs/libero_original_hdf5_inventory.json",
        "source_parquet_alignment": project_root / "configs/libero_source_parquet_alignment.json",
        "qualification": project_root / "scripts/qualify_libero_expert_replay.py",
        "qualification_launcher": project_root / "scripts" / qualification_launcher,
        "simulator_preflight": project_root / "scripts/preflight_libero_env.py",
        "simulator_preflight_launcher": project_root / "scripts/run_libero_preflight.sh",
        "training_launcher": project_root / "scripts" / training_launcher,
        "training_program": project_root / "scripts/train_libero.py",
    }
    source_files = {name: stable_file_sha256(path) for name, path in named_paths.items()}
    configs = {
        "direct_regression": stable_file_sha256(
            project_root
            / "configs"
            / ("libero_direct_regression_single_gpu.toml" if single_gpu else "libero_direct_regression.toml")
        ),
        "rectified_flow": stable_file_sha256(
            project_root / "configs" / ("libero_single_gpu.toml" if single_gpu else "libero.toml")
        ),
    }
    return {
        "config_file_sha256": configs,
        "project_source_tree_sha256": training_source_tree_sha256(project_root),
        "source_files_sha256": source_files,
    }


def validate_normalization(value: Mapping[str, Any]) -> None:
    require(value.get("schema") == "duo-vla-libero-normalization-v1", "normalization schema mismatch")
    recorded = _require_sha256(value.get("content_sha256"), "normalization content_sha256")
    unsigned = {name: item for name, item in value.items() if name != "content_sha256"}
    require(hmac.compare_digest(canonical_sha256(unsigned), recorded), "normalization semantic self-hash mismatch")
    require(recorded == NORMALIZATION_CONTENT_SHA256, "normalization content identity mismatch")
    dataset = value.get("dataset")
    counts = value.get("counts")
    action = value.get("action")
    require(isinstance(dataset, Mapping), "normalization dataset identity is invalid")
    require(dataset.get("id") == "HuggingFaceVLA/libero", "normalization dataset ID mismatch")
    require(dataset.get("revision") == DATASET_REVISION, "normalization dataset revision mismatch")
    require(isinstance(counts, Mapping), "normalization counts are invalid")
    require(
        {name: counts.get(name) for name in ("tasks", "total_episodes", "total_frames")}
        == {"tasks": 40, "total_episodes": 1693, "total_frames": 273465},
        "normalization dataset counts mismatch",
    )
    require(isinstance(action, Mapping), "normalization action contract is invalid")
    require(action.get("observed_gripper_values") == [-1.0, 1.0], "normalization gripper set mismatch")


def dataset_content_inventory_sha256(value: Mapping[str, Any]) -> str:
    files = value.get("files")
    require(isinstance(files, Mapping), "dataset tree file inventory is invalid")
    records: list[dict[str, Any]] = []
    for name in sorted(set(files) - {".gitattributes"}):
        entry = files[name]
        require(isinstance(entry, Mapping), f"dataset tree entry is invalid: {name}")
        if "lfs_sha256" in entry:
            algorithm = "sha256"
            size = entry.get("lfs_size")
            digest = entry.get("lfs_sha256")
        else:
            algorithm = "git-sha1"
            size = entry.get("size")
            digest = entry.get("blob_id")
        require(type(size) is int and size >= 0, f"dataset tree entry byte count is invalid: {name}")
        if algorithm == "sha256":
            _require_sha256(digest, f"dataset tree entry digest {name}")
        else:
            require(
                isinstance(digest, str)
                and len(digest) == 40
                and all(character in "0123456789abcdef" for character in digest),
                f"dataset tree Git digest is invalid: {name}",
            )
        records.append({"algorithm": algorithm, "bytes": size, "digest": digest, "path": name})
    return canonical_sha256(records)


def validate_dataset_tree(value: Mapping[str, Any]) -> None:
    require(set(value) == {"files", "format_version"}, "dataset tree metadata fields differ")
    require(value["format_version"] == 1, "dataset tree format version mismatch")
    files = value["files"]
    require(isinstance(files, Mapping), "dataset tree file inventory is invalid")
    require(len(files) == DATASET_TREE_FILE_COUNT, "dataset tree file count mismatch")
    sizes = [item.get("size") for item in files.values() if isinstance(item, Mapping)]
    require(
        len(sizes) == DATASET_TREE_FILE_COUNT
        and all(type(size) is int and size >= 0 for size in sizes)
        and sum(sizes) == DATASET_TREE_TOTAL_BYTES,
        "dataset tree total byte count mismatch",
    )
    require(
        dataset_content_inventory_sha256(value) == DATASET_CONTENT_INVENTORY_SHA256,
        "dataset content inventory identity mismatch",
    )


def task_identities(attestation: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(attestation.get("schema") == SIMULATOR_ATTESTATION_SCHEMA, "simulator attestation schema mismatch")
    require(attestation.get("status") == "ok", "simulator attestation did not pass")
    require(
        attestation.get("environment_constructed") is True, "simulator attestation did not construct an environment"
    )
    require(attestation.get("task_inventory_count") == 40, "simulator task inventory count mismatch")
    require(attestation.get("task_inventory_sha256") == TASK_INVENTORY_SHA256, "simulator task inventory mismatch")
    inventory = attestation.get("task_inventory")
    require(isinstance(inventory, list) and len(inventory) == 40, "simulator task inventory is incomplete")
    require(
        canonical_sha256(inventory) == attestation["task_inventory_sha256"] == TASK_INVENTORY_SHA256,
        "simulator task inventory content does not match its declared identity",
    )
    tasks: list[dict[str, Any]] = []
    fields = {"instruction", "suite", "task_id", "task_name"}
    for index, item in enumerate(inventory):
        require(isinstance(item, Mapping), f"simulator task inventory entry {index} is invalid")
        task = {name: item.get(name) for name in fields}
        require(
            isinstance(task["suite"], str)
            and type(task["task_id"]) is int
            and isinstance(task["task_name"], str)
            and bool(task["task_name"])
            and isinstance(task["instruction"], str)
            and bool(task["instruction"]),
            f"simulator task inventory entry {index} has invalid identity fields",
        )
        require(item.get("reset_count") == 50, f"simulator task {index} reset count mismatch")
        tasks.append(task)
    expected_pairs = [(suite, task_id) for suite in SUITES for task_id in range(10)]
    require(
        [(task["suite"], task["task_id"]) for task in tasks] == expected_pairs,
        "simulator task inventory is not in canonical 40-task order",
    )
    return tasks


def simulator_runtime_sha256(attestation: Mapping[str, Any]) -> str:
    names = (
        "assets",
        "backend",
        "eval_venv_identity",
        "distribution_records",
        "egl_device",
        "evaluator_lock_sha256",
        "installed_distributions",
        "manifest_sha256",
        "module_origins",
        "opengl",
        "packages",
        "process",
        "project_sources",
        "site_packages",
        "source",
        "torch",
    )
    runtime = {name: attestation.get(name) for name in names}
    require(
        all(value is not None for value in runtime.values()), "simulator attestation runtime identity is incomplete"
    )
    return canonical_sha256(runtime)


def validate_dataset_snapshot_report(value: Mapping[str, Any]) -> None:
    require(
        set(value)
        == {
            "content_inventory_sha256",
            "files_verified",
            "revision",
            "snapshot",
            "total_bytes",
            "tree_metadata_sha256",
        },
        "live dataset snapshot report fields differ",
    )
    require(value["revision"] == DATASET_REVISION, "live dataset snapshot revision mismatch")
    require(
        value["tree_metadata_sha256"] == DATASET_TREE_METADATA_SHA256
        and value["content_inventory_sha256"] == DATASET_CONTENT_INVENTORY_SHA256,
        "live dataset snapshot content identity mismatch",
    )
    require(
        value["files_verified"] == DATASET_SNAPSHOT_FILES_VERIFIED
        and value["total_bytes"] == DATASET_SNAPSHOT_TOTAL_BYTES,
        "live dataset snapshot verified counts mismatch",
    )
    require(isinstance(value["snapshot"], str) and bool(value["snapshot"]), "live dataset snapshot path is invalid")


def validate_train_venv_identity(value: Any) -> dict[str, Any]:
    from duo_vla.runtime_integrity import require_matching_train_venv

    require(isinstance(value, dict), "train-venv identity must be an object")
    try:
        return require_matching_train_venv(value, dict(value))
    except RuntimeError as exc:
        raise ReplayEvidenceError(str(exc)) from exc


def build_expected_inputs(
    project_root: Path,
    *,
    simulator_attestation: Mapping[str, Any],
    simulator_attestation_raw_sha256: str,
    dataset_tree: Mapping[str, Any],
    dataset_tree_raw_sha256: str,
    normalization: Mapping[str, Any],
    normalization_raw_sha256: str,
    dataset_snapshot: Mapping[str, Any],
    train_venv_identity: Mapping[str, Any],
    original_hdf5_inventory: Mapping[str, Any] | None = None,
    original_hdf5_inventory_raw_sha256: str | None = None,
) -> dict[str, Any]:
    task_identities(simulator_attestation)
    validate_dataset_tree(dataset_tree)
    validate_normalization(normalization)
    require(dataset_tree_raw_sha256 == DATASET_TREE_METADATA_SHA256, "dataset tree metadata raw identity mismatch")
    require(normalization_raw_sha256 == NORMALIZATION_RAW_SHA256, "normalization raw identity mismatch")
    validate_dataset_snapshot_report(dataset_snapshot)
    validated_train_venv = validate_train_venv_identity(dict(train_venv_identity))
    if original_hdf5_inventory is None:
        inventory_path = project_root / "configs/libero_original_hdf5_inventory.json"
        original_hdf5_inventory, _records, observed_inventory_raw_sha256 = load_original_hdf5_inventory(inventory_path)
        if original_hdf5_inventory_raw_sha256 is None:
            original_hdf5_inventory_raw_sha256 = observed_inventory_raw_sha256
    else:
        validate_original_hdf5_inventory(original_hdf5_inventory)
    _require_sha256(original_hdf5_inventory_raw_sha256, "original HDF5 inventory raw SHA-256")
    require(
        original_hdf5_inventory.get("content_sha256") == ORIGINAL_HDF5_CONTENT_SHA256,
        "original HDF5 inventory content identity mismatch",
    )
    train_venv_root = Path(validated_train_venv["root"])
    identity = source_identity(project_root, single_gpu=train_venv_root.name == "train-single-gpu")
    return {
        "config_file_sha256": identity["config_file_sha256"],
        "dataset_content_inventory_sha256": DATASET_CONTENT_INVENTORY_SHA256,
        "dataset_revision": DATASET_REVISION,
        "dataset_snapshot_files_verified": DATASET_SNAPSHOT_FILES_VERIFIED,
        "dataset_snapshot_total_bytes": DATASET_SNAPSHOT_TOTAL_BYTES,
        "dataset_tree_file_count": DATASET_TREE_FILE_COUNT,
        "dataset_tree_metadata_sha256": dataset_tree_raw_sha256,
        "dataset_tree_total_bytes": DATASET_TREE_TOTAL_BYTES,
        "normalization_content_sha256": NORMALIZATION_CONTENT_SHA256,
        "normalization_raw_sha256": normalization_raw_sha256,
        "original_hdf5_file_count": ORIGINAL_HDF5_FILE_COUNT,
        "original_hdf5_inventory_content_sha256": ORIGINAL_HDF5_CONTENT_SHA256,
        "original_hdf5_inventory_raw_sha256": original_hdf5_inventory_raw_sha256,
        "original_hdf5_repository_id": ORIGINAL_HDF5_REPOSITORY_ID,
        "original_hdf5_revision": ORIGINAL_HDF5_REVISION,
        "original_hdf5_total_bytes": ORIGINAL_HDF5_TOTAL_BYTES,
        "project_source_tree_sha256": identity["project_source_tree_sha256"],
        "simulator_attestation_raw_sha256": simulator_attestation_raw_sha256,
        "simulator_attestation_sha256": canonical_sha256(simulator_attestation),
        "simulator_runtime_sha256": simulator_runtime_sha256(simulator_attestation),
        "source_files_sha256": identity["source_files_sha256"],
        "task_inventory_sha256": TASK_INVENTORY_SHA256,
        "train_venv_identity": json.loads(canonical_json_bytes(validated_train_venv).decode("ascii")),
    }


def canonical_gate_results() -> dict[str, dict[str, Any]]:
    return {
        "schema_revision_counts": {
            "dataset_revision": DATASET_REVISION,
            "declared_file_pointer_mismatches": 1690,
            "episodes": 1693,
            "frames": 273465,
            "physical_data_files": 377,
            "tasks": 40,
        },
        "exact_gripper_set": {"observed_values": [-1.0, 1.0], "unexpected_value_count": 0, "zero_value_count": 0},
        "controller_impulse_directions": {
            "action_dimension": 7,
            "checked_directions": list(IMPULSE_DIRECTIONS),
            "controller": "OSC_POSE",
            "direction_match_count": len(IMPULSE_DIRECTIONS),
            "gripper_close": 1.0,
            "gripper_open": -1.0,
        },
        "normalization_round_trip": {
            "continuous_action_dimensions": 6,
            "gripper_sign_exact": True,
            "max_abs_error": 0.0,
            "samples": 1,
            "state_dimensions": 8,
        },
        "episode_boundary_chunk_fixture": {
            "checked_terminal_anchors": 1,
            "cross_boundary_count": 0,
            "horizon": 8,
            "padding_mask_exact": True,
            "padding_value": "zeros",
        },
        "pre_action_observation_alignment": {
            "action_index": "t",
            "checked_tasks": 40,
            "checked_transitions": 40,
            "exact_match_count": 40,
            "observation_index": "t",
            "post_action_pairing_rejected": True,
        },
        "camera_transform_parity": {
            "checked_tasks": 40,
            "dataset_transform": "already_rotated_no_additional_transform",
            "dtype": "uint8",
            "image_shape": [256, 256, 3],
            "pixel_parity_all": True,
            "positive_stride_contiguous": True,
            "processor_order": ["agentview", "eye_in_hand"],
            "rollout_transform": "rotate_180_once",
        },
        "deterministic_fixed_state_reset": {
            "all_serialized_state_repeats_equal": True,
            "checked_tasks": 40,
            "environment_seed": ENVIRONMENT_SEED,
            "official_fixed_states_used": False,
            "resets_per_task": 2,
        },
        "pre_dispatch_integrity_controls": {
            "checked_tasks": 40,
            "mutations": {name: {"mutation_detected": True} for name in PRE_DISPATCH_MUTATIONS},
        },
    }
