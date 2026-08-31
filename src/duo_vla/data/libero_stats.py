"""Exact, revision-bound normalization statistics for the pinned LIBERO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from duo_vla.data.libero import LiberoParquetDataset, _pyarrow_parquet
from duo_vla.data.sampling import EpisodeSplit, make_task_stratified_episode_split
from duo_vla.normalization import ActionNormalizer, PercentileNormalizer

LIBERO_DATASET_ID = "HuggingFaceVLA/libero"
LIBERO_DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
LIBERO_STATS_SCHEMA = "duo-vla-libero-normalization-v1"
DEFAULT_SPLIT_SEED = 1729


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _content_hash(payload: dict[str, Any]) -> str:
    without_hash = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_json(without_hash).encode()).hexdigest()


def _indices_hash(indices: tuple[int, ...]) -> str:
    serialized = ",".join(map(str, indices)).encode()
    return hashlib.sha256(serialized).hexdigest()


def percentile_bounds(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate source values and return exact linear q01/q99 bounds."""

    states = np.asarray(states)
    actions = np.asarray(actions)
    if states.ndim != 2 or states.shape[1] != 8:
        raise ValueError("states must have shape [frames, 8]")
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("actions must have shape [frames, 7]")
    if states.shape[0] == 0 or states.shape[0] != actions.shape[0]:
        raise ValueError("states/actions must have the same positive frame count")
    if not np.issubdtype(states.dtype, np.floating) or not np.issubdtype(actions.dtype, np.floating):
        raise TypeError("states and actions must be floating point")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("states/actions contain non-finite values")
    if np.any(np.abs(actions[:, :6]) > 1.0 + 1e-6):
        raise ValueError("LIBERO continuous action values must be in [-1, 1]")
    gripper_values = np.unique(actions[:, 6])
    if not np.array_equal(gripper_values, np.array([-1.0, 1.0], dtype=gripper_values.dtype)):
        raise ValueError("LIBERO gripper values must contain exactly {-1, +1}")

    state_q01, state_q99 = np.quantile(states, (0.01, 0.99), axis=0, method="linear")
    action_q01, action_q99 = np.quantile(actions[:, :6], (0.01, 0.99), axis=0, method="linear")
    return state_q01, state_q99, action_q01, action_q99


def compute_libero_normalization_artifact(
    dataset: LiberoParquetDataset,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    validation_fraction: float = 0.1,
    dataset_revision: str = LIBERO_DATASET_REVISION,
) -> dict[str, Any]:
    """Scan every source row once, validate storage semantics, and fit train-only percentiles."""

    split = make_task_stratified_episode_split(
        dataset.episodes,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )
    training_lookup = np.zeros(len(dataset.episodes), dtype=np.bool_)
    training_lookup[np.asarray(split.train_episode_indices, dtype=np.int64)] = True
    training_frames = sum(dataset.episodes[index].length for index in split.train_episode_indices)
    states = np.empty((training_frames, 8), dtype=np.float32)
    actions = np.empty((training_frames, 7), dtype=np.float32)

    episode_starts = np.asarray([episode.global_start for episode in dataset.episodes], dtype=np.int64)
    episode_lengths = np.asarray([episode.length for episode in dataset.episodes], dtype=np.int64)
    task_index_by_name = {task: index for index, task in dataset.task_by_index.items()}
    episode_tasks = np.asarray([task_index_by_name[episode.task] for episode in dataset.episodes], dtype=np.int64)
    observed_episode_lengths = np.zeros(len(dataset.episodes), dtype=np.int64)
    training_task_frames: Counter[str] = Counter()
    cursor = 0

    for data_file in dataset.data_files:
        path = dataset.root / "data" / f"chunk-{data_file.chunk_index:03d}" / f"file-{data_file.file_index:03d}.parquet"
        table = _pyarrow_parquet().read_table(
            path,
            columns=[
                "index",
                "episode_index",
                "frame_index",
                "task_index",
                "observation.state",
                "action",
            ],
        )
        indices = np.asarray(table.column("index").to_numpy(), dtype=np.int64)
        expected_indices = np.arange(data_file.global_start, data_file.global_stop, dtype=np.int64)
        if not np.array_equal(indices, expected_indices):
            raise ValueError(f"global index mismatch in {path}")
        episode_indices = np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)
        if np.any((episode_indices < 0) | (episode_indices >= len(dataset.episodes))):
            raise ValueError(f"unknown episode index in {path}")
        frame_indices = np.asarray(table.column("frame_index").to_numpy(), dtype=np.int64)
        if not np.array_equal(frame_indices, indices - episode_starts[episode_indices]):
            raise ValueError(f"episode-local frame index mismatch in {path}")
        task_indices = np.asarray(table.column("task_index").to_numpy(), dtype=np.int64)
        if not np.array_equal(task_indices, episode_tasks[episode_indices]):
            raise ValueError(f"task index mismatch in {path}")
        observed_episode_lengths += np.bincount(episode_indices, minlength=len(dataset.episodes))

        file_states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        file_actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        if file_states.shape != (len(table), 8) or file_actions.shape != (len(table), 7):
            raise ValueError(f"state/action shape mismatch in {path}")
        if not np.isfinite(file_states).all() or not np.isfinite(file_actions).all():
            raise ValueError(f"non-finite state/action value in {path}")
        selected = training_lookup[episode_indices]
        selected_count = int(selected.sum())
        states[cursor : cursor + selected_count] = file_states[selected]
        actions[cursor : cursor + selected_count] = file_actions[selected]
        for task_index, count in zip(*np.unique(task_indices[selected], return_counts=True), strict=True):
            training_task_frames[dataset.task_by_index[int(task_index)]] += int(count)
        cursor += selected_count

    if not np.array_equal(observed_episode_lengths, episode_lengths):
        raise ValueError("physical rows do not match every declared episode length")
    if cursor != training_frames:
        raise ValueError("collected training frame count does not match the episode split")
    state_q01, state_q99, action_q01, action_q99 = percentile_bounds(states, actions)
    split_payload = _split_payload(split, split_seed=split_seed, validation_fraction=validation_fraction)
    payload: dict[str, Any] = {
        "schema": LIBERO_STATS_SCHEMA,
        "dataset": {
            "id": LIBERO_DATASET_ID,
            "revision": dataset_revision,
            "physical_data_files": len(dataset.data_files),
            "declared_file_pointer_mismatches": dataset.declared_file_mismatch_count,
        },
        "split": split_payload,
        "counts": {
            "total_episodes": len(dataset.episodes),
            "total_frames": int(dataset.info["total_frames"]),
            "tasks": len(dataset.task_by_index),
            "training_episodes": len(split.train_episode_indices),
            "training_frames": training_frames,
            "validation_episodes": len(split.validation_episode_indices),
            "validation_frames": int(dataset.info["total_frames"]) - training_frames,
            "training_frames_by_task": dict(sorted(training_task_frames.items())),
        },
        "state": {
            "dimension": 8,
            "q01": state_q01.tolist(),
            "q99": state_q99.tolist(),
            "constant_dimensions": np.flatnonzero(np.abs(state_q99 - state_q01) < 1e-6).tolist(),
        },
        "action": {
            "dimension": 7,
            "continuous_dimensions": [0, 1, 2, 3, 4, 5],
            "gripper_index": 6,
            "observed_gripper_values": [-1.0, 1.0],
            "q01": action_q01.tolist(),
            "q99": action_q99.tolist(),
            "constant_dimensions": np.flatnonzero(np.abs(action_q99 - action_q01) < 1e-6).tolist(),
        },
        "algorithm": {
            "quantile": "numpy.quantile(method=linear)",
            "source_dtype": "float32",
            "result_dtype": "float64",
            "row_order": "physical parquet global index ascending",
            "gripper_excluded_from_percentiles": True,
        },
    }
    payload["content_sha256"] = _content_hash(payload)
    return payload


def _split_payload(split: EpisodeSplit, *, split_seed: int, validation_fraction: float) -> dict[str, Any]:
    return {
        "algorithm": "task-stratified stable sha256 episode ordering",
        "seed": split_seed,
        "validation_fraction": validation_fraction,
        "train_episode_indices": list(split.train_episode_indices),
        "validation_episode_indices": list(split.validation_episode_indices),
        "train_episode_sha256": _indices_hash(split.train_episode_indices),
        "validation_episode_sha256": _indices_hash(split.validation_episode_indices),
    }


def save_libero_normalization_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    if payload.get("schema") != LIBERO_STATS_SCHEMA or payload.get("content_sha256") != _content_hash(payload):
        raise ValueError("normalization payload schema or content hash is invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def load_libero_normalizers(
    path: str | Path,
    *,
    expected_revision: str = LIBERO_DATASET_REVISION,
) -> tuple[PercentileNormalizer, ActionNormalizer, dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema") != LIBERO_STATS_SCHEMA:
        raise ValueError("unsupported LIBERO normalization artifact schema")
    if payload.get("content_sha256") != _content_hash(payload):
        raise ValueError("LIBERO normalization artifact content hash mismatch")
    dataset = payload.get("dataset", {})
    if dataset.get("id") != LIBERO_DATASET_ID or dataset.get("revision") != expected_revision:
        raise ValueError("LIBERO normalization artifact dataset identity mismatch")
    state = payload.get("state", {})
    action = payload.get("action", {})
    state_normalizer = PercentileNormalizer(
        torch.tensor(state.get("q01"), dtype=torch.float32),
        torch.tensor(state.get("q99"), dtype=torch.float32),
    )
    action_normalizer = ActionNormalizer(
        PercentileNormalizer(
            torch.tensor(action.get("q01"), dtype=torch.float32),
            torch.tensor(action.get("q99"), dtype=torch.float32),
        )
    )
    return state_normalizer, action_normalizer, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--dataset-revision", default=LIBERO_DATASET_REVISION)
    args = parser.parse_args()
    dataset = LiberoParquetDataset(args.snapshot_root)
    artifact = compute_libero_normalization_artifact(
        dataset,
        split_seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        dataset_revision=args.dataset_revision,
    )
    save_libero_normalization_artifact(args.output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
