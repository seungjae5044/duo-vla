"""Pinned LeRobot-v3 LIBERO parquet reader with episode-safe samples."""

from __future__ import annotations

import io
import json
import re
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from duo_vla.benchmarks.common import CanonicalObservation, PaddedActionChunk
from duo_vla.benchmarks.libero import LiberoProtocol, make_libero_action_chunk, validate_libero_dataset_actions
from duo_vla.data.sampling import LiberoAnchor


@dataclass(frozen=True, slots=True)
class LiberoEpisode:
    episode_index: int
    chunk_index: int
    file_index: int
    global_start: int
    global_stop: int
    task: str

    @property
    def length(self) -> int:
        return self.global_stop - self.global_start


@dataclass(frozen=True, slots=True)
class LiberoDataFile:
    chunk_index: int
    file_index: int
    global_start: int
    global_stop: int

    @property
    def length(self) -> int:
        return self.global_stop - self.global_start


@dataclass(frozen=True, slots=True)
class LiberoTrainingSample:
    observation: CanonicalObservation
    instruction: str
    action_chunk: PaddedActionChunk
    episode_index: int
    frame_index: int
    task_index: int


class LiberoParquetDataset:
    """Random-access local reader that never infers episode boundaries from file boundaries."""

    def __init__(self, snapshot_root: str | Path, *, max_cached_files: int = 4) -> None:
        if max_cached_files <= 0:
            raise ValueError("max_cached_files must be positive")
        self.root = Path(snapshot_root)
        self.max_cached_files = max_cached_files
        self.protocol = LiberoProtocol()
        self.info = _load_json(self.root / "meta" / "info.json")
        self._validate_info()
        pyarrow_parquet = _pyarrow_parquet()
        task_table = pyarrow_parquet.read_table(self.root / "meta" / "tasks.parquet")
        task_rows = task_table.to_pylist()
        self.task_by_index = {int(row["task_index"]): str(row["__index_level_0__"]) for row in task_rows}
        if len(self.task_by_index) != 40 or len(set(self.task_by_index.values())) != 40:
            raise ValueError("LIBERO task mapping must be a 40-entry bijection")

        episode_table = pyarrow_parquet.read_table(
            self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
            columns=[
                "episode_index",
                "data/chunk_index",
                "data/file_index",
                "dataset_from_index",
                "dataset_to_index",
                "tasks",
                "length",
            ],
        )
        episodes: list[LiberoEpisode] = []
        for row in episode_table.to_pylist():
            tasks = row["tasks"]
            if not isinstance(tasks, list) or len(tasks) != 1 or tasks[0] not in self.task_by_index.values():
                raise ValueError("every LIBERO episode must contain exactly one canonical task")
            episode = LiberoEpisode(
                episode_index=int(row["episode_index"]),
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
                global_start=int(row["dataset_from_index"]),
                global_stop=int(row["dataset_to_index"]),
                task=str(tasks[0]),
            )
            if episode.length != int(row["length"]) or episode.length <= 0:
                raise ValueError("LIBERO episode length metadata is inconsistent")
            episodes.append(episode)
        if len(episodes) != self.info["total_episodes"]:
            raise ValueError("LIBERO episode metadata count does not match info.json")
        self.episodes = tuple(episodes)
        self.data_files = self._discover_data_files()
        self._data_file_by_episode = resolve_episode_data_files(self.episodes, self.data_files)
        self.declared_file_mismatch_count = sum(
            (episode.chunk_index, episode.file_index)
            != (
                self._data_file_by_episode[episode.episode_index].chunk_index,
                self._data_file_by_episode[episode.episode_index].file_index,
            )
            for episode in self.episodes
        )
        self._data_file_cache: OrderedDict[tuple[int, int], Any] = OrderedDict()

    def sample(self, episode_index: int, frame_index: int, *, horizon: int = 8) -> LiberoTrainingSample:
        episode = self.episodes[episode_index]
        if episode.episode_index != episode_index:
            raise ValueError("episode metadata is not indexed contiguously")
        if not 0 <= frame_index < episode.length:
            raise IndexError("frame_index is outside the episode")
        table = self._read_episode_rows(episode)
        actions = self._validated_episode_actions(table)
        return self._sample_from_episode_rows(episode, frame_index, table, actions, horizon=horizon)

    def sample_many(
        self,
        anchors: Sequence[LiberoAnchor],
        *,
        horizon: int = 8,
    ) -> tuple[LiberoTrainingSample, ...]:
        """Read distinct anchors once per physical parquet while preserving requested order."""

        requested = tuple(anchors)
        if not requested:
            raise ValueError("anchors must not be empty")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        seen: set[tuple[int, int]] = set()
        grouped: dict[LiberoDataFile, list[tuple[int, LiberoAnchor, LiberoEpisode]]] = {}
        for position, anchor in enumerate(requested):
            if not 0 <= anchor.episode_index < len(self.episodes):
                raise IndexError("anchor episode_index is outside the dataset")
            episode = self.episodes[anchor.episode_index]
            if episode.episode_index != anchor.episode_index:
                raise ValueError("episode metadata is not indexed contiguously")
            if not 0 <= anchor.frame_index < episode.length:
                raise IndexError("anchor frame_index is outside the episode")
            if anchor.task != episode.task:
                raise ValueError("anchor task does not match episode metadata")
            identity = (anchor.episode_index, anchor.frame_index)
            if identity in seen:
                raise ValueError("anchors must not contain duplicate episode/frame pairs")
            seen.add(identity)
            physical_file = self._data_file_by_episode[episode.episode_index]
            grouped.setdefault(physical_file, []).append((position, anchor, episode))

        samples_by_position: dict[int, LiberoTrainingSample] = {}
        for physical_file, entries in grouped.items():
            data_file_table = self._read_data_file(physical_file.chunk_index, physical_file.file_index)
            episode_cache: dict[int, tuple[Any, torch.Tensor]] = {}
            for position, anchor, episode in entries:
                if episode.episode_index not in episode_cache:
                    episode_table = self._episode_rows_from_table(episode, physical_file, data_file_table)
                    episode_cache[episode.episode_index] = (
                        episode_table,
                        self._validated_episode_actions(episode_table),
                    )
                episode_table, actions = episode_cache[episode.episode_index]
                samples_by_position[position] = self._sample_from_episode_rows(
                    episode,
                    anchor.frame_index,
                    episode_table,
                    actions,
                    horizon=horizon,
                )
        return tuple(samples_by_position[position] for position in range(len(requested)))

    def _sample_from_episode_rows(
        self,
        episode: LiberoEpisode,
        frame_index: int,
        table: Any,
        actions: torch.Tensor,
        *,
        horizon: int,
    ) -> LiberoTrainingSample:
        row = table.slice(frame_index, 1).to_pylist()[0]
        task_index = int(row["task_index"])
        instruction = self.task_by_index.get(task_index)
        if instruction != episode.task:
            raise ValueError("row task_index does not match episode language")
        if int(row["frame_index"]) != frame_index:
            raise ValueError("row frame_index is not contiguous within its episode")
        observation = CanonicalObservation(
            third_person=_decode_rgb(row["observation.images.image"]),
            wrist=_decode_rgb(row["observation.images.image2"]),
            state=torch.tensor(row["observation.state"], dtype=torch.float32),
        )
        return LiberoTrainingSample(
            observation=observation,
            instruction=instruction,
            action_chunk=make_libero_action_chunk(actions, frame_index, horizon=horizon),
            episode_index=episode.episode_index,
            frame_index=frame_index,
            task_index=task_index,
        )

    @staticmethod
    def _validated_episode_actions(table: Any) -> torch.Tensor:
        actions = torch.tensor(table.column("action").to_pylist(), dtype=torch.float32)
        return validate_libero_dataset_actions(actions)

    def _read_episode_rows(self, episode: LiberoEpisode):
        physical_file = self._data_file_by_episode[episode.episode_index]
        table = self._read_data_file(physical_file.chunk_index, physical_file.file_index)
        return self._episode_rows_from_table(episode, physical_file, table)

    @staticmethod
    def _episode_rows_from_table(episode: LiberoEpisode, physical_file: LiberoDataFile, table: Any):
        offset = episode.global_start - physical_file.global_start
        episode_rows = table.slice(offset, episode.length)
        expected_indices = np.arange(episode.global_start, episode.global_stop)
        indices = np.asarray(episode_rows.column("index").to_numpy())
        episode_indices = np.asarray(episode_rows.column("episode_index").to_numpy())
        frame_indices = np.asarray(episode_rows.column("frame_index").to_numpy())
        if not np.array_equal(indices, expected_indices):
            raise ValueError("episode rows have incorrect global indices")
        if not np.array_equal(episode_indices, np.full(episode.length, episode.episode_index)):
            raise ValueError("episode rows have incorrect episode indices")
        if not np.array_equal(frame_indices, np.arange(episode.length)):
            raise ValueError("episode rows have incorrect frame indices")
        return episode_rows

    def _discover_data_files(self) -> tuple[LiberoDataFile, ...]:
        """Index physical parquet spans; v3 episode metadata file indices are not reliable locators."""

        pattern = re.compile(r"chunk-(\d{3})/file-(\d{3})\.parquet$")
        files: list[LiberoDataFile] = []
        for path in sorted((self.root / "data").glob("chunk-*/file-*.parquet")):
            relative = path.relative_to(self.root / "data").as_posix()
            match = pattern.fullmatch(relative)
            if match is None:
                raise ValueError(f"unexpected LIBERO parquet path: {relative}")
            indices = np.asarray(_pyarrow_parquet().read_table(path, columns=["index"]).column("index").to_numpy())
            if indices.size == 0 or not np.array_equal(indices, np.arange(indices[0], indices[0] + indices.size)):
                raise ValueError(f"LIBERO parquet indices are empty or non-contiguous: {path}")
            files.append(
                LiberoDataFile(
                    chunk_index=int(match.group(1)),
                    file_index=int(match.group(2)),
                    global_start=int(indices[0]),
                    global_stop=int(indices[-1]) + 1,
                )
            )
        ordered = tuple(sorted(files, key=lambda item: item.global_start))
        expected_start = 0
        for data_file in ordered:
            if data_file.global_start != expected_start or data_file.length <= 0:
                raise ValueError("physical LIBERO parquet spans are missing, overlapping, or out of order")
            expected_start = data_file.global_stop
        if expected_start != self.info["total_frames"]:
            raise ValueError("physical LIBERO parquet spans do not cover the declared frame count")
        return ordered

    def _read_data_file(self, chunk_index: int, file_index: int):
        key = (chunk_index, file_index)
        if key in self._data_file_cache:
            self._data_file_cache.move_to_end(key)
            return self._data_file_cache[key]
        path = self.root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"LIBERO data shard is not downloaded: {path}")
        table = _pyarrow_parquet().read_table(path)
        self._data_file_cache[key] = table
        if len(self._data_file_cache) > self.max_cached_files:
            self._data_file_cache.popitem(last=False)
        return table

    def _validate_info(self) -> None:
        expected = {
            "codebase_version": "v3.0",
            "total_episodes": 1693,
            "total_frames": 273465,
            "total_tasks": 40,
        }
        mismatches = {
            key: (value, self.info.get(key)) for key, value in expected.items() if self.info.get(key) != value
        }
        if mismatches:
            raise ValueError(f"pinned LIBERO metadata changed: {mismatches}")
        features = self.info.get("features", {})
        expected_shapes = {
            "observation.images.image": [256, 256, 3],
            "observation.images.image2": [256, 256, 3],
            "observation.state": [8],
            "action": [7],
        }
        for key, shape in expected_shapes.items():
            if features.get(key, {}).get("shape") != shape:
                raise ValueError(f"unexpected LIBERO feature shape for {key}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required LIBERO metadata is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def resolve_episode_data_files(
    episodes: tuple[LiberoEpisode, ...],
    data_files: tuple[LiberoDataFile, ...],
) -> dict[int, LiberoDataFile]:
    """Resolve each episode by global index range, independently of stale metadata file indices."""

    if not episodes or not data_files:
        raise ValueError("episodes and data_files must not be empty")
    starts = [data_file.global_start for data_file in data_files]
    if starts != sorted(starts) or len(starts) != len(set(starts)):
        raise ValueError("data file spans must have unique sorted starts")
    for previous, current in pairwise(data_files):
        if previous.length <= 0 or previous.global_stop != current.global_start:
            raise ValueError("data file spans must be positive and contiguous")
    if data_files[-1].length <= 0:
        raise ValueError("data file spans must be positive and contiguous")
    resolved: dict[int, LiberoDataFile] = {}
    for episode in episodes:
        offset = bisect_right(starts, episode.global_start) - 1
        if offset < 0:
            raise ValueError(f"episode {episode.episode_index} begins before every physical data file")
        data_file = data_files[offset]
        if not (data_file.global_start <= episode.global_start and episode.global_stop <= data_file.global_stop):
            raise ValueError(f"episode {episode.episode_index} crosses or falls outside a physical data file")
        if episode.episode_index in resolved:
            raise ValueError("episode indices must be unique")
        resolved[episode.episode_index] = data_file
    return resolved


def _decode_rgb(encoded: dict[str, Any]) -> np.ndarray:
    if not isinstance(encoded, dict) or not isinstance(encoded.get("bytes"), bytes):
        raise ValueError("LeRobot image cell does not contain encoded bytes")
    with Image.open(io.BytesIO(encoded["bytes"])) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if rgb.shape != (256, 256, 3):
        raise ValueError(f"decoded LIBERO image has unexpected shape {rgb.shape}")
    return rgb


def _pyarrow_parquet():
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - minimal core installs intentionally omit pyarrow
        raise ImportError("install duo-vla's data dependencies to read LIBERO parquet files") from exc
    return parquet
