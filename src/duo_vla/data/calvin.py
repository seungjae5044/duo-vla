"""Strict random-access reader for the official CALVIN per-timestep NPZ layout."""

from __future__ import annotations

import hashlib
import io
import os
import sqlite3
import stat
import zlib
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from duo_vla.benchmarks.calvin import (
    calvin_state,
    calvin_training_observation,
    make_calvin_action_chunk,
    validate_calvin_rel_actions,
)
from duo_vla.benchmarks.common import CanonicalObservation, PaddedActionChunk

if TYPE_CHECKING:
    from duo_vla.data.calvin_stats import AuthenticatedCalvinDatasetGeneration


@dataclass(frozen=True, slots=True)
class CalvinEpisode:
    """One inclusive interval from ``ep_start_end_ids.npy``."""

    episode_index: int
    global_start: int
    global_end_inclusive: int
    scene: str | None = None

    @property
    def length(self) -> int:
        return self.global_end_inclusive - self.global_start + 1


@dataclass(frozen=True, slots=True)
class CalvinAnnotation:
    """One language-labelled half-open interval contained in an episode."""

    annotation_index: int
    episode_index: int
    global_start: int
    global_end_exclusive: int
    instruction: str
    task: str

    @property
    def length(self) -> int:
        return self.global_end_exclusive - self.global_start


@dataclass(frozen=True, slots=True)
class CalvinAnchor:
    annotation_index: int
    global_index: int
    task: str


@dataclass(frozen=True, slots=True)
class CalvinTrainingSample:
    observation: CanonicalObservation
    instruction: str
    action_chunk: PaddedActionChunk
    annotation_index: int
    episode_index: int
    global_index: int
    task: str


@dataclass(frozen=True, slots=True)
class CalvinEpisodeSplit:
    train_episode_indices: tuple[int, ...]
    validation_episode_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        train = set(self.train_episode_indices)
        validation = set(self.validation_episode_indices)
        if not train or not validation:
            raise ValueError("both CALVIN train and validation episode splits must be non-empty")
        if train & validation:
            raise ValueError("CALVIN train and validation episode splits overlap")


class CalvinNpzDataset:
    """Read raw CALVIN frames without importing its legacy training stack.

    The official archive stores one compressed NPZ per control timestep. Language
    Play-episode intervals use inclusive ends.  Language annotations use the
    official half-open ``[start, end)`` convention.
    """

    def __init__(
        self,
        split_root: str | Path,
        *,
        max_cached_frames: int = 512,
        expected_scenes: Sequence[str] | None = None,
        verify_frame_files: bool = False,
        verify_archive: bool = True,
        authenticated_generation: AuthenticatedCalvinDatasetGeneration | None = None,
        storage_mode: str | None = None,
    ) -> None:
        if max_cached_frames <= 0:
            raise ValueError("max_cached_frames must be positive")
        self.root = Path(split_root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"CALVIN split directory is missing: {self.root}")
        self._member_connection: sqlite3.Connection | None = None
        self._member_index_descriptor: int | None = None
        self._archive_reader: object | None = None
        self._authenticated_generation = authenticated_generation
        self._authenticated_frame_mode = False
        self._closed = False
        try:
            authenticated_metadata: dict[str, bytes] | None = None
            if authenticated_generation is not None:
                from duo_vla.data.calvin_stats import (
                    CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
                    CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION,
                    AuthenticatedCalvinDatasetGeneration,
                    calvin_member_index_path,
                    open_calvin_archive_reader,
                    verify_calvin_dataset_generation,
                )

                if not isinstance(authenticated_generation, AuthenticatedCalvinDatasetGeneration):
                    raise TypeError("authenticated_generation must be an AuthenticatedCalvinDatasetGeneration")
                authenticated_mode = authenticated_generation.storage.mode
                if storage_mode is not None and storage_mode != authenticated_mode:
                    raise ValueError("explicit CALVIN storage mode differs from the authenticated generation")
                storage_mode = authenticated_mode
                if storage_mode == CALVIN_STORAGE_MODE_ARCHIVE_DIRECT:
                    if verify_frame_files:
                        raise ValueError("verify_frame_files is an extracted-v3 option, not an archive-direct option")
                    reader = open_calvin_archive_reader(self.root, authenticated_generation)
                    self._archive_reader = reader
                    authenticated_metadata = {
                        relative: reader.read_authenticated_metadata_bytes(relative)
                        for relative in (
                            "training/scene_info.npy",
                            "training/ep_start_end_ids.npy",
                            "training/lang_annotations/auto_lang_ann.npy",
                        )
                    }
                elif storage_mode == CALVIN_STORAGE_MODE_VERIFIED_EXTRACTION:
                    verify_calvin_dataset_generation(self.root, authenticated_generation)
                    self._authenticated_frame_mode = True
                    self._pin_authenticated_member_index(
                        calvin_member_index_path(self.root),
                        expected_bytes=authenticated_generation.member_index_bytes,
                        expected_sha256=authenticated_generation.member_index_sha256,
                    )
                    critical_files = dict(authenticated_generation.critical_files)
                    metadata_paths = (
                        self.root / "scene_info.npy",
                        self.root / "ep_start_end_ids.npy",
                        self.root / "lang_annotations" / "auto_lang_ann.npy",
                    )
                    authenticated_metadata = {}
                    for metadata_path in metadata_paths:
                        relative = metadata_path.relative_to(self.root.parent).as_posix()
                        expected_sha256 = critical_files.get(relative)
                        if expected_sha256 is None:
                            raise ValueError(f"authenticated CALVIN generation has no metadata identity: {relative}")
                        authenticated_metadata[relative] = _read_authenticated_metadata_bytes(
                            metadata_path,
                            relative=relative,
                            expected_sha256=expected_sha256,
                        )
                else:
                    raise ValueError(f"unsupported authenticated CALVIN storage mode: {storage_mode!r}")
            elif verify_frame_files:
                # This gate runs before loading pickle-bearing metadata below.
                from duo_vla.data.calvin_stats import calvin_member_index_path, load_calvin_dataset_manifest

                manifest = load_calvin_dataset_manifest(
                    self.root,
                    verify_archive=verify_archive,
                    allow_legacy_v3=True,
                )
                member_identity = manifest["extraction"]["member_index"]
                member_path = calvin_member_index_path(self.root)
                if member_path.stat().st_size != member_identity["bytes"]:
                    raise ValueError("CALVIN member-index file size changed")
                digest = hashlib.sha256()
                with member_path.open("rb") as handle:
                    while block := handle.read(8 * 1024 * 1024):
                        digest.update(block)
                if digest.hexdigest() != member_identity["sha256"]:
                    raise ValueError("CALVIN member-index file hash changed")
                self._member_connection = sqlite3.connect(member_path.as_uri() + "?mode=ro&immutable=1", uri=True)
                self._authenticated_frame_mode = True
                storage_mode = "verified-extraction"
            elif storage_mode != "verified-extraction":
                raise ValueError(
                    "unauthenticated CALVIN fixtures must explicitly select storage_mode='verified-extraction'"
                )
            self.storage_mode = storage_mode
            self.max_cached_frames = max_cached_frames
            scene_info_path = self.root / "scene_info.npy"
            if authenticated_metadata is not None:
                scene_relative = scene_info_path.relative_to(self.root.parent).as_posix()
                self.scene_intervals = _load_scene_intervals(
                    scene_info_path,
                    authenticated_bytes=authenticated_metadata[scene_relative],
                )
            else:
                self.scene_intervals = _load_scene_intervals(scene_info_path) if scene_info_path.is_file() else None
            if expected_scenes is not None:
                expected = set(expected_scenes)
                if not expected or self.scene_intervals is None:
                    raise ValueError("expected CALVIN scenes require a non-empty scene_info.npy")
                observed = set(self.scene_intervals)
                if observed != expected:
                    raise ValueError(
                        "CALVIN split scene identity mismatch: "
                        f"expected={sorted(expected)}, observed={sorted(observed)}"
                    )
            episode_path = self.root / "ep_start_end_ids.npy"
            episode_bytes = None
            if authenticated_metadata is not None:
                episode_bytes = authenticated_metadata[episode_path.relative_to(self.root.parent).as_posix()]
            self.episodes = _load_episodes(
                episode_path,
                self.scene_intervals,
                authenticated_bytes=episode_bytes,
            )
            self._episode_starts = tuple(episode.global_start for episode in self.episodes)
            annotation_path = self.root / "lang_annotations" / "auto_lang_ann.npy"
            annotation_bytes = None
            if authenticated_metadata is not None:
                annotation_bytes = authenticated_metadata[annotation_path.relative_to(self.root.parent).as_posix()]
            self.annotations = _load_annotations(
                annotation_path,
                self.episodes,
                self._episode_starts,
                authenticated_bytes=annotation_bytes,
            )
            self.tasks = tuple(sorted({annotation.task for annotation in self.annotations}))
            if not self.tasks:
                raise ValueError("CALVIN language annotations contain no tasks")
            if self.storage_mode == "verified-extraction":
                self._file_prefix, self._file_suffix, self._file_digits = _infer_frame_naming(
                    self.root,
                    self.episodes[0].global_start,
                )
            else:
                self._file_prefix = self._file_suffix = ""
                self._file_digits = 7
            self._frame_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
            if self.storage_mode == "verified-extraction":
                for index in {
                    self.episodes[0].global_start,
                    self.episodes[-1].global_end_inclusive,
                    self.annotations[0].global_start,
                    self.annotations[-1].global_end_exclusive - 1,
                }:
                    if not self._frame_path(index).is_file():
                        raise FileNotFoundError(
                            f"CALVIN frame interval references a missing file: {self._frame_path(index)}"
                        )
        except BaseException:
            self.close()
            raise

    def sample(self, anchor: CalvinAnchor, *, horizon: int = 8) -> CalvinTrainingSample:
        return self.sample_many((anchor,), horizon=horizon)[0]

    def sample_many(
        self,
        anchors: Sequence[CalvinAnchor],
        *,
        horizon: int = 8,
    ) -> tuple[CalvinTrainingSample, ...]:
        self._ensure_open()
        requested = tuple(anchors)
        if not requested:
            raise ValueError("anchors must not be empty")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        identities: set[tuple[int, int]] = set()
        validated: list[tuple[int, CalvinAnchor, CalvinAnnotation, CalvinEpisode, int]] = []
        for request_index, anchor in enumerate(requested):
            if not 0 <= anchor.annotation_index < len(self.annotations):
                raise IndexError("anchor annotation_index is outside the dataset")
            annotation = self.annotations[anchor.annotation_index]
            if annotation.annotation_index != anchor.annotation_index:
                raise ValueError("annotation metadata is not indexed contiguously")
            if not annotation.global_start <= anchor.global_index < annotation.global_end_exclusive:
                raise IndexError("anchor global_index is outside its language interval")
            if anchor.task != annotation.task:
                raise ValueError("anchor task does not match annotation metadata")
            identity = (anchor.annotation_index, anchor.global_index)
            if identity in identities:
                raise ValueError("anchors must not contain duplicate annotation/frame pairs")
            identities.add(identity)
            episode = self.episodes[annotation.episode_index]
            final_index = min(
                anchor.global_index + horizon - 1,
                annotation.global_end_exclusive - 1,
                episode.global_end_inclusive,
            )
            validated.append((request_index, anchor, annotation, episode, final_index))

        # All anchors are validated before the first archive or filesystem read.
        # Reads are canonicalized independently from request order and results are
        # restored to the caller's exact ordering below.
        anchor_frames = {
            global_index: self._read_frame(global_index)
            for global_index in sorted({anchor.global_index for _, anchor, _, _, _ in validated})
        }
        action_indices = {
            global_index
            for _, anchor, _, _, final_index in validated
            for global_index in range(anchor.global_index, final_index + 1)
        }
        actions_by_index = {global_index: self._read_action(global_index) for global_index in sorted(action_indices)}

        samples: list[CalvinTrainingSample | None] = [None] * len(validated)
        for request_index, anchor, annotation, episode, final_index in validated:
            frame = anchor_frames[anchor.global_index]
            actions = validate_calvin_rel_actions(
                torch.from_numpy(
                    np.stack([actions_by_index[index] for index in range(anchor.global_index, final_index + 1)])
                ).float()
            )
            chunk = make_calvin_action_chunk(
                actions,
                0,
                annotation_end_exclusive=len(actions),
                episode_end_inclusive=len(actions) - 1,
                horizon=horizon,
            )
            samples[request_index] = CalvinTrainingSample(
                observation=calvin_training_observation(frame),
                instruction=annotation.instruction,
                action_chunk=chunk,
                annotation_index=annotation.annotation_index,
                episode_index=episode.episode_index,
                global_index=anchor.global_index,
                task=annotation.task,
            )
        if any(sample is None for sample in samples):
            raise AssertionError("CALVIN sample request-order restoration is incomplete")
        return tuple(sample for sample in samples if sample is not None)

    def close(self) -> None:
        self._closed = True
        archive_reader = self._archive_reader
        self._archive_reader = None
        connection = self._member_connection
        self._member_connection = None
        descriptor = self._member_index_descriptor
        self._member_index_descriptor = None
        try:
            if archive_reader is not None:
                archive_reader.close()
        finally:
            try:
                if connection is not None:
                    connection.close()
            finally:
                if descriptor is not None:
                    os.close(descriptor)

    def __enter__(self) -> CalvinNpzDataset:
        self._ensure_open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _frame_path(self, global_index: int) -> Path:
        return Path(f"{self._file_prefix}{global_index:0{self._file_digits}d}{self._file_suffix}")

    def _read_frame(self, global_index: int) -> dict[str, np.ndarray]:
        self._ensure_open()
        cached = self._frame_cache.get(global_index)
        if cached is not None:
            self._frame_cache.move_to_end(global_index)
            return cached
        required = ("rgb_static", "rgb_gripper", "robot_obs", "rel_actions")
        frame = self._load_frame_arrays(global_index, required)
        _validate_frame(frame, self._frame_display_path(global_index))
        self._frame_cache[global_index] = frame
        if len(self._frame_cache) > self.max_cached_frames:
            self._frame_cache.popitem(last=False)
        return frame

    def _read_action(self, global_index: int) -> np.ndarray:
        self._ensure_open()
        cached = self._frame_cache.get(global_index)
        if cached is not None:
            self._frame_cache.move_to_end(global_index)
            return cached["rel_actions"]
        arrays = self._load_frame_arrays(global_index, ("rel_actions",))
        action = arrays["rel_actions"]
        _validate_action(action, self._frame_display_path(global_index))
        return action

    def read_state_action(self, global_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Read only arrays needed by the train-only statistics scan.

        CALVIN stores each timestep as a compressed NPZ.  Loading RGB arrays
        while fitting seven state percentiles needlessly decompresses both
        cameras for every frame in the 517 GB archive.
        """

        self._ensure_open()
        cached = self._frame_cache.get(global_index)
        if cached is not None:
            self._frame_cache.move_to_end(global_index)
            return cached["robot_obs"], cached["rel_actions"]
        required = ("robot_obs", "rel_actions")
        arrays = self._load_frame_arrays(global_index, required)
        robot_obs = arrays["robot_obs"]
        rel_actions = arrays["rel_actions"]
        _validate_state_action(robot_obs, rel_actions, self._frame_display_path(global_index))
        return robot_obs, rel_actions

    def iter_state_actions_physical(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield v4 training frames in authenticated archive data-offset order."""

        self._ensure_open()
        if self.storage_mode != "archive-direct" or self._archive_reader is None:
            raise RuntimeError("physical CALVIN state/action iteration requires archive-direct storage")
        reader = self._archive_reader
        for record in reader.iter_frame_records_physical(split="training"):
            if record.global_index is None:
                raise AssertionError("archive training-frame record has no global index")
            arrays = self._load_archive_member_arrays(record.path, ("robot_obs", "rel_actions"))
            robot_obs = arrays["robot_obs"]
            rel_actions = arrays["rel_actions"]
            _validate_state_action(robot_obs, rel_actions, Path(f"archive:{record.path}"))
            yield record.global_index, robot_obs, rel_actions
        self._ensure_open()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("CALVIN dataset is closed")
        if self.storage_mode == "archive-direct":
            if self._archive_reader is None:
                raise RuntimeError("authenticated CALVIN archive reader is unavailable")
            # This property revalidates the pinned root/archive/index identities,
            # including when a requested anchor is already in the local cache.
            _ = self._archive_reader.archive_file_identity

    def _pin_authenticated_member_index(
        self,
        path: Path,
        *,
        expected_bytes: int,
        expected_sha256: str,
    ) -> None:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"authenticated CALVIN member index is missing: {path}") from exc
        except OSError as exc:
            raise ValueError("authenticated CALVIN member index must be a regular non-symlink file") from exc
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("authenticated CALVIN member index must be a regular non-symlink file")
            if file_stat.st_size != expected_bytes:
                raise ValueError("CALVIN member index size differs from the authenticated generation")
            digest = hashlib.sha256()
            remaining = expected_bytes
            while remaining:
                block = os.read(descriptor, min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("CALVIN member index ended before its authenticated byte length")
                digest.update(block)
                remaining -= len(block)
            if os.read(descriptor, 1):
                raise ValueError("CALVIN member index exceeds its authenticated byte length")
            if digest.hexdigest() != expected_sha256:
                raise ValueError("CALVIN member index content differs from the authenticated generation")
            os.lseek(descriptor, 0, os.SEEK_SET)
            pinned_path = Path(f"/proc/self/fd/{descriptor}")
            try:
                connection = sqlite3.connect(pinned_path.as_uri() + "?mode=ro&immutable=1", uri=True)
            except sqlite3.Error as exc:
                raise ValueError("could not open the pinned authenticated CALVIN member index") from exc
        except BaseException:
            os.close(descriptor)
            raise
        self._member_index_descriptor = descriptor
        self._member_connection = connection

    def _frame_display_path(self, global_index: int) -> Path:
        if self.storage_mode == "archive-direct":
            return Path(f"archive:training/episode_{global_index:07d}.npz")
        return self._frame_path(global_index)

    def _load_frame_arrays(self, global_index: int, required: tuple[str, ...]) -> dict[str, np.ndarray]:
        if self.storage_mode == "archive-direct":
            return self._load_archive_member_arrays(
                f"training/episode_{global_index:07d}.npz",
                required,
            )
        path = self._frame_path(global_index)
        if self._authenticated_frame_mode:
            source: Path | io.BytesIO = io.BytesIO(self._read_authenticated_frame_bytes(path))
        else:
            if not path.is_file():
                raise FileNotFoundError(f"CALVIN frame is missing: {path}")
            source = path
        try:
            with np.load(source, allow_pickle=False) as archive:
                missing = [key for key in required if key not in archive]
                if missing:
                    raise ValueError(f"CALVIN frame is missing required arrays {missing}: {path}")
                return {key: np.asarray(archive[key]).copy() for key in required}
        finally:
            if isinstance(source, io.BytesIO):
                source.close()

    def _load_archive_member_arrays(
        self,
        relative: str,
        required: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        self._ensure_open()
        if self._archive_reader is None:
            raise RuntimeError("authenticated CALVIN archive reader is unavailable")
        source = io.BytesIO(self._archive_reader.read_member_bytes(relative))
        try:
            with np.load(source, allow_pickle=False) as archive:
                missing = [key for key in required if key not in archive]
                if missing:
                    raise ValueError(f"CALVIN frame is missing required arrays {missing}: archive:{relative}")
                return {key: np.asarray(archive[key]).copy() for key in required}
        finally:
            source.close()

    def _read_authenticated_frame_bytes(self, path: Path) -> bytes:
        self._ensure_open()
        connection = self._member_connection
        if connection is None:
            raise RuntimeError("authenticated CALVIN frame reader is unavailable")
        relative = path.relative_to(self.root.parent).as_posix()
        row = connection.execute(
            "SELECT bytes, crc32, sha256 FROM members WHERE path = ?",
            (relative,),
        ).fetchone()
        if (
            row is None
            or len(row) != 3
            or type(row[0]) is not int
            or row[0] <= 0
            or type(row[1]) is not int
            or not 0 <= row[1] <= 0xFFFFFFFF
            or not isinstance(row[2], str)
            or len(row[2]) != 64
            or any(character not in "0123456789abcdef" for character in row[2])
        ):
            raise ValueError(f"CALVIN member index has no valid frame identity: {relative}")
        expected_bytes, expected_crc32, expected_sha256 = row
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"CALVIN frame is missing: {path}") from exc
        except OSError as exc:
            raise ValueError(f"CALVIN frame must be a regular non-symlink file: {relative}") from exc
        digest = hashlib.sha256()
        crc32 = 0
        content = bytearray()
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"CALVIN frame must be a regular non-symlink file: {relative}")
            if file_stat.st_size != expected_bytes:
                raise ValueError(f"CALVIN extracted frame size changed: {relative}")
            remaining = expected_bytes
            while remaining:
                block = os.read(descriptor, min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"CALVIN extracted frame ended before its indexed byte length: {relative}")
                content.extend(block)
                digest.update(block)
                crc32 = zlib.crc32(block, crc32)
                remaining -= len(block)
            if os.read(descriptor, 1):
                raise ValueError(f"CALVIN extracted frame exceeds its indexed byte length: {relative}")
        finally:
            os.close(descriptor)
        if digest.hexdigest() != expected_sha256 or crc32 & 0xFFFFFFFF != expected_crc32:
            raise ValueError(f"CALVIN extracted frame content changed: {relative}")
        return bytes(content)


class CalvinTaskUniformAnchorSampler:
    """Draw ``task -> annotation -> timestep`` from selected whole episodes."""

    def __init__(
        self,
        annotations: Sequence[CalvinAnnotation],
        episode_indices: Sequence[int],
    ) -> None:
        selected = set(episode_indices)
        if not selected or len(selected) != len(tuple(episode_indices)):
            raise ValueError("episode_indices must be non-empty and distinct")
        by_task: dict[str, list[CalvinAnnotation]] = defaultdict(list)
        for annotation in annotations:
            if annotation.episode_index in selected:
                by_task[annotation.task].append(annotation)
        if not by_task:
            raise ValueError("selected CALVIN episodes contain no language annotations")
        self._tasks = tuple(sorted(by_task))
        self._annotations_by_task = {
            task: tuple(sorted(values, key=lambda item: item.annotation_index)) for task, values in by_task.items()
        }
        self._population_size = sum(
            annotation.length for values in self._annotations_by_task.values() for annotation in values
        )

    @property
    def tasks(self) -> tuple[str, ...]:
        return self._tasks

    @property
    def population_size(self) -> int:
        return self._population_size

    def draw(self, generator: torch.Generator) -> CalvinAnchor:
        task = self._tasks[int(torch.randint(len(self._tasks), (), generator=generator).item())]
        annotations = self._annotations_by_task[task]
        annotation = annotations[int(torch.randint(len(annotations), (), generator=generator).item())]
        offset = int(torch.randint(annotation.length, (), generator=generator).item())
        return CalvinAnchor(annotation.annotation_index, annotation.global_start + offset, task)


def make_calvin_episode_split(
    episodes: Sequence[CalvinEpisode],
    annotations: Sequence[CalvinAnnotation],
    *,
    validation_fraction: float = 0.1,
    seed: int = 1729,
) -> CalvinEpisodeSplit:
    """Hold out complete play episodes and fail closed if either side loses a task."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    if len(episodes) < 2:
        raise ValueError("at least two CALVIN episodes are required")
    by_scene: dict[str, list[CalvinEpisode]] = defaultdict(list)
    for episode in episodes:
        by_scene[episode.scene or "__unspecified__"].append(episode)
    train_values: list[int] = []
    validation_values: list[int] = []
    for scene in sorted(by_scene):
        candidates = by_scene[scene]
        if len(candidates) < 2:
            raise ValueError(f"CALVIN scene {scene!r} needs at least two episodes for a whole-episode split")
        ordered = sorted(
            candidates,
            key=lambda episode: (
                hashlib.sha256(f"calvin:{seed}:{scene}:{episode.episode_index}".encode()).digest(),
                episode.episode_index,
            ),
        )
        validation_count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
        validation_values.extend(episode.episode_index for episode in ordered[:validation_count])
        train_values.extend(episode.episode_index for episode in ordered[validation_count:])
    validation = tuple(sorted(validation_values))
    train = tuple(sorted(train_values))
    split = CalvinEpisodeSplit(train, validation)
    all_tasks = {annotation.task for annotation in annotations}
    train_tasks = {annotation.task for annotation in annotations if annotation.episode_index in set(train)}
    validation_tasks = {annotation.task for annotation in annotations if annotation.episode_index in set(validation)}
    if not all_tasks or train_tasks != all_tasks or validation_tasks != all_tasks:
        raise ValueError(
            "deterministic episode split does not preserve every CALVIN task on both sides: "
            f"train_missing={sorted(all_tasks - train_tasks)}, "
            f"validation_missing={sorted(all_tasks - validation_tasks)}"
        )
    return split


def _read_authenticated_metadata_bytes(
    path: Path,
    *,
    relative: str,
    expected_sha256: str,
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"authenticated CALVIN metadata is missing: {relative}") from exc
    except OSError as exc:
        raise ValueError(f"authenticated CALVIN metadata must be a regular non-symlink file: {relative}") from exc
    content = bytearray()
    digest = hashlib.sha256()
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"authenticated CALVIN metadata must be a regular non-symlink file: {relative}")
        remaining = file_stat.st_size
        while remaining:
            block = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"authenticated CALVIN metadata ended before its opened byte length: {relative}")
            content.extend(block)
            digest.update(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ValueError(f"authenticated CALVIN metadata exceeds its opened byte length: {relative}")
    finally:
        os.close(descriptor)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"CALVIN metadata content differs from the authenticated generation: {relative}")
    return bytes(content)


def _load_episodes(
    path: Path,
    scene_intervals: Mapping[str, tuple[int, int]] | None,
    *,
    authenticated_bytes: bytes | None = None,
) -> tuple[CalvinEpisode, ...]:
    if authenticated_bytes is None:
        if not path.is_file():
            raise FileNotFoundError(f"CALVIN episode metadata is missing: {path}")
        intervals = np.asarray(np.load(path, allow_pickle=False))
    else:
        with io.BytesIO(authenticated_bytes) as source:
            intervals = np.asarray(np.load(source, allow_pickle=False))
    if intervals.ndim != 2 or intervals.shape[1] != 2 or not np.issubdtype(intervals.dtype, np.integer):
        raise ValueError("CALVIN ep_start_end_ids.npy must have integer shape [episodes, 2]")
    episodes_list: list[CalvinEpisode] = []
    for index, (start_value, end_value) in enumerate(intervals.tolist()):
        start, end = int(start_value), int(end_value)
        scene = None
        if scene_intervals is not None:
            matches = [
                name
                for name, (scene_start, scene_end) in scene_intervals.items()
                if scene_start <= start <= end <= scene_end
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"CALVIN episode {index} must be contained in exactly one scene interval, got {sorted(matches)}"
                )
            scene = matches[0]
        episodes_list.append(CalvinEpisode(index, start, end, scene))
    episodes = tuple(episodes_list)
    if not episodes or any(episode.length <= 0 for episode in episodes):
        raise ValueError("CALVIN episode intervals must be non-empty and positive")
    for previous, current in pairwise(episodes):
        if current.global_start <= previous.global_end_inclusive:
            raise ValueError("CALVIN episode intervals overlap or are out of order")
    return episodes


def _load_scene_intervals(
    path: Path,
    *,
    authenticated_bytes: bytes | None = None,
) -> dict[str, tuple[int, int]]:
    if authenticated_bytes is None:
        payload = np.load(path, allow_pickle=True).item()
    else:
        with io.BytesIO(authenticated_bytes) as source:
            payload = np.load(source, allow_pickle=True).item()
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("CALVIN scene_info.npy must contain a non-empty mapping")
    result: dict[str, tuple[int, int]] = {}
    for scene, interval in payload.items():
        if not isinstance(scene, str) or not scene:
            raise ValueError("CALVIN scene_info.npy has an invalid scene name")
        values = np.asarray(interval)
        if values.shape != (2,):
            raise ValueError(f"CALVIN scene interval for {scene!r} must contain two indices")
        try:
            start, end = (int(value) for value in values.tolist())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CALVIN scene interval for {scene!r} is not integral") from exc
        if start > end:
            raise ValueError(f"CALVIN scene interval for {scene!r} is reversed")
        result[scene] = (start, end)
    return result


def _load_annotations(
    path: Path,
    episodes: tuple[CalvinEpisode, ...],
    episode_starts: tuple[int, ...],
    *,
    authenticated_bytes: bytes | None = None,
) -> tuple[CalvinAnnotation, ...]:
    if authenticated_bytes is None:
        if not path.is_file():
            raise FileNotFoundError(f"CALVIN language metadata is missing: {path}")
        payload = np.load(path, allow_pickle=True).item()
    else:
        with io.BytesIO(authenticated_bytes) as source:
            payload = np.load(source, allow_pickle=True).item()
    if not isinstance(payload, Mapping):
        raise ValueError("CALVIN auto_lang_ann.npy must contain a mapping")
    try:
        intervals = np.asarray(payload["info"]["indx"])
        instructions = tuple(payload["language"]["ann"])
        tasks = tuple(payload["language"]["task"])
    except (KeyError, TypeError) as exc:
        raise ValueError("CALVIN language metadata has an invalid schema") from exc
    if intervals.ndim != 2 or intervals.shape[1] != 2 or not np.issubdtype(intervals.dtype, np.integer):
        raise ValueError("CALVIN annotation indices must have integer shape [annotations, 2]")
    if not (len(intervals) == len(instructions) == len(tasks)) or len(intervals) == 0:
        raise ValueError("CALVIN annotation fields must have the same positive length")
    result: list[CalvinAnnotation] = []
    for index, ((start_value, end_value), instruction_value, task_value) in enumerate(
        zip(intervals.tolist(), instructions, tasks, strict=True)
    ):
        start, end = int(start_value), int(end_value)
        if not isinstance(instruction_value, str) or not instruction_value.strip():
            raise ValueError(f"CALVIN annotation {index} has no raw instruction string")
        if not isinstance(task_value, str) or not task_value.strip():
            raise ValueError(f"CALVIN annotation {index} has no task string")
        episode_offset = bisect_right(episode_starts, start) - 1
        if episode_offset < 0:
            raise ValueError(f"CALVIN annotation {index} starts outside every episode")
        episode = episodes[episode_offset]
        if not episode.global_start <= start < end <= episode.global_end_inclusive + 1:
            raise ValueError(f"CALVIN annotation {index} crosses or falls outside its underlying episode")
        result.append(
            CalvinAnnotation(
                annotation_index=index,
                episode_index=episode.episode_index,
                global_start=start,
                global_end_exclusive=end,
                instruction=instruction_value,
                task=task_value,
            )
        )
    return tuple(result)


def _infer_frame_naming(root: Path, first_global_index: int) -> tuple[Path, str, int]:
    # The official archive contract is exact.  Inferring from the first
    # directory entry lets a stale alternate-prefix file redirect all reads.
    first = root / f"episode_{first_global_index:07d}.npz"
    if not first.is_file():
        raise FileNotFoundError(f"CALVIN split is missing the canonical first NPZ frame: {first}")
    return root / "episode_", ".npz", 7


def _validate_frame(frame: Mapping[str, np.ndarray], path: Path) -> None:
    expected_shapes = {
        "rgb_static": (200, 200, 3),
        "rgb_gripper": (84, 84, 3),
        "robot_obs": (15,),
        "rel_actions": (7,),
    }
    for key, shape in expected_shapes.items():
        value = frame[key]
        if value.shape != shape:
            raise ValueError(f"CALVIN {key} has shape {value.shape}, expected {shape}: {path}")
    if frame["rgb_static"].dtype != np.uint8 or frame["rgb_gripper"].dtype != np.uint8:
        raise TypeError(f"CALVIN RGB arrays must be uint8: {path}")
    _validate_state_action(frame["robot_obs"], frame["rel_actions"], path)


def _validate_state_action(robot_obs: np.ndarray, rel_actions: np.ndarray, path: Path) -> None:
    expected_shapes = {"robot_obs": (15,), "rel_actions": (7,)}
    for key, value in (("robot_obs", robot_obs), ("rel_actions", rel_actions)):
        if value.shape != expected_shapes[key]:
            raise ValueError(f"CALVIN {key} has shape {value.shape}, expected {expected_shapes[key]}: {path}")
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"CALVIN {key} must contain finite floating values: {path}")
    # Reuse the benchmark contracts for exact state/gripper/action validation.
    robot_tensor = torch.from_numpy(robot_obs)
    if robot_tensor.shape != (15,):  # pragma: no cover - shape is checked above
        raise AssertionError("CALVIN state validation received an invalid shape")
    calvin_state(robot_tensor)
    validate_calvin_rel_actions(torch.from_numpy(rel_actions).reshape(1, 7))


def _validate_action(rel_actions: np.ndarray, path: Path) -> None:
    if rel_actions.shape != (7,):
        raise ValueError(f"CALVIN rel_actions has shape {rel_actions.shape}, expected (7,): {path}")
    if not np.issubdtype(rel_actions.dtype, np.floating) or not np.isfinite(rel_actions).all():
        raise ValueError(f"CALVIN rel_actions must contain finite floating values: {path}")
    validate_calvin_rel_actions(torch.from_numpy(rel_actions).reshape(1, 7))
