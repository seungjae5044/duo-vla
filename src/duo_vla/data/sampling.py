"""Deterministic episode splits and task-uniform LIBERO anchor sampling."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Generator


class EpisodeLike(Protocol):
    episode_index: int
    length: int
    task: str


@dataclass(frozen=True, slots=True)
class EpisodeSplit:
    train_episode_indices: tuple[int, ...]
    validation_episode_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        train = set(self.train_episode_indices)
        validation = set(self.validation_episode_indices)
        if not train or not validation:
            raise ValueError("both train and validation splits must be non-empty")
        if train & validation:
            raise ValueError("train and validation episode splits overlap")


@dataclass(frozen=True, slots=True)
class LiberoAnchor:
    episode_index: int
    frame_index: int
    task: str


def make_task_stratified_episode_split(
    episodes: Sequence[EpisodeLike],
    *,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> EpisodeSplit:
    """Hold out complete episodes within every task using a stable hash order."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    if not episodes:
        raise ValueError("episodes must not be empty")
    by_task: dict[str, list[EpisodeLike]] = defaultdict(list)
    seen_indices: set[int] = set()
    for episode in episodes:
        if episode.episode_index in seen_indices:
            raise ValueError("episode indices must be unique")
        if episode.length <= 0 or not episode.task:
            raise ValueError("episodes must have positive length and a task")
        seen_indices.add(episode.episode_index)
        by_task[episode.task].append(episode)

    train: list[int] = []
    validation: list[int] = []
    for task in sorted(by_task):
        task_episodes = by_task[task]
        if len(task_episodes) < 2:
            raise ValueError(f"task {task!r} needs at least two episodes for an episode-level split")
        ordered = sorted(
            task_episodes,
            key=lambda episode: (
                hashlib.sha256(f"{seed}:{episode.episode_index}".encode()).digest(),
                episode.episode_index,
            ),
        )
        validation_count = min(
            len(ordered) - 1,
            max(1, round(len(ordered) * validation_fraction)),
        )
        validation.extend(episode.episode_index for episode in ordered[:validation_count])
        train.extend(episode.episode_index for episode in ordered[validation_count:])
    return EpisodeSplit(tuple(sorted(train)), tuple(sorted(validation)))


class TaskUniformAnchorSampler:
    """Draw `task -> episode -> frame`, each uniformly at its own level."""

    def __init__(self, episodes: Sequence[EpisodeLike], episode_indices: Sequence[int]) -> None:
        episode_by_index = {episode.episode_index: episode for episode in episodes}
        if len(episode_by_index) != len(episodes):
            raise ValueError("episode indices must be unique")
        by_task: dict[str, list[EpisodeLike]] = defaultdict(list)
        selected: set[int] = set()
        for index in episode_indices:
            if index in selected:
                raise ValueError("episode_indices must not contain duplicates")
            try:
                episode = episode_by_index[index]
            except KeyError as exc:
                raise ValueError(f"unknown episode index {index}") from exc
            if episode.length <= 0:
                raise ValueError("selected episodes must have positive length")
            selected.add(index)
            by_task[episode.task].append(episode)
        if not by_task:
            raise ValueError("at least one episode must be selected")
        self._tasks = tuple(sorted(by_task))
        self._episodes_by_task = {
            task: tuple(sorted(task_episodes, key=lambda episode: episode.episode_index))
            for task, task_episodes in by_task.items()
        }
        self._population_size = sum(
            episode.length for task_episodes in self._episodes_by_task.values() for episode in task_episodes
        )

    @property
    def tasks(self) -> tuple[str, ...]:
        return self._tasks

    @property
    def population_size(self) -> int:
        """Number of distinct ``(episode, frame)`` anchors in this sampler."""

        return self._population_size

    def draw(self, generator: Generator) -> LiberoAnchor:
        task_offset = int(torch.randint(len(self._tasks), (), generator=generator).item())
        task = self._tasks[task_offset]
        candidates = self._episodes_by_task[task]
        episode_offset = int(torch.randint(len(candidates), (), generator=generator).item())
        episode = candidates[episode_offset]
        frame_index = int(torch.randint(episode.length, (), generator=generator).item())
        return LiberoAnchor(episode.episode_index, frame_index, task)

    def draw_many(self, count: int, generator: Generator) -> tuple[LiberoAnchor, ...]:
        if count <= 0:
            raise ValueError("count must be positive")
        return tuple(self.draw(generator) for _ in range(count))
