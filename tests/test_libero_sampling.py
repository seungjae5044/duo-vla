from dataclasses import dataclass

import pytest
import torch

from duo_vla.data.sampling import TaskUniformAnchorSampler, make_task_stratified_episode_split


@dataclass(frozen=True)
class Episode:
    episode_index: int
    length: int
    task: str


def _episodes() -> tuple[Episode, ...]:
    return tuple(
        Episode(episode_index=task_index * 10 + episode_index, length=episode_index + 2, task=f"task-{task_index}")
        for task_index, episode_count in enumerate((2, 5, 9))
        for episode_index in range(episode_count)
    )


def test_episode_split_is_deterministic_disjoint_and_task_stratified() -> None:
    episodes = _episodes()
    first = make_task_stratified_episode_split(episodes, validation_fraction=0.2, seed=7)
    second = make_task_stratified_episode_split(episodes, validation_fraction=0.2, seed=7)
    assert first == second
    assert set(first.train_episode_indices).isdisjoint(first.validation_episode_indices)
    assert set(first.train_episode_indices) | set(first.validation_episode_indices) == {
        episode.episode_index for episode in episodes
    }
    by_index = {episode.episode_index: episode for episode in episodes}
    assert {by_index[index].task for index in first.train_episode_indices} == {"task-0", "task-1", "task-2"}
    assert {by_index[index].task for index in first.validation_episode_indices} == {"task-0", "task-1", "task-2"}


def test_task_uniform_sampler_is_reproducible_and_not_frame_uniform() -> None:
    episodes = _episodes()
    sampler = TaskUniformAnchorSampler(episodes, [episode.episode_index for episode in episodes])
    assert sampler.population_size == sum(episode.length for episode in episodes)
    first = sampler.draw_many(6_000, torch.Generator().manual_seed(19))
    second = sampler.draw_many(6_000, torch.Generator().manual_seed(19))
    assert first == second
    counts = {task: sum(anchor.task == task for anchor in first) for task in sampler.tasks}
    assert max(counts.values()) - min(counts.values()) < 200
    by_index = {episode.episode_index: episode for episode in episodes}
    assert all(0 <= anchor.frame_index < by_index[anchor.episode_index].length for anchor in first)


def test_sampler_rejects_unknown_and_duplicate_episode_indices() -> None:
    episodes = _episodes()
    with pytest.raises(ValueError, match="unknown"):
        TaskUniformAnchorSampler(episodes, [999])
    with pytest.raises(ValueError, match="duplicates"):
        TaskUniformAnchorSampler(episodes, [0, 0])


def test_split_rejects_task_with_only_one_episode() -> None:
    with pytest.raises(ValueError, match="at least two"):
        make_task_stratified_episode_split((Episode(0, 3, "only"),))
