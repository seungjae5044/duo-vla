from __future__ import annotations

import pytest

from duo_vla.data.libero import LiberoDataFile, LiberoEpisode, LiberoParquetDataset, resolve_episode_data_files


def _episode(index: int, start: int, stop: int, declared_file: int) -> LiberoEpisode:
    return LiberoEpisode(index, 0, declared_file, start, stop, f"task-{index}")


def test_episode_mapping_uses_global_ranges_not_declared_file_indices() -> None:
    files = (
        LiberoDataFile(0, 0, 0, 10),
        LiberoDataFile(0, 1, 10, 25),
    )
    episodes = (
        _episode(0, 0, 4, 0),
        _episode(1, 12, 20, 0),
    )

    resolved = resolve_episode_data_files(episodes, files)

    assert resolved[0].file_index == 0
    assert resolved[1].file_index == 1


def test_episode_mapping_rejects_episode_crossing_physical_files() -> None:
    files = (
        LiberoDataFile(0, 0, 0, 10),
        LiberoDataFile(0, 1, 10, 20),
    )
    with pytest.raises(ValueError, match="crosses"):
        resolve_episode_data_files((_episode(0, 8, 12, 0),), files)


@pytest.mark.parametrize(
    "files",
    [
        (LiberoDataFile(0, 0, 0, 9), LiberoDataFile(0, 1, 10, 20)),
        (LiberoDataFile(0, 0, 0, 11), LiberoDataFile(0, 1, 10, 20)),
        (LiberoDataFile(0, 0, 0, 0),),
    ],
)
def test_episode_mapping_rejects_gap_overlap_and_empty_spans(files: tuple[LiberoDataFile, ...]) -> None:
    with pytest.raises(ValueError, match="positive and contiguous"):
        resolve_episode_data_files((_episode(0, 1, 2, 0),), files)


def test_dataset_rejects_nonpositive_cache_capacity() -> None:
    with pytest.raises(ValueError, match="max_cached_files"):
        LiberoParquetDataset("unused", max_cached_files=0)
