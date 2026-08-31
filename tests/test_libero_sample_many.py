from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

pyarrow = pytest.importorskip("pyarrow")

from duo_vla.data import libero as libero_module  # noqa: E402
from duo_vla.data.libero import LiberoDataFile, LiberoEpisode, LiberoParquetDataset  # noqa: E402
from duo_vla.data.sampling import LiberoAnchor  # noqa: E402


def _row(global_index: int, episode_index: int, frame_index: int) -> dict[str, object]:
    task_index = episode_index
    return {
        "observation.images.image": {"bytes": b"image"},
        "observation.images.image2": {"bytes": b"wrist"},
        "observation.state": [float(global_index), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "action": [0.01 * global_index, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "index": global_index,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "task_index": task_index,
    }


def _dataset(monkeypatch: pytest.MonkeyPatch):
    episodes = tuple(
        LiberoEpisode(
            episode_index=index,
            chunk_index=0,
            file_index=99,
            global_start=2 * index,
            global_stop=2 * index + 2,
            task=f"task-{index}",
        )
        for index in range(4)
    )
    files = (
        LiberoDataFile(0, 0, 0, 4),
        LiberoDataFile(0, 1, 4, 8),
    )
    tables = {
        (0, 0): pyarrow.Table.from_pylist(
            [_row(global_index, global_index // 2, global_index % 2) for global_index in range(4)]
        ),
        (0, 1): pyarrow.Table.from_pylist(
            [_row(global_index, global_index // 2, global_index % 2) for global_index in range(4, 8)]
        ),
    }
    reads: Counter[tuple[int, int]] = Counter()

    dataset = object.__new__(LiberoParquetDataset)
    dataset.episodes = episodes
    dataset.task_by_index = {index: f"task-{index}" for index in range(4)}
    dataset._data_file_by_episode = {episode.episode_index: files[episode.episode_index // 2] for episode in episodes}

    def read_data_file(chunk_index: int, file_index: int):
        key = (chunk_index, file_index)
        reads[key] += 1
        return tables[key]

    dataset._read_data_file = read_data_file
    monkeypatch.setattr(
        libero_module,
        "_decode_rgb",
        lambda _encoded: np.zeros((256, 256, 3), dtype=np.uint8),
    )
    return dataset, tables, reads


def test_sample_many_groups_physical_reads_and_preserves_anchor_order(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, _, reads = _dataset(monkeypatch)
    anchors = (
        LiberoAnchor(2, 1, "task-2"),
        LiberoAnchor(0, 0, "task-0"),
        LiberoAnchor(3, 0, "task-3"),
        LiberoAnchor(1, 1, "task-1"),
    )

    samples = dataset.sample_many(anchors, horizon=2)

    assert [(sample.episode_index, sample.frame_index) for sample in samples] == [
        (anchor.episode_index, anchor.frame_index) for anchor in anchors
    ]
    assert [sample.observation.state[0].item() for sample in samples] == [5.0, 0.0, 6.0, 3.0]
    assert reads == Counter({(0, 0): 1, (0, 1): 1})


@pytest.mark.parametrize(
    ("anchors", "error", "message"),
    [
        (
            (LiberoAnchor(0, 0, "task-0"), LiberoAnchor(0, 0, "task-0")),
            ValueError,
            "duplicate",
        ),
        ((LiberoAnchor(-1, 0, "task-0"),), IndexError, "episode_index"),
        ((LiberoAnchor(4, 0, "task-4"),), IndexError, "episode_index"),
        ((LiberoAnchor(0, -1, "task-0"),), IndexError, "frame_index"),
        ((LiberoAnchor(0, 2, "task-0"),), IndexError, "frame_index"),
        ((LiberoAnchor(0, 0, "wrong-task"),), ValueError, "task"),
    ],
)
def test_sample_many_rejects_invalid_anchors_before_io(
    monkeypatch: pytest.MonkeyPatch,
    anchors: tuple[LiberoAnchor, ...],
    error: type[Exception],
    message: str,
) -> None:
    dataset, _, reads = _dataset(monkeypatch)

    with pytest.raises(error, match=message):
        dataset.sample_many(anchors)

    assert not reads


def test_sample_many_reuses_episode_row_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, tables, _ = _dataset(monkeypatch)
    rows = tables[(0, 0)].to_pylist()
    rows[1]["frame_index"] = 7
    tables[(0, 0)] = pyarrow.Table.from_pylist(rows)

    with pytest.raises(ValueError, match="incorrect frame indices"):
        dataset.sample_many((LiberoAnchor(0, 0, "task-0"),))


def test_sample_many_rejects_empty_batch_and_invalid_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, _, reads = _dataset(monkeypatch)

    with pytest.raises(ValueError, match="must not be empty"):
        dataset.sample_many(())
    with pytest.raises(ValueError, match="horizon"):
        dataset.sample_many((LiberoAnchor(0, 0, "task-0"),), horizon=0)

    assert not reads
