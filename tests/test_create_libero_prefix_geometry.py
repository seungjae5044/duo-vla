from __future__ import annotations

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/create_libero_prefix_geometry.py"
SPEC = importlib.util.spec_from_file_location("create_libero_prefix_geometry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def _write_suite_map(root: Path, mapping: dict[str, list[str]]) -> None:
    path = root / "libero/libero/benchmark/libero_suite_task_map.py"
    path.parent.mkdir(parents=True)
    path.write_text(f"libero_task_map = {mapping!r}\n", encoding="utf-8")


def test_official_inventory_is_literal_and_mirrors_language_conversion(tmp_path: Path) -> None:
    mapping = {suite: [f"{suite}_task_{index}" for index in range(10)] for suite in GENERATOR.LIBERO_SUITES}
    mapping["libero_10"][0] = "KITCHEN_SCENE10_put_the_mug_on_the_plate"
    _write_suite_map(tmp_path, mapping)

    instructions = GENERATOR.official_libero_instructions(tmp_path)

    assert len(instructions) == 40
    assert "put the mug on the plate" in instructions
    assert "libero spatial task 0" in instructions


def test_official_inventory_rejects_executable_suite_map(tmp_path: Path) -> None:
    path = tmp_path / "libero/libero/benchmark/libero_suite_task_map.py"
    path.parent.mkdir(parents=True)
    path.write_text("libero_task_map = build_map()\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not literal data"):
        GENERATOR.official_libero_instructions(tmp_path)


def test_dataset_inventory_requires_canonical_indices(tmp_path: Path) -> None:
    metadata = tmp_path / "meta"
    metadata.mkdir()
    table = pa.table(
        {
            "task_index": list(range(40)),
            "__index_level_0__": [f"instruction {index}" for index in range(40)],
        }
    )
    pq.write_table(table, metadata / "tasks.parquet")

    assert GENERATOR.dataset_libero_instructions(tmp_path)[0] == "instruction 0"
    changed = table.set_column(0, "task_index", pa.array([1, *range(1, 40)]))
    pq.write_table(changed, metadata / "tasks.parquet")
    with pytest.raises(RuntimeError, match="indices are not canonical"):
        GENERATOR.dataset_libero_instructions(tmp_path)
