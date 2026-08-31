#!/usr/bin/env python3
"""Create the authenticated fixed-prefix geometry artifact for LIBERO."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoProcessor

from duo_vla.backbones.loading import DEFAULT_DIFFUSION_GEMMA_SPEC
from duo_vla.data.libero_stats import LIBERO_DATASET_REVISION
from duo_vla.hf_snapshot import verify_huggingface_snapshot
from duo_vla.prefix_geometry import (
    CameraGeometry,
    SnapshotTreeIdentity,
    create_prefix_geometry_contract,
    save_prefix_geometry_contract,
)

LIBERO_DATASET_ID = "HuggingFaceVLA/libero"
LIBERO_SOURCE_REVISION = "8561c60eea2fb93096146f240194649df73d8b1e"
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIBERO_FIXED_PREFIX_WIDTH = 545
LIBERO_CAMERAS = (
    CameraGeometry("agentview", 256, 256),
    CameraGeometry("eye_in_hand", 256, 256),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def authenticate_libero_source(source_root: Path) -> dict[str, str]:
    """Require the pinned clean tracked LIBERO source used by the evaluator."""

    root = source_root.resolve()
    _require(root.is_dir(), f"LIBERO source root is missing: {root}")
    revision = _git_output(root, "rev-parse", "HEAD")
    _require(revision == LIBERO_SOURCE_REVISION, f"LIBERO source revision mismatch: {revision}")
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=no")
    _require(not status, "LIBERO tracked source tree is dirty")
    map_path = root / "libero/libero/benchmark/libero_suite_task_map.py"
    _require(map_path.is_file(), f"LIBERO suite map is missing: {map_path}")
    return {
        "revision": revision,
        "suite_map_path": map_path.relative_to(root).as_posix(),
        "suite_map_sha256": _sha256(map_path),
    }


def _literal_suite_map(path: Path) -> dict[str, list[str]]:
    """Read the official task map as data without importing simulator code."""

    try:
        syntax = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise RuntimeError(f"cannot parse LIBERO suite map: {path}") from exc
    assignments = [
        node
        for node in syntax.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "libero_task_map" for target in node.targets)
    ]
    _require(len(assignments) == 1, "LIBERO source must define exactly one literal libero_task_map")
    try:
        value = ast.literal_eval(assignments[0].value)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("LIBERO libero_task_map is not literal data") from exc
    _require(isinstance(value, dict), "LIBERO libero_task_map must be an object")
    result: dict[str, list[str]] = {}
    for suite, tasks in value.items():
        _require(isinstance(suite, str) and isinstance(tasks, list), "LIBERO suite map schema is invalid")
        _require(tasks and all(isinstance(task, str) and task for task in tasks), f"invalid LIBERO suite {suite!r}")
        _require(len(tasks) == len(set(tasks)), f"LIBERO suite {suite!r} contains duplicate task names")
        result[suite] = list(tasks)
    return result


def _language_from_task_name(task: str) -> str:
    """Mirror the pinned evaluator's ``grab_language_from_filename`` exactly."""

    filename = f"{task}.bddl"
    if filename[0].isupper():
        marker = filename.find("SCENE")
        _require(marker >= 0, f"uppercase LIBERO task has no SCENE marker: {task!r}")
        offset = 8 if "SCENE10" in filename else 7
        language = " ".join(filename[marker + offset :].split("_"))
    else:
        language = " ".join(filename.split("_"))
    suffix = language.find(".bddl")
    _require(suffix > 0, f"LIBERO task filename cannot be converted to language: {task!r}")
    return language[:suffix]


def official_libero_instructions(source_root: Path) -> tuple[str, ...]:
    map_path = source_root.resolve() / "libero/libero/benchmark/libero_suite_task_map.py"
    suite_map = _literal_suite_map(map_path)
    _require(all(suite in suite_map for suite in LIBERO_SUITES), "LIBERO source is missing a benchmark suite")
    instructions = tuple(_language_from_task_name(task) for suite in LIBERO_SUITES for task in suite_map[suite])
    _require(len(instructions) == 40, f"LIBERO evaluation inventory must contain 40 tasks, got {len(instructions)}")
    _require(len(set(instructions)) == 40, "LIBERO evaluation instruction inventory contains duplicates")
    return tuple(sorted(instructions))


def dataset_libero_instructions(snapshot_root: Path) -> tuple[str, ...]:
    tasks_path = snapshot_root.resolve() / "meta/tasks.parquet"
    _require(tasks_path.is_file(), f"LIBERO tasks metadata is missing: {tasks_path}")
    try:
        table = pq.read_table(tasks_path, columns=["task_index", "__index_level_0__"])
        rows = table.to_pylist()
    except Exception as exc:
        raise RuntimeError("cannot read LIBERO task metadata") from exc
    indices = [row.get("task_index") for row in rows]
    instructions = [row.get("__index_level_0__") for row in rows]
    _require(indices == list(range(40)), f"LIBERO task indices are not canonical 0..39: {indices}")
    _require(
        len(instructions) == 40 and all(isinstance(value, str) and value for value in instructions),
        "LIBERO training instruction inventory is invalid",
    )
    _require(len(set(instructions)) == 40, "LIBERO training instruction inventory contains duplicates")
    return tuple(sorted(instructions))


def parse_args() -> argparse.Namespace:
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla"))
    hf_home = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-snapshot",
        type=Path,
        default=hf_home / "hub/datasets--HuggingFaceVLA--libero/snapshots" / LIBERO_DATASET_REVISION,
    )
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        default=hf_home
        / "hub/models--google--diffusiongemma-26B-A4B-it/snapshots"
        / DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
    )
    parser.add_argument(
        "--libero-source",
        type=Path,
        default=cache_root / "simulators/libero/source",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=cache_root / "contracts/prefix-geometry/libero-v2.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_snapshot = args.dataset_snapshot.resolve()
    model_snapshot = args.model_snapshot.resolve()
    source_root = args.libero_source.resolve()
    output = Path(os.path.abspath(os.fspath(args.output)))
    _require(output.parent.is_dir(), f"output parent must already exist: {output.parent}")

    source_identity = authenticate_libero_source(source_root)
    dataset_report = verify_huggingface_snapshot(
        dataset_snapshot,
        expected_revision=LIBERO_DATASET_REVISION,
    )
    model_report = verify_huggingface_snapshot(
        model_snapshot,
        expected_revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
    )
    official_instructions = official_libero_instructions(source_root)
    dataset_instructions = dataset_libero_instructions(dataset_snapshot)
    _require(
        official_instructions == dataset_instructions,
        "LIBERO training and official evaluation instruction inventories differ",
    )

    processor = AutoProcessor.from_pretrained(
        DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
        revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
        local_files_only=True,
    )
    model_identity = SnapshotTreeIdentity.from_huggingface_report(
        DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
        model_report,
    )
    contract = create_prefix_geometry_contract(
        processor,
        model_identity=model_identity,
        processor_identity=model_identity,
        ordered_cameras=LIBERO_CAMERAS,
        instructions=official_instructions,
        fixed_physical_prefix_width=LIBERO_FIXED_PREFIX_WIDTH,
        padding_side="left",
    )
    _require(
        contract["geometry"]["maximum_valid_prefix_length"] + 1 == LIBERO_FIXED_PREFIX_WIDTH,
        "LIBERO fixed width must reserve exactly one padding sentinel beyond the authenticated maximum",
    )
    content_sha256 = save_prefix_geometry_contract(output, contract)
    tasks_path = dataset_snapshot / "meta/tasks.parquet"
    print(
        json.dumps(
            {
                "artifact": str(output),
                "content_sha256": content_sha256,
                "dataset": {
                    "id": LIBERO_DATASET_ID,
                    "inventory_sha256": dataset_report["content_inventory_sha256"],
                    "revision": LIBERO_DATASET_REVISION,
                    "tasks_parquet_sha256": _sha256(tasks_path),
                    "tree_metadata_sha256": dataset_report["tree_metadata_sha256"],
                },
                "fixed_physical_prefix_width": LIBERO_FIXED_PREFIX_WIDTH,
                "instruction_count": len(official_instructions),
                "instruction_inventory_sha256": contract["instruction_inventory"]["sha256"],
                "maximum_valid_prefix_length": contract["geometry"]["maximum_valid_prefix_length"],
                "model": model_identity.to_dict(),
                "official_source": source_identity,
                "ordered_instruction_sha256": _canonical_sha256(list(official_instructions)),
                "status": "ok",
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
