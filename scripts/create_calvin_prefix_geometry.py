#!/usr/bin/env python3
"""Create the authenticated sentinel-padded prefix-geometry artifact for CALVIN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoProcessor

from duo_vla.backbones.loading import DEFAULT_DIFFUSION_GEMMA_SPEC
from duo_vla.data.calvin import CalvinNpzDataset
from duo_vla.data.calvin_stats import (
    CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
    AuthenticatedCalvinDatasetGeneration,
    authenticate_calvin_dataset_generation,
)
from duo_vla.hf_snapshot import verify_huggingface_snapshot
from duo_vla.prefix_geometry import (
    CameraGeometry,
    SnapshotTreeIdentity,
    build_prefix_geometry_contract,
    instruction_inventory_sha256,
    load_prefix_geometry_contract,
    measure_prefix_valid_lengths,
    measure_unbounded_prefix_valid_lengths,
    save_prefix_geometry_contract,
)

CALVIN_DATASET = "task_ABC_D"
CALVIN_SOURCE_REVISION = "fa03f01f19c65920e18cf37398a9ce859274af76"
CALVIN_EXPECTED_TRAINING_SCENES = ("calvin_scene_A", "calvin_scene_B", "calvin_scene_C")
CALVIN_VALIDATION_ANNOTATIONS_PATH = Path("calvin_models/conf/annotations/new_playtable_validation.yaml")
CALVIN_VALIDATION_ANNOTATIONS_SHA256 = "d14b0bf960f65158c5815b10ff11d2464073f26ac9b62c9f54e5c90ca352ccfa"
CALVIN_CAMERAS = (
    CameraGeometry("rgb_static", 200, 200),
    CameraGeometry("rgb_gripper", 84, 84),
)
CALVIN_PADDING_SIDE = "left"
CALVIN_TRAINING_ANNOTATION_PATH = "training/lang_annotations/auto_lang_ann.npy"


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
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """PyYAML safe loader which also rejects repeated mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    if not isinstance(node, yaml.nodes.MappingNode):
        raise yaml.constructor.ConstructorError(
            None,
            None,
            "expected a YAML mapping node",
            node.start_mark,
        )
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable YAML mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate YAML key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def parse_validation_annotations_yaml(raw: bytes | str) -> dict[str, str]:
    """Safely select the first exact validation phrase for every official task."""

    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RuntimeError("CALVIN validation annotation YAML is not UTF-8") from exc
    elif isinstance(raw, str):
        text = raw
    else:
        raise TypeError("CALVIN validation annotation YAML must be bytes or text")
    try:
        payload = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"cannot safely parse CALVIN validation annotation YAML: {exc}") from exc
    _require(isinstance(payload, dict) and bool(payload), "CALVIN validation annotation YAML must be a mapping")

    selected: dict[str, str] = {}
    for task, phrases in payload.items():
        _require(
            isinstance(task, str) and bool(task.strip()),
            "CALVIN validation annotation task names must be nonempty strings",
        )
        _require(
            task == task.strip(),
            f"CALVIN validation annotation task has surrounding whitespace: {task!r}",
        )
        _require(
            isinstance(phrases, list) and bool(phrases),
            f"CALVIN validation annotation phrases for {task!r} must be a nonempty list",
        )
        _require(
            all(isinstance(phrase, str) and bool(phrase.strip()) for phrase in phrases),
            f"CALVIN validation annotation phrases for {task!r} must be nonempty strings",
        )
        selected[task] = phrases[0]
    return {task: selected[task] for task in sorted(selected)}


def _authenticated_validation_yaml(source_root: Path) -> tuple[bytes, dict[str, str]]:
    root = source_root.resolve()
    path = root / CALVIN_VALIDATION_ANNOTATIONS_PATH
    _require(path.is_file(), f"official CALVIN validation annotation YAML is missing: {path}")
    _require(not path.is_symlink(), f"official CALVIN validation annotation YAML must not be a symlink: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read official CALVIN validation annotation YAML: {path}") from exc
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    _require(
        observed_sha256 == CALVIN_VALIDATION_ANNOTATIONS_SHA256,
        "official CALVIN validation annotation YAML SHA-256 mismatch",
    )
    return raw, {
        "path": CALVIN_VALIDATION_ANNOTATIONS_PATH.as_posix(),
        "sha256": observed_sha256,
    }


def authenticate_calvin_source(source_root: Path) -> dict[str, str]:
    """Require the clean pinned parent CALVIN checkout and validation YAML."""

    root = source_root.resolve()
    _require(root.is_dir(), f"CALVIN source root is missing: {root}")
    revision = _git_output(root, "rev-parse", "HEAD")
    _require(revision == CALVIN_SOURCE_REVISION, f"CALVIN source revision mismatch: {revision}")
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    _require(not status, "CALVIN source checkout is not clean")
    _, yaml_identity = _authenticated_validation_yaml(root)
    return {
        "revision": revision,
        "validation_annotations_path": yaml_identity["path"],
        "validation_annotations_sha256": yaml_identity["sha256"],
    }


def official_calvin_evaluation_inventory(source_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load the authenticated first official validation phrase for each task."""

    raw, identity = _authenticated_validation_yaml(source_root)
    return parse_validation_annotations_yaml(raw), identity


def _exact_unique_text(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    _require(not isinstance(values, (str, bytes)), f"{name} must be a sequence")
    materialized = tuple(values)
    _require(bool(materialized), f"{name} must not be empty")
    _require(
        all(isinstance(value, str) and bool(value.strip()) for value in materialized),
        f"{name} must contain only nonempty exact strings",
    )
    _require(len(materialized) == len(set(materialized)), f"{name} contains duplicate exact strings")
    return tuple(sorted(materialized))


def training_calvin_inventory(dataset: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return every unique exact training instruction and task from loaded annotations."""

    annotations = getattr(dataset, "annotations", None)
    _require(
        isinstance(annotations, Sequence) and not isinstance(annotations, (str, bytes)) and bool(annotations),
        "CALVIN dataset annotations must be a nonempty sequence",
    )
    instructions: set[str] = set()
    tasks: set[str] = set()
    for index, annotation in enumerate(annotations):
        instruction = getattr(annotation, "instruction", None)
        task = getattr(annotation, "task", None)
        _require(
            isinstance(instruction, str) and bool(instruction.strip()),
            f"CALVIN training annotation {index} has no exact instruction",
        )
        _require(
            isinstance(task, str) and bool(task.strip()),
            f"CALVIN training annotation {index} has no exact task",
        )
        instructions.add(instruction)
        tasks.add(task)
    normalized_instructions = tuple(sorted(instructions))
    normalized_tasks = tuple(sorted(tasks))
    declared_tasks = _exact_unique_text(getattr(dataset, "tasks", ()), name="CALVIN dataset tasks")
    _require(
        normalized_tasks == declared_tasks,
        "CALVIN dataset task inventory differs from its annotation task inventory",
    )
    return normalized_instructions, normalized_tasks


def union_calvin_instruction_inventories(
    *,
    training_instructions: Sequence[str],
    training_tasks: Sequence[str],
    evaluation_by_task: Mapping[str, str],
) -> tuple[str, ...]:
    """Require equal task vocabularies and union exact train/evaluation phrases."""

    normalized_training_instructions = _exact_unique_text(
        training_instructions,
        name="CALVIN training instructions",
    )
    normalized_training_tasks = _exact_unique_text(training_tasks, name="CALVIN training tasks")
    _require(
        isinstance(evaluation_by_task, Mapping) and bool(evaluation_by_task),
        "CALVIN evaluation task-to-instruction inventory must be a nonempty mapping",
    )
    evaluation_tasks = _exact_unique_text(tuple(evaluation_by_task), name="CALVIN evaluation tasks")
    _require(
        all(isinstance(value, str) and bool(value.strip()) for value in evaluation_by_task.values()),
        "CALVIN evaluation instructions must be nonempty exact strings",
    )
    missing_from_evaluation = sorted(set(normalized_training_tasks) - set(evaluation_tasks))
    missing_from_training = sorted(set(evaluation_tasks) - set(normalized_training_tasks))
    _require(
        not missing_from_evaluation and not missing_from_training,
        "CALVIN official evaluation and training task sets differ: "
        f"missing_from_evaluation={missing_from_evaluation}, missing_from_training={missing_from_training}",
    )
    return tuple(sorted(set(normalized_training_instructions) | set(evaluation_by_task.values())))


def measure_sentinel_padded_prefix_geometry(
    processor: Any,
    *,
    instructions: Sequence[str],
    ordered_cameras: Sequence[CameraGeometry | Mapping[str, Any]],
    padding_side: str,
    expected_fixed_prefix_width: int | None = None,
) -> tuple[int, dict[str, int]]:
    """Discover the exact maximum and verify ``P=max+1`` fixed-width probes.

    The always-present padding sentinel prevents SDPA from choosing its
    mask-elision path for an all-maximum-length replicated serving batch while
    choosing an explicit mask for a mixed training batch.
    """

    if expected_fixed_prefix_width is not None:
        _require(
            type(expected_fixed_prefix_width) is int and expected_fixed_prefix_width > 0,
            "expected fixed prefix width must be a positive integer",
        )
    unbounded = measure_unbounded_prefix_valid_lengths(
        processor,
        instructions=instructions,
        ordered_cameras=ordered_cameras,
        padding_side=padding_side,
    )
    _require(bool(unbounded), "CALVIN unbounded prefix measurement returned no instructions")
    fixed_physical_prefix_width = max(unbounded.values()) + 1
    if expected_fixed_prefix_width is not None:
        _require(
            fixed_physical_prefix_width == expected_fixed_prefix_width,
            "discovered CALVIN sentinel-padded prefix width differs from --expected-fixed-prefix-width: "
            f"discovered={fixed_physical_prefix_width}, expected={expected_fixed_prefix_width}",
        )
    fixed = measure_prefix_valid_lengths(
        processor,
        instructions=instructions,
        ordered_cameras=ordered_cameras,
        fixed_physical_prefix_width=fixed_physical_prefix_width,
        padding_side=padding_side,
    )
    _require(
        fixed == unbounded,
        "CALVIN fixed-width prefix measurements differ from the authenticated unbounded discovery pass",
    )
    return fixed_physical_prefix_width, fixed


def parse_args() -> argparse.Namespace:
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla"))
    hf_home = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=cache_root / "data/calvin/task_ABC_D/training",
    )
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        default=hf_home
        / "hub/models--google--diffusiongemma-26B-A4B-it/snapshots"
        / DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
    )
    parser.add_argument(
        "--calvin-source",
        type=Path,
        default=Path(os.environ.get("CALVIN_SOURCE_ROOT", cache_root / "simulators/calvin")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=cache_root / "contracts/prefix-geometry/calvin-abc-to-d-v1.json",
    )
    parser.add_argument(
        "--expected-fixed-prefix-width",
        type=int,
        help="Optional independent pin; must equal the discovered maximum valid length plus one sentinel.",
    )
    return parser.parse_args()


def _inventory_report(instructions: Sequence[str]) -> dict[str, str | int]:
    normalized = _exact_unique_text(instructions, name="report instruction inventory")
    return {
        "count": len(normalized),
        "sha256": instruction_inventory_sha256(normalized),
    }


def authenticated_dataset_report(
    dataset_manifest: Mapping[str, Any],
    generation: AuthenticatedCalvinDatasetGeneration,
) -> dict[str, Any]:
    """Bind prefix qualification output to the complete v4 storage identity."""

    _require(
        generation.storage.mode == CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
        "CALVIN prefix geometry requires archive-direct v4 production storage",
    )
    _require(
        dataset_manifest.get("schema") == generation.storage.manifest_schema
        and dataset_manifest.get("content_sha256") == generation.storage.manifest_content_sha256,
        "CALVIN prefix dataset manifest differs from its authenticated generation",
    )
    critical_files = dict(generation.critical_files)
    critical_annotation_sha256 = critical_files.get(CALVIN_TRAINING_ANNOTATION_PATH)
    _require(
        isinstance(critical_annotation_sha256, str),
        "authenticated CALVIN generation omits the critical training annotation hash",
    )
    return {
        "critical_annotation_path": CALVIN_TRAINING_ANNOTATION_PATH,
        "critical_annotation_sha256": critical_annotation_sha256,
        "manifest": {
            "content_sha256": dataset_manifest["content_sha256"],
            "file_sha256": generation.dataset_manifest_file_sha256,
            "schema": dataset_manifest["schema"],
        },
        "name": CALVIN_DATASET,
        "storage": generation.storage.to_dict(),
        "storage_identity_sha256": generation.storage.content_sha256,
    }


def main() -> None:
    args = parse_args()
    training_root = args.training_root.resolve()
    model_snapshot = args.model_snapshot.resolve()
    source_root = args.calvin_source.resolve()
    output = Path(os.path.abspath(os.fspath(args.output)))
    _require(output.parent.is_dir(), f"output parent must already exist: {output.parent}")

    dataset_manifest, authenticated_generation = authenticate_calvin_dataset_generation(training_root)
    dataset_report = authenticated_dataset_report(dataset_manifest, authenticated_generation)
    with CalvinNpzDataset(
        training_root,
        expected_scenes=CALVIN_EXPECTED_TRAINING_SCENES,
        authenticated_generation=authenticated_generation,
    ) as dataset:
        training_annotation_count = len(dataset.annotations)
        training_instructions, training_tasks = training_calvin_inventory(dataset)

    source_identity = authenticate_calvin_source(source_root)
    evaluation_by_task, validation_yaml_identity = official_calvin_evaluation_inventory(source_root)
    instructions = union_calvin_instruction_inventories(
        training_instructions=training_instructions,
        training_tasks=training_tasks,
        evaluation_by_task=evaluation_by_task,
    )
    evaluation_instructions = tuple(sorted(set(evaluation_by_task.values())))

    model_report = verify_huggingface_snapshot(
        model_snapshot,
        expected_revision=DEFAULT_DIFFUSION_GEMMA_SPEC.revision,
    )
    processor = AutoProcessor.from_pretrained(
        model_snapshot,
        local_files_only=True,
    )
    model_identity = SnapshotTreeIdentity.from_huggingface_report(
        DEFAULT_DIFFUSION_GEMMA_SPEC.model_id,
        model_report,
    )
    fixed_physical_prefix_width, measured = measure_sentinel_padded_prefix_geometry(
        processor,
        instructions=instructions,
        ordered_cameras=CALVIN_CAMERAS,
        padding_side=CALVIN_PADDING_SIDE,
        expected_fixed_prefix_width=args.expected_fixed_prefix_width,
    )
    contract = build_prefix_geometry_contract(
        model_identity=model_identity,
        processor_identity=model_identity,
        ordered_cameras=CALVIN_CAMERAS,
        instruction_lengths=measured,
        fixed_physical_prefix_width=fixed_physical_prefix_width,
        padding_side=CALVIN_PADDING_SIDE,
    )
    _require(
        contract["geometry"]["maximum_valid_prefix_length"] == fixed_physical_prefix_width - 1,
        "CALVIN fixed prefix width must reserve exactly one padding sentinel beyond its maximum",
    )
    content_sha256 = save_prefix_geometry_contract(output, contract)
    reloaded = load_prefix_geometry_contract(
        output,
        expected_content_sha256=content_sha256,
        expected_model_identity=model_identity,
        expected_processor_identity=model_identity,
        expected_ordered_cameras=CALVIN_CAMERAS,
        expected_instructions=instructions,
        expected_fixed_physical_prefix_width=fixed_physical_prefix_width,
    )
    _require(reloaded == contract, "published CALVIN prefix geometry differs after strict reload")

    print(
        json.dumps(
            {
                "artifact": str(output),
                "content_sha256": content_sha256,
                "dataset": dataset_report,
                "fixed_physical_prefix_width": fixed_physical_prefix_width,
                "instruction_inventories": {
                    "evaluation": _inventory_report(evaluation_instructions),
                    "training": {
                        **_inventory_report(training_instructions),
                        "annotation_count": training_annotation_count,
                    },
                    "union": _inventory_report(instructions),
                },
                "maximum_valid_prefix_length": contract["geometry"]["maximum_valid_prefix_length"],
                "model": model_identity.to_dict(),
                "official_source": source_identity,
                "official_task_phrase_sha256": _canonical_sha256(evaluation_by_task),
                "task_inventory": {
                    "count": len(training_tasks),
                    "sha256": _canonical_sha256(list(training_tasks)),
                },
                "validation_annotations": validation_yaml_identity,
                "status": "ok",
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
