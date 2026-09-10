#!/usr/bin/env python3
"""Bind simulator replay evidence to the pinned LeRobot LIBERO snapshot.

This stage runs in the closed training environment because that environment
contains the pinned parquet decoder.  It matches each simulator-selected source
trajectory by its complete float32 action sequence, then requires the complete
pre-action RGB/state sequence to match the training episode byte-for-byte after
the one documented 180-degree simulator camera transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import site
import stat
import sys
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _validate_project_source_root(project_root: Path) -> Path:
    source_path = project_root / "src"
    source_identity = os.lstat(source_path)
    if not stat.S_ISDIR(source_identity.st_mode):
        raise RuntimeError("project src import root must be a real directory")
    source_root = source_path.resolve()
    entries = list(os.scandir(source_root))
    if len(entries) != 1 or entries[0].name != "duo_vla" or not entries[0].is_dir(follow_symlinks=False):
        raise RuntimeError("project src import root must contain only the real duo_vla package directory")

    def visit(directory: Path, *, in_cache: bool) -> None:
        for entry in os.scandir(directory):
            mode = entry.stat(follow_symlinks=False).st_mode
            path = directory / entry.name
            if stat.S_ISDIR(mode):
                visit(path, in_cache=in_cache or entry.name == "__pycache__")
            elif stat.S_ISREG(mode):
                allowed = entry.name.endswith(".pyc") if in_cache else entry.name.endswith(".py")
                if not allowed:
                    raise RuntimeError(f"project package import tree contains a forbidden file: {path}")
            else:
                raise RuntimeError(f"project package import tree contains a linked or special entry: {path}")

    visit(source_root / "duo_vla", in_cache=False)
    source_text = str(source_root)
    sys.path[:] = [source_text] + [
        entry for entry in sys.path if str(Path(entry or os.getcwd()).resolve()) != source_text
    ]
    return source_root


def _validate_project_module_origins(source_root: Path, required_modules: set[str]) -> dict[str, str]:
    package_root = (source_root / "duo_vla").resolve(strict=True)
    loaded = {name: module for name, module in sys.modules.items() if name == "duo_vla" or name.startswith("duo_vla.")}
    missing = sorted(required_modules - set(loaded))
    if missing:
        raise RuntimeError(f"required checkout modules are not loaded: {missing}")
    origins: dict[str, str] = {}
    for name, module in sorted(loaded.items()):
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        module_file = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not isinstance(module_file, str):
            raise RuntimeError(f"checkout module has no file origin: {name}")
        resolved_origin = Path(origin).resolve(strict=True)
        resolved_file = Path(module_file).resolve(strict=True)
        if resolved_origin != resolved_file or not resolved_file.is_relative_to(package_root):
            raise RuntimeError(f"checkout module origin escapes authenticated source root: {name}")
        origins[name] = str(resolved_file)
    return origins


_PROJECT_SOURCE_ROOT = _validate_project_source_root(Path(__file__).resolve().parents[1])

from duo_vla.data.batching import collate_libero_samples  # noqa: E402
from duo_vla.data.libero import LiberoParquetDataset, _decode_rgb, _pyarrow_parquet  # noqa: E402
from duo_vla.data.libero_stats import LIBERO_DATASET_REVISION, load_libero_normalizers  # noqa: E402
from duo_vla.hf_snapshot import verify_huggingface_snapshot  # noqa: E402
from duo_vla.libero_replay_evidence import (  # noqa: E402
    DATASET_CONTENT_INVENTORY_SHA256,
    DATASET_SNAPSHOT_FILES_VERIFIED,
    DATASET_SNAPSHOT_TOTAL_BYTES,
    DATASET_TREE_METADATA_SHA256,
    EVIDENCE_SCHEMA,
    GATE_NAMES,
    NORMALIZATION_CONTENT_SHA256,
    NORMALIZATION_RAW_SHA256,
    ORIGINAL_HDF5_CONTENT_SHA256,
    PARQUET_BINDING_SCHEMA,
    PARQUET_TASK_SCHEMA,
    PRE_DISPATCH_MUTATIONS,
    SIMULATOR_STAGE_SCHEMA,
    SIMULATOR_TASK_SCHEMA,
    SUITES,
    TASK_INVENTORY_SHA256,
    ObservationSequenceDigester,
    ReplayEvidenceError,
    action_sequence_sha256,
    build_expected_inputs,
    canonical_gate_results,
    canonical_sha256,
    load_original_hdf5_inventory,
    load_source_parquet_alignment,
    observation_alignment_frame,
    observation_alignment_metrics,
    raw_evidence_record,
    read_stable_json,
    require,
    simulator_runtime_sha256,
    stable_regular_file_identity,
    task_identities,
    task_slug,
    validate_normalization,
    validate_observation_alignment_metrics,
)
from duo_vla.runtime_integrity import (  # noqa: E402
    content_address_train_venv,
    require_matching_train_venv,
)

_REQUIRED_PROJECT_MODULES = {
    "duo_vla",
    "duo_vla.data.batching",
    "duo_vla.data.libero",
    "duo_vla.data.libero_stats",
    "duo_vla.hf_snapshot",
    "duo_vla.libero_replay_evidence",
    "duo_vla.runtime_integrity",
}
_validate_project_module_origins(_PROJECT_SOURCE_ROOT, _REQUIRED_PROJECT_MODULES)

REQUIRED_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "RAYON_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "TZ": "UTC",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def validate_process_environment(project_root: Path, cache_root: Path) -> dict[str, Any]:
    configured_train_venv = os.environ.get("DUO_VLA_TRAIN_VENV", str(cache_root / "venvs/train"))
    train_venv = Path(configured_train_venv).resolve()
    allowed_train_venvs = {
        (cache_root / "venvs/train").resolve(),
        (cache_root / "venvs/train-single-gpu").resolve(),
    }
    require(train_venv in allowed_train_venvs, "parquet binder train venv is not an allowed execution profile")
    source_root = _validate_project_source_root(project_root)
    _validate_project_module_origins(source_root, _REQUIRED_PROJECT_MODULES)
    expected = {
        **REQUIRED_ENVIRONMENT,
        "DUO_VLA_CACHE_ROOT": str(cache_root),
        "DUO_VLA_PROJECT_ROOT": str(project_root),
        "DUO_VLA_TRAIN_VENV": str(train_venv),
        "HF_HOME": "/root/.cache/huggingface",
        "HOME": "/root",
        "PATH": f"{train_venv / 'bin'}:/usr/bin:/bin",
    }
    require(dict(os.environ) == expected, "parquet binder process environment is not the exact closed allowlist")
    require(Path(sys.prefix).resolve() == train_venv, "parquet binder is not running in the pinned train venv")
    require(sys.flags.safe_path == 1, "parquet binder requires Python safe-path mode")
    require(sys.flags.dont_write_bytecode == 1 and sys.dont_write_bytecode, "parquet binder requires -B")
    require(sys.flags.no_user_site == 1 and not site.ENABLE_USER_SITE, "parquet binder requires no user site")
    require(sys.pycache_prefix == "/dev/null", "parquet binder requires an impossible pycache lookup prefix")
    expected_invocation_flags = ["-P", "-B", "-X", "pycache_prefix=/dev/null"]
    require(
        sys.orig_argv[1:5] == expected_invocation_flags,
        "parquet binder must be invoked with the exact safe Python flags",
    )
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    compact_version = f"python{sys.version_info.major}{sys.version_info.minor}"
    expected_sys_path = [
        str(source_root),
        str(Path(sys.base_prefix) / "lib" / f"{compact_version}.zip"),
        str(Path(sys.base_prefix) / "lib" / version),
        str(Path(sys.base_exec_prefix) / "lib" / version / "lib-dynload"),
        str(train_venv / "lib" / version / "site-packages"),
    ]
    require(sys.path == expected_sys_path, f"parquet binder import search path differs: {sys.path}")
    return {
        "environment": expected,
        "environment_sha256": canonical_sha256(expected),
        "python_base_exec_prefix": sys.base_exec_prefix,
        "python_base_prefix": sys.base_prefix,
        "python_executable": sys.executable,
        "python_flags": {
            "dont_write_bytecode": bool(sys.dont_write_bytecode),
            "no_user_site": bool(sys.flags.no_user_site),
            "safe_path": bool(sys.flags.safe_path),
        },
        "python_invocation_flags": expected_invocation_flags,
        "python_prefix": str(train_venv),
        "python_pycache_prefix": sys.pycache_prefix,
        "python_version": sys.version.split()[0],
        "sys_path": expected_sys_path,
    }


def _write_exclusive_bytes(path: Path, payload: bytes, *, mode: int = 0o444) -> str:
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True).encode("ascii") + b"\n"
    return _write_exclusive_bytes(path, payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        str(path),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_exact_fields(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping) and set(value) == fields, f"{name} fields differ")
    return value


def _load_stage(
    stage_path: Path,
    expected_sha256: str,
    project_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    stage, stage_sha256 = read_stable_json(
        stage_path,
        name="expert replay simulator stage",
        expected_sha256=expected_sha256,
    )
    require(stage_sha256 == expected_sha256, "simulator stage SHA-256 differs")
    _require_exact_fields(
        stage,
        {"collector", "gates", "inputs", "schema", "source_scan", "status", "task_records"},
        "simulator stage",
    )
    require(
        stage.get("schema") == SIMULATOR_STAGE_SCHEMA and stage.get("status") == "complete",
        "simulator stage is incomplete",
    )
    root = stage_path.parent
    require(stage_path == root / "simulator-stage.json", "simulator stage must use its canonical path")
    commit, commit_sha256 = read_stable_json(
        root / "simulator-stage.commit.json",
        name="simulator stage commit",
    )
    require(
        commit
        == {
            "schema": "duo-vla-libero-expert-replay-simulator-stage-commit-v1",
            "simulator_stage_sha256": stage_sha256,
            "task_record_root_sha256": canonical_sha256(stage["task_records"]),
        },
        "simulator stage commit differs",
    )
    collector = stage.get("collector")
    alignment, training_source_indices, alignment_raw_sha256 = load_source_parquet_alignment(
        project_root / "configs/libero_source_parquet_alignment.json"
    )
    require(
        collector
        == {
            "path": "scripts/collect_libero_expert_replay.py",
            "sha256": stable_regular_file_identity(project_root / "scripts/collect_libero_expert_replay.py")["sha256"],
            "source_parquet_alignment_content_sha256": alignment["content_sha256"],
            "source_parquet_alignment_raw_sha256": alignment_raw_sha256,
            "shared_contract_sha256": stable_regular_file_identity(
                project_root / "src/duo_vla/libero_replay_evidence.py"
            )["sha256"],
        },
        "simulator stage collector source differs",
    )
    gates = _require_exact_fields(
        stage.get("gates"),
        {"controller_impulse_directions", "exact_gripper_set"},
        "simulator stage gates",
    )
    controller = _require_exact_fields(
        gates["controller_impulse_directions"],
        {
            "action_dimension",
            "checked_directions",
            "controller",
            "direction_match_count",
            "gripper_close",
            "gripper_open",
            "raw",
        },
        "simulator controller gate",
    )
    _require_exact_fields(controller["raw"], {"gripper_apertures", "responses"}, "simulator controller raw evidence")
    exact_gripper = _require_exact_fields(
        gates["exact_gripper_set"],
        {"observed_values", "unexpected_value_count", "zero_value_count"},
        "simulator gripper gate",
    )
    require(
        exact_gripper == {"observed_values": [-1.0, 1.0], "unexpected_value_count": 0, "zero_value_count": 0},
        "simulator gripper gate differs",
    )
    _require_exact_fields(
        stage.get("inputs"),
        {
            "original_hdf5_inventory_content_sha256",
            "original_hdf5_inventory_raw_sha256",
            "simulator_attestation_raw_sha256",
            "simulator_attestation_sha256",
            "simulator_runtime_sha256",
            "task_inventory_sha256",
        },
        "simulator stage inputs",
    )
    records = stage.get("task_records")
    require(isinstance(records, list) and len(records) == 40, "simulator stage task records are incomplete")
    source_scan = stage.get("source_scan")
    require(isinstance(source_scan, list) and len(source_scan) == 40, "simulator stage source scan is incomplete")
    expected_tasks = [(suite, task_id) for suite in SUITES for task_id in range(10)]
    observed_files = {"simulator-stage.json", "simulator-stage.commit.json"}
    task_documents: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        _require_exact_fields(
            record,
            {"bytes", "path", "sha256", "suite", "task_id"},
            f"simulator task record {index}",
        )
        suite, task_id = expected_tasks[index]
        relative = f"raw/{task_slug(suite, task_id)}.json"
        require(
            record.get("suite") == suite and record.get("task_id") == task_id and record.get("path") == relative,
            f"simulator task record {index} identity differs",
        )
        identity = stable_regular_file_identity(root / relative, expected_bytes=record.get("bytes"))
        require(identity["sha256"] == record.get("sha256"), f"simulator task record {index} hash differs")
        task_document, _raw = read_stable_json(
            root / relative,
            name=f"simulator task evidence {suite}:{task_id}",
            expected_sha256=record["sha256"],
        )
        _validate_simulator_task(task_document, suite=suite, task_id=task_id)
        require(
            all(
                attempt["training_linked"]
                == (attempt["source_episode_index"] in training_source_indices[(suite, task_id)])
                for attempt in task_document["attempts"]
            ),
            f"simulator training-link flags differ from the pinned alignment: {suite}:{task_id}",
        )
        scan_summary = _require_exact_fields(
            source_scan[index],
            {"gripper_counts", "path", "raw_transition_count", "retained_transition_count", "suite", "task_id"},
            f"simulator source-scan summary {index}",
        )
        expected_scan_summary = {
            "gripper_counts": task_document["source_scan"]["gripper_counts"],
            "path": task_document["source_file"]["path"],
            "raw_transition_count": task_document["source_scan"]["raw_transition_count"],
            "retained_transition_count": task_document["source_scan"]["retained_transition_count"],
            "suite": suite,
            "task_id": task_id,
        }
        require(scan_summary == expected_scan_summary, f"simulator source-scan summary {index} differs")
        require(
            task_document["collector_source_sha256"] == collector["sha256"],
            f"simulator task collector identity {index} differs",
        )
        task_documents.append(task_document)
        observed_files.add(relative)
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        identity = path.lstat()
        if stat.S_ISREG(identity.st_mode):
            require(identity.st_nlink == 1, f"simulator stage file has multiple hard links: {relative}")
            actual_files.add(relative)
        elif stat.S_ISDIR(identity.st_mode):
            actual_directories.add(relative)
        else:
            raise ReplayEvidenceError(f"simulator stage contains a special or linked entry: {relative}")
    require(actual_files == observed_files, "simulator stage contains missing or unexpected files")
    require(actual_directories == {"raw"}, "simulator stage contains missing or unexpected directories")
    return stage, task_documents, commit_sha256


def _validate_simulator_task(value: Mapping[str, Any], *, suite: str, task_id: int) -> None:
    fields = {
        "attempts",
        "collector_source_sha256",
        "pre_dispatch_integrity_controls",
        "reset_determinism",
        "schema",
        "selected",
        "source_file",
        "source_scan",
        "task",
    }
    require(set(value) == fields and value["schema"] == SIMULATOR_TASK_SCHEMA, "simulator task evidence fields differ")
    task = _require_exact_fields(value["task"], {"instruction", "suite", "task_id", "task_name"}, "simulator task")
    require(task["suite"] == suite and task["task_id"] == task_id, "simulator task identity differs")
    attempts = value["attempts"]
    require(isinstance(attempts, list) and bool(attempts), "simulator replay attempt list is empty")
    for index, attempt in enumerate(attempts):
        _require_exact_fields(
            attempt,
            {"action_sequence_sha256", "source_episode_index", "step_count", "success", "training_linked"},
            f"simulator replay attempt {index}",
        )
    require(
        all(not (item.get("success") is True and item.get("training_linked") is True) for item in attempts[:-1]),
        "a successful training-linked replay was skipped before selection",
    )
    require(
        attempts[-1].get("success") is True and attempts[-1].get("training_linked") is True,
        "selected replay did not succeed with a training link",
    )
    require(
        [item.get("source_episode_index") for item in attempts] == list(range(len(attempts))),
        "simulator replay attempts are not in canonical source order",
    )
    selected = _require_exact_fields(
        value["selected"],
        {
            "action_sequence_sha256",
            "alignment_probe",
            "initial_state_sha256",
            "inverted_gripper_action_sequence_sha256",
            "observation_alignment_features",
            "observation_sequence_sha256",
            "source_episode_index",
            "step_count",
            "success",
            "swapped_observation_sequence_sha256",
            "trajectory_sha256",
            "zero_action_sequence_sha256",
        },
        "simulator selected replay",
    )
    self_alignment = observation_alignment_metrics(
        selected["observation_alignment_features"],
        selected["observation_alignment_features"].get("frames", []),
    )
    require(
        self_alignment["passed"] is True and self_alignment["frames"] == selected["step_count"],
        "simulator observation-alignment feature structure differs",
    )
    _require_exact_fields(
        selected["alignment_probe"],
        {
            "post_action_observation_sha256",
            "pre_action_observation_sha256",
            "retained_transition_index",
            "source_transition_index",
        },
        "simulator alignment probe",
    )
    require(
        selected.get("success") is True
        and selected.get("source_episode_index") == attempts[-1]["source_episode_index"]
        and selected.get("action_sequence_sha256") == attempts[-1]["action_sequence_sha256"]
        and selected.get("step_count") == attempts[-1]["step_count"],
        "simulator selected replay differs from the first successful training-linked attempt",
    )
    reset = _require_exact_fields(
        value["reset_determinism"],
        {
            "environment_seed",
            "first_observation_sha256",
            "first_simulator_state_sha256",
            "probe_environment_count",
            "second_observation_sha256",
            "second_simulator_state_sha256",
            "seed_calls_per_environment",
            "settle_steps",
        },
        "simulator reset evidence",
    )
    require(
        reset.get("environment_seed") == 0
        and reset.get("probe_environment_count") == 2
        and reset.get("seed_calls_per_environment") == 1
        and reset.get("settle_steps") == 10
        and reset.get("first_observation_sha256") == reset.get("second_observation_sha256")
        and reset.get("first_simulator_state_sha256") == reset.get("second_simulator_state_sha256"),
        "simulator deterministic reset evidence differs",
    )
    controls = value["pre_dispatch_integrity_controls"]
    require(set(controls) == set(PRE_DISPATCH_MUTATIONS), "simulator pre-dispatch mutation inventory differs")
    require(
        all(item == {"mutation_detected": True} for item in controls.values()),
        "simulator pre-dispatch integrity mutation was not detected",
    )
    source_file = _require_exact_fields(
        value["source_file"],
        {"bytes", "path", "sha256", "suite", "task_id", "task_name"},
        "simulator source file",
    )
    require(source_file["suite"] == suite and source_file["task_id"] == task_id, "simulator source file differs")
    source_scan = _require_exact_fields(
        value["source_scan"],
        {"demonstrations", "gripper_counts", "raw_transition_count", "retained_transition_count"},
        "simulator source scan",
    )
    demonstrations = source_scan["demonstrations"]
    require(isinstance(demonstrations, list) and bool(demonstrations), "simulator source demonstrations are empty")
    for index, demonstration in enumerate(demonstrations):
        _require_exact_fields(
            demonstration,
            {
                "action_sequence_sha256",
                "initial_state_sha256",
                "raw_transition_count",
                "retained_transition_count",
                "source_action_dtype",
                "source_episode_index",
                "source_state_sequence_sha256",
            },
            f"simulator source demonstration {index}",
        )
        require(demonstration["source_episode_index"] == index, "simulator source demonstrations are not ordered")
        try:
            action_dtype = np.dtype(demonstration["source_action_dtype"])
        except (TypeError, ValueError) as exc:
            raise ReplayEvidenceError("simulator source action dtype is invalid") from exc
        require(
            action_dtype.str == demonstration["source_action_dtype"] and np.issubdtype(action_dtype, np.floating),
            "simulator source action dtype is not canonical floating point",
        )


def _episode_action_matches(
    dataset: LiberoParquetDataset,
    wanted: Mapping[tuple[str, str, int], tuple[str, int]],
) -> dict[tuple[str, int], int]:
    """Scan only action columns, returning a unique parquet episode per replay."""

    episodes_by_file: dict[Any, list[Any]] = {}
    for episode in dataset.episodes:
        physical = dataset._data_file_by_episode[episode.episode_index]
        episodes_by_file.setdefault(physical, []).append(episode)
    matches: dict[tuple[str, int], list[int]] = {identity: [] for identity in wanted.values()}
    parquet = _pyarrow_parquet()
    for physical in dataset.data_files:
        path = dataset.root / "data" / f"chunk-{physical.chunk_index:03d}" / f"file-{physical.file_index:03d}.parquet"
        table = parquet.read_table(path, columns=["action"])
        for episode in episodes_by_file[physical]:
            offset = episode.global_start - physical.global_start
            episode_rows = table.slice(offset, episode.length)
            actions = dataset._validated_episode_actions(episode_rows).cpu().numpy()
            key = (episode.task, action_sequence_sha256(actions), episode.length)
            if key in wanted:
                matches[wanted[key]].append(episode.episode_index)
    unique: dict[tuple[str, int], int] = {}
    for identity, episode_indices in matches.items():
        require(len(episode_indices) == 1, f"simulator trajectory does not have one unique parquet match: {identity}")
        unique[identity] = episode_indices[0]
    return unique


def _bind_task(dataset: LiberoParquetDataset, task_document: Mapping[str, Any], episode_index: int) -> dict[str, Any]:
    task = task_document["task"]
    selected = task_document["selected"]
    episode = dataset.episodes[episode_index]
    require(episode.task == task["instruction"], "matched parquet episode language differs")
    physical = dataset._data_file_by_episode[episode_index]
    parquet = _pyarrow_parquet()
    path = dataset.root / "data" / f"chunk-{physical.chunk_index:03d}" / f"file-{physical.file_index:03d}.parquet"
    table = parquet.read_table(
        path,
        columns=[
            "action",
            "episode_index",
            "frame_index",
            "index",
            "observation.images.image",
            "observation.images.image2",
            "observation.state",
            "task_index",
        ],
    )
    rows = dataset._episode_rows_from_table(episode, physical, table)
    actions = dataset._validated_episode_actions(rows).cpu().numpy()
    task_indices = np.asarray(rows.column("task_index").to_numpy(), dtype=np.int64)
    require(
        np.array_equal(task_indices, np.full(episode.length, task_indices[0], dtype=np.int64)),
        "matched parquet task indices differ within the episode",
    )
    dataset_task_index = int(task_indices[0])
    require(dataset.task_by_index.get(dataset_task_index) == episode.task, "matched parquet task mapping differs")
    action_digest = action_sequence_sha256(actions)
    require(action_digest == selected["action_sequence_sha256"], "matched parquet action sequence differs")
    observations = ObservationSequenceDigester()
    dataset_alignment_frames: list[dict[str, Any]] = []
    for row in rows.to_pylist():
        agentview = _decode_rgb(row["observation.images.image"])
        wrist = _decode_rgb(row["observation.images.image2"])
        state = np.asarray(row["observation.state"], dtype=np.float32)
        observations.update(agentview, wrist, state)
        dataset_alignment_frames.append(observation_alignment_frame(agentview, wrist, state))
    dataset_observation_digest = observations.hexdigest()
    alignment_metrics = validate_observation_alignment_metrics(
        observation_alignment_metrics(
            selected["observation_alignment_features"],
            dataset_alignment_frames,
        )
    )
    require(
        alignment_metrics["passed"] is True,
        f"matched parquet pre-action RGB/state alignment differs: {task['suite']}:{task['task_id']}",
    )
    return {
        "action_sequence_sha256": action_digest,
        "dataset_observation_sequence_sha256": dataset_observation_digest,
        "dataset_episode_index": episode_index,
        "dataset_global_start": episode.global_start,
        "dataset_global_stop": episode.global_stop,
        "dataset_task_index": dataset_task_index,
        "instruction": episode.task,
        "observation_alignment_metrics": alignment_metrics,
        "observation_sequence_sha256": selected["observation_sequence_sha256"],
        "schema": PARQUET_TASK_SCHEMA,
        "source_episode_index": selected["source_episode_index"],
        "step_count": episode.length,
        "suite": task["suite"],
        "task_id": task["task_id"],
        "task_name": task["task_name"],
        "trajectory_sha256": selected["trajectory_sha256"],
    }


def _scan_training_grippers(dataset: LiberoParquetDataset) -> tuple[list[float], int, int]:
    observed: set[float] = set()
    validated_episodes = 0
    validated_actions = 0
    parquet = _pyarrow_parquet()
    episodes_by_file: dict[Any, list[Any]] = {}
    for episode in dataset.episodes:
        physical = dataset._data_file_by_episode[episode.episode_index]
        episodes_by_file.setdefault(physical, []).append(episode)
    for physical in dataset.data_files:
        path = dataset.root / "data" / f"chunk-{physical.chunk_index:03d}" / f"file-{physical.file_index:03d}.parquet"
        table = parquet.read_table(path, columns=["action"])
        for episode in episodes_by_file[physical]:
            offset = episode.global_start - physical.global_start
            actions = dataset._validated_episode_actions(table.slice(offset, episode.length))
            observed.update(float(value) for value in torch.unique(actions[:, -1]).tolist())
            validated_episodes += 1
            validated_actions += int(actions.shape[0])
    require(validated_episodes == 1693, "full action scan did not validate all 1693 episodes")
    require(validated_actions == 273465, "full action scan did not validate all 273465 actions")
    return sorted(observed), validated_episodes, validated_actions


def _production_dataset_gates(
    dataset: LiberoParquetDataset,
    episode_index: int,
    normalization_path: Path,
    expected_normalization: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exercise production sampling and normalization on one matched episode."""

    state_normalizer, action_normalizer, loaded_normalization = load_libero_normalizers(normalization_path)
    require(
        loaded_normalization == expected_normalization,
        "production normalization loader returned different content",
    )
    episode = dataset.episodes[episode_index]
    sample = dataset.sample(episode_index, episode.length - 1, horizon=8)
    require(
        sample.episode_index == episode_index
        and sample.frame_index == episode.length - 1
        and sample.instruction == episode.task,
        "production terminal sample identity differs",
    )
    chunk = sample.action_chunk
    require(chunk.actions.shape == (8, 7), "production terminal action chunk shape differs")
    require(chunk.valid_mask.shape == (8,), "production terminal validity-mask shape differs")
    require(chunk.valid_mask.dtype == torch.bool, "production terminal validity mask is not boolean")
    expected_mask = torch.tensor([True, False, False, False, False, False, False, False])
    require(torch.equal(chunk.valid_mask.cpu(), expected_mask), "production terminal validity mask differs")
    require(bool(torch.equal(chunk.actions[1:], torch.zeros_like(chunk.actions[1:]))), "terminal padding is not zero")
    episode_rows = dataset._read_episode_rows(episode)
    episode_actions = dataset._validated_episode_actions(episode_rows)
    require(
        torch.equal(chunk.actions[0], episode_actions[-1]),
        "production terminal chunk anchor differs from the underlying episode action",
    )

    batch = collate_libero_samples(
        (sample,),
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
    )
    require(batch.action_valid_mask.dtype == torch.bool, "production batch validity mask is not boolean")
    require(torch.equal(batch.action_valid_mask[0].cpu(), expected_mask), "production batch validity mask differs")
    require(
        bool(torch.equal(batch.clean_actions[0, 1:], torch.zeros_like(batch.clean_actions[0, 1:]))),
        "production normalized terminal padding is not zero",
    )

    state = sample.observation.state
    action = chunk.actions[0]
    normalized_state = batch.states[0]
    restored_state = state_normalizer.unnormalize(normalized_state)
    state_lower = state_normalizer.lower.to(device=state.device, dtype=state.dtype)
    state_upper = state_normalizer.upper.to(device=state.device, dtype=state.dtype)
    expected_state = state.clamp(state_lower, state_upper)

    normalized_action = batch.clean_actions[0, 0]
    restored_action = action_normalizer.unnormalize(normalized_action)
    action_lower = action_normalizer.continuous.lower.to(device=action.device, dtype=action.dtype)
    action_upper = action_normalizer.continuous.upper.to(device=action.device, dtype=action.dtype)
    expected_action = action.clone()
    expected_action[:6] = action[:6].clamp(action_lower, action_upper)
    expected_action[6] = 1.0 if action[6] >= 0 else -1.0
    gripper_sign_exact = bool(normalized_action[6] == action[6] and restored_action[6] == action[6])
    require(gripper_sign_exact, "production action normalizer changed the binary gripper sign")
    max_abs_error = max(
        float(torch.max(torch.abs(restored_state - expected_state)).item()),
        float(torch.max(torch.abs(restored_action - expected_action)).item()),
    )
    require(max_abs_error <= 1e-6, "production normalization round-trip error exceeds tolerance")
    return (
        {
            "continuous_action_dimensions": 6,
            "gripper_sign_exact": True,
            "max_abs_error": max_abs_error,
            "samples": 1,
            "state_dimensions": 8,
        },
        {
            "checked_terminal_anchors": 1,
            "cross_boundary_count": 0,
            "horizon": 8,
            "padding_mask_exact": True,
            "padding_value": "zeros",
        },
    )


def _build_gate_results(
    stage: Mapping[str, Any],
    task_documents: Sequence[Mapping[str, Any]],
    *,
    normalization_round_trip: Mapping[str, Any],
    episode_boundary_chunk_fixture: Mapping[str, Any],
) -> dict[str, Any]:
    template = canonical_gate_results()
    controller = dict(stage["gates"]["controller_impulse_directions"])
    controller.pop("raw")
    resets = [task["reset_determinism"] for task in task_documents]
    require(
        all(item["first_observation_sha256"] == item["second_observation_sha256"] for item in resets),
        "reset evidence differs",
    )
    mutations = {name: {"mutation_detected": True} for name in PRE_DISPATCH_MUTATIONS}
    return {
        "schema_revision_counts": template["schema_revision_counts"],
        "exact_gripper_set": stage["gates"]["exact_gripper_set"],
        "controller_impulse_directions": controller,
        "normalization_round_trip": dict(normalization_round_trip),
        "episode_boundary_chunk_fixture": dict(episode_boundary_chunk_fixture),
        "pre_action_observation_alignment": template["pre_action_observation_alignment"],
        "camera_transform_parity": template["camera_transform_parity"],
        "deterministic_fixed_state_reset": template["deterministic_fixed_state_reset"],
        "pre_dispatch_integrity_controls": {"checked_tasks": 40, "mutations": mutations},
    }


def bind(args: argparse.Namespace, *, project_root: Path | None = None) -> tuple[Path, str]:
    root = Path(__file__).resolve().parents[1] if project_root is None else project_root.resolve()
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")).resolve()
    process_identity = validate_process_environment(root, cache_root)
    train_venv = Path(process_identity["python_prefix"])
    train_venv_identity = content_address_train_venv(train_venv)

    stage_path = Path(os.path.abspath(args.simulator_stage))
    require(stage_path.resolve() == stage_path, "simulator stage path must not contain linked path components")
    stage, task_documents, simulator_stage_commit_sha256 = _load_stage(
        stage_path,
        args.simulator_stage_sha256,
        root,
    )
    evidence_root = stage_path.parent
    inventory, _records, inventory_raw_sha256 = load_original_hdf5_inventory(args.inventory.resolve())
    require(inventory_raw_sha256 == args.inventory_sha256, "original HDF5 inventory raw SHA-256 differs")
    require(
        stage["inputs"]["original_hdf5_inventory_content_sha256"] == ORIGINAL_HDF5_CONTENT_SHA256
        and stage["inputs"]["original_hdf5_inventory_raw_sha256"] == inventory_raw_sha256,
        "simulator stage original-HDF5 identity differs",
    )
    attestation, attestation_raw_sha256 = read_stable_json(
        args.simulator_attestation.resolve(),
        name="simulator attestation",
        expected_sha256=args.simulator_attestation_sha256,
    )
    tasks = task_identities(attestation)
    require(attestation["task_inventory_sha256"] == TASK_INVENTORY_SHA256, "simulator task inventory differs")
    require(
        stage["inputs"]["simulator_attestation_raw_sha256"] == attestation_raw_sha256
        and stage["inputs"]["simulator_attestation_sha256"] == canonical_sha256(attestation)
        and stage["inputs"]["simulator_runtime_sha256"] == simulator_runtime_sha256(attestation),
        "simulator stage attestation identity differs",
    )
    dataset_tree, dataset_tree_raw_sha256 = read_stable_json(
        args.dataset_tree_metadata.resolve(),
        name="dataset tree metadata",
        expected_sha256=args.dataset_tree_metadata_sha256,
    )
    normalization, normalization_raw_sha256 = read_stable_json(
        args.normalization_artifact.resolve(), name="normalization artifact"
    )
    validate_normalization(normalization)
    require(
        normalization_raw_sha256 == NORMALIZATION_RAW_SHA256
        and normalization["content_sha256"] == NORMALIZATION_CONTENT_SHA256,
        "normalization artifact identity differs",
    )
    snapshot_report = verify_huggingface_snapshot(
        args.snapshot_root.resolve(),
        expected_revision=LIBERO_DATASET_REVISION,
    )
    require(
        snapshot_report["tree_metadata_sha256"] == DATASET_TREE_METADATA_SHA256
        and snapshot_report["content_inventory_sha256"] == DATASET_CONTENT_INVENTORY_SHA256,
        "live LIBERO training snapshot differs",
    )
    require(
        snapshot_report["files_verified"] == DATASET_SNAPSHOT_FILES_VERIFIED
        and snapshot_report["total_bytes"] == DATASET_SNAPSHOT_TOTAL_BYTES,
        "live LIBERO training snapshot count/size differs",
    )
    expected_inputs = build_expected_inputs(
        root,
        simulator_attestation=attestation,
        simulator_attestation_raw_sha256=attestation_raw_sha256,
        dataset_tree=dataset_tree,
        dataset_tree_raw_sha256=dataset_tree_raw_sha256,
        normalization=normalization,
        normalization_raw_sha256=normalization_raw_sha256,
        dataset_snapshot=snapshot_report,
        train_venv_identity=train_venv_identity,
        original_hdf5_inventory=inventory,
        original_hdf5_inventory_raw_sha256=inventory_raw_sha256,
    )

    dataset = LiberoParquetDataset(args.snapshot_root.resolve())
    wanted: dict[tuple[str, str, int], tuple[str, int]] = {}
    for task_document in task_documents:
        task = task_document["task"]
        selected = task_document["selected"]
        key = (task["instruction"], selected["action_sequence_sha256"], selected["step_count"])
        require(key not in wanted, "selected simulator action identity is duplicated")
        wanted[key] = (task["suite"], task["task_id"])
    matched_episodes = _episode_action_matches(dataset, wanted)
    training_grippers, validated_episodes, validated_actions = _scan_training_grippers(dataset)
    require(training_grippers == [-1.0, 1.0], "training parquet gripper set is not exactly {-1,+1}")
    normalization_round_trip, episode_boundary_chunk_fixture = _production_dataset_gates(
        dataset,
        matched_episodes[(SUITES[0], 0)],
        args.normalization_artifact.resolve(),
        normalization,
    )

    binding_dir = evidence_root / "binding"
    binding_dir.mkdir(mode=0o755, exist_ok=False)
    binding_records: list[dict[str, Any]] = []
    binding_documents: list[dict[str, Any]] = []
    try:
        for task_document in task_documents:
            task = task_document["task"]
            identity = (task["suite"], task["task_id"])
            document = _bind_task(dataset, task_document, matched_episodes[identity])
            relative = f"binding/{task_slug(*identity)}.json"
            digest = _write_exclusive_json(evidence_root / relative, document)
            binding_records.append(
                {
                    "bytes": (evidence_root / relative).stat().st_size,
                    "path": relative,
                    "sha256": digest,
                    "suite": identity[0],
                    "task_id": identity[1],
                }
            )
            binding_documents.append(document)

        require(
            len({document["dataset_episode_index"] for document in binding_documents}) == 40,
            "matched parquet episodes are not unique",
        )
        require(
            sorted(document["dataset_task_index"] for document in binding_documents) == list(range(40)),
            "matched parquet task indices are not the canonical 0..39 inventory",
        )
        spans = sorted(
            (document["dataset_global_start"], document["dataset_global_stop"]) for document in binding_documents
        )
        require(
            all(start < stop for start, stop in spans)
            and all(
                left_stop <= right_start for (_left_start, left_stop), (right_start, _right_stop) in pairwise(spans)
            ),
            "matched parquet episode spans overlap",
        )

        stage2 = {
            "binder": {
                "path": "scripts/bind_libero_expert_replay.py",
                "sha256": stable_regular_file_identity(Path(__file__))["sha256"],
                "shared_contract_sha256": stable_regular_file_identity(root / "src/duo_vla/libero_replay_evidence.py")[
                    "sha256"
                ],
            },
            "gates": {
                "episode_boundary_chunk_fixture": dict(episode_boundary_chunk_fixture),
                "normalization_round_trip": dict(normalization_round_trip),
                "training_gripper_set": {
                    "observed_values": training_grippers,
                    "unexpected_value_count": 0,
                    "validated_actions": validated_actions,
                    "validated_episodes": validated_episodes,
                    "zero_value_count": 0,
                },
            },
            "inputs": {
                "dataset_content_inventory_sha256": snapshot_report["content_inventory_sha256"],
                "dataset_snapshot_files_verified": snapshot_report["files_verified"],
                "dataset_snapshot_total_bytes": snapshot_report["total_bytes"],
                "dataset_tree_sha256": snapshot_report["tree_metadata_sha256"],
                "original_hdf5_inventory_content_sha256": inventory["content_sha256"],
                "original_hdf5_inventory_raw_sha256": inventory_raw_sha256,
                "simulator_stage_commit_sha256": simulator_stage_commit_sha256,
                "simulator_stage_sha256": args.simulator_stage_sha256,
                "train_venv_root_sha256": train_venv_identity["root_sha256"],
            },
            "process": process_identity,
            "schema": PARQUET_BINDING_SCHEMA,
            "status": "complete",
            "task_records": binding_records,
            "train_venv": train_venv_identity,
        }
        require(
            set(stage2)
            == {
                "binder",
                "gates",
                "inputs",
                "process",
                "schema",
                "status",
                "task_records",
                "train_venv",
            },
            "parquet binding stage fields differ",
        )
        require(
            set(stage2["gates"])
            == {"episode_boundary_chunk_fixture", "normalization_round_trip", "training_gripper_set"},
            "parquet binding gate fields differ",
        )
        require(
            set(stage2["process"])
            == {
                "environment",
                "environment_sha256",
                "python_base_exec_prefix",
                "python_base_prefix",
                "python_executable",
                "python_flags",
                "python_invocation_flags",
                "python_prefix",
                "python_pycache_prefix",
                "python_version",
                "sys_path",
            },
            "parquet binding process fields differ",
        )
        require(
            set(stage2["inputs"])
            == {
                "dataset_content_inventory_sha256",
                "dataset_snapshot_files_verified",
                "dataset_snapshot_total_bytes",
                "dataset_tree_sha256",
                "original_hdf5_inventory_content_sha256",
                "original_hdf5_inventory_raw_sha256",
                "simulator_stage_commit_sha256",
                "simulator_stage_sha256",
                "train_venv_root_sha256",
            },
            "parquet binding input fields differ",
        )
        stage2_digest = _write_exclusive_json(evidence_root / "parquet-binding.json", stage2)

        raw_records: list[dict[str, Any]] = [
            raw_evidence_record(evidence_root, "parquet-binding.json", "parquet-binding"),
            raw_evidence_record(evidence_root, "simulator-stage.json", "simulator-stage"),
            raw_evidence_record(
                evidence_root,
                "simulator-stage.commit.json",
                "simulator-stage-commit",
            ),
        ]
        for record_index, simulator_record in enumerate(stage["task_records"]):
            binding_record = binding_records[record_index]
            identity = (simulator_record["suite"], simulator_record["task_id"])
            raw_records.append(
                raw_evidence_record(evidence_root, binding_record["path"], f"parquet-{task_slug(*identity)}")
            )
            raw_records.append(
                raw_evidence_record(evidence_root, simulator_record["path"], f"simulator-{task_slug(*identity)}")
            )
        raw_records.sort(key=lambda item: item["id"])
        expected_raw_ids = {"parquet-binding", "simulator-stage", "simulator-stage-commit"}
        expected_raw_ids.update(f"parquet-{task_slug(suite, task_id)}" for suite in SUITES for task_id in range(10))
        expected_raw_ids.update(f"simulator-{task_slug(suite, task_id)}" for suite in SUITES for task_id in range(10))
        require(
            len(raw_records) == 83 and {item["id"] for item in raw_records} == expected_raw_ids,
            "parquet binding raw-evidence inventory differs",
        )
        parquet_binding_record = next(item for item in raw_records if item["id"] == "parquet-binding")
        require(stage2_digest == parquet_binding_record["sha256"], "parquet binding publication digest differs")
        task_raw_ids = [
            item["id"]
            for item in raw_records
            if item["id"].startswith("parquet-libero") or item["id"].startswith("simulator-libero")
        ]
        gate_results = _build_gate_results(
            stage,
            task_documents,
            normalization_round_trip=normalization_round_trip,
            episode_boundary_chunk_fixture=episode_boundary_chunk_fixture,
        )
        simulator_commit_evidence = ["simulator-stage-commit"]
        gate_evidence = {
            "schema_revision_counts": ["parquet-binding"],
            "exact_gripper_set": ["parquet-binding", "simulator-stage", *simulator_commit_evidence],
            "controller_impulse_directions": ["simulator-stage", *simulator_commit_evidence],
            "normalization_round_trip": ["parquet-binding"],
            "episode_boundary_chunk_fixture": ["parquet-binding"],
            "pre_action_observation_alignment": [*task_raw_ids, *simulator_commit_evidence],
            "camera_transform_parity": [*task_raw_ids, *simulator_commit_evidence],
            "deterministic_fixed_state_reset": [
                item["id"] for item in raw_records if item["id"].startswith("simulator-libero")
            ]
            + simulator_commit_evidence,
            "pre_dispatch_integrity_controls": [
                item["id"] for item in raw_records if item["id"].startswith("simulator-libero")
            ]
            + simulator_commit_evidence,
        }
        gates = [
            {
                "evidence_ids": sorted(gate_evidence[name]),
                "name": name,
                "passed": True,
                "result": gate_results[name],
                "result_sha256": canonical_sha256(gate_results[name]),
            }
            for name in GATE_NAMES
        ]
        task_by_identity = {(task["suite"], task["task_id"]): task for task in tasks}
        demonstrations: list[dict[str, Any]] = []
        for task_index, simulator_task in enumerate(task_documents):
            binding_task = binding_documents[task_index]
            task = simulator_task["task"]
            identity = (task["suite"], task["task_id"])
            expected_task = task_by_identity[identity]
            selected = simulator_task["selected"]
            require(
                binding_task["action_sequence_sha256"] == selected["action_sequence_sha256"]
                and binding_task["observation_sequence_sha256"] == selected["observation_sequence_sha256"]
                and binding_task["trajectory_sha256"] == selected["trajectory_sha256"],
                "parquet binding changed before evidence publication",
            )
            require(
                task["task_name"] == expected_task["task_name"] and task["instruction"] == expected_task["instruction"],
                "simulator evidence task metadata differs from the attestation",
            )
            demonstrations.append(
                {
                    "action_sequence_sha256": selected["action_sequence_sha256"],
                    "demonstration_id": f"{task_slug(*identity)}-source-{selected['source_episode_index']:04d}",
                    "initial_state_sha256": selected["initial_state_sha256"],
                    "instruction": task["instruction"],
                    "observation_sequence_sha256": selected["observation_sequence_sha256"],
                    "raw_evidence_ids": sorted(
                        [f"parquet-{task_slug(*identity)}", f"simulator-{task_slug(*identity)}"]
                    ),
                    "regenerated": True,
                    "source_episode_index": selected["source_episode_index"],
                    "step_count": selected["step_count"],
                    "success": True,
                    "suite": task["suite"],
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "trajectory_sha256": selected["trajectory_sha256"],
                }
            )
        demonstrations.sort(key=lambda item: item["demonstration_id"])
        evidence = {
            "demonstrations": demonstrations,
            "gates": gates,
            "inputs": expected_inputs,
            "raw_evidence": raw_records,
            "schema": EVIDENCE_SCHEMA,
        }
        evidence_payload = (
            json.dumps(evidence, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True).encode("ascii") + b"\n"
        )
        evidence_digest = hashlib.sha256(evidence_payload).hexdigest()
        snapshot_report_after = verify_huggingface_snapshot(
            args.snapshot_root.resolve(),
            expected_revision=LIBERO_DATASET_REVISION,
        )
        require(snapshot_report_after == snapshot_report, "LIBERO training snapshot changed during binding")
        train_venv_identity_after = content_address_train_venv(train_venv)
        try:
            require_matching_train_venv(train_venv_identity, train_venv_identity_after)
        except RuntimeError as exc:
            raise RuntimeError("train venv/base-Python runtime changed during parquet binding") from exc
        inventory_after, _records_after, inventory_raw_sha256_after = load_original_hdf5_inventory(
            args.inventory.resolve()
        )
        require(
            inventory_after == inventory and inventory_raw_sha256_after == inventory_raw_sha256,
            "original HDF5 inventory changed during parquet binding",
        )
        attestation_after, attestation_raw_sha256_after = read_stable_json(
            args.simulator_attestation.resolve(),
            name="simulator attestation publication recheck",
            expected_sha256=args.simulator_attestation_sha256,
        )
        require(
            attestation_after == attestation and attestation_raw_sha256_after == attestation_raw_sha256,
            "simulator attestation changed during parquet binding",
        )
        dataset_tree_after, dataset_tree_raw_sha256_after = read_stable_json(
            args.dataset_tree_metadata.resolve(),
            name="dataset tree metadata publication recheck",
            expected_sha256=args.dataset_tree_metadata_sha256,
        )
        require(
            dataset_tree_after == dataset_tree and dataset_tree_raw_sha256_after == dataset_tree_raw_sha256,
            "dataset tree metadata changed during parquet binding",
        )
        normalization_after, normalization_raw_sha256_after = read_stable_json(
            args.normalization_artifact.resolve(),
            name="normalization artifact publication recheck",
            expected_sha256=NORMALIZATION_RAW_SHA256,
        )
        require(
            normalization_after == normalization and normalization_raw_sha256_after == normalization_raw_sha256,
            "normalization artifact changed during parquet binding",
        )
        expected_inputs_after = build_expected_inputs(
            root,
            simulator_attestation=attestation_after,
            simulator_attestation_raw_sha256=attestation_raw_sha256_after,
            dataset_tree=dataset_tree_after,
            dataset_tree_raw_sha256=dataset_tree_raw_sha256_after,
            normalization=normalization_after,
            normalization_raw_sha256=normalization_raw_sha256_after,
            dataset_snapshot=snapshot_report_after,
            train_venv_identity=train_venv_identity_after,
            original_hdf5_inventory=inventory_after,
            original_hdf5_inventory_raw_sha256=inventory_raw_sha256_after,
        )
        require(expected_inputs_after == expected_inputs, "qualification inputs changed during parquet binding")
        stage_after, stage_sha256_after = read_stable_json(
            stage_path,
            name="simulator stage publication recheck",
            expected_sha256=args.simulator_stage_sha256,
        )
        require(stage_after == stage and stage_sha256_after == args.simulator_stage_sha256, "simulator stage changed")
        require(
            {
                "path": "scripts/bind_libero_expert_replay.py",
                "sha256": stable_regular_file_identity(Path(__file__))["sha256"],
                "shared_contract_sha256": stable_regular_file_identity(root / "src/duo_vla/libero_replay_evidence.py")[
                    "sha256"
                ],
            }
            == stage2["binder"],
            "binder source changed during parquet binding",
        )
        require(
            validate_process_environment(root, cache_root) == stage2["process"],
            "binder process identity changed during parquet binding",
        )
        raw_records_after = [raw_evidence_record(evidence_root, record["path"], record["id"]) for record in raw_records]
        require(raw_records_after == raw_records, "raw evidence changed before qualification publication")
        _write_exclusive_bytes(
            evidence_root / "evidence.json.sha256",
            f"{evidence_digest}  evidence.json\n".encode("ascii"),
        )
        _fsync_directory(binding_dir)
        _fsync_directory(evidence_root)
        _write_exclusive_bytes(evidence_root / "evidence.json", evidence_payload)
        _fsync_directory(evidence_root)
        return evidence_root / "evidence.json", evidence_digest
    except BaseException as exc:
        try:
            _write_exclusive_json(
                evidence_root / "binding-failed.json",
                {
                    "error_type": type(exc).__name__,
                    "schema": "duo-vla-libero-expert-replay-parquet-failure-v1",
                    "status": "failed",
                },
            )
            _fsync_directory(evidence_root)
        except BaseException:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-stage", type=Path, required=True)
    parser.add_argument("--simulator-stage-sha256", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--simulator-attestation", type=Path, required=True)
    parser.add_argument("--simulator-attestation-sha256", required=True)
    parser.add_argument("--dataset-tree-metadata", type=Path, required=True)
    parser.add_argument("--dataset-tree-metadata-sha256", required=True)
    parser.add_argument("--normalization-artifact", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    path, digest = bind(parse_args(argv))
    print(json.dumps({"evidence_manifest": str(path), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
