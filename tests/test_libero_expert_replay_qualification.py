"""Adversarial tests for the fail-closed LIBERO expert replay gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from duo_vla import libero_replay_evidence as SHARED_CONTRACT

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify_libero_expert_replay.py"
SPEC = importlib.util.spec_from_file_location("qualify_libero_expert_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUALIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFY)


def test_authenticated_shared_contract_matches_qualification_contract() -> None:
    assert SHARED_CONTRACT.source_identity(ROOT) == QUALIFY.source_identity(ROOT)
    assert SHARED_CONTRACT.canonical_gate_results() == QUALIFY.canonical_gate_results()
    assert inspect.signature(SHARED_CONTRACT.build_expected_inputs) == inspect.signature(QUALIFY.build_expected_inputs)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _task_inventory() -> list[dict[str, Any]]:
    source_inventory = json.loads((ROOT / "configs/libero_original_hdf5_inventory.json").read_text(encoding="utf-8"))
    task_names = {(item["suite"], item["task_id"]): item["task_name"] for item in source_inventory["files"]}
    return [
        {
            "bddl_sha256": _sha(f"bddl:{suite}:{task_id}"),
            "instruction": f"instruction {suite} {task_id}",
            "reset_count": 50,
            "reset_dtype": "float64",
            "reset_shape": [100],
            "reset_state_sha256": [_sha(f"state:{suite}:{task_id}:{index}") for index in range(50)],
            "suite": suite,
            "task_id": task_id,
            "task_name": task_names[(suite, task_id)],
        }
        for suite in QUALIFY.SUITES
        for task_id in range(10)
    ]


def _base_python_runtime_identity(venv_root: Path, *, version: str) -> dict[str, Any]:
    major, minor, *_rest = version.split(".")
    base = venv_root.parent / f"base-python-{major}.{minor}"
    unsigned = {
        "base_prefix": str(base),
        "configured_home": str(base / "bin"),
        "configured_home_resolved": str(base / "bin"),
        "content_inventory_sha256": _sha(f"base-content:{version}"),
        "files_verified": 1,
        "pyvenv_cfg_bytes": 1,
        "pyvenv_cfg_sha256": _sha(f"pyvenv:{version}"),
        "resolved_executable": str(base / "bin" / f"python{major}.{minor}"),
        "resolved_executable_bytes": 1,
        "resolved_executable_sha256": _sha(f"python:{version}"),
        "schema": "duo-vla-base-python-runtime-identity-v1",
        "startup_hooks": [],
        "startup_hooks_sha256": _sha(f"base-hooks:{version}"),
        "symlinks_verified": 0,
        "total_bytes": 1,
        "tree_metadata_sha256": _sha(f"base-tree:{version}"),
        "venv_python": str(venv_root / "bin/python"),
        "venv_python_link_target": str(base / "bin" / f"python{major}.{minor}"),
        "venv_root": str(venv_root),
    }
    return {**unsigned, "root_sha256": QUALIFY.canonical_sha256(unsigned)}


def _venv_identity(root: Path, *, version: str, schema: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyvenv.cfg").write_text(f"version_info = {version}\n", encoding="utf-8")
    base = _base_python_runtime_identity(root.resolve(), version=version)
    values = {
        "base_python_runtime": base,
        "content_inventory_sha256": _sha(f"venv-content:{schema}"),
        "files_verified": 1,
        "root": str(root.resolve()),
        "schema": schema,
        "startup_hooks": [],
        "startup_hooks_sha256": _sha(f"venv-hooks:{schema}"),
        "symlinks_verified": 0,
        "total_bytes": (root / "pyvenv.cfg").stat().st_size,
        "tree_metadata_sha256": _sha(f"venv-tree:{schema}"),
    }
    root_payload = {
        "base_python_runtime_root_sha256": base["root_sha256"],
        "content_inventory_sha256": values["content_inventory_sha256"],
        "files_verified": values["files_verified"],
        "schema": schema,
        "startup_hooks_sha256": values["startup_hooks_sha256"],
        "symlinks_verified": values["symlinks_verified"],
        "total_bytes": values["total_bytes"],
        "tree_metadata_sha256": values["tree_metadata_sha256"],
    }
    return {**values, "root_sha256": QUALIFY.canonical_sha256(root_payload)}


def _attestation(tasks: list[dict[str, Any]], tmp_path: Path) -> dict[str, Any]:
    eval_venv = _venv_identity(
        tmp_path / "cache/venvs/libero-eval",
        version="3.12.13",
        schema="duo-vla-eval-venv-identity-v1",
    )
    base = eval_venv["base_python_runtime"]
    version = "python3.12"
    process = {
        "environment": {"PYTHONSAFEPATH": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        "python_base_exec_prefix": str(Path(base["venv_python_link_target"]).parent.parent),
        "python_base_prefix": base["base_prefix"],
        "python_executable": str(Path(eval_venv["root"]) / "bin/python"),
        "python_flags": {"dont_write_bytecode": True, "no_user_site": True, "safe_path": True},
        "python_invocation_flags": ["-P", "-B", "-X", "pycache_prefix=/dev/null"],
        "python_prefix": eval_venv["root"],
        "python_pycache_prefix": "/dev/null",
        "python_version": "3.12.13",
        "sys_path": [
            str(ROOT / "src"),
            str(Path(base["base_prefix"]) / "lib/python312.zip"),
            str(Path(base["base_prefix"]) / f"lib/{version}"),
            str(Path(base["base_prefix"]) / f"lib/{version}/lib-dynload"),
            str(Path(eval_venv["root"]) / f"lib/{version}/site-packages"),
        ],
    }
    project_sources = {
        name: _sha(f"project-source:{name}")
        for name in (
            "bridge",
            "evaluator",
            "launcher",
            "preflight",
            "preflight_launcher",
            "replay_binder_launcher",
            "replay_collector_launcher",
            "replay_contract",
            "replay_qualification",
            "replay_qualification_launcher",
        )
    }
    return {
        "assets": {"path": str(tmp_path / "assets"), "root_sha256": _sha("assets")},
        "backend": "mujoco.egl",
        "distribution_records": {"identity": _sha("records")},
        "egl_device": "0",
        "environment_constructed": True,
        "eval_venv_identity": eval_venv,
        "evaluator_lock_sha256": _sha("lock"),
        "installed_distributions": {"count": 1, "inventory_sha256": _sha("distributions"), "packages": []},
        "manifest_sha256": _sha("manifest"),
        "module_origins": {"inventory_sha256": _sha("modules")},
        "opengl": {"identity_sha256": _sha("opengl")},
        "packages": {"hf-libero": "0.1.4"},
        "process": process,
        "project_sources": project_sources,
        "schema": QUALIFY.SIMULATOR_ATTESTATION_SCHEMA,
        "site_packages": {
            "declared_files": 1,
            "startup_files": {},
            "symlinks": {},
            "unregistered_files": 0,
        },
        "source": {"revision": "source-revision"},
        "status": "ok",
        "task_inventory": tasks,
        "task_inventory_count": 40,
        "task_inventory_sha256": QUALIFY.canonical_sha256(tasks),
        "torch": "cpu-test",
    }


def _normalization() -> dict[str, Any]:
    unsigned = {
        "action": {"observed_gripper_values": [-1.0, 1.0]},
        "counts": {"tasks": 40, "total_episodes": 1693, "total_frames": 273465},
        "dataset": {"id": "HuggingFaceVLA/libero", "revision": QUALIFY.DATASET_REVISION},
        "schema": "duo-vla-libero-normalization-v1",
    }
    return {**unsigned, "content_sha256": QUALIFY.canonical_sha256(unsigned)}


def _train_venv_identity(root: Path) -> dict[str, Any]:
    return _venv_identity(root, version="3.11.15", schema="duo-vla-train-venv-identity-v2")


def _validator_runtime(attestation: dict[str, Any]) -> dict[str, Any]:
    value = {
        name: copy.deepcopy(attestation[name])
        for name in QUALIFY._VALIDATOR_RUNTIME_FIELDS - {"schema", "simulator_runtime_sha256"}
    }
    value.update(
        {
            "schema": QUALIFY.VALIDATOR_RUNTIME_SCHEMA,
            "simulator_runtime_sha256": QUALIFY.simulator_runtime_sha256(attestation),
        }
    )
    return value


def _inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Path]]:
    tasks = _task_inventory()
    task_hash = QUALIFY.canonical_sha256(tasks)
    monkeypatch.setattr(QUALIFY, "TASK_INVENTORY_SHA256", task_hash)
    attestation = _attestation(tasks, tmp_path)
    dataset_tree = {"files": {"data.parquet": {"blob_id": "a" * 40, "size": 7}}, "format_version": 1}
    dataset_tree_path = tmp_path / "dataset-tree.json"
    dataset_tree_path.write_text(json.dumps(dataset_tree, sort_keys=True), encoding="utf-8")
    dataset_tree_raw = hashlib.sha256(dataset_tree_path.read_bytes()).hexdigest()
    monkeypatch.setattr(QUALIFY, "DATASET_TREE_METADATA_SHA256", dataset_tree_raw)
    monkeypatch.setattr(QUALIFY, "DATASET_TREE_FILE_COUNT", 1)
    monkeypatch.setattr(QUALIFY, "DATASET_TREE_TOTAL_BYTES", 7)
    monkeypatch.setattr(
        QUALIFY,
        "DATASET_CONTENT_INVENTORY_SHA256",
        QUALIFY.dataset_content_inventory_sha256(dataset_tree),
    )
    monkeypatch.setattr(QUALIFY, "DATASET_SNAPSHOT_FILES_VERIFIED", 1)
    monkeypatch.setattr(QUALIFY, "DATASET_SNAPSHOT_TOTAL_BYTES", 7)
    snapshot_root = tmp_path / QUALIFY.DATASET_REVISION
    snapshot_root.mkdir()
    dataset_snapshot = {
        "content_inventory_sha256": QUALIFY.DATASET_CONTENT_INVENTORY_SHA256,
        "files_verified": 1,
        "revision": QUALIFY.DATASET_REVISION,
        "snapshot": str(snapshot_root.resolve()),
        "total_bytes": 7,
        "tree_metadata_sha256": dataset_tree_raw,
    }
    train_venv = _train_venv_identity(tmp_path / "cache/venvs/train")
    monkeypatch.setenv("DUO_VLA_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(QUALIFY, "verify_huggingface_snapshot", lambda *_args, **_kwargs: dataset_snapshot)
    monkeypatch.setattr(QUALIFY, "content_address_train_venv", lambda _root: train_venv)
    validator_runtime = _validator_runtime(attestation)
    monkeypatch.setattr(
        QUALIFY,
        "validator_runtime_identity",
        lambda _root, _cache_root, _attestation: copy.deepcopy(validator_runtime),
    )
    normalization = _normalization()
    normalization_path = tmp_path / "normalization.json"
    normalization_path.write_text(json.dumps(normalization, sort_keys=True), encoding="utf-8")
    normalization_raw = hashlib.sha256(normalization_path.read_bytes()).hexdigest()
    monkeypatch.setattr(QUALIFY, "NORMALIZATION_CONTENT_SHA256", normalization["content_sha256"])
    monkeypatch.setattr(QUALIFY, "NORMALIZATION_RAW_SHA256", normalization_raw)
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(json.dumps(attestation, sort_keys=True), encoding="utf-8")
    attestation_raw = hashlib.sha256(attestation_path.read_bytes()).hexdigest()
    expected = QUALIFY.build_expected_inputs(
        ROOT,
        simulator_attestation=attestation,
        simulator_attestation_raw_sha256=attestation_raw,
        dataset_tree=dataset_tree,
        dataset_tree_raw_sha256=dataset_tree_raw,
        normalization=normalization,
        normalization_raw_sha256=normalization_raw,
        dataset_snapshot=dataset_snapshot,
        train_venv_identity=train_venv,
    )
    return (
        expected,
        tasks,
        attestation,
        {
            "attestation": attestation_path,
            "dataset_tree": dataset_tree_path,
            "normalization": normalization_path,
            "snapshot": snapshot_root,
        },
    )


def _evidence(expected: dict[str, Any], tasks: list[dict[str, Any]], raw_sha256: str) -> dict[str, Any]:
    results = QUALIFY.canonical_gate_results()
    gates = [
        {
            "evidence_ids": ["all-evidence"],
            "name": name,
            "passed": True,
            "result": results[name],
            "result_sha256": QUALIFY.canonical_sha256(results[name]),
        }
        for name in QUALIFY.GATE_NAMES
    ]
    demonstrations = []
    for task in tasks:
        label = f"{task['suite']}:{task['task_id']}"
        demonstrations.append(
            {
                "action_sequence_sha256": _sha(f"actions:{label}"),
                "demonstration_id": label,
                "initial_state_sha256": _sha(f"initial:{label}"),
                "instruction": task["instruction"],
                "observation_sequence_sha256": _sha(f"observations:{label}"),
                "raw_evidence_ids": ["all-evidence"],
                "regenerated": True,
                "source_episode_index": task["task_id"],
                "step_count": 10,
                "success": True,
                "suite": task["suite"],
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "trajectory_sha256": _sha(f"trajectory:{label}"),
            }
        )
    demonstrations.sort(key=lambda item: item["demonstration_id"])
    return {
        "demonstrations": demonstrations,
        "gates": gates,
        "inputs": expected,
        "raw_evidence": [{"bytes": 12, "id": "all-evidence", "path": "raw/all.bin", "sha256": raw_sha256}],
        "schema": QUALIFY.EVIDENCE_SCHEMA,
    }


def _valid_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Path]]:
    expected, tasks, attestation, paths = _inputs(monkeypatch, tmp_path)
    evidence_root = tmp_path / "bundle"
    evidence_root.mkdir()
    raw_dir = evidence_root / "raw"
    raw_dir.mkdir()
    binding_dir = evidence_root / "binding"
    binding_dir.mkdir()
    inventory = json.loads((ROOT / "configs/libero_original_hdf5_inventory.json").read_text(encoding="utf-8"))
    inventory_by_task = {(item["suite"], item["task_id"]): item for item in inventory["files"]}

    def write_json(relative: str, value: dict[str, Any]) -> dict[str, Any]:
        path = evidence_root / relative
        path.write_text(json.dumps(value, allow_nan=False, sort_keys=True), encoding="utf-8")
        raw = path.read_bytes()
        return {"bytes": len(raw), "path": relative, "sha256": hashlib.sha256(raw).hexdigest()}

    simulator_records: list[dict[str, Any]] = []
    binding_records: list[dict[str, Any]] = []
    demonstrations: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    stage_source_scan: list[dict[str, Any]] = []
    for dataset_task_index, task in enumerate(tasks):
        suite = task["suite"]
        task_id = task["task_id"]
        slug = QUALIFY.task_slug(suite, task_id)
        action_sha256 = _sha(f"actions:{slug}")
        initial_sha256 = _sha(f"initial:{slug}")
        observation_sha256 = _sha(f"observations:{slug}")
        trajectory_sha256 = QUALIFY.canonical_sha256(
            {
                "action_sequence_sha256": action_sha256,
                "initial_state_sha256": initial_sha256,
                "observation_sequence_sha256": observation_sha256,
                "source_episode_index": 0,
                "suite": suite,
                "task_id": task_id,
            }
        )
        selected = {
            "action_sequence_sha256": action_sha256,
            "alignment_probe": {
                "post_action_observation_sha256": _sha(f"post:{slug}"),
                "pre_action_observation_sha256": _sha(f"pre:{slug}"),
                "retained_transition_index": 0,
                "source_transition_index": 0,
            },
            "initial_state_sha256": initial_sha256,
            "inverted_gripper_action_sequence_sha256": _sha(f"inverted:{slug}"),
            "observation_sequence_sha256": observation_sha256,
            "source_episode_index": 0,
            "step_count": 10,
            "success": True,
            "swapped_observation_sequence_sha256": _sha(f"swapped:{slug}"),
            "trajectory_sha256": trajectory_sha256,
            "zero_action_sequence_sha256": _sha(f"zero:{slug}"),
        }
        attempt = {
            "action_sequence_sha256": action_sha256,
            "source_episode_index": 0,
            "step_count": 10,
            "success": True,
        }
        controls = {name: {"mutation_detected": True} for name in QUALIFY.PRE_DISPATCH_MUTATIONS}
        source_scan = {
            "demonstrations": [
                {
                    "action_sequence_sha256": action_sha256,
                    "initial_state_sha256": initial_sha256,
                    "raw_transition_count": 10,
                    "retained_transition_count": 10,
                    "source_action_dtype": "<f8",
                    "source_episode_index": 0,
                    "source_state_sequence_sha256": _sha(f"source-states:{slug}"),
                }
            ],
            "gripper_counts": {"-1": 5, "1": 5},
            "raw_transition_count": 10,
            "retained_transition_count": 10,
        }
        simulator_task = {
            "attempts": [attempt],
            "collector_source_sha256": expected["source_files_sha256"]["expert_replay_collector"],
            "pre_dispatch_integrity_controls": controls,
            "reset_determinism": {
                "environment_seed": QUALIFY.ENVIRONMENT_SEED,
                "first_observation_sha256": _sha(f"reset-observation:{slug}"),
                "first_simulator_state_sha256": _sha(f"reset-state:{slug}"),
                "probe_environment_count": 2,
                "seed_calls_per_environment": 1,
                "second_observation_sha256": _sha(f"reset-observation:{slug}"),
                "second_simulator_state_sha256": _sha(f"reset-state:{slug}"),
                "settle_steps": 10,
            },
            "schema": QUALIFY.SIMULATOR_TASK_SCHEMA,
            "selected": selected,
            "source_file": inventory_by_task[(suite, task_id)],
            "source_scan": source_scan,
            "task": {
                "instruction": task["instruction"],
                "suite": suite,
                "task_id": task_id,
                "task_name": task["task_name"],
            },
        }
        simulator_file = write_json(f"raw/{slug}.json", simulator_task)
        simulator_records.append({**simulator_file, "suite": suite, "task_id": task_id})
        simulator_id = f"simulator-{slug}"
        raw_records.append({**simulator_file, "id": simulator_id})
        stage_source_scan.append(
            {
                "gripper_counts": source_scan["gripper_counts"],
                "path": inventory_by_task[(suite, task_id)]["path"],
                "raw_transition_count": 10,
                "retained_transition_count": 10,
                "suite": suite,
                "task_id": task_id,
            }
        )

        parquet_task = {
            "action_sequence_sha256": action_sha256,
            "dataset_episode_index": dataset_task_index,
            "dataset_global_start": dataset_task_index * 10,
            "dataset_global_stop": dataset_task_index * 10 + 10,
            "dataset_task_index": dataset_task_index,
            "instruction": task["instruction"],
            "observation_sequence_sha256": observation_sha256,
            "schema": QUALIFY.PARQUET_TASK_SCHEMA,
            "source_episode_index": 0,
            "step_count": 10,
            "suite": suite,
            "task_id": task_id,
            "task_name": task["task_name"],
            "trajectory_sha256": trajectory_sha256,
        }
        parquet_file = write_json(f"binding/{slug}.json", parquet_task)
        binding_records.append({**parquet_file, "suite": suite, "task_id": task_id})
        parquet_id = f"parquet-{slug}"
        raw_records.append({**parquet_file, "id": parquet_id})
        demonstrations.append(
            {
                "action_sequence_sha256": action_sha256,
                "demonstration_id": f"{slug}-source-0000",
                "initial_state_sha256": initial_sha256,
                "instruction": task["instruction"],
                "observation_sequence_sha256": observation_sha256,
                "raw_evidence_ids": sorted([parquet_id, simulator_id]),
                "regenerated": True,
                "source_episode_index": 0,
                "step_count": 10,
                "success": True,
                "suite": suite,
                "task_id": task_id,
                "task_name": task["task_name"],
                "trajectory_sha256": trajectory_sha256,
            }
        )

    simulator_stage = {
        "collector": {
            "path": "scripts/collect_libero_expert_replay.py",
            "sha256": expected["source_files_sha256"]["expert_replay_collector"],
            "shared_contract_sha256": expected["source_files_sha256"]["expert_replay_contract"],
        },
        "inputs": {
            "original_hdf5_inventory_content_sha256": expected["original_hdf5_inventory_content_sha256"],
            "original_hdf5_inventory_raw_sha256": expected["original_hdf5_inventory_raw_sha256"],
            "simulator_attestation_raw_sha256": expected["simulator_attestation_raw_sha256"],
            "simulator_attestation_sha256": expected["simulator_attestation_sha256"],
            "simulator_runtime_sha256": expected["simulator_runtime_sha256"],
            "task_inventory_sha256": expected["task_inventory_sha256"],
        },
        "schema": QUALIFY.SIMULATOR_STAGE_SCHEMA,
        "gates": {
            "controller_impulse_directions": {
                **QUALIFY.canonical_gate_results()["controller_impulse_directions"],
                "raw": {
                    "gripper_apertures": {"close": 0.0, "open": 1.0},
                    "responses": [
                        {
                            "direction": direction,
                            "leakage": 0.0,
                            "primary_delta": 1.0 if direction.startswith("+") else -1.0,
                        }
                        for direction in QUALIFY.IMPULSE_DIRECTIONS
                    ],
                },
            },
            "exact_gripper_set": QUALIFY.canonical_gate_results()["exact_gripper_set"],
        },
        "source_scan": stage_source_scan,
        "status": "complete",
        "task_records": simulator_records,
    }
    simulator_stage_file = write_json("simulator-stage.json", simulator_stage)
    raw_records.append({**simulator_stage_file, "id": "simulator-stage"})
    simulator_commit = {
        "schema": "duo-vla-libero-expert-replay-simulator-stage-commit-v1",
        "simulator_stage_sha256": simulator_stage_file["sha256"],
        "task_record_root_sha256": QUALIFY.canonical_sha256(simulator_records),
    }
    simulator_commit_file = write_json("simulator-stage.commit.json", simulator_commit)
    raw_records.append({**simulator_commit_file, "id": "simulator-stage-commit"})
    parquet_stage = {
        "binder": {
            "path": "scripts/bind_libero_expert_replay.py",
            "sha256": expected["source_files_sha256"]["expert_replay_binder"],
            "shared_contract_sha256": expected["source_files_sha256"]["expert_replay_contract"],
        },
        "gates": {
            "episode_boundary_chunk_fixture": QUALIFY.canonical_gate_results()["episode_boundary_chunk_fixture"],
            "normalization_round_trip": QUALIFY.canonical_gate_results()["normalization_round_trip"],
            "training_gripper_set": {
                **QUALIFY.canonical_gate_results()["exact_gripper_set"],
                "validated_actions": 273465,
                "validated_episodes": 1693,
            },
        },
        "inputs": {
            "dataset_content_inventory_sha256": expected["dataset_content_inventory_sha256"],
            "dataset_snapshot_files_verified": expected["dataset_snapshot_files_verified"],
            "dataset_snapshot_total_bytes": expected["dataset_snapshot_total_bytes"],
            "dataset_tree_sha256": expected["dataset_tree_metadata_sha256"],
            "original_hdf5_inventory_content_sha256": expected["original_hdf5_inventory_content_sha256"],
            "original_hdf5_inventory_raw_sha256": expected["original_hdf5_inventory_raw_sha256"],
            "simulator_stage_commit_sha256": simulator_commit_file["sha256"],
            "simulator_stage_sha256": simulator_stage_file["sha256"],
            "train_venv_root_sha256": expected["train_venv_identity"]["root_sha256"],
        },
        "process": QUALIFY._expected_binder_process(
            ROOT,
            Path(expected["train_venv_identity"]["root"]),
            expected["train_venv_identity"],
        ),
        "schema": QUALIFY.PARQUET_BINDING_SCHEMA,
        "status": "complete",
        "task_records": binding_records,
        "train_venv": expected["train_venv_identity"],
    }
    parquet_stage_file = write_json("parquet-binding.json", parquet_stage)
    raw_records.append({**parquet_stage_file, "id": "parquet-binding"})
    raw_records.sort(key=lambda item: item["id"])

    results = QUALIFY.canonical_gate_results()
    gate_evidence = QUALIFY._canonical_gate_evidence_ids()
    evidence = {
        "demonstrations": sorted(demonstrations, key=lambda item: item["demonstration_id"]),
        "gates": [
            {
                "evidence_ids": gate_evidence[name],
                "name": name,
                "passed": True,
                "result": results[name],
                "result_sha256": QUALIFY.canonical_sha256(results[name]),
            }
            for name in QUALIFY.GATE_NAMES
        ],
        "inputs": expected,
        "raw_evidence": raw_records,
        "schema": QUALIFY.EVIDENCE_SCHEMA,
    }
    evidence_path = evidence_root / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, allow_nan=False, sort_keys=True), encoding="utf-8")
    evidence_digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    (evidence_root / "evidence.json.sha256").write_text(
        f"{evidence_digest}  evidence.json\n",
        encoding="ascii",
    )
    paths["evidence"] = evidence_path
    return evidence, attestation, tasks, paths


def _run_args(paths: dict[str, Path], output: Path) -> argparse.Namespace:
    original_hdf5_inventory = ROOT / "configs/libero_original_hdf5_inventory.json"
    return argparse.Namespace(
        dataset_tree_metadata=paths["dataset_tree"],
        dataset_tree_metadata_sha256=hashlib.sha256(paths["dataset_tree"].read_bytes()).hexdigest(),
        evidence_manifest=paths["evidence"],
        evidence_manifest_sha256=hashlib.sha256(paths["evidence"].read_bytes()).hexdigest(),
        normalization_artifact=paths["normalization"],
        normalization_artifact_sha256=hashlib.sha256(paths["normalization"].read_bytes()).hexdigest(),
        original_hdf5_inventory=original_hdf5_inventory,
        original_hdf5_inventory_sha256=hashlib.sha256(original_hdf5_inventory.read_bytes()).hexdigest(),
        output_dir=output,
        snapshot_root=paths["snapshot"],
        simulator_attestation=paths["attestation"],
        simulator_attestation_sha256=hashlib.sha256(paths["attestation"].read_bytes()).hexdigest(),
    )


def test_qualification_publishes_complete_40_of_40_report_exclusively(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _evidence_value, attestation, _tasks, paths = _valid_bundle(monkeypatch, tmp_path)
    output = tmp_path / "qualification"

    report_path, report_sha256 = QUALIFY.run_qualification(_run_args(paths, output), project_root=ROOT)

    assert report_path == output / "qualification.json"
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == report_sha256
    report, observed = QUALIFY.load_qualification_report(
        report_path,
        expected_raw_sha256=report_sha256,
        project_root=ROOT,
        simulator_attestation=attestation,
        simulator_attestation_raw_sha256=hashlib.sha256(paths["attestation"].read_bytes()).hexdigest(),
    )
    assert observed == report_sha256
    assert report["replay_summary"] == {
        "demonstration_count": 40,
        "successful_demonstration_count": 40,
        "successful_task_count": 40,
        "task_count": 40,
    }
    assert report["status"] == "passed"
    with pytest.raises(FileExistsError, match="already exists"):
        QUALIFY.run_qualification(_run_args(paths, output), project_root=ROOT)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda value: value["demonstrations"].pop(), "exactly 40"),
        (lambda value: value["demonstrations"].append(copy.deepcopy(value["demonstrations"][0])), "exactly 40"),
        (lambda value: value["demonstrations"][0].update(success=False), "did not succeed"),
        (lambda value: value["demonstrations"][0].update(regenerated=False), "not regenerated"),
        (
            lambda value: value["gates"][5]["result"].update(observation_index="t+1"),
            "result self-hash mismatch",
        ),
        (
            lambda value: value["gates"][6]["result"].update(rollout_transform="rotate_180_twice"),
            "result self-hash mismatch",
        ),
        (
            lambda value: value["gates"][2]["result"].update(direction_match_count=11),
            "result self-hash mismatch",
        ),
        (
            lambda value: value["gates"][7]["result"].update(official_fixed_states_used=True),
            "result self-hash mismatch",
        ),
        (lambda value: value["inputs"].update(dataset_revision="mutable-main"), "inputs differ"),
    ),
)
def test_evidence_rejects_missing_task_failure_or_mutated_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    evidence, _attestation_value, tasks, _paths = _valid_bundle(monkeypatch, tmp_path)
    changed = copy.deepcopy(evidence)
    mutate(changed)

    with pytest.raises(QUALIFY.QualificationError, match=message):
        QUALIFY.validate_evidence_document(
            changed,
            expected_inputs=evidence["inputs"],
            task_inventory=tasks,
            raw_evidence_root=None,
        )


def test_gate_semantics_fail_even_when_mutation_has_a_matching_self_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence, _attestation_value, tasks, _paths = _valid_bundle(monkeypatch, tmp_path)
    changed = copy.deepcopy(evidence)
    gate = changed["gates"][5]
    gate["result"]["observation_index"] = "t+1"
    gate["result_sha256"] = QUALIFY.canonical_sha256(gate["result"])

    with pytest.raises(QUALIFY.QualificationError, match="pre-action observation alignment"):
        QUALIFY.validate_evidence_document(
            changed,
            expected_inputs=evidence["inputs"],
            task_inventory=tasks,
            raw_evidence_root=None,
        )


def test_attestation_task_inventory_self_hash_is_recomputed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _expected, _tasks, attestation, _paths = _inputs(monkeypatch, tmp_path)
    changed = copy.deepcopy(attestation)
    changed["task_inventory"][0]["instruction"] += " mutated"

    with pytest.raises(QUALIFY.QualificationError, match="content does not match"):
        QUALIFY._task_identities(changed)


def test_raw_evidence_hash_escape_and_symlink_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence, _attestation_value, tasks, paths = _valid_bundle(monkeypatch, tmp_path)
    changed_hash = copy.deepcopy(evidence)
    changed_hash["raw_evidence"][0]["sha256"] = "0" * 64
    with pytest.raises(QUALIFY.QualificationError, match="raw evidence hash mismatch"):
        QUALIFY.validate_evidence_document(
            changed_hash,
            expected_inputs=evidence["inputs"],
            task_inventory=tasks,
            raw_evidence_root=paths["evidence"].parent,
        )

    changed_escape = copy.deepcopy(evidence)
    changed_escape["raw_evidence"][0]["path"] = "../outside.bin"
    with pytest.raises(QUALIFY.QualificationError, match="83 canonical IDs and paths"):
        QUALIFY.validate_evidence_document(
            changed_escape,
            expected_inputs=evidence["inputs"],
            task_inventory=tasks,
            raw_evidence_root=paths["evidence"].parent,
        )

    target = tmp_path / "target.bin"
    target.write_bytes(b"raw evidence")
    changed_link = copy.deepcopy(evidence)
    linked_record = changed_link["raw_evidence"][0]
    link = paths["evidence"].parent / linked_record["path"]
    link.unlink()
    link.symlink_to(target)
    with pytest.raises(QUALIFY.QualificationError, match="without following links"):
        QUALIFY.validate_evidence_document(
            changed_link,
            expected_inputs=evidence["inputs"],
            task_inventory=tasks,
            raw_evidence_root=paths["evidence"].parent,
        )


def test_strict_json_and_report_self_hash_reject_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _evidence_value, _attestation_value, _tasks, paths = _valid_bundle(monkeypatch, tmp_path)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    with pytest.raises(QUALIFY.QualificationError, match="strict finite"):
        QUALIFY.read_stable_json(duplicate, name="duplicate")
    overflow = tmp_path / "overflow.json"
    overflow.write_text('{"value":1e999}', encoding="utf-8")
    with pytest.raises(QUALIFY.QualificationError, match="non-finite"):
        QUALIFY.read_stable_json(overflow, name="overflow")

    report_path, report_sha256 = QUALIFY.run_qualification(
        _run_args(paths, tmp_path / "qualification"),
        project_root=ROOT,
    )
    report = json.loads(report_path.read_text(encoding="ascii"))
    report["replay_summary"]["task_count"] = 39
    with pytest.raises(QUALIFY.QualificationError, match="self-hash"):
        QUALIFY.validate_qualification_report_document(report)
    with pytest.raises(QUALIFY.QualificationError, match="externally recorded"):
        QUALIFY.load_qualification_report(report_path, expected_raw_sha256="0" * 64)
    assert report_sha256 != "0" * 64


def test_durable_report_rejects_self_consistent_one_record_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence, _attestation, tasks, _paths = _valid_bundle(monkeypatch, tmp_path)
    changed = copy.deepcopy(evidence)
    changed["raw_evidence"] = [next(record for record in changed["raw_evidence"] if record["id"] == "parquet-binding")]
    for gate in changed["gates"]:
        gate["evidence_ids"] = ["parquet-binding"]
    for demo in changed["demonstrations"]:
        demo["raw_evidence_ids"] = ["parquet-binding"]
    with pytest.raises(QUALIFY.QualificationError, match="83 canonical IDs and paths"):
        QUALIFY.validate_evidence_document(
            changed,
            expected_inputs=evidence["inputs"],
            task_inventory=tasks,
            raw_evidence_root=None,
        )


def test_validator_runtime_mutation_prevents_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _evidence, attestation, _tasks, paths = _valid_bundle(monkeypatch, tmp_path)
    expected_runtime = _validator_runtime(attestation)
    calls = 0

    def changing_runtime(_root: Path, _cache_root: Path, _attestation: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        value = copy.deepcopy(expected_runtime)
        if calls == 2:
            value["packages"]["hf-libero"] = "mutated"
        return value

    monkeypatch.setattr(QUALIFY, "validator_runtime_identity", changing_runtime)
    output = tmp_path / "runtime-mutation-output"
    with pytest.raises(QUALIFY.QualificationError, match="validator runtime changed"):
        QUALIFY.run_qualification(_run_args(paths, output), project_root=ROOT)
    assert calls == 2
    assert not output.exists()


def test_report_rejects_validator_startup_flag_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _evidence, _attestation, _tasks, paths = _valid_bundle(monkeypatch, tmp_path)
    report_path, _digest = QUALIFY.run_qualification(
        _run_args(paths, tmp_path / "runtime-report"),
        project_root=ROOT,
    )
    report = json.loads(report_path.read_text(encoding="ascii"))
    report["validator_runtime_identity"]["process"]["python_flags"]["safe_path"] = False
    unsigned = {name: value for name, value in report.items() if name != "content_sha256"}
    report["content_sha256"] = QUALIFY.canonical_sha256(unsigned)
    with pytest.raises(QUALIFY.QualificationError, match="startup flags differ"):
        QUALIFY.validate_qualification_report_document(report)


def test_qualification_launcher_is_closed() -> None:
    launcher = (ROOT / "scripts/run_qualify_libero_expert_replay.sh").read_text(encoding="utf-8")
    assert launcher.startswith("#!/bin/bash -p\nset -euo pipefail\n")
    assert 'export PATH="/usr/bin:/bin"' in launcher
    assert "exec /usr/bin/env -i" in launcher
    assert "PYTHONPATH" not in launcher
    assert '"PYTHONSAFEPATH=1"' in launcher
    assert '"PYTHONDONTWRITEBYTECODE=1"' in launcher
    assert " -P -B -X pycache_prefix=/dev/null " in launcher
    assert os.access(ROOT / "scripts/run_qualify_libero_expert_replay.sh", os.X_OK)
