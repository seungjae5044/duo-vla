"""Fail-closed tests for the sealed official LIBERO evaluation contract."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import create_libero_preregistration as CREATOR  # noqa: E402
import evaluate_libero as EVALUATOR  # noqa: E402
import preflight_libero_env as PREFLIGHT  # noqa: E402
import qualify_libero_expert_replay as QUALIFY  # noqa: E402

from duo_vla.run_config import load_resolved_toml, save_resolved_config  # noqa: E402
from duo_vla.run_journal import create_run_journal, record_latest_checkpoint  # noqa: E402


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_evaluator_prioritizes_checkout_and_rejects_preloaded_duo_shadow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = str((ROOT / "src").resolve())
    shadow_site = tmp_path / "site-packages"
    shadow_site.mkdir()
    monkeypatch.setattr(EVALUATOR.sys, "path", [str(shadow_site), source_root, source_root])
    assert EVALUATOR._activate_project_source_root() == ROOT / "src"
    assert EVALUATOR.sys.path == [source_root, str(shadow_site)]

    shadow_file = shadow_site / "duo_vla/compatible.py"
    shadow_file.parent.mkdir()
    shadow_file.write_text("VALUE = 'compatible'\n", encoding="utf-8")
    shadow_module = SimpleNamespace(
        __file__=str(shadow_file),
        __spec__=SimpleNamespace(origin=str(shadow_file)),
    )
    monkeypatch.setitem(EVALUATOR.sys.modules, "duo_vla.compatible_shadow", shadow_module)
    with pytest.raises(RuntimeError, match="escapes authenticated source root"):
        EVALUATOR._validate_project_module_origins(EVALUATOR._REQUIRED_PROJECT_MODULES)


def _base_python_runtime_identity(venv_root: str) -> dict[str, Any]:
    identity = {
        "base_prefix": "/usr/local",
        "configured_home": "/usr/local/bin",
        "configured_home_resolved": "/usr/local/bin",
        "content_inventory_sha256": _digest("base-python-content"),
        "files_verified": 100,
        "pyvenv_cfg_bytes": 128,
        "pyvenv_cfg_sha256": _digest("pyvenv-cfg"),
        "resolved_executable": "/usr/local/bin/python3.11",
        "resolved_executable_bytes": 20_000,
        "resolved_executable_sha256": _digest("base-python-executable"),
        "schema": EVALUATOR.BASE_PYTHON_RUNTIME_IDENTITY_SCHEMA,
        "startup_hooks": [],
        "startup_hooks_sha256": EVALUATOR.canonical_sha256([]),
        "symlinks_verified": 4,
        "total_bytes": 1_000_000,
        "tree_metadata_sha256": _digest("base-python-tree"),
        "venv_python": f"{venv_root}/bin/python",
        "venv_python_link_target": "/usr/local/bin/python3.11",
        "venv_root": venv_root,
    }
    identity["root_sha256"] = EVALUATOR.canonical_sha256(identity)
    return identity


def _venv_identity(root: str, schema: str, label: str) -> dict[str, Any]:
    base_python_runtime = _base_python_runtime_identity(root)
    identity = {
        "base_python_runtime": base_python_runtime,
        "content_inventory_sha256": _digest(f"{label}-content"),
        "files_verified": 10,
        "root": root,
        "schema": schema,
        "startup_hooks": [],
        "startup_hooks_sha256": EVALUATOR.canonical_sha256([]),
        "symlinks_verified": 2,
        "total_bytes": 1_000,
        "tree_metadata_sha256": _digest(f"{label}-tree"),
    }
    identity["root_sha256"] = EVALUATOR.canonical_sha256(
        {
            "base_python_runtime_root_sha256": base_python_runtime["root_sha256"],
            "content_inventory_sha256": identity["content_inventory_sha256"],
            "files_verified": identity["files_verified"],
            "schema": identity["schema"],
            "startup_hooks_sha256": identity["startup_hooks_sha256"],
            "symlinks_verified": identity["symlinks_verified"],
            "total_bytes": identity["total_bytes"],
            "tree_metadata_sha256": identity["tree_metadata_sha256"],
        }
    )
    return identity


def _rebind_venv_root(identity: dict[str, Any]) -> None:
    identity["root_sha256"] = EVALUATOR.canonical_sha256(
        {
            "base_python_runtime_root_sha256": identity["base_python_runtime"]["root_sha256"],
            "content_inventory_sha256": identity["content_inventory_sha256"],
            "files_verified": identity["files_verified"],
            "schema": identity["schema"],
            "startup_hooks_sha256": identity["startup_hooks_sha256"],
            "symlinks_verified": identity["symlinks_verified"],
            "total_bytes": identity["total_bytes"],
            "tree_metadata_sha256": identity["tree_metadata_sha256"],
        }
    )


def _train_venv_identity() -> dict[str, Any]:
    return _venv_identity(
        "/root/.cache/duo-vla/venvs/train",
        EVALUATOR.TRAIN_VENV_IDENTITY_SCHEMA,
        "train-venv",
    )


def _eval_venv_identity() -> dict[str, Any]:
    return _venv_identity(
        "/root/.cache/duo-vla/venvs/libero-eval",
        EVALUATOR.EVAL_VENV_IDENTITY_SCHEMA,
        "eval-venv",
    )


def _validator_runtime_identity(*, simulator_runtime_sha256: str) -> dict[str, Any]:
    return {
        "distribution_records": {"identity": _digest("records")},
        "eval_venv_identity": _eval_venv_identity(),
        "installed_distributions": {
            "count": 1,
            "inventory_sha256": _digest("distributions"),
            "packages": [{"name": "hf-libero", "version": "0.1.4"}],
        },
        "module_origins": {"identity": _digest("modules")},
        "packages": {"hf-libero": "0.1.4"},
        "process": {
            "environment": {
                "PYTHONSAFEPATH": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "python_base_exec_prefix": "/usr/local",
            "python_base_prefix": "/usr/local",
            "python_executable": "/root/.cache/duo-vla/venvs/libero-eval/bin/python",
            "python_flags": {"dont_write_bytecode": True, "no_user_site": True, "safe_path": True},
            "python_invocation_flags": ["-P", "-B", "-X", "pycache_prefix=/dev/null"],
            "python_prefix": "/root/.cache/duo-vla/venvs/libero-eval",
            "python_pycache_prefix": "/dev/null",
            "python_version": "3.12.13",
            "sys_path": [
                str(ROOT / "src"),
                "/usr/local/lib/python312.zip",
                "/usr/local/lib/python3.12",
                "/usr/local/lib/python3.12/lib-dynload",
                "/root/.cache/duo-vla/venvs/libero-eval/lib/python3.12/site-packages",
            ],
        },
        "project_sources": {
            "bridge": _digest("bridge"),
            "evaluator": _digest("evaluator"),
            "launcher": _digest("launcher"),
            "preflight": _digest("preflight"),
            "preflight_launcher": _digest("preflight-launcher"),
            "replay_binder_launcher": _digest("binder-launcher"),
            "replay_collector_launcher": _digest("collector-launcher"),
            "replay_contract": _digest("replay-contract"),
            "replay_qualification": _digest("replay-qualification"),
            "replay_qualification_launcher": _digest("qualification-launcher"),
        },
        "schema": EVALUATOR.VALIDATOR_RUNTIME_IDENTITY_SCHEMA,
        "simulator_runtime_sha256": simulator_runtime_sha256,
        "site_packages": {
            "declared_files": 1,
            "startup_files": {},
            "symlinks": {},
            "unregistered_files": 0,
        },
    }


def _validator_runtime_from_attestation(attestation: dict[str, Any]) -> dict[str, Any]:
    names = EVALUATOR._VALIDATOR_RUNTIME_IDENTITY_FIELDS - {"schema", "simulator_runtime_sha256"}
    return {
        **{name: copy.deepcopy(attestation[name]) for name in names},
        "schema": EVALUATOR.VALIDATOR_RUNTIME_IDENTITY_SCHEMA,
        "simulator_runtime_sha256": QUALIFY.simulator_runtime_sha256(attestation),
    }


def _cells(episode_sha256: str) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for seed in EVALUATOR.OFFICIAL_TRAIN_SEEDS:
        for objective, nfes in (
            ("rectified_flow", EVALUATOR.OFFICIAL_FLOW_NFES),
            ("direct_regression", (1,)),
        ):
            checkpoint = {
                "dataset_content_inventory_sha256": EVALUATOR.DATASET_CONTENT_INVENTORY_SHA256,
                "dataset_tree_sha256": EVALUATOR.DATASET_TREE_METADATA_SHA256,
                "manifest_sha256": _digest(f"checkpoint:{seed}:{objective}"),
                "source_tree_sha256": "f" * 64,
                "update": EVALUATOR.FINAL_CHECKPOINT_UPDATE,
            }
            policy_contract_sha256 = _digest(f"contract:{objective}")
            runtime_sha256 = _digest(f"runtime:{seed}:{objective}")
            for nfe in nfes:
                for execution_horizon in EVALUATOR.OFFICIAL_EXECUTION_HORIZONS:
                    selected_policy = {
                        "inference_seed_behavior": (
                            "episode_identity_gaussian_noise"
                            if objective == "rectified_flow"
                            else "episode_identity_echo_only"
                        ),
                        "nfe": nfe,
                        "objective": objective,
                        "sampler": "euler_uniform" if objective == "rectified_flow" else "single_forward",
                    }
                    cells.append(
                        {
                            "cell_id": EVALUATOR.official_cell_id(seed, objective, nfe, execution_horizon),
                            "checkpoint": checkpoint,
                            "episode_matrix_sha256": episode_sha256,
                            "execution_geometry": copy.deepcopy(EVALUATOR.LIBERO_EXECUTION_GEOMETRY),
                            "execution_horizon": execution_horizon,
                            **selected_policy,
                            "latency_runtime_sha256": _digest("latency-runtime"),
                            "policy_contract_sha256": policy_contract_sha256,
                            "policy_warmup_calls": EVALUATOR.OFFICIAL_POLICY_WARMUP_CALLS,
                            "serving_policy_sha256": EVALUATOR.canonical_sha256(selected_policy),
                            "serving_runtime_sha256": runtime_sha256,
                            "train_seed": seed,
                        }
                    )
    return cells


def _manifest(*, token: str = "sealed", attestation_sha256: str = "a" * 64) -> dict[str, Any]:
    contamination = EVALUATOR.load_contamination_contract(ROOT)
    episodes = EVALUATOR.official_episode_matrix(contamination)
    episode_sha256 = EVALUATOR.canonical_sha256(episodes)
    roots = {
        "claims": {"device": 1, "inode": 2, "path": "/sealed/libero-official/claims"},
        "runs": {"device": 1, "inode": 1, "path": "/sealed/libero-official/runs"},
        "schema": EVALUATOR.OUTPUT_ROOTS_SCHEMA,
    }
    freeze_sha256 = hashlib.sha256(token.encode()).hexdigest()
    simulator_runtime_sha256 = _digest("simulator-runtime")
    cells = _cells(episode_sha256)
    for cell in cells:
        cell["output_claim"] = EVALUATOR.derive_output_claim(cell["cell_id"], roots, freeze_sha256)
    return {
        "aggregation_python_version": EVALUATOR.AGGREGATION_PYTHON_VERSION,
        "aggregator_sha256": EVALUATOR.sha256_file(ROOT / "scripts/aggregate_libero_official.py"),
        "benchmark_protocol": EVALUATOR.PROTOCOL,
        "cells": cells,
        "contamination": contamination,
        "episode_count": EVALUATOR.OFFICIAL_PRIMARY_EPISODES,
        "episode_matrix_sha256": episode_sha256,
        "episodes": episodes,
        "evaluation_seed": 123,
        "expert_replay_qualification": {
            "config_file_sha256": {
                "direct_regression": _digest("config-direct"),
                "rectified_flow": _digest("config-flow"),
            },
            "content_sha256": _digest("qualification-content"),
            "dataset_content_inventory_sha256": EVALUATOR.DATASET_CONTENT_INVENTORY_SHA256,
            "dataset_snapshot_files_verified": EVALUATOR.DATASET_SNAPSHOT_FILES_VERIFIED,
            "dataset_snapshot_total_bytes": EVALUATOR.DATASET_SNAPSHOT_TOTAL_BYTES,
            "dataset_tree_metadata_sha256": EVALUATOR.DATASET_TREE_METADATA_SHA256,
            "demonstration_count": 40,
            "evidence_manifest_raw_sha256": _digest("evidence-manifest"),
            "gates_sha256": _digest("gates"),
            "kind": "libero-40-task-regenerated-expert-replay",
            "normalization_content_sha256": EVALUATOR.NORMALIZATION_SHA256,
            "normalization_raw_sha256": QUALIFY.NORMALIZATION_RAW_SHA256,
            "original_hdf5_file_count": QUALIFY.ORIGINAL_HDF5_FILE_COUNT,
            "original_hdf5_inventory_content_sha256": QUALIFY.ORIGINAL_HDF5_CONTENT_SHA256,
            "original_hdf5_inventory_raw_sha256": EVALUATOR.sha256_file(
                ROOT / "configs/libero_original_hdf5_inventory.json"
            ),
            "original_hdf5_repository_id": QUALIFY.ORIGINAL_HDF5_REPOSITORY_ID,
            "original_hdf5_revision": QUALIFY.ORIGINAL_HDF5_REVISION,
            "original_hdf5_total_bytes": QUALIFY.ORIGINAL_HDF5_TOTAL_BYTES,
            "project_source_tree_sha256": "f" * 64,
            "raw_evidence_root_sha256": _digest("raw-evidence"),
            "report_sha256": _digest("qualification-report"),
            "schema": "duo-vla-libero-expert-replay-qualification-v1",
            "simulator_attestation_raw_sha256": _digest("attestation-raw"),
            "simulator_attestation_sha256": attestation_sha256,
            "simulator_runtime_sha256": simulator_runtime_sha256,
            "status": "passed",
            "successful_demonstration_count": 40,
            "successful_task_count": 40,
            "task_count": 40,
            "task_inventory_sha256": QUALIFY.TASK_INVENTORY_SHA256,
            "train_venv_identity": _train_venv_identity(),
            "validator_runtime_identity": _validator_runtime_identity(
                simulator_runtime_sha256=simulator_runtime_sha256
            ),
        },
        "execution_horizons": list(EVALUATOR.OFFICIAL_EXECUTION_HORIZONS),
        "final_checkpoint_update": EVALUATOR.FINAL_CHECKPOINT_UPDATE,
        "final_freeze_token_sha256": freeze_sha256,
        "official_output_roots": roots,
        "official_resets_per_task": EVALUATOR.OFFICIAL_RESETS_PER_TASK,
        "policy_warmup_calls": EVALUATOR.OFFICIAL_POLICY_WARMUP_CALLS,
        "schema": EVALUATOR.PREREGISTRATION_SCHEMA,
        "simulator_attestation_sha256": attestation_sha256,
        "suites": list(EVALUATOR.SUITES),
        "task_ids": list(range(10)),
        "training_seeds": list(EVALUATOR.OFFICIAL_TRAIN_SEEDS),
    }


def _creator_attestation() -> dict[str, Any]:
    tasks = [
        {
            "instruction": f"instruction {suite} {task_id}",
            "reset_count": 50,
            "suite": suite,
            "task_id": task_id,
            "task_name": f"task_{suite}_{task_id}",
        }
        for suite in EVALUATOR.SUITES
        for task_id in range(10)
    ]
    runtime = _validator_runtime_identity(simulator_runtime_sha256="0" * 64)
    project_sources = runtime["project_sources"]
    project_sources.update(
        {
            "bridge": EVALUATOR.sha256_file(ROOT / "scripts/libero_bridge.py"),
            "evaluator": EVALUATOR.sha256_file(ROOT / "scripts/evaluate_libero.py"),
            "preflight": EVALUATOR.sha256_file(ROOT / "scripts/preflight_libero_env.py"),
        }
    )
    return {
        "assets": {"root_sha256": _digest("assets")},
        "backend": "mujoco.egl",
        "distribution_records": runtime["distribution_records"],
        "egl_device": "0",
        "environment_constructed": True,
        "eval_venv_identity": runtime["eval_venv_identity"],
        "evaluator_lock_sha256": _digest("lock"),
        "installed_distributions": runtime["installed_distributions"],
        "manifest_sha256": _digest("manifest"),
        "module_origins": runtime["module_origins"],
        "opengl": {"identity": _digest("opengl")},
        "packages": runtime["packages"],
        "process": runtime["process"],
        "project_sources": project_sources,
        "schema": QUALIFY.SIMULATOR_ATTESTATION_SCHEMA,
        "site_packages": runtime["site_packages"],
        "source": {"revision": "pinned"},
        "status": "ok",
        "task_inventory": tasks,
        "task_inventory_count": 40,
        "task_inventory_sha256": QUALIFY.TASK_INVENTORY_SHA256,
        "torch": "cpu-test",
    }


def _write_creator_qualification(
    tmp_path: Path,
    *,
    attestation: dict[str, Any],
    attestation_raw_sha256: str,
) -> tuple[Path, str, str]:
    source = QUALIFY.source_identity(ROOT)
    inputs = {
        "config_file_sha256": source["config_file_sha256"],
        "dataset_content_inventory_sha256": QUALIFY.DATASET_CONTENT_INVENTORY_SHA256,
        "dataset_revision": QUALIFY.DATASET_REVISION,
        "dataset_snapshot_files_verified": QUALIFY.DATASET_SNAPSHOT_FILES_VERIFIED,
        "dataset_snapshot_total_bytes": QUALIFY.DATASET_SNAPSHOT_TOTAL_BYTES,
        "dataset_tree_file_count": QUALIFY.DATASET_TREE_FILE_COUNT,
        "dataset_tree_metadata_sha256": QUALIFY.DATASET_TREE_METADATA_SHA256,
        "dataset_tree_total_bytes": QUALIFY.DATASET_TREE_TOTAL_BYTES,
        "normalization_content_sha256": QUALIFY.NORMALIZATION_CONTENT_SHA256,
        "normalization_raw_sha256": QUALIFY.NORMALIZATION_RAW_SHA256,
        "original_hdf5_file_count": QUALIFY.ORIGINAL_HDF5_FILE_COUNT,
        "original_hdf5_inventory_content_sha256": QUALIFY.ORIGINAL_HDF5_CONTENT_SHA256,
        "original_hdf5_inventory_raw_sha256": EVALUATOR.sha256_file(
            ROOT / "configs/libero_original_hdf5_inventory.json"
        ),
        "original_hdf5_repository_id": QUALIFY.ORIGINAL_HDF5_REPOSITORY_ID,
        "original_hdf5_revision": QUALIFY.ORIGINAL_HDF5_REVISION,
        "original_hdf5_total_bytes": QUALIFY.ORIGINAL_HDF5_TOTAL_BYTES,
        "project_source_tree_sha256": source["project_source_tree_sha256"],
        "simulator_attestation_raw_sha256": attestation_raw_sha256,
        "simulator_attestation_sha256": QUALIFY.canonical_sha256(attestation),
        "simulator_runtime_sha256": QUALIFY.simulator_runtime_sha256(attestation),
        "source_files_sha256": source["source_files_sha256"],
        "task_inventory_sha256": QUALIFY.TASK_INVENTORY_SHA256,
        "train_venv_identity": _train_venv_identity(),
    }
    task_inventory = [
        {name: task[name] for name in ("instruction", "suite", "task_id", "task_name")}
        for task in attestation["task_inventory"]
    ]
    results = QUALIFY.canonical_gate_results()
    evidence = {
        "demonstrations": sorted(
            [
                {
                    "action_sequence_sha256": _digest(f"actions:{task['suite']}:{task['task_id']}"),
                    "demonstration_id": f"{task['suite']}:{task['task_id']:02d}",
                    "initial_state_sha256": _digest(f"initial:{task['suite']}:{task['task_id']}"),
                    "instruction": task["instruction"],
                    "observation_sequence_sha256": _digest(f"observations:{task['suite']}:{task['task_id']}"),
                    "raw_evidence_ids": ["all-evidence"],
                    "regenerated": True,
                    "source_episode_index": task["task_id"],
                    "step_count": 10,
                    "success": True,
                    "suite": task["suite"],
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "trajectory_sha256": _digest(f"trajectory:{task['suite']}:{task['task_id']}"),
                }
                for task in task_inventory
            ],
            key=lambda item: item["demonstration_id"],
        ),
        "gates": [
            {
                "evidence_ids": ["all-evidence"],
                "name": name,
                "passed": True,
                "result": results[name],
                "result_sha256": QUALIFY.canonical_sha256(results[name]),
            }
            for name in QUALIFY.GATE_NAMES
        ],
        "inputs": inputs,
        "raw_evidence": [{"bytes": 1, "id": "all-evidence", "path": "raw.bin", "sha256": _digest("raw")}],
        "schema": QUALIFY.EVIDENCE_SCHEMA,
    }
    summary = QUALIFY.validate_evidence_document(
        evidence,
        expected_inputs=inputs,
        task_inventory=task_inventory,
        raw_evidence_root=None,
    )
    report = QUALIFY.build_report(
        evidence,
        evidence_manifest_raw_sha256=_digest("evidence-manifest"),
        expected_inputs=inputs,
        task_inventory=task_inventory,
        replay_summary=summary,
        validator_runtime=_validator_runtime_from_attestation(attestation),
        validator_source=source,
    )
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="ascii")
    digest = EVALUATOR.sha256_file(path)
    path.with_suffix(".json.sha256").write_text(f"{digest}  qualification.json\n", encoding="ascii")
    return path, digest, source["project_source_tree_sha256"]


def test_contamination_contract_forces_exact_1999_episode_primary_matrix() -> None:
    contamination = EVALUATOR.load_contamination_contract(ROOT)
    episodes = EVALUATOR.official_episode_matrix(contamination)

    assert len(episodes) == 1999
    assert EVALUATOR.OFFICIAL_EXCLUDED_EPISODE not in episodes
    assert sum(episode["suite"] == "libero_goal" and episode["task_id"] == 7 for episode in episodes) == 49
    assert episodes[0] == {"reset_id": 0, "suite": "libero_spatial", "task_id": 0}
    assert episodes[-1] == {"reset_id": 49, "suite": "libero_10", "task_id": 9}


def test_preregistration_binds_raw_sha_token_attestation_and_exact_24_cells(tmp_path: Path) -> None:
    token = "external final freeze token"
    manifest = _manifest(token=token)
    path = tmp_path / "preregistered.json"
    path.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
    raw_sha256 = EVALUATOR.sha256_file(path)

    loaded, cell, observed_sha256 = EVALUATOR.load_preregistration(
        path,
        cell_id="seed-1-flow-nfe-5-k-4",
        execution_horizon=4,
        evaluation_seed=123,
        final_freeze_token=token,
        preregistration_sha256=raw_sha256,
        simulator_attestation_sha256="a" * 64,
        contamination=manifest["contamination"],
    )

    assert loaded == manifest
    assert cell["train_seed"] == 1
    assert cell["objective"] == "rectified_flow"
    assert cell["nfe"] == 5
    assert observed_sha256 == raw_sha256

    with pytest.raises(RuntimeError, match="freeze token"):
        EVALUATOR.load_preregistration(
            path,
            cell_id=cell["cell_id"],
            execution_horizon=4,
            evaluation_seed=123,
            final_freeze_token="post-hoc",
            preregistration_sha256=raw_sha256,
            simulator_attestation_sha256="a" * 64,
            contamination=manifest["contamination"],
        )


def test_preregistration_creator_derives_episode_and_serving_policy_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    attestation = _creator_attestation()
    synthetic_task_inventory_sha256 = QUALIFY.canonical_sha256(attestation["task_inventory"])
    attestation["task_inventory_sha256"] = synthetic_task_inventory_sha256
    monkeypatch.setattr(QUALIFY, "TASK_INVENTORY_SHA256", synthetic_task_inventory_sha256)
    monkeypatch.setattr(EVALUATOR, "TASK_INVENTORY_SHA256", synthetic_task_inventory_sha256)
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(json.dumps(attestation, sort_keys=True), encoding="utf-8")
    attestation_sha256 = EVALUATOR.canonical_sha256(attestation)
    qualified_source = QUALIFY.source_identity(ROOT)["project_source_tree_sha256"]
    qualified_identity = copy.deepcopy(_manifest(attestation_sha256=attestation_sha256)["expert_replay_qualification"])
    qualified_identity["project_source_tree_sha256"] = qualified_source
    qualified_identity["simulator_attestation_raw_sha256"] = EVALUATOR.sha256_file(attestation_path)
    qualified_identity["simulator_runtime_sha256"] = QUALIFY.simulator_runtime_sha256(attestation)
    qualified_identity["task_inventory_sha256"] = synthetic_task_inventory_sha256
    qualified_identity["validator_runtime_identity"] = _validator_runtime_from_attestation(attestation)
    qualification_path = tmp_path / "qualification.json"
    qualification_path.write_text("{}\n", encoding="ascii")
    qualification_sha256 = EVALUATOR.sha256_file(qualification_path)
    monkeypatch.setattr(
        CREATOR,
        "load_qualification_report",
        lambda *_args, **_kwargs: ({"validated": True}, qualification_sha256),
    )
    monkeypatch.setattr(
        CREATOR,
        "qualification_identity",
        lambda *_args, **_kwargs: copy.deepcopy(qualified_identity),
    )
    cells = []
    for cell in manifest["cells"]:
        value = {
            name: field
            for name, field in cell.items()
            if name not in {"episode_matrix_sha256", "output_claim", "serving_policy_sha256"}
        }
        value["checkpoint"] = {**value["checkpoint"], "source_tree_sha256": qualified_source}
        cells.append(value)
    cells_path = tmp_path / "cells.json"
    cells_path.write_text(json.dumps({"cells": cells}, sort_keys=True))
    output = tmp_path / "preregistered.json"
    run_root = tmp_path / "official-runs"
    claim_root = tmp_path / "official-claims"
    run_root.mkdir()
    claim_root.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_libero_preregistration.py",
            "--cells",
            str(cells_path),
            "--simulator-attestation",
            str(attestation_path),
            "--expert-replay-qualification",
            str(qualification_path),
            "--expert-replay-qualification-sha256",
            qualification_sha256,
            "--evaluation-seed",
            "123",
            "--final-freeze-token",
            "sealed",
            "--official-output-root",
            str(run_root),
            "--official-claim-root",
            str(claim_root),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(CREATOR, "validate_evaluator_process_environment", lambda _root: {})
    CREATOR.main()
    created = json.loads(output.read_text())
    assert created["episode_matrix_sha256"] == manifest["episode_matrix_sha256"]
    assert created["aggregator_sha256"] == EVALUATOR.sha256_file(ROOT / "scripts/aggregate_libero_official.py")
    assert all(cell["serving_policy_sha256"] for cell in created["cells"])
    assert all(cell["output_claim"] for cell in created["cells"])
    assert output.with_suffix(".json.sha256").is_file()


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda value: value["cells"].pop(), "exact 24-cell"),
        (lambda value: value["episodes"].append(dict(EVALUATOR.OFFICIAL_EXCLUDED_EPISODE)), "episode matrix changed"),
        (lambda value: value["cells"][0].update(nfe=7), "flow cell NFE"),
        (lambda value: value["cells"][0].update(policy_warmup_calls=1), "warm-up count"),
        (lambda value: value.update(policy_warmup_calls=1), "policy warm-up count"),
        (
            lambda value: value["cells"][0].update(latency_runtime_sha256="0" * 64),
            "hardware/software runtime identity",
        ),
        (
            lambda value: value["cells"][0]["checkpoint"].update(update=29_999),
            "checkpoint update",
        ),
    ),
)
def test_preregistration_rejects_incomplete_or_mutated_factor_matrix(mutate: Any, message: str) -> None:
    manifest = _manifest()
    mutate(manifest)
    with pytest.raises(RuntimeError, match=message):
        EVALUATOR.validate_preregistration_manifest(
            manifest,
            contamination=manifest["contamination"],
            simulator_attestation_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda value: value["expert_replay_qualification"].update(successful_task_count=39),
            "40/40 successful replay",
        ),
        (
            lambda value: value["expert_replay_qualification"].update(dataset_content_inventory_sha256="0" * 64),
            "dataset identity mismatch",
        ),
        (
            lambda value: value["expert_replay_qualification"].update(dataset_snapshot_files_verified=381),
            "dataset snapshot counts mismatch",
        ),
        (
            lambda value: value["expert_replay_qualification"]["train_venv_identity"].update(root_sha256="0" * 64),
            "semantic root hash differs",
        ),
        (
            lambda value: value["expert_replay_qualification"].update(simulator_attestation_sha256="0" * 64),
            "simulator attestation mismatch",
        ),
        (
            lambda value: value["expert_replay_qualification"].update(original_hdf5_revision="0" * 40),
            "original HDF5 repository identity mismatch",
        ),
        (
            lambda value: value["expert_replay_qualification"].update(original_hdf5_inventory_content_sha256="0" * 64),
            "original HDF5 inventory identity mismatch",
        ),
        (
            lambda value: value["cells"][0]["checkpoint"].update(source_tree_sha256="0" * 64),
            "qualified source tree",
        ),
    ),
)
def test_preregistration_rejects_forged_expert_replay_identity(mutate: Any, message: str) -> None:
    manifest = _manifest()
    mutate(manifest)
    with pytest.raises(RuntimeError, match=message):
        EVALUATOR.validate_preregistration_manifest(
            manifest,
            contamination=manifest["contamination"],
            simulator_attestation_sha256="a" * 64,
        )


def _committed_checkpoint(
    tmp_path: Path,
    *,
    selected_cell: dict[str, Any],
    task: str | None = None,
    examples_seen: int = EVALUATOR.OFFICIAL_TRAINING_EXAMPLES,
    train_episode_count: int = EVALUATOR.OFFICIAL_TRAIN_EPISODES,
    validation_episode_count: int = EVALUATOR.OFFICIAL_VALIDATION_EPISODES,
    optimization_overrides: dict[str, Any] | None = None,
    training_overrides: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    run_root = tmp_path / "run"
    run_root.mkdir()
    checkpoint = run_root / "checkpoints/update-030000"
    checkpoint.mkdir(parents=True)
    canonical_name = (
        "libero_direct_regression.toml" if selected_cell["objective"] == "direct_regression" else "libero.toml"
    )
    training_environment = {"authenticated_runtime": {"train_venv": _train_venv_identity()}}
    resolved_config = load_resolved_toml(ROOT / "configs" / canonical_name)
    resolved_config["optimization"].update(optimization_overrides or {})
    resolved_config["training"].update(training_overrides or {})
    resolved_config.update(
        {
            "artifact_trees": {
                "dataset_content_inventory_sha256": selected_cell["checkpoint"]["dataset_content_inventory_sha256"],
                "dataset_tree_sha256": selected_cell["checkpoint"]["dataset_tree_sha256"],
            },
            "execution_environment": training_environment,
            "run": {"max_cached_files": 128, "seed": selected_cell["train_seed"], "task": task},
            "source_tree_sha256": selected_cell["checkpoint"]["source_tree_sha256"],
        }
    )
    resolved_path = checkpoint / "artifacts/resolved_config.json"
    config_sha256 = save_resolved_config(resolved_path, resolved_config)
    resolved_sha256 = EVALUATOR.sha256_file(resolved_path)
    journal = create_run_journal(run_root, config_sha256=config_sha256)
    manifest = {
        "artifacts": {
            "resolved_config": {
                "bytes": resolved_path.stat().st_size,
                "path": "artifacts/resolved_config.json",
                "sha256": resolved_sha256,
            }
        },
        "config_sha256": config_sha256,
        "dataset_content_inventory_sha256": selected_cell["checkpoint"]["dataset_content_inventory_sha256"],
        "dataset_tree_sha256": selected_cell["checkpoint"]["dataset_tree_sha256"],
        "execution_environment": training_environment,
        "execution_environment_sha256": EVALUATOR.canonical_sha256(training_environment),
        "kind": "resumable-libero-training",
        "last_metrics": {
            "examples_seen": examples_seen,
            "gradient_norm": 1.0,
            "objective": selected_cell["objective"],
            "train_loss": 0.25,
            "update": EVALUATOR.FINAL_CHECKPOINT_UPDATE,
            "update_seconds": 42.0,
            "validation_loss": 0.3,
        },
        "parent_manifest_sha256": None,
        "policy_contract": {
            "inference_seed_behavior": selected_cell["inference_seed_behavior"],
            "nfe": 10,
            "objective": selected_cell["objective"],
            "sampler": selected_cell["sampler"],
        },
        "policy_contract_sha256": selected_cell["policy_contract_sha256"],
        "run_seed": selected_cell["train_seed"],
        "run_uuid": journal.run_uuid,
        "schema": "duo-vla-checkpoint-v1",
        "source_tree_sha256": selected_cell["checkpoint"]["source_tree_sha256"],
        "task": task,
        "train_episode_count": train_episode_count,
        "trainer_state": {
            "examples_seen": examples_seen,
            "next_update": EVALUATOR.FINAL_CHECKPOINT_UPDATE,
            "schema": "duo-vla-trainer-state-v1",
        },
        "validation_episode_count": validation_episode_count,
    }
    path = checkpoint / "manifest.json"
    path.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
    manifest_sha256 = EVALUATOR.sha256_file(path)
    record_latest_checkpoint(
        run_root,
        checkpoint=checkpoint,
        update=EVALUATOR.FINAL_CHECKPOINT_UPDATE,
        manifest_sha256=manifest_sha256,
        parent_manifest_sha256=None,
        last_metrics=manifest["last_metrics"],
    )
    return checkpoint, manifest_sha256


def test_official_health_binds_selected_nfe_and_final_journal_tip(tmp_path: Path) -> None:
    manifest = _manifest()
    selected = next(cell for cell in manifest["cells"] if cell["cell_id"] == "seed-1-flow-nfe-5-k-4")
    checkpoint_path, manifest_sha256 = _committed_checkpoint(tmp_path, selected_cell=selected)
    selected["checkpoint"]["manifest_sha256"] = manifest_sha256
    contract = {
        "inference_seed_behavior": selected["inference_seed_behavior"],
        "nfe": selected["nfe"],
        "objective": selected["objective"],
        "sampler": selected["sampler"],
    }
    training_environment = {"authenticated_runtime": {"train_venv": _train_venv_identity()}}
    health = {
        **contract,
        "checkpoint": {
            "execution_geometry": copy.deepcopy(EVALUATOR.LIBERO_EXECUTION_GEOMETRY),
            "kind": "resumable-libero-training",
            "manifest_sha256": manifest_sha256,
            "path": str(checkpoint_path),
            "policy_contract": {**contract, "nfe": 10},
            "policy_contract_sha256": selected["policy_contract_sha256"],
            "source_tree_sha256": selected["checkpoint"]["source_tree_sha256"],
            "train_seed": selected["train_seed"],
            "train_venv": _train_venv_identity(),
            "training_execution_environment": training_environment,
            "training_execution_environment_sha256": EVALUATOR.canonical_sha256(training_environment),
        },
        "dataset_revision": EVALUATOR.DATASET_REVISION,
        "execution_geometry": copy.deepcopy(EVALUATOR.LIBERO_EXECUTION_GEOMETRY),
        "mode": "real",
        "model_revision": EVALUATOR.MODEL_REVISION,
        "normalization_content_sha256": EVALUATOR.NORMALIZATION_SHA256,
        "prefix_cache_scope": "request",
        "latency_runtime_sha256": selected["latency_runtime_sha256"],
        "serving_runtime_sha256": selected["serving_runtime_sha256"],
        "train_seed": selected["train_seed"],
    }

    report = EVALUATOR.validate_official_policy_health(
        health,
        selected_cell=selected,
        execution_horizon=4,
        expert_replay_qualification=manifest["expert_replay_qualification"],
    )

    assert report["update"] == EVALUATOR.FINAL_CHECKPOINT_UPDATE
    assert report["run_journal_latest"] is True
    changed = copy.deepcopy(health)
    changed["nfe"] = 10
    with pytest.raises(RuntimeError, match="serving policy"):
        EVALUATOR.validate_official_policy_health(
            changed,
            selected_cell=selected,
            execution_horizon=4,
            expert_replay_qualification=manifest["expert_replay_qualification"],
        )
    changed = copy.deepcopy(health)
    changed_venv = changed["checkpoint"]["train_venv"]
    changed_venv["content_inventory_sha256"] = _digest("different-train-venv-content")
    _rebind_venv_root(changed_venv)
    with pytest.raises(RuntimeError, match="differs from expert replay qualification"):
        EVALUATOR.validate_official_policy_health(
            changed,
            selected_cell=selected,
            execution_horizon=4,
            expert_replay_qualification=manifest["expert_replay_qualification"],
        )
    changed = copy.deepcopy(health)
    nested_venv = changed["checkpoint"]["training_execution_environment"]["authenticated_runtime"]["train_venv"]
    nested_venv["tree_metadata_sha256"] = _digest("different-train-venv-tree")
    _rebind_venv_root(nested_venv)
    changed["checkpoint"]["training_execution_environment_sha256"] = EVALUATOR.canonical_sha256(
        changed["checkpoint"]["training_execution_environment"]
    )
    with pytest.raises(RuntimeError, match="authenticated train-venv differs"):
        EVALUATOR.validate_official_policy_health(
            changed,
            selected_cell=selected,
            execution_horizon=4,
            expert_replay_qualification=manifest["expert_replay_qualification"],
        )


@pytest.mark.parametrize(
    ("checkpoint_kwargs", "message"),
    (
        ({"task": "turn on the stove"}, "full 40-task multitask"),
        ({"optimization_overrides": {"total_updates": 60_000}}, "optimization recipe"),
        ({"optimization_overrides": {"warmup_updates": 999}}, "optimization recipe"),
        ({"training_overrides": {"validation_samples": 8}}, "training recipe"),
        ({"examples_seen": EVALUATOR.OFFICIAL_TRAINING_EXAMPLES - 64}, "trainer state"),
        ({"train_episode_count": EVALUATOR.OFFICIAL_TRAIN_EPISODES - 1}, "train episode count"),
        (
            {"validation_episode_count": EVALUATOR.OFFICIAL_VALIDATION_EPISODES - 1},
            "validation episode count",
        ),
    ),
)
def test_official_health_rejects_noncanonical_training_checkpoint(
    tmp_path: Path,
    checkpoint_kwargs: dict[str, Any],
    message: str,
) -> None:
    preregistration = _manifest()
    selected = next(cell for cell in preregistration["cells"] if cell["cell_id"] == "seed-0-flow-nfe-10-k-4")
    checkpoint_path, manifest_sha256 = _committed_checkpoint(
        tmp_path,
        selected_cell=selected,
        **checkpoint_kwargs,
    )
    selected["checkpoint"]["manifest_sha256"] = manifest_sha256
    contract = {
        "inference_seed_behavior": selected["inference_seed_behavior"],
        "nfe": selected["nfe"],
        "objective": selected["objective"],
        "sampler": selected["sampler"],
    }
    training_environment = {"authenticated_runtime": {"train_venv": _train_venv_identity()}}
    health = {
        **contract,
        "checkpoint": {
            "execution_geometry": copy.deepcopy(EVALUATOR.LIBERO_EXECUTION_GEOMETRY),
            "kind": "resumable-libero-training",
            "manifest_sha256": manifest_sha256,
            "path": str(checkpoint_path),
            "policy_contract": contract,
            "policy_contract_sha256": selected["policy_contract_sha256"],
            "source_tree_sha256": selected["checkpoint"]["source_tree_sha256"],
            "train_seed": selected["train_seed"],
            "train_venv": _train_venv_identity(),
            "training_execution_environment": training_environment,
            "training_execution_environment_sha256": EVALUATOR.canonical_sha256(training_environment),
        },
        "dataset_revision": EVALUATOR.DATASET_REVISION,
        "execution_geometry": copy.deepcopy(EVALUATOR.LIBERO_EXECUTION_GEOMETRY),
        "mode": "real",
        "model_revision": EVALUATOR.MODEL_REVISION,
        "normalization_content_sha256": EVALUATOR.NORMALIZATION_SHA256,
        "prefix_cache_scope": "request",
        "latency_runtime_sha256": selected["latency_runtime_sha256"],
        "serving_runtime_sha256": selected["serving_runtime_sha256"],
        "train_seed": selected["train_seed"],
    }

    with pytest.raises(RuntimeError, match=message):
        EVALUATOR.validate_official_policy_health(
            health,
            selected_cell=selected,
            execution_horizon=4,
            expert_replay_qualification=preregistration["expert_replay_qualification"],
        )


def test_official_mode_requires_full_primary_selection_and_seal() -> None:
    args = EVALUATOR.parse_args(
        [
            "--mode",
            "official-score",
            "--suite",
            "all",
            "--task-ids",
            "all",
            "--init-state-ids",
            "all",
            "--evaluation-seed",
            "123",
            "--execution-horizon",
            "4",
            "--output-dir",
            "/tmp/output",
            "--preregistration-manifest",
            "/tmp/preregistered.json",
            "--preregistration-sha256",
            "a" * 64,
            "--cell-id",
            "seed-0-flow-nfe-10-k-4",
            "--final-freeze-token",
            "sealed",
        ]
    )
    EVALUATOR.validate_mode_arguments(args)

    args.task_ids = "0"
    with pytest.raises(RuntimeError, match="task-ids all"):
        EVALUATOR.validate_mode_arguments(args)


def test_official_summary_states_clean_denominator_and_exclusion() -> None:
    contamination = EVALUATOR.load_contamination_contract(ROOT)
    task_metrics = [
        {
            "episodes": 49 if (suite, task_id) == ("libero_goal", 7) else 50,
            "suite": suite,
            "task_id": task_id,
        }
        for suite in EVALUATOR.SUITES
        for task_id in range(10)
    ]
    summary = {
        "complete_40_task_macro": True,
        "episodes": 1999,
        "policy_calls": 1,
        "task_metrics": task_metrics,
    }

    result = EVALUATOR.bind_official_summary(
        summary,
        contamination=contamination,
        episode_matrix_sha256="e" * 64,
    )

    assert result["reporting"]["denominator"] == 1999
    assert result["reporting"]["full_official_episode_count"] == 2000
    assert result["reporting"]["excluded_episodes"] == [EVALUATOR.OFFICIAL_EXCLUDED_EPISODE]
    assert result["reporting"]["non_blind_full_set_reported"] is False
    assert result["reporting"]["policy_warmup_calls"] == EVALUATOR.OFFICIAL_POLICY_WARMUP_CALLS
    assert result["reporting"]["policy_warmup_included_in_latency"] is False
    assert result["reporting"]["latency_scope"] == "episode_policy_calls_only"


def test_eval_launchers_scrub_injection_and_do_not_append_pythonpath() -> None:
    for name in (
        "run_create_libero_preregistration.sh",
        "run_aggregate_libero_official.sh",
        "run_libero_eval.sh",
        "run_libero_preflight.sh",
    ):
        source = (ROOT / "scripts" / name).read_text()
        assert source.startswith("#!/bin/bash -p\nset -euo pipefail")
        assert "unset BASH_ENV CDPATH ENV GLOBIGNORE" in source
        assert "exec /usr/bin/env -i" in source
        assert "PYTHONPATH" not in source
        assert "${PYTHONPATH:+" not in source
        assert '"MUJOCO_EGL_DEVICE_ID=0"' in source
        assert '"HF_HOME=/root/.cache/huggingface"' in source
        assert '"LANG=C.UTF-8"' in source
        assert '"LC_ALL=C.UTF-8"' in source
        assert '"PYTHONSAFEPATH=1"' in source
        assert '"PYTHONDONTWRITEBYTECODE=1"' in source
        assert "-P -B -X pycache_prefix=/dev/null" in source


def test_evaluator_rejects_direct_python_environment_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONWARNINGS", "error")
    with pytest.raises(RuntimeError, match="injection_overrides"):
        EVALUATOR.validate_evaluator_process_environment(ROOT)


def test_evaluator_rejects_unlisted_render_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    cache_root = Path("/root/.cache/duo-vla")
    expected = {
        **EVALUATOR._REQUIRED_EVALUATOR_ENVIRONMENT,
        "DUO_VLA_CACHE_ROOT": str(cache_root),
        "HF_HOME": "/root/.cache/huggingface",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LIBERO_CONFIG_PATH": str(cache_root / "simulators/libero/config"),
        "PATH": f"{cache_root}/venvs/libero-eval/bin:/usr/bin:/bin",
    }
    monkeypatch.setattr(EVALUATOR.os, "environ", {**expected, "LIBGL_ALWAYS_SOFTWARE": "1"})
    monkeypatch.setattr(EVALUATOR.sys, "prefix", str(cache_root / "venvs/libero-eval"))
    monkeypatch.setattr(
        EVALUATOR.sys,
        "flags",
        type("Flags", (), {"dont_write_bytecode": 1, "no_user_site": 1, "safe_path": 1})(),
    )
    monkeypatch.setattr(EVALUATOR.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(EVALUATOR.sys, "pycache_prefix", "/dev/null")
    version = f"python{EVALUATOR.sys.version_info.major}.{EVALUATOR.sys.version_info.minor}"
    compact_version = f"python{EVALUATOR.sys.version_info.major}{EVALUATOR.sys.version_info.minor}"
    monkeypatch.setattr(
        EVALUATOR.sys,
        "path",
        [
            str(ROOT / "src"),
            str(Path(EVALUATOR.sys.base_prefix) / "lib" / f"{compact_version}.zip"),
            str(Path(EVALUATOR.sys.base_prefix) / "lib" / version),
            str(Path(EVALUATOR.sys.base_exec_prefix) / "lib" / version / "lib-dynload"),
            str(cache_root / "venvs/libero-eval" / "lib" / version / "site-packages"),
        ],
    )
    with pytest.raises(RuntimeError, match="exact closed launcher allowlist"):
        EVALUATOR.validate_evaluator_process_environment(ROOT)


class _WarmupClient:
    def __init__(self, *, drift: bool = False) -> None:
        self.replan_ids: list[int] = []
        self.execution_horizons: list[int] = []
        self.drift = drift

    def predict(self, **request: Any) -> tuple[Any, dict[str, Any]]:
        import numpy as np

        self.replan_ids.append(request["replan_id"])
        self.execution_horizons.append(request["execution_horizon"])
        actions = np.zeros((8, 7), dtype=np.float32)
        if self.drift and len(self.replan_ids) == 2:
            actions[0, 0] = 1.0
        response = {
            "evaluation_seed": request["evaluation_seed"],
            "inference_seed": 7,
            "inference_seed_behavior": "episode_identity_gaussian_noise",
            "nfe": 10,
            "objective": "rectified_flow",
            "policy_seconds": 0.01,
            "reset_id": request["reset_id"],
            "reset_source": request["reset_source"],
            "reset_state_sha256": request["reset_state_sha256"],
            "sampler": "euler_uniform",
        }
        return actions, response


def test_warmups_reserve_replan_520_and_reject_drift_before_scoring() -> None:
    health = {"train_seed": 0}
    client = _WarmupClient()
    reports = EVALUATOR.run_policy_warmups(
        client,
        health,
        count=2,
        execution_horizon=4,
        evaluation_seed=123,
    )
    assert client.replan_ids == [520, 520]
    assert client.execution_horizons == [1, 4]
    assert [report["replan_id"] for report in reports] == [520, 520]
    assert [report["execution_horizon"] for report in reports] == [1, 4]

    drifting = _WarmupClient(drift=True)
    with pytest.raises(RuntimeError, match="was not K-independent"):
        EVALUATOR.run_policy_warmups(
            drifting,
            health,
            count=2,
            execution_horizon=4,
            evaluation_seed=123,
        )


def test_external_claim_survives_journal_setup_failure_and_prevents_retry(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    claims_root = tmp_path / "claims"
    runs_root.mkdir()
    claims_root.mkdir()
    roots = EVALUATOR.capture_official_output_roots(runs_root.resolve(), claims_root.resolve())
    freeze_sha256 = _digest("freeze")
    cell_id = "seed-0-flow-nfe-10-k-4"
    output_claim = EVALUATOR.derive_output_claim(cell_id, roots, freeze_sha256)
    preregistration_sha256 = _digest("preregistration")
    claim = EVALUATOR.build_claim_record(
        output_claim,
        cell_id=cell_id,
        preregistration_sha256=preregistration_sha256,
        final_freeze_token_sha256=freeze_sha256,
        created_utc="2026-01-01T00:00:00+00:00",
    )
    payload = (json.dumps(claim, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    claim_path = Path(output_claim["claim_path"])
    claim_sha256 = EVALUATOR.publish_bytes_and_sha256_exclusive(claim_path, payload)
    Path(output_claim["output_dir"]).mkdir()
    journal = EVALUATOR.EvaluationJournal(
        Path(output_claim["output_dir"]),
        {"cell": {"cell_id": cell_id}, "preregistration_sha256": preregistration_sha256},
        claim_path=claim_path,
        claim_json_sha256=claim_sha256,
    )
    with pytest.raises(FileExistsError):
        journal.start()
    assert claim_path.is_file()
    assert claim_path.with_suffix(".json.sha256").is_file()
    with pytest.raises(FileExistsError):
        EVALUATOR.publish_bytes_and_sha256_exclusive(claim_path, payload)


def test_official_claim_and_running_journal_precede_simulator_and_policy_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    claims_root = tmp_path / "claims"
    runs_root.mkdir()
    claims_root.mkdir()
    roots = EVALUATOR.capture_official_output_roots(runs_root.resolve(), claims_root.resolve())
    manifest = _manifest(attestation_sha256="a" * 64)
    manifest["official_output_roots"] = roots
    for cell in manifest["cells"]:
        cell["output_claim"] = EVALUATOR.derive_output_claim(
            cell["cell_id"],
            roots,
            manifest["final_freeze_token_sha256"],
        )
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
    preregistration_sha256 = EVALUATOR.sha256_file(preregistration)
    cell = next(value for value in manifest["cells"] if value["cell_id"] == "seed-0-flow-nfe-10-k-4")
    claim_path = Path(cell["output_claim"]["claim_path"])
    output_dir = Path(cell["output_claim"]["output_dir"])

    monkeypatch.setattr(EVALUATOR, "validate_evaluator_process_environment", lambda _root: {})
    monkeypatch.setattr(EVALUATOR.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(EVALUATOR, "capture_evaluator_source_identities", lambda _root: {})
    monkeypatch.setattr(EVALUATOR, "require_evaluator_sources_unchanged", lambda *_args, **_kwargs: None)

    def fail_after_claim(_root: Path, *, construct_environment: bool) -> dict[str, Any]:
        assert construct_environment is True
        assert claim_path.is_file()
        assert claim_path.with_suffix(".json.sha256").is_file()
        running = json.loads((output_dir / "run.json").read_text())
        assert running["status"] == "running"
        raise RuntimeError("simulator setup failed after claim")

    monkeypatch.setattr(EVALUATOR, "run_exact_simulator_preflight", fail_after_claim)

    class _ForbiddenPolicyClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("policy connection occurred before simulator failure")

    monkeypatch.setattr(EVALUATOR, "PolicyClient", _ForbiddenPolicyClient)
    with pytest.raises(RuntimeError, match="simulator setup failed after claim"):
        EVALUATOR.main(
            [
                "--mode",
                "official-score",
                "--socket",
                str(tmp_path / "policy.sock"),
                "--suite",
                "all",
                "--task-ids",
                "all",
                "--init-state-ids",
                "all",
                "--evaluation-seed",
                "123",
                "--execution-horizon",
                "4",
                "--preregistration-manifest",
                str(preregistration),
                "--preregistration-sha256",
                preregistration_sha256,
                "--cell-id",
                cell["cell_id"],
                "--final-freeze-token",
                "sealed",
                "--output-dir",
                str(output_dir),
            ]
        )
    failed = json.loads((output_dir / "run.json").read_text())
    assert failed["status"] == "failed"
    assert failed["error"]["message"] == "simulator setup failed after claim"


def test_output_claim_rejects_alternate_paths_and_live_root_replacement(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    claims_root = tmp_path / "claims"
    runs_root.mkdir()
    claims_root.mkdir()
    roots = EVALUATOR.capture_official_output_roots(runs_root.resolve(), claims_root.resolve())
    symlink_root = tmp_path / "runs-link"
    symlink_root.symlink_to(runs_root, target_is_directory=True)
    with pytest.raises(RuntimeError, match="canonical real path"):
        EVALUATOR.capture_official_output_roots(symlink_root, claims_root.resolve())
    claim = EVALUATOR.derive_output_claim("seed-0-flow-nfe-10-k-4", roots, _digest("freeze"))
    changed = dict(claim)
    changed["output_dir"] = str(runs_root / "alternate")
    with pytest.raises(RuntimeError, match="not canonical"):
        EVALUATOR.validate_output_claim(
            changed,
            cell_id="seed-0-flow-nfe-10-k-4",
            roots=roots,
            final_freeze_token_sha256=_digest("freeze"),
        )

    runs_root.rename(tmp_path / "retired-runs")
    runs_root.mkdir()
    with pytest.raises(RuntimeError, match="identity drifted"):
        EVALUATOR.validate_official_output_roots(roots, require_live=True)


def test_simulator_manifest_parser_rejects_duplicate_and_extra_fields(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="strict finite UTF-8 JSON"):
        PREFLIGHT.load_strict_manifest(duplicate)

    extra = tmp_path / "extra.json"
    extra.write_text(
        json.dumps(
            {
                "assets": {},
                "environment": {},
                "extra": True,
                "paths": {},
                "schema": "duo-vla-libero-simulator-v1",
                "source": {},
                "training_data_downloaded": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="top-level fields changed"):
        PREFLIGHT.load_strict_manifest(extra)
