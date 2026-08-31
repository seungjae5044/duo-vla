"""Focused tests for authenticated official LIBERO matrix aggregation."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate_libero_official as AGGREGATOR  # noqa: E402
import evaluate_libero as EVALUATOR  # noqa: E402


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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


def _validator_runtime_identity() -> dict[str, Any]:
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
            "environment": {"PYTHONSAFEPATH": "1", "PYTHONDONTWRITEBYTECODE": "1"},
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
        "simulator_runtime_sha256": _digest("simulator-runtime"),
        "site_packages": {
            "declared_files": 1,
            "startup_files": {},
            "symlinks": {},
            "unregistered_files": 0,
        },
    }


def _simulator_runtime_sha256(attestation: dict[str, Any]) -> str:
    return EVALUATOR.canonical_sha256(
        {name: attestation[name] for name in EVALUATOR._SIMULATOR_RUNTIME_IDENTITY_FIELDS}
    )


def _validator_runtime_from_attestation(attestation: dict[str, Any]) -> dict[str, Any]:
    fields = EVALUATOR._VALIDATOR_RUNTIME_IDENTITY_FIELDS - {"schema", "simulator_runtime_sha256"}
    return {
        **{name: copy.deepcopy(attestation[name]) for name in fields},
        "schema": EVALUATOR.VALIDATOR_RUNTIME_IDENTITY_SCHEMA,
        "simulator_runtime_sha256": _simulator_runtime_sha256(attestation),
    }


def _policy_contract(objective: str) -> dict[str, Any]:
    flow = objective == "rectified_flow"
    return {
        "action_dim": 7,
        "action_horizon": 8,
        "clip_final_normalized_actions": True,
        "clip_intermediate_actions": False,
        "inference_seed_behavior": "episode_identity_gaussian_noise" if flow else "episode_identity_echo_only",
        "nfe": 10 if flow else 1,
        "objective": objective,
        "sampler": "euler_uniform" if flow else "single_forward",
        "schema": "duo-vla-policy-contract-v1",
        "training_input": "linear_noise_to_clean" if flow else "zero_action_canvas",
        "training_target": "velocity_clean_minus_noise" if flow else "clean_action",
        "training_timestep": "uniform_per_chunk" if flow else "fixed_one",
    }


def _simulator_attestation() -> dict[str, Any]:
    runtime = _validator_runtime_identity()
    project_sources = runtime["project_sources"]
    project_sources.update(
        {
            "bridge": AGGREGATOR._STARTUP_AGGREGATION_SOURCE_IDENTITIES["libero_bridge.py"]["sha256"],
            "evaluator": AGGREGATOR._STARTUP_AGGREGATION_SOURCE_IDENTITIES["evaluate_libero.py"]["sha256"],
            "preflight": AGGREGATOR._STARTUP_AGGREGATION_SOURCE_IDENTITIES["preflight_libero_env.py"]["sha256"],
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
        "schema": EVALUATOR.SIMULATOR_ATTESTATION_SCHEMA,
        "site_packages": runtime["site_packages"],
        "source": {"revision": "pinned"},
        "status": "ok",
        "task_inventory": [
            {"suite": suite, "task_id": task_id, "task_name": f"{suite}-task-{task_id}"}
            for suite in EVALUATOR.SUITES
            for task_id in range(10)
        ],
        "torch": "cpu-test",
    }


def _manifest(tmp_path: Path | None = None) -> dict[str, Any]:
    contamination = EVALUATOR.load_contamination_contract(ROOT)
    episodes = EVALUATOR.official_episode_matrix(contamination)
    episode_sha256 = EVALUATOR.canonical_sha256(episodes)
    if tmp_path is None:
        roots = {
            "claims": {"device": 1, "inode": 2, "path": "/sealed/libero-official/claims"},
            "runs": {"device": 1, "inode": 1, "path": "/sealed/libero-official/runs"},
            "schema": EVALUATOR.OUTPUT_ROOTS_SCHEMA,
        }
    else:
        runs_root = tmp_path / "official-runs"
        claims_root = tmp_path / "official-claims"
        runs_root.mkdir()
        claims_root.mkdir()
        roots = EVALUATOR.capture_official_output_roots(runs_root.resolve(), claims_root.resolve())
    freeze_sha256 = _digest("sealed")
    cells: list[dict[str, Any]] = []
    for seed in EVALUATOR.OFFICIAL_TRAIN_SEEDS:
        for objective, nfes in (
            ("rectified_flow", EVALUATOR.OFFICIAL_FLOW_NFES),
            ("direct_regression", (1,)),
        ):
            contract_sha256 = EVALUATOR.canonical_sha256(_policy_contract(objective))
            checkpoint = {
                "dataset_content_inventory_sha256": EVALUATOR.DATASET_CONTENT_INVENTORY_SHA256,
                "dataset_tree_sha256": EVALUATOR.DATASET_TREE_METADATA_SHA256,
                "manifest_sha256": _digest(f"checkpoint:{seed}:{objective}"),
                "source_tree_sha256": "f" * 64,
                "update": EVALUATOR.FINAL_CHECKPOINT_UPDATE,
            }
            for nfe in nfes:
                for execution_horizon in EVALUATOR.OFFICIAL_EXECUTION_HORIZONS:
                    cell_id = EVALUATOR.official_cell_id(seed, objective, nfe, execution_horizon)
                    selected = {
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
                            "cell_id": cell_id,
                            "checkpoint": checkpoint,
                            "episode_matrix_sha256": episode_sha256,
                            "execution_geometry": copy.deepcopy(EVALUATOR.LIBERO_EXECUTION_GEOMETRY),
                            "execution_horizon": execution_horizon,
                            **selected,
                            "latency_runtime_sha256": _digest("latency-runtime"),
                            "policy_contract_sha256": contract_sha256,
                            "policy_warmup_calls": EVALUATOR.OFFICIAL_POLICY_WARMUP_CALLS,
                            "output_claim": EVALUATOR.derive_output_claim(cell_id, roots, freeze_sha256),
                            "serving_policy_sha256": EVALUATOR.canonical_sha256(selected),
                            "serving_runtime_sha256": _digest(f"runtime:{seed}:{objective}"),
                            "train_seed": seed,
                        }
                    )
    simulator = _simulator_attestation()
    return {
        "aggregation_python_version": EVALUATOR.AGGREGATION_PYTHON_VERSION,
        "aggregator_sha256": AGGREGATOR._STARTUP_AGGREGATION_SOURCE_IDENTITIES["aggregate_libero_official.py"][
            "sha256"
        ],
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
            "normalization_raw_sha256": "e" * 64,
            "original_hdf5_file_count": EVALUATOR.ORIGINAL_HDF5_FILE_COUNT,
            "original_hdf5_inventory_content_sha256": EVALUATOR.ORIGINAL_HDF5_CONTENT_SHA256,
            "original_hdf5_inventory_raw_sha256": EVALUATOR.ORIGINAL_HDF5_INVENTORY_RAW_SHA256,
            "original_hdf5_repository_id": EVALUATOR.ORIGINAL_HDF5_REPOSITORY_ID,
            "original_hdf5_revision": EVALUATOR.ORIGINAL_HDF5_REVISION,
            "original_hdf5_total_bytes": EVALUATOR.ORIGINAL_HDF5_TOTAL_BYTES,
            "project_source_tree_sha256": "f" * 64,
            "raw_evidence_root_sha256": _digest("raw-evidence"),
            "report_sha256": _digest("qualification-report"),
            "schema": "duo-vla-libero-expert-replay-qualification-v1",
            "simulator_attestation_raw_sha256": _digest("attestation-raw"),
            "simulator_attestation_sha256": EVALUATOR.canonical_sha256(simulator),
            "simulator_runtime_sha256": _simulator_runtime_sha256(simulator),
            "status": "passed",
            "successful_demonstration_count": 40,
            "successful_task_count": 40,
            "task_count": 40,
            "task_inventory_sha256": "d00c211a09f34003089ba5a4dbbbb0e11af2543f4bba9cb1901a04a2a25e0117",
            "train_venv_identity": _train_venv_identity(),
            "validator_runtime_identity": _validator_runtime_from_attestation(simulator),
        },
        "execution_horizons": list(EVALUATOR.OFFICIAL_EXECUTION_HORIZONS),
        "final_checkpoint_update": EVALUATOR.FINAL_CHECKPOINT_UPDATE,
        "final_freeze_token_sha256": freeze_sha256,
        "official_output_roots": roots,
        "official_resets_per_task": EVALUATOR.OFFICIAL_RESETS_PER_TASK,
        "policy_warmup_calls": EVALUATOR.OFFICIAL_POLICY_WARMUP_CALLS,
        "schema": EVALUATOR.PREREGISTRATION_SCHEMA,
        "simulator_attestation_sha256": EVALUATOR.canonical_sha256(simulator),
        "suites": list(EVALUATOR.SUITES),
        "task_ids": list(range(10)),
        "training_seeds": list(EVALUATOR.OFFICIAL_TRAIN_SEEDS),
    }


def _episode(planned: dict[str, Any], *, execution_horizon: int = 4) -> dict[str, Any]:
    return {
        "action_clip_fraction": 0.0,
        "action_clipped_channels": 0,
        "action_continuous_channels": 6,
        "elapsed_seconds": 0.5,
        "environment_seed": EVALUATOR.ENVIRONMENT_SEED,
        "evaluation_seed": 123,
        "execution_horizon": execution_horizon,
        "init_state_id": planned["reset_id"],
        "normalized_action_clip_fraction": 0.0,
        "policy_budget": EVALUATOR.POLICY_BUDGETS[planned["suite"]],
        "policy_calls": 1,
        "policy_latency_p50_seconds": 0.1,
        "policy_latency_p95_seconds": 0.1,
        "policy_latency_seconds": [0.1],
        "policy_steps": 1,
        "reset_id": planned["reset_id"],
        "reset_source": "official",
        "reset_state_sha256": None,
        "server_latency_p50_seconds": 0.08,
        "server_latency_p95_seconds": 0.08,
        "server_latency_seconds": [0.08],
        "settle_steps": 10,
        "simulator_done": False,
        "steps_to_success": 1,
        "success": True,
        "suite": planned["suite"],
        "task_id": planned["task_id"],
        "task_name": f"{planned['suite']}-task-{planned['task_id']}",
    }


def _write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_episode_validator_requires_exact_order_denominator_and_invariants() -> None:
    manifest = _manifest()
    records = [_episode(planned) for planned in manifest["episodes"]]

    checked = AGGREGATOR.validate_episode_records(
        records,
        planned_episodes=manifest["episodes"],
        evaluation_seed=123,
        execution_horizon=4,
    )
    assert len(checked) == 1999

    records[0], records[1] = records[1], records[0]
    with pytest.raises(RuntimeError, match="pre-registered order"):
        AGGREGATOR.validate_episode_records(
            records,
            planned_episodes=manifest["episodes"],
            evaluation_seed=123,
            execution_horizon=4,
        )

    records = [_episode(planned) for planned in manifest["episodes"]]
    records[0]["server_latency_p50_seconds"] = 0.2
    records[0]["server_latency_p95_seconds"] = 0.2
    records[0]["server_latency_seconds"] = [0.2]
    with pytest.raises(RuntimeError, match="paired server latency"):
        AGGREGATOR.validate_episode_records(
            records,
            planned_episodes=manifest["episodes"],
            evaluation_seed=123,
            execution_horizon=4,
        )


def test_run_artifact_authenticates_hashes_cell_and_recomputes_summary(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    cell = next(value for value in manifest["cells"] if value["cell_id"] == "seed-0-flow-nfe-10-k-4")
    output_dir = Path(cell["output_claim"]["output_dir"])
    output_dir.mkdir()
    preregistration_sha256 = _digest("preregistration")
    claim_record = EVALUATOR.build_claim_record(
        cell["output_claim"],
        cell_id=cell["cell_id"],
        preregistration_sha256=preregistration_sha256,
        final_freeze_token_sha256=manifest["final_freeze_token_sha256"],
        created_utc="2026-01-01T00:00:00+00:00",
    )
    claim_payload = (json.dumps(claim_record, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    claim_sha256 = EVALUATOR.publish_bytes_and_sha256_exclusive(
        Path(cell["output_claim"]["claim_path"]),
        claim_payload,
    )
    records = [_episode(planned) for planned in manifest["episodes"]]
    episodes_payload = b"".join(
        (json.dumps(record, allow_nan=False, sort_keys=True) + "\n").encode() for record in records
    )
    (output_dir / "episodes.jsonl").write_bytes(episodes_payload)
    episodes_sha256 = hashlib.sha256(episodes_payload).hexdigest()
    summary = EVALUATOR.bind_official_summary(
        EVALUATOR.summarize_episodes(records, execution_horizon=4),
        contamination=manifest["contamination"],
        episode_matrix_sha256=manifest["episode_matrix_sha256"],
    )
    summary_sha256 = _write_json(output_dir / "summary.json", summary)
    train_venv = _train_venv_identity()
    training_environment = {
        "authenticated_runtime": {"train_venv": copy.deepcopy(train_venv)},
        "schema": "test-training-environment-v1",
    }
    contract = _policy_contract("rectified_flow")
    health_checkpoint = {
        "dataset_content_inventory_sha256": cell["checkpoint"]["dataset_content_inventory_sha256"],
        "dataset_files_verified": EVALUATOR.DATASET_SNAPSHOT_FILES_VERIFIED,
        "dataset_total_bytes": EVALUATOR.DATASET_SNAPSHOT_TOTAL_BYTES,
        "dataset_tree_sha256": cell["checkpoint"]["dataset_tree_sha256"],
        "execution_geometry": copy.deepcopy(EVALUATOR.LIBERO_EXECUTION_GEOMETRY),
        "kind": "resumable-libero-training",
        "manifest_sha256": cell["checkpoint"]["manifest_sha256"],
        "model_content_inventory_sha256": _digest("model-content"),
        "model_files_verified": 10,
        "model_total_bytes": 1_000,
        "model_tree_sha256": _digest("model-tree"),
        "path": "/sealed/run/checkpoints/update-030000",
        "policy_contract": contract,
        "policy_contract_sha256": EVALUATOR.canonical_sha256(contract),
        "source_tree_sha256": cell["checkpoint"]["source_tree_sha256"],
        "train_seed": 0,
        "train_venv": copy.deepcopy(train_venv),
        "training_execution_environment": training_environment,
        "training_execution_environment_sha256": EVALUATOR.canonical_sha256(training_environment),
    }
    health = {
        "action_dim": 7,
        "action_horizon": 8,
        "checkpoint": health_checkpoint,
        "dataset_revision": EVALUATOR.DATASET_REVISION,
        "execution_geometry": copy.deepcopy(EVALUATOR.LIBERO_EXECUTION_GEOMETRY),
        "inference_seed_behavior": cell["inference_seed_behavior"],
        "latency_runtime_sha256": cell["latency_runtime_sha256"],
        "mode": "real",
        "model_revision": EVALUATOR.MODEL_REVISION,
        "nfe": cell["nfe"],
        "normalization_content_sha256": EVALUATOR.NORMALIZATION_SHA256,
        "objective": cell["objective"],
        "operation": "health",
        "prefix_cache_scope": "request",
        "protocol": EVALUATOR.PROTOCOL,
        "request_id": "test-health",
        "sampler": cell["sampler"],
        "schema": "duo-vla-libero-policy-ipc-v5",
        "serving_runtime_sha256": cell["serving_runtime_sha256"],
        "state_dim": 8,
        "status": "ok",
        "train_seed": 0,
    }
    simulator = _simulator_attestation()
    warmup = {
        "count": 2,
        "included_in_episode_latency": False,
        "reports": [
            {
                "actions_sha256": _digest("actions"),
                "evaluation_seed": 123,
                "execution_horizon": EVALUATOR.OFFICIAL_EXECUTION_HORIZONS[index],
                "inference_seed": 7,
                "inference_seed_behavior": cell["inference_seed_behavior"],
                "nfe": cell["nfe"],
                "objective": cell["objective"],
                "policy_seconds": 0.1,
                "replan_id": EVALUATOR.OFFICIAL_POLICY_WARMUP_REPLAN_ID,
                "reset_id": 0,
                "reset_source": "official",
                "reset_state_sha256": None,
                "sampler": cell["sampler"],
                "status": "ok",
                "warmup_index": index,
            }
            for index in range(2)
        ],
    }
    warmup.update(
        {
            "k_independent_response_sha256": EVALUATOR.canonical_sha256(
                EVALUATOR.warmup_k_independent_response(warmup["reports"][0])
            ),
            "replan_id": EVALUATOR.OFFICIAL_POLICY_WARMUP_REPLAN_ID,
            "validated_before_scoring": True,
        }
    )
    run = {
        "cell": cell,
        "checkpoint": {
            "manifest_sha256": cell["checkpoint"]["manifest_sha256"],
            "path": "/sealed/run/checkpoints/update-030000",
            "run_journal_latest": True,
            "run_root": "/sealed/run",
            "source_tree_sha256": cell["checkpoint"]["source_tree_sha256"],
            "train_venv": copy.deepcopy(train_venv),
            "update": 30_000,
        },
        "claim_json_sha256": claim_sha256,
        "contamination": manifest["contamination"],
        "created_utc": "2026-01-01T00:00:00+00:00",
        "episode_count": 1999,
        "episode_matrix_sha256": manifest["episode_matrix_sha256"],
        "episode_records": 1999,
        "episodes_jsonl_sha256": episodes_sha256,
        "evaluation_seed": 123,
        "evaluator_environment": {
            **EVALUATOR._REQUIRED_EVALUATOR_ENVIRONMENT,
            "DUO_VLA_CACHE_ROOT": "/root/.cache/duo-vla",
            "HF_HOME": "/root/.cache/huggingface",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "LIBERO_CONFIG_PATH": "/root/.cache/duo-vla/simulators/libero/config",
            "PATH": "/root/.cache/duo-vla/venvs/libero-eval/bin:/usr/bin:/bin",
        },
        "execution_horizon": 4,
        "final_freeze_token_sha256": manifest["final_freeze_token_sha256"],
        "finished_utc": "2026-01-02T00:00:00+00:00",
        "init_state_ids": list(range(50)),
        "mode": "official-score",
        "output_claim": cell["output_claim"],
        "policy_health": health,
        "policy_socket": "/sealed/policy.sock",
        "policy_warmup": warmup,
        "preregistration_manifest": "/sealed/preregistration.json",
        "preregistration_sha256": preregistration_sha256,
        "protocol": EVALUATOR.PROTOCOL,
        "reset_identity": {
            "bank": None,
            "id_field": "published_init_state_id",
            "source": "official",
            "state_sha256": None,
        },
        "reset_source": "official",
        "schema": EVALUATOR.OFFICIAL_RUN_SCHEMA,
        "simulator_attestation_sha256": EVALUATOR.canonical_sha256(simulator),
        "simulator_preflight": simulator,
        "status": "complete",
        "summary_json_sha256": summary_sha256,
        "suites": list(EVALUATOR.SUITES),
        "task_ids": list(range(10)),
        "validator_runtime_identity": copy.deepcopy(
            manifest["expert_replay_qualification"]["validator_runtime_identity"]
        ),
    }
    run_sha256 = _write_json(output_dir / "run.json", run)
    completion = {
        "cell_id": cell["cell_id"],
        "claim_json_sha256": claim_sha256,
        "episode_records": 1999,
        "episodes_jsonl_sha256": episodes_sha256,
        "preregistration_sha256": preregistration_sha256,
        "run_json_sha256": run_sha256,
        "schema": EVALUATOR.COMPLETION_SCHEMA,
        "summary_json_sha256": summary_sha256,
    }
    completion_payload = (json.dumps(completion, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    completion_sha256 = EVALUATOR.publish_bytes_and_sha256_exclusive(
        output_dir / "completion.json",
        completion_payload,
    )
    entry = {
        "cell_id": cell["cell_id"],
        "claim_json_sha256": claim_sha256,
        "completion_json_sha256": completion_sha256,
        "episodes_jsonl_sha256": episodes_sha256,
        "run_json_sha256": run_sha256,
        "summary_json_sha256": summary_sha256,
    }

    result = AGGREGATOR.validate_run_artifact(
        entry,
        preregistration=manifest,
        preregistration_sha256=run["preregistration_sha256"],
        registered_cell=cell,
    )
    assert result["summary"]["episodes"] == 1999
    assert result["metrics"]["overall_40_task_macro_success"] == 1.0
    assert result["metrics"]["policy_latency_p50_seconds"] == 0.1
    assert result["metrics"]["server_latency_p95_seconds"] == 0.08
    assert result["metrics"]["episode_throughput_per_hour"] == 7200.0
    assert result["metrics"]["policy_call_throughput_per_second"] == 2.0

    original_train_venv = copy.deepcopy(run["checkpoint"]["train_venv"])
    run["checkpoint"]["train_venv"]["content_inventory_sha256"] = _digest("different-train-venv-content")
    run["checkpoint"]["train_venv"]["root_sha256"] = EVALUATOR.canonical_sha256(
        {
            "base_python_runtime_root_sha256": run["checkpoint"]["train_venv"]["base_python_runtime"]["root_sha256"],
            "content_inventory_sha256": run["checkpoint"]["train_venv"]["content_inventory_sha256"],
            "files_verified": run["checkpoint"]["train_venv"]["files_verified"],
            "schema": run["checkpoint"]["train_venv"]["schema"],
            "startup_hooks_sha256": run["checkpoint"]["train_venv"]["startup_hooks_sha256"],
            "symlinks_verified": run["checkpoint"]["train_venv"]["symlinks_verified"],
            "total_bytes": run["checkpoint"]["train_venv"]["total_bytes"],
            "tree_metadata_sha256": run["checkpoint"]["train_venv"]["tree_metadata_sha256"],
        }
    )
    entry["run_json_sha256"] = _write_json(output_dir / "run.json", run)
    with pytest.raises(RuntimeError, match="differs from expert replay qualification"):
        AGGREGATOR.validate_run_artifact(
            entry,
            preregistration=manifest,
            preregistration_sha256=run["preregistration_sha256"],
            registered_cell=cell,
        )
    run["checkpoint"]["train_venv"] = original_train_venv
    entry["run_json_sha256"] = _write_json(output_dir / "run.json", run)

    external_link = tmp_path / "episodes-hardlink.jsonl"
    external_link.hardlink_to(output_dir / "episodes.jsonl")
    with pytest.raises(RuntimeError, match="linked"):
        AGGREGATOR.validate_run_artifact(
            entry,
            preregistration=manifest,
            preregistration_sha256=run["preregistration_sha256"],
            registered_cell=cell,
        )
    external_link.unlink()

    entry["run_json_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="externally supplied SHA-256"):
        AGGREGATOR.validate_run_artifact(
            entry,
            preregistration=manifest,
            preregistration_sha256=run["preregistration_sha256"],
            registered_cell=cell,
        )
    entry["run_json_sha256"] = run_sha256

    run["checkpoint"]["manifest_sha256"] = "0" * 64
    entry["run_json_sha256"] = _write_json(output_dir / "run.json", run)
    with pytest.raises(RuntimeError, match="checkpoint manifest_sha256 drifted"):
        AGGREGATOR.validate_run_artifact(
            entry,
            preregistration=manifest,
            preregistration_sha256=run["preregistration_sha256"],
            registered_cell=cell,
        )
    run["checkpoint"]["manifest_sha256"] = cell["checkpoint"]["manifest_sha256"]
    entry["run_json_sha256"] = _write_json(output_dir / "run.json", run)

    summary["overall_pooled_success_rate"] = 0.0
    entry["summary_json_sha256"] = _write_json(output_dir / "summary.json", summary)
    with pytest.raises(RuntimeError, match="run summary digest drifted"):
        AGGREGATOR.validate_run_artifact(
            entry,
            preregistration=manifest,
            preregistration_sha256=run["preregistration_sha256"],
            registered_cell=cell,
        )


def test_matrix_requires_exact_inventory_and_keeps_k_separate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    preregistration_sha256 = _digest("preregistration")
    for cell in manifest["cells"]:
        Path(cell["output_claim"]["output_dir"]).mkdir()
        claim = EVALUATOR.build_claim_record(
            cell["output_claim"],
            cell_id=cell["cell_id"],
            preregistration_sha256=preregistration_sha256,
            final_freeze_token_sha256=manifest["final_freeze_token_sha256"],
            created_utc="2026-01-01T00:00:00+00:00",
        )
        payload = (json.dumps(claim, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
        EVALUATOR.publish_bytes_and_sha256_exclusive(Path(cell["output_claim"]["claim_path"]), payload)
    runs = [
        {
            "cell_id": cell["cell_id"],
            "claim_json_sha256": _digest(f"claim:{cell['cell_id']}"),
            "completion_json_sha256": _digest(f"completion:{cell['cell_id']}"),
            "episodes_jsonl_sha256": _digest(f"episodes:{cell['cell_id']}"),
            "run_json_sha256": _digest(f"run:{cell['cell_id']}"),
            "summary_json_sha256": _digest(f"summary:{cell['cell_id']}"),
        }
        for cell in manifest["cells"]
    ]
    inventory = {
        "preregistration_sha256": preregistration_sha256,
        "runs": runs,
        "schema": AGGREGATOR.RUN_INVENTORY_SCHEMA,
    }

    def fake_validate(entry: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        cell = kwargs["registered_cell"]
        value = cell["train_seed"] + cell["execution_horizon"] / 10
        return {
            "cell_id": cell["cell_id"],
            "execution_horizon": cell["execution_horizon"],
            "latency_runtime_sha256": cell["latency_runtime_sha256"],
            "metrics": {metric: value for metric in AGGREGATOR._COMPARISON_METRICS},
            "nfe": cell["nfe"],
            "objective": cell["objective"],
            "train_seed": cell["train_seed"],
            "warmup_output": {
                "actions_sha256": _digest(f"warmup:{cell['train_seed']}:{cell['objective']}:{cell['nfe']}"),
                "count": EVALUATOR.OFFICIAL_POLICY_WARMUP_CALLS,
                "inference_seed": 7,
            },
        }

    monkeypatch.setattr(AGGREGATOR, "validate_run_artifact", fake_validate)
    result = AGGREGATOR.aggregate_matrix(
        manifest,
        preregistration_sha256,
        inventory,
        _digest("inventory"),
    )
    assert result["cell_count"] == 24
    assert result["comparison_count"] == 8
    k1 = next(value for value in result["comparisons"] if value["comparison_id"] == "flow-nfe-10-k-1")
    k4 = next(value for value in result["comparisons"] if value["comparison_id"] == "flow-nfe-10-k-4")
    assert k1["metrics"]["overall_pooled_success_rate"]["mean"] == pytest.approx(1.1)
    assert k4["metrics"]["overall_pooled_success_rate"]["mean"] == pytest.approx(1.4)
    assert k1["metrics"]["overall_pooled_success_rate"]["sample_std"] == pytest.approx(1.0)
    assert "policy_latency_p95_seconds" in k1["metrics"]
    assert "episode_throughput_per_hour" in k1["metrics"]
    assert result["latency_runtime_sha256"] == _digest("latency-runtime")
    assert result["policy_warmup_calls"] == EVALUATOR.OFFICIAL_POLICY_WARMUP_CALLS
    assert result["policy_warmup_included_in_latency"] is False

    runs[0]["output_dir"] = "/tmp/cherry-picked-run"
    with pytest.raises(RuntimeError, match="fields differ"):
        AGGREGATOR.aggregate_matrix(
            manifest,
            preregistration_sha256,
            inventory,
            _digest("inventory-output-dir"),
        )
    runs[0].pop("output_dir")

    extra = Path(manifest["official_output_roots"]["runs"]["path"]) / "unregistered-attempt"
    extra.mkdir()
    with pytest.raises(RuntimeError, match="run root inventory is not exact"):
        AGGREGATOR.aggregate_matrix(
            manifest,
            preregistration_sha256,
            inventory,
            _digest("inventory-extra-attempt"),
        )
    extra.rmdir()

    def mismatched_cross_k(entry: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        value = fake_validate(entry, **kwargs)
        if kwargs["registered_cell"]["execution_horizon"] == 4:
            value["warmup_output"]["actions_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(AGGREGATOR, "validate_run_artifact", mismatched_cross_k)
    with pytest.raises(RuntimeError, match="differs across K"):
        AGGREGATOR.aggregate_matrix(
            manifest,
            preregistration_sha256,
            inventory,
            _digest("inventory-cross-k"),
        )
    monkeypatch.setattr(AGGREGATOR, "validate_run_artifact", fake_validate)

    changed_source = copy.deepcopy(manifest)
    changed_source["aggregator_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="pre-registered aggregator"):
        AGGREGATOR.aggregate_matrix(
            changed_source,
            preregistration_sha256,
            inventory,
            _digest("inventory-source-change"),
        )

    inventory["runs"] = runs[:-1]
    with pytest.raises(RuntimeError, match="exactly 24 runs"):
        AGGREGATOR.aggregate_matrix(
            manifest,
            preregistration_sha256,
            inventory,
            _digest("inventory-short"),
        )


def test_exclusive_publication_writes_bound_sidecar_and_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "matrix.json"
    digest = AGGREGATOR.publish_bytes_and_sha256_exclusive(output, b"payload\n")
    assert output.read_bytes() == b"payload\n"
    assert output.with_suffix(".json.sha256").read_text() == f"{digest}  matrix.json\n"
    with pytest.raises(FileExistsError):
        AGGREGATOR.publish_bytes_and_sha256_exclusive(output, b"replacement\n")
    assert output.read_bytes() == b"payload\n"

    interrupted = tmp_path / "interrupted.json"

    def reject_commit() -> None:
        raise RuntimeError("commit rejected")

    with pytest.raises(RuntimeError, match="commit rejected"):
        AGGREGATOR.publish_bytes_and_sha256_exclusive(interrupted, b"uncommitted\n", commit_guard=reject_commit)
    assert not interrupted.exists()
    assert not interrupted.with_suffix(".json.sha256").exists()
    assert not list(tmp_path.glob(".interrupted.json*"))
