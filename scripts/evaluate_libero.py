#!/usr/bin/env python3
"""Deterministic LIBERO rollout evaluator backed by the persistent Duo-VLA policy socket."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from libero_bridge import (
    ACTION_DIM,
    IMAGE_SHAPE,
    LIBERO_EXECUTION_GEOMETRY,
    PROTOCOL,
    SUITES,
    PolicyClient,
    validate_execution_geometry,
)

from duo_vla.benchmarks.libero_dev_states import (
    canonical_official_state_hashes,
    load_bank,
    sequence_root_sha256,
    state_sha256,
)
from duo_vla.run_journal import validate_resume_checkpoint

CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")
CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
ENVIRONMENT_SEED = 7
SETTLE_STEPS = 10
POLICY_BUDGETS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
NORMALIZATION_SHA256 = "a972b5d95a8aaa8ae7582bafcbc071261979cb46c2a3515b4da7a7cf0156ac73"
OPEN_GRIPPER_NOOP = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
PREREGISTRATION_SCHEMA = "duo-vla-libero-official-preregistration-v1"
OFFICIAL_RUN_SCHEMA = "duo-vla-libero-official-run-v1"
OFFICIAL_SUMMARY_SCHEMA = "duo-vla-libero-official-summary-v1"
CONTAMINATION_SCHEMA = "duo-vla-evaluation-contamination-v1"
CONTAMINATION_LEDGER_SHA256 = "014b3db5dfc8bce9027a20f6deb25cd701a41e6f4204d7cc458416cbc53a2992"
FINAL_CHECKPOINT_UPDATE = 30_000
OFFICIAL_TRAIN_SEEDS = (0, 1, 2)
OFFICIAL_EXECUTION_HORIZONS = (1, 4)
OFFICIAL_FLOW_NFES = (1, 5, 10)
OFFICIAL_FLOW_CHECKPOINT_NFE = 10
OFFICIAL_RESETS_PER_TASK = 50
OFFICIAL_FULL_EPISODES = 2_000
OFFICIAL_PRIMARY_EPISODES = 1_999
OFFICIAL_POLICY_WARMUP_CALLS = 2
OFFICIAL_EXCLUDED_EPISODE = {"reset_id": 0, "suite": "libero_goal", "task_id": 7}
_PREREGISTRATION_FIELDS = {
    "benchmark_protocol",
    "cells",
    "contamination",
    "episode_count",
    "episode_matrix_sha256",
    "episodes",
    "evaluation_seed",
    "execution_horizons",
    "final_checkpoint_update",
    "final_freeze_token_sha256",
    "official_resets_per_task",
    "schema",
    "simulator_attestation_sha256",
    "suites",
    "task_ids",
    "training_seeds",
}
_CELL_FIELDS = {
    "cell_id",
    "checkpoint",
    "episode_matrix_sha256",
    "execution_geometry",
    "execution_horizon",
    "inference_seed_behavior",
    "nfe",
    "objective",
    "policy_contract_sha256",
    "policy_warmup_calls",
    "sampler",
    "serving_policy_sha256",
    "serving_runtime_sha256",
    "train_seed",
}
_CHECKPOINT_FIELDS = {"manifest_sha256", "source_tree_sha256", "update"}
_CONTAMINATION_FIELDS = {
    "excluded_episodes",
    "full_episode_count",
    "ledger_schema",
    "ledger_sha256",
    "primary_episode_count",
    "report_label",
}
_REQUIRED_EVALUATOR_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "MUJOCO_EGL_DEVICE_ID": "0",
    "MUJOCO_GL": "egl",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYOPENGL_PLATFORM": "egl",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
}
_EVALUATOR_ALGORITHM_PREFIXES = (
    "CUBLAS_",
    "CUDA_",
    "CUDNN_",
    "MKL_",
    "MUJOCO_",
    "NCCL_",
    "NPY_",
    "NUMEXPR_",
    "OMP_",
    "OPENBLAS_",
    "PYOPENGL_",
    "PYTORCH_",
    "TORCH_",
)
_ALLOWED_EVALUATOR_ALGORITHM_ENVIRONMENT = frozenset(_REQUIRED_EVALUATOR_ENVIRONMENT)
_EVALUATOR_INJECTION_PREFIXES = ("LD_", "MALLOC_", "OPENSSL_", "PYTHON")
_ALLOWED_EVALUATOR_INJECTION_ENVIRONMENT = {"PYTHONHASHSEED", "PYTHONNOUSERSITE", "PYTHONPATH"}


@dataclass(frozen=True, slots=True)
class DevelopmentResetBank:
    manifest: dict[str, Any]
    manifest_sha256: str
    states_by_task: dict[tuple[str, int], np.ndarray]
    task_records: dict[tuple[str, int], dict[str, Any]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"value is not finite canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(value)
    require(
        observed == expected,
        f"{name} fields differ: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}",
    )


def _require_sha256(value: Any, name: str) -> None:
    require(
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{name} must be 64 lowercase hexadecimal characters",
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field {name!r}")
        result[name] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_strict_json(
    path: Path,
    *,
    name: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read {name}: {path}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, f"expected {name} SHA-256")
        require(hmac.compare_digest(digest, expected_sha256), f"{name} differs from its externally supplied SHA-256")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"{name} is not strict finite UTF-8 JSON") from exc
    require(isinstance(value, dict), f"{name} root must be an object")
    return value, digest


def validate_evaluator_process_environment(project_root: Path) -> dict[str, str]:
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla"))
    forbidden = ("GCONV_PATH", "GLIBC_TUNABLES", "LOCPATH")
    present_forbidden = [name for name in forbidden if os.environ.get(name)]
    injection_overrides = sorted(
        name
        for name in os.environ
        if name.startswith(_EVALUATOR_INJECTION_PREFIXES) and name not in _ALLOWED_EVALUATOR_INJECTION_ENVIRONMENT
    )
    algorithm_overrides = sorted(
        name
        for name in os.environ
        if name.startswith(_EVALUATOR_ALGORITHM_PREFIXES) and name not in _ALLOWED_EVALUATOR_ALGORITHM_ENVIRONMENT
    )
    require(
        not present_forbidden and not injection_overrides and not algorithm_overrides,
        "LIBERO evaluator environment contains injection/algorithm overrides: "
        f"forbidden={present_forbidden}, injection_overrides={injection_overrides}, "
        f"algorithm_overrides={algorithm_overrides}",
    )
    observed = {name: os.environ.get(name) for name in _REQUIRED_EVALUATOR_ENVIRONMENT}
    require(observed == _REQUIRED_EVALUATOR_ENVIRONMENT, f"LIBERO evaluator environment differs: {observed}")
    expected_pythonpath = f"{(project_root / 'src').resolve()}:{(project_root / 'scripts').resolve()}"
    require(
        os.environ.get("PYTHONPATH") == expected_pythonpath,
        f"LIBERO evaluator requires PYTHONPATH={expected_pythonpath}",
    )
    expected_prefix = (cache_root / "venvs/libero-eval").resolve()
    require(Path(sys.prefix).resolve() == expected_prefix, f"LIBERO evaluator must run from {expected_prefix}")
    expected_path = f"{expected_prefix / 'bin'}:/usr/bin:/bin"
    require(os.environ.get("PATH") == expected_path, f"LIBERO evaluator requires PATH={expected_path}")
    require(os.environ.get("HF_HOME") == "/root/.cache/huggingface", "LIBERO evaluator requires pinned HF_HOME")
    expected_libero_config = str((cache_root / "simulators/libero/config").resolve())
    require(
        os.environ.get("LIBERO_CONFIG_PATH") == expected_libero_config,
        f"LIBERO evaluator requires LIBERO_CONFIG_PATH={expected_libero_config}",
    )
    expected_environment = {
        **_REQUIRED_EVALUATOR_ENVIRONMENT,
        "DUO_VLA_CACHE_ROOT": str(cache_root),
        "HF_HOME": "/root/.cache/huggingface",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LIBERO_CONFIG_PATH": expected_libero_config,
        "PATH": expected_path,
        "PYTHONPATH": expected_pythonpath,
    }
    require(
        dict(os.environ) == expected_environment,
        "LIBERO evaluator process environment is not the exact closed launcher allowlist",
    )
    return expected_environment


def load_contamination_contract(project_root: Path) -> dict[str, Any]:
    """Authenticate the immutable exposure ledger and return its scoring rule."""

    ledger_path = project_root / "docs/evaluation_contamination.json"
    companion_path = project_root / "docs/evaluation_contamination.sha256"
    ledger, digest = _read_strict_json(
        ledger_path,
        name="LIBERO contamination ledger",
        expected_sha256=CONTAMINATION_LEDGER_SHA256,
    )
    try:
        companion = companion_path.read_text(encoding="ascii").strip().split()
    except OSError as exc:
        raise RuntimeError(f"cannot read contamination SHA-256 companion: {companion_path}") from exc
    require(
        companion == [CONTAMINATION_LEDGER_SHA256, "docs/evaluation_contamination.json"],
        "contamination SHA-256 companion differs from the frozen ledger",
    )
    _require_exact_keys(ledger, {"primary_reporting_rule", "records", "schema"}, "contamination ledger")
    require(ledger["schema"] == CONTAMINATION_SCHEMA, "contamination ledger schema mismatch")
    require(
        isinstance(ledger["records"], list) and bool(ledger["records"]),
        "contamination ledger has no exposure records",
    )
    rule = ledger["primary_reporting_rule"]
    require(isinstance(rule, dict), "contamination primary reporting rule must be an object")
    _require_exact_keys(
        rule,
        {
            "clean_holdout_episodes_per_policy_seed_k",
            "exclude_symmetrically_across",
            "excluded_cells",
            "official_2000_episode_result_label",
        },
        "contamination primary reporting rule",
    )
    require(
        rule["clean_holdout_episodes_per_policy_seed_k"] == OFFICIAL_PRIMARY_EPISODES,
        "contamination clean-holdout denominator changed",
    )
    require(
        rule["exclude_symmetrically_across"] == ["policy_objective", "flow_nfe", "execution_horizon", "training_seed"],
        "contamination symmetry factors changed",
    )
    excluded = rule["excluded_cells"]
    require(isinstance(excluded, list) and len(excluded) == 1, "contamination ledger must exclude exactly one cell")
    cell = excluded[0]
    require(isinstance(cell, dict), "contamination excluded cell must be an object")
    _require_exact_keys(cell, {"init_state_id", "suite", "task_id", "task_name"}, "contamination excluded cell")
    require(
        cell
        == {
            "init_state_id": 0,
            "suite": "libero_goal",
            "task_id": 7,
            "task_name": "turn_on_the_stove",
        },
        "contamination excluded episode changed",
    )
    require(
        rule["official_2000_episode_result_label"] == "non_blind_full_set_comparability_only",
        "contamination full-set label changed",
    )
    return {
        "excluded_episodes": [dict(OFFICIAL_EXCLUDED_EPISODE)],
        "full_episode_count": OFFICIAL_FULL_EPISODES,
        "ledger_schema": CONTAMINATION_SCHEMA,
        "ledger_sha256": digest,
        "primary_episode_count": OFFICIAL_PRIMARY_EPISODES,
        "report_label": "primary_clean_holdout",
    }


def official_episode_matrix(contamination: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the canonical 1,999-episode primary matrix in published order."""

    _require_exact_keys(contamination, _CONTAMINATION_FIELDS, "contamination contract")
    excluded = {(item["suite"], item["task_id"], item["reset_id"]) for item in contamination["excluded_episodes"]}
    episodes = [
        {"reset_id": reset_id, "suite": suite, "task_id": task_id}
        for suite in SUITES
        for task_id in range(10)
        for reset_id in range(OFFICIAL_RESETS_PER_TASK)
        if (suite, task_id, reset_id) not in excluded
    ]
    require(len(episodes) == OFFICIAL_PRIMARY_EPISODES, "canonical LIBERO primary episode count changed")
    require(
        len(episodes) + len(excluded) == OFFICIAL_FULL_EPISODES,
        "canonical LIBERO full episode count changed",
    )
    return episodes


def official_cell_id(train_seed: int, objective: str, nfe: int, execution_horizon: int) -> str:
    label = "flow" if objective == "rectified_flow" else "direct"
    return f"seed-{train_seed}-{label}-nfe-{nfe}-k-{execution_horizon}"


def _official_factor_matrix() -> set[tuple[int, str, int, int]]:
    return {
        (seed, objective, nfe, execution_horizon)
        for seed in OFFICIAL_TRAIN_SEEDS
        for objective, nfes in (("rectified_flow", OFFICIAL_FLOW_NFES), ("direct_regression", (1,)))
        for nfe in nfes
        for execution_horizon in OFFICIAL_EXECUTION_HORIZONS
    }


def _validate_registered_cell(cell: Any, *, episode_matrix_sha256: str) -> dict[str, Any]:
    require(isinstance(cell, dict), "pre-registration cell must be an object")
    _require_exact_keys(cell, _CELL_FIELDS, "pre-registration cell")
    checkpoint = cell["checkpoint"]
    require(isinstance(checkpoint, dict), "pre-registration checkpoint identity must be an object")
    _require_exact_keys(checkpoint, _CHECKPOINT_FIELDS, "pre-registration checkpoint identity")
    _require_sha256(checkpoint["manifest_sha256"], "checkpoint manifest_sha256")
    _require_sha256(checkpoint["source_tree_sha256"], "checkpoint source_tree_sha256")
    require(
        type(checkpoint["update"]) is int and checkpoint["update"] == FINAL_CHECKPOINT_UPDATE,
        f"official checkpoint update must equal {FINAL_CHECKPOINT_UPDATE}",
    )
    _require_sha256(cell["policy_contract_sha256"], "policy_contract_sha256")
    _require_sha256(cell["serving_policy_sha256"], "serving_policy_sha256")
    _require_sha256(cell["serving_runtime_sha256"], "serving_runtime_sha256")
    _require_sha256(cell["episode_matrix_sha256"], "episode_matrix_sha256")
    require(
        hmac.compare_digest(cell["episode_matrix_sha256"], episode_matrix_sha256),
        "cell episode matrix differs from the frozen primary matrix",
    )
    require(cell["execution_geometry"] == LIBERO_EXECUTION_GEOMETRY, "cell execution geometry mismatch")
    train_seed = cell["train_seed"]
    objective = cell["objective"]
    nfe = cell["nfe"]
    execution_horizon = cell["execution_horizon"]
    require(type(train_seed) is int and train_seed in OFFICIAL_TRAIN_SEEDS, "cell train_seed must be one of {0,1,2}")
    require(
        type(execution_horizon) is int and execution_horizon in OFFICIAL_EXECUTION_HORIZONS,
        "cell execution_horizon must be one of {1,4}",
    )
    require(
        type(cell["policy_warmup_calls"]) is int and cell["policy_warmup_calls"] == OFFICIAL_POLICY_WARMUP_CALLS,
        f"official policy warm-up count must equal {OFFICIAL_POLICY_WARMUP_CALLS}",
    )
    if objective == "rectified_flow":
        require(type(nfe) is int and nfe in OFFICIAL_FLOW_NFES, "flow cell NFE must be one of {1,5,10}")
        require(cell["sampler"] == "euler_uniform", "flow cell sampler mismatch")
        require(
            cell["inference_seed_behavior"] == "episode_identity_gaussian_noise",
            "flow cell inference seed behavior mismatch",
        )
    elif objective == "direct_regression":
        require(type(nfe) is int and nfe == 1, "direct cell NFE must equal one")
        require(cell["sampler"] == "single_forward", "direct cell sampler mismatch")
        require(
            cell["inference_seed_behavior"] == "episode_identity_echo_only",
            "direct cell inference seed behavior mismatch",
        )
    else:
        raise RuntimeError(f"unsupported pre-registration objective {objective!r}")
    selected_policy = {name: cell[name] for name in ("inference_seed_behavior", "nfe", "objective", "sampler")}
    require(
        hmac.compare_digest(cell["serving_policy_sha256"], canonical_sha256(selected_policy)),
        "selected serving-policy SHA-256 mismatch",
    )
    require(
        cell["cell_id"] == official_cell_id(train_seed, objective, nfe, execution_horizon),
        "pre-registration cell_id is not canonical",
    )
    return dict(cell)


def validate_preregistration_manifest(
    manifest: Any,
    *,
    contamination: Mapping[str, Any],
    simulator_attestation_sha256: str,
) -> list[dict[str, Any]]:
    require(isinstance(manifest, dict), "pre-registration manifest must be an object")
    _require_exact_keys(manifest, _PREREGISTRATION_FIELDS, "pre-registration manifest")
    require(manifest["schema"] == PREREGISTRATION_SCHEMA, "pre-registration schema mismatch")
    require(manifest["benchmark_protocol"] == PROTOCOL, "pre-registration protocol mismatch")
    require(manifest["training_seeds"] == list(OFFICIAL_TRAIN_SEEDS), "pre-registration training seeds changed")
    require(
        manifest["execution_horizons"] == list(OFFICIAL_EXECUTION_HORIZONS),
        "pre-registration execution horizons changed",
    )
    require(manifest["suites"] == list(SUITES), "pre-registration suite order changed")
    require(manifest["task_ids"] == list(range(10)), "pre-registration task set changed")
    require(
        manifest["official_resets_per_task"] == OFFICIAL_RESETS_PER_TASK,
        "pre-registration official reset count changed",
    )
    require(
        manifest["final_checkpoint_update"] == FINAL_CHECKPOINT_UPDATE,
        "pre-registration final checkpoint update changed",
    )
    require(
        type(manifest["evaluation_seed"]) is int and 0 <= manifest["evaluation_seed"] < 2**63,
        "pre-registration evaluation_seed is invalid",
    )
    _require_sha256(manifest["final_freeze_token_sha256"], "final_freeze_token_sha256")
    _require_sha256(manifest["simulator_attestation_sha256"], "simulator_attestation_sha256")
    _require_sha256(simulator_attestation_sha256, "current simulator attestation SHA-256")
    require(
        hmac.compare_digest(manifest["simulator_attestation_sha256"], simulator_attestation_sha256),
        "current simulator attestation differs from pre-registration",
    )
    require(manifest["contamination"] == dict(contamination), "pre-registration contamination contract changed")
    episodes = official_episode_matrix(contamination)
    require(manifest["episode_count"] == OFFICIAL_PRIMARY_EPISODES, "pre-registration episode count changed")
    require(manifest["episodes"] == episodes, "pre-registration episode matrix changed")
    episode_sha256 = canonical_sha256(episodes)
    _require_sha256(manifest["episode_matrix_sha256"], "pre-registration episode_matrix_sha256")
    require(
        hmac.compare_digest(manifest["episode_matrix_sha256"], episode_sha256),
        "pre-registration episode matrix SHA-256 mismatch",
    )
    cells = manifest["cells"]
    require(isinstance(cells, list), "pre-registration cells must be a list")
    checked = [_validate_registered_cell(cell, episode_matrix_sha256=episode_sha256) for cell in cells]
    ids = [cell["cell_id"] for cell in checked]
    require(len(ids) == len(set(ids)), "pre-registration cell_id values must be unique")
    factors = {(cell["train_seed"], cell["objective"], cell["nfe"], cell["execution_horizon"]) for cell in checked}
    require(factors == _official_factor_matrix(), "pre-registration does not contain the exact 24-cell factor matrix")
    require(len(checked) == len(factors) == 24, "pre-registration must contain exactly 24 policy cells")

    for seed in OFFICIAL_TRAIN_SEEDS:
        for objective in ("rectified_flow", "direct_regression"):
            group = [cell for cell in checked if cell["train_seed"] == seed and cell["objective"] == objective]
            checkpoint_identities = {canonical_json_bytes(cell["checkpoint"]) for cell in group}
            contract_ids = {cell["policy_contract_sha256"] for cell in group}
            runtime_ids = {cell["serving_runtime_sha256"] for cell in group}
            require(len(checkpoint_identities) == 1, "NFE/K cells must share one final checkpoint per seed/objective")
            require(len(contract_ids) == 1, "NFE/K cells must share one policy contract per seed/objective")
            require(len(runtime_ids) == 1, "NFE/K cells must share one serving runtime per seed/objective")
        flow_manifest = next(
            cell["checkpoint"]["manifest_sha256"]
            for cell in checked
            if cell["train_seed"] == seed and cell["objective"] == "rectified_flow"
        )
        direct_manifest = next(
            cell["checkpoint"]["manifest_sha256"]
            for cell in checked
            if cell["train_seed"] == seed and cell["objective"] == "direct_regression"
        )
        require(flow_manifest != direct_manifest, "flow and direct cells cannot share a checkpoint manifest")
    require(
        len({cell["checkpoint"]["source_tree_sha256"] for cell in checked}) == 1,
        "all official checkpoints must share one qualified source tree",
    )
    require(
        len({cell["checkpoint"]["manifest_sha256"] for cell in checked}) == 6,
        "official matrix must contain one distinct final checkpoint per seed/objective",
    )
    contract_ids_by_objective = {
        objective: {cell["policy_contract_sha256"] for cell in checked if cell["objective"] == objective}
        for objective in ("rectified_flow", "direct_regression")
    }
    require(
        all(len(values) == 1 for values in contract_ids_by_objective.values()),
        "each objective must share one policy contract across all training seeds",
    )
    require(
        len(set.union(*contract_ids_by_objective.values())) == 2,
        "flow and direct objectives must have distinct policy contracts",
    )
    return checked


def load_preregistration(
    path: Path,
    *,
    cell_id: str,
    execution_horizon: int,
    evaluation_seed: int,
    final_freeze_token: str,
    preregistration_sha256: str,
    simulator_attestation_sha256: str,
    contamination: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest, manifest_sha256 = _read_strict_json(
        path,
        name="LIBERO pre-registration manifest",
        expected_sha256=preregistration_sha256,
    )
    cells = validate_preregistration_manifest(
        manifest,
        contamination=contamination,
        simulator_attestation_sha256=simulator_attestation_sha256,
    )
    require(manifest["evaluation_seed"] == evaluation_seed, "CLI evaluation seed differs from pre-registration")
    require(isinstance(final_freeze_token, str) and bool(final_freeze_token.strip()), "final freeze token is empty")
    token_sha256 = hashlib.sha256(final_freeze_token.encode("utf-8")).hexdigest()
    require(
        hmac.compare_digest(token_sha256, manifest["final_freeze_token_sha256"]),
        "final freeze token does not match pre-registration",
    )
    matching = [cell for cell in cells if cell["cell_id"] == cell_id]
    require(len(matching) == 1, "selected --cell-id is absent from pre-registration")
    selected = matching[0]
    require(
        selected["execution_horizon"] == execution_horizon,
        "CLI execution horizon differs from the pre-registered cell",
    )
    return dict(manifest), selected, manifest_sha256


def load_development_reset_bank(bank_dir: Path) -> DevelopmentResetBank:
    manifest, artifacts = load_bank(bank_dir)
    runtime_manifest_path = (
        Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")) / "simulators/libero/manifest.json"
    )
    try:
        runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot authenticate current LIBERO runtime: {runtime_manifest_path}") from exc
    expected_simulator = {
        "assets_revision": runtime_manifest["assets"]["revision"],
        "egl_device": str(int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"))),
        "packages": runtime_manifest["environment"]["packages"],
        "python": runtime_manifest["environment"]["python"],
        "schema": runtime_manifest["schema"],
        "source_revision": runtime_manifest["source"]["revision"],
    }
    require(manifest["simulator"] == expected_simulator, "development reset bank simulator identity mismatch")
    states_by_task: dict[tuple[str, int], np.ndarray] = {}
    task_records: dict[tuple[str, int], dict[str, Any]] = {}
    for record in manifest["tasks"]:
        identity = (record["suite"], record["task_id"])
        path = record["artifact"]["path"]
        require(path in artifacts, f"development reset bank is missing {path}")
        states_by_task[identity] = artifacts[path]
        task_records[identity] = record
    manifest_path = bank_dir / "manifest.json"
    return DevelopmentResetBank(
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        states_by_task=states_by_task,
        task_records=task_records,
    )


def rotate_eval_rgb_180(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    require(values.shape == IMAGE_SHAPE, f"camera image must have shape {IMAGE_SHAPE}, got {values.shape}")
    require(values.dtype == np.uint8, f"camera image must have uint8 dtype, got {values.dtype}")
    return np.flip(values, axis=(0, 1)).copy()


def libero_state(observation: dict[str, Any]) -> np.ndarray:
    from robosuite.utils.transform_utils import quat2axisangle

    position = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
    quaternion = np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy()
    gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
    require(position.shape == (3,), f"unexpected EEF position shape: {position.shape}")
    require(quaternion.shape == (4,), f"unexpected EEF quaternion shape: {quaternion.shape}")
    require(gripper.shape == (2,), f"unexpected gripper shape: {gripper.shape}")
    axis_angle = np.asarray(quat2axisangle(quaternion), dtype=np.float32)
    state = np.concatenate((position, axis_angle, gripper)).astype(np.float32, copy=False)
    require(state.shape == (8,) and bool(np.isfinite(state).all()), "constructed LIBERO state is invalid")
    return state.copy()


def libero_env_action(action: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    require(
        values.shape == (ACTION_DIM,) and bool(np.isfinite(values).all()),
        "policy action must contain seven finite values",
    )
    clip_count = int(np.count_nonzero(np.abs(values[:6]) > 1.0))
    output = values.copy()
    output[:6] = np.clip(output[:6], -1.0, 1.0)
    output[6] = 1.0 if output[6] >= 0 else -1.0
    return output, clip_count


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def wilson95(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    require(0 <= successes <= trials, "invalid success counts")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def parse_selection(text: str, *, upper: int, name: str) -> tuple[int, ...]:
    if text == "all":
        return tuple(range(upper))
    selected: set[int] = set()
    for component in text.split(","):
        component = component.strip()
        if not component:
            raise ValueError(f"empty component in {name}")
        if "-" in component:
            fields = component.split("-")
            if len(fields) != 2:
                raise ValueError(f"invalid range {component!r} in {name}")
            start, stop = map(int, fields)
            if stop < start:
                raise ValueError(f"descending range {component!r} in {name}")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(component))
    if not selected or min(selected) < 0 or max(selected) >= upper:
        raise ValueError(f"{name} must select values in [0, {upper})")
    return tuple(sorted(selected))


def run_exact_simulator_preflight(project_root: Path, *, construct_environment: bool) -> dict[str, Any]:
    script = project_root / "scripts/preflight_libero_env.py"
    command = [sys.executable, str(script)]
    if not construct_environment:
        command.append("--imports-only")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"exact LIBERO simulator preflight failed ({completed.returncode}):\n{completed.stdout}{completed.stderr}"
        )
    lines = completed.stdout.splitlines()
    json_start = next((index for index, line in enumerate(lines) if line.startswith("{")), None)
    require(json_start is not None, f"simulator preflight did not emit JSON: {completed.stdout}")
    report = json.loads("\n".join(lines[json_start:]))
    require(report.get("status") == "ok", "simulator preflight did not report success")
    require(report.get("schema") == "duo-vla-libero-simulator-attestation-v2", "simulator attestation schema mismatch")
    require(
        report.get("environment_constructed") is construct_environment,
        "simulator preflight environment-construction status mismatch",
    )
    return report


def validate_policy_health(health: dict[str, Any], *, allow_fake_policy: bool) -> None:
    mode = health.get("mode")
    require(mode in {"fake", "real"}, "policy server reported an unknown mode")
    require(health.get("prefix_cache_scope") == "request", "policy prefix cache scope mismatch")
    if mode == "fake":
        require(allow_fake_policy, "refusing test-only fake policy without --allow-fake-policy")
        for name in (
            "checkpoint",
            "execution_geometry",
            "model_revision",
            "normalization_content_sha256",
            "serving_runtime_sha256",
        ):
            require(health.get(name) is None, f"fake policy cannot claim {name}")
        return
    require(health.get("model_revision") == MODEL_REVISION, "policy model revision mismatch")
    require(health.get("dataset_revision") == DATASET_REVISION, "policy dataset revision mismatch")
    require(
        health.get("normalization_content_sha256") == NORMALIZATION_SHA256,
        "policy normalization content hash mismatch",
    )
    serving_runtime_sha256 = health.get("serving_runtime_sha256")
    require(
        isinstance(serving_runtime_sha256, str)
        and len(serving_runtime_sha256) == 64
        and all(character in "0123456789abcdef" for character in serving_runtime_sha256),
        "policy serving runtime hash is invalid",
    )
    checkpoint = health.get("checkpoint")
    require(isinstance(checkpoint, dict), "real policy health has no checkpoint identity")
    require(checkpoint.get("kind") == "resumable-libero-training", "real policy checkpoint kind mismatch")
    require(health.get("train_seed") in OFFICIAL_TRAIN_SEEDS, "real policy train seed must be one of {0,1,2}")
    require(checkpoint.get("train_seed") == health.get("train_seed"), "checkpoint and policy train seeds disagree")
    execution_geometry = validate_execution_geometry(health.get("execution_geometry"))
    require(
        checkpoint.get("execution_geometry") == execution_geometry == LIBERO_EXECUTION_GEOMETRY,
        "checkpoint and live execution geometry disagree",
    )
    require(
        isinstance(checkpoint.get("manifest_sha256"), str) and len(checkpoint["manifest_sha256"]) == 64,
        "policy checkpoint manifest hash is invalid",
    )
    require(
        isinstance(checkpoint.get("policy_contract_sha256"), str) and len(checkpoint["policy_contract_sha256"]) == 64,
        "policy checkpoint contract hash is invalid",
    )
    checkpoint_contract = checkpoint.get("policy_contract")
    require(isinstance(checkpoint_contract, dict), "real policy checkpoint has no policy contract")
    wire_contract = {name: health[name] for name in ("objective", "sampler", "nfe", "inference_seed_behavior")}
    for name in ("objective", "sampler", "inference_seed_behavior"):
        require(checkpoint_contract.get(name) == wire_contract[name], "checkpoint and policy-server contracts disagree")
    if wire_contract["objective"] == "rectified_flow":
        require(wire_contract["nfe"] in {1, 5, 10}, "unsupported rectified-flow serving NFE")
    else:
        require(checkpoint_contract.get("nfe") == wire_contract["nfe"], "direct-regression NFE was overridden")


def _authenticate_final_checkpoint(
    health: Mapping[str, Any],
    selected_cell: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = health["checkpoint"]
    registered = selected_cell["checkpoint"]
    require(isinstance(checkpoint, Mapping), "policy health checkpoint identity must be an object")
    checkpoint_dir = Path(checkpoint["path"]).resolve()
    require(checkpoint_dir.parent.name == "checkpoints", "checkpoint is not inside a canonical checkpoints directory")
    run_root = checkpoint_dir.parent.parent
    record = validate_resume_checkpoint(run_root, checkpoint_dir)
    require(record.update == registered["update"], "run-journal checkpoint update differs from pre-registration")
    require(
        hmac.compare_digest(record.manifest_sha256, registered["manifest_sha256"]),
        "run-journal checkpoint manifest differs from pre-registration",
    )
    manifest, manifest_sha256 = _read_strict_json(
        checkpoint_dir / "manifest.json",
        name="final checkpoint manifest",
        expected_sha256=registered["manifest_sha256"],
    )
    require(manifest.get("schema") == "duo-vla-checkpoint-v1", "final checkpoint schema mismatch")
    require(manifest.get("kind") == "resumable-libero-training", "final checkpoint kind mismatch")
    require(manifest.get("run_seed") == selected_cell["train_seed"], "final checkpoint train seed mismatch")
    require(
        manifest.get("source_tree_sha256") == registered["source_tree_sha256"],
        "final checkpoint source tree differs from pre-registration",
    )
    require(
        manifest.get("policy_contract_sha256") == selected_cell["policy_contract_sha256"],
        "final checkpoint policy contract differs from pre-registration",
    )
    trainer_state = manifest.get("trainer_state")
    require(isinstance(trainer_state, dict), "final checkpoint has no trainer state")
    require(
        trainer_state.get("next_update") == registered["update"] == FINAL_CHECKPOINT_UPDATE,
        "final checkpoint trainer update mismatch",
    )
    last_metrics = manifest.get("last_metrics")
    require(
        isinstance(last_metrics, dict) and last_metrics.get("update") == registered["update"],
        "final checkpoint metric update mismatch",
    )
    manifest_contract = manifest.get("policy_contract")
    require(isinstance(manifest_contract, dict), "final checkpoint has no policy contract")
    for name in ("inference_seed_behavior", "objective", "sampler"):
        require(
            manifest_contract.get(name) == selected_cell[name],
            f"final checkpoint {name} differs from pre-registration",
        )
    expected_checkpoint_nfe = 1 if selected_cell["objective"] == "direct_regression" else OFFICIAL_FLOW_CHECKPOINT_NFE
    require(
        manifest_contract.get("nfe") == expected_checkpoint_nfe,
        f"final checkpoint NFE must equal {expected_checkpoint_nfe} for {selected_cell['objective']}",
    )
    return {
        "manifest_sha256": manifest_sha256,
        "path": str(checkpoint_dir),
        "run_journal_latest": True,
        "run_root": str(run_root),
        "source_tree_sha256": manifest["source_tree_sha256"],
        "update": record.update,
    }


def validate_official_policy_health(
    health: dict[str, Any],
    *,
    selected_cell: Mapping[str, Any],
    execution_horizon: int,
) -> dict[str, Any]:
    """Bind live policy/runtime/checkpoint identity to one frozen official cell."""

    validate_policy_health(health, allow_fake_policy=False)
    require(health["mode"] == "real", "official-score refuses fake policy health")
    require(execution_horizon == selected_cell["execution_horizon"], "official execution horizon changed")
    require(health["train_seed"] == selected_cell["train_seed"], "live train seed differs from pre-registration")
    checkpoint = health["checkpoint"]
    require(
        hmac.compare_digest(checkpoint["manifest_sha256"], selected_cell["checkpoint"]["manifest_sha256"]),
        "live checkpoint manifest differs from pre-registration",
    )
    require(
        hmac.compare_digest(checkpoint["source_tree_sha256"], selected_cell["checkpoint"]["source_tree_sha256"]),
        "live checkpoint source tree differs from pre-registration",
    )
    require(
        hmac.compare_digest(checkpoint["policy_contract_sha256"], selected_cell["policy_contract_sha256"]),
        "live checkpoint policy contract differs from pre-registration",
    )
    require(
        hmac.compare_digest(health["serving_runtime_sha256"], selected_cell["serving_runtime_sha256"]),
        "live serving runtime differs from pre-registration",
    )
    require(
        health["execution_geometry"] == selected_cell["execution_geometry"] == LIBERO_EXECUTION_GEOMETRY,
        "live execution geometry differs from pre-registration",
    )
    selected_policy = {name: health[name] for name in ("inference_seed_behavior", "nfe", "objective", "sampler")}
    registered_policy = {
        name: selected_cell[name] for name in ("inference_seed_behavior", "nfe", "objective", "sampler")
    }
    require(selected_policy == registered_policy, "live serving policy differs from pre-registration")
    require(
        hmac.compare_digest(canonical_sha256(selected_policy), selected_cell["serving_policy_sha256"]),
        "live serving-policy SHA-256 differs from pre-registration",
    )
    return _authenticate_final_checkpoint(health, selected_cell)


def run_episode(
    environment: Any,
    *,
    initial_state: np.ndarray,
    client: PolicyClient,
    train_seed: int,
    evaluation_seed: int,
    suite: str,
    task_id: int,
    task_name: str,
    instruction: str,
    init_state_id: int,
    execution_horizon: int,
    policy_budget: int,
    reset_source: str = "official",
    reset_state_sha256: str | None = None,
    environment_seed: int = ENVIRONMENT_SEED,
    expected_settled_state_sha256: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    require(reset_source in {"official", "clean-dev"}, "unknown LIBERO reset source")
    if reset_source == "official":
        require(reset_state_sha256 is None, "official reset must not carry a state hash")
        require(expected_settled_state_sha256 is None, "official reset must not carry a settled-state hash")
    else:
        require(reset_state_sha256 is not None, "clean-dev reset requires a state hash")
        require(expected_settled_state_sha256 is not None, "clean-dev reset requires a settled-state hash")
    environment.seed(environment_seed)
    environment.reset()
    observation = environment.set_init_state(initial_state)
    action_queue: deque[np.ndarray] = deque()
    policy_latencies: list[float] = []
    server_latencies: list[float] = []
    normalized_clip_fractions: list[float] = []
    policy_steps = 0
    policy_calls = 0
    settle_executed = 0
    environment_clip_count = 0
    success = False
    simulator_done = False

    while settle_executed < SETTLE_STEPS and not success:
        observation, _, done, _ = environment.step(OPEN_GRIPPER_NOOP.copy())
        simulator_done = simulator_done or bool(done)
        settle_executed += 1
        success = bool(environment.check_success())

    if expected_settled_state_sha256 is not None:
        observed_settled_hash = state_sha256(environment.get_sim_state())
        require(
            observed_settled_hash == expected_settled_state_sha256,
            "clean-dev mandatory settle trajectory differs from the authenticated bank",
        )

    while policy_steps < policy_budget and not success:
        if not action_queue:
            agentview = rotate_eval_rgb_180(np.asarray(observation[CAMERA_KEYS[0]]))
            wrist = rotate_eval_rgb_180(np.asarray(observation[CAMERA_KEYS[1]]))
            state = libero_state(observation)
            request_started = time.perf_counter()
            actions, response = client.predict(
                suite=suite,
                task_id=task_id,
                reset_source=reset_source,
                reset_id=init_state_id,
                reset_state_sha256=reset_state_sha256,
                replan_id=policy_calls,
                execution_horizon=execution_horizon,
                train_seed=train_seed,
                evaluation_seed=evaluation_seed,
                instruction=instruction,
                agentview_rgb=agentview,
                wrist_rgb=wrist,
                state=state,
            )
            policy_latencies.append(time.perf_counter() - request_started)
            require(response["evaluation_seed"] == evaluation_seed, "policy response evaluation seed drifted")
            require(response["reset_source"] == reset_source, "policy response reset source drifted")
            require(response["reset_id"] == init_state_id, "policy response reset id drifted")
            require(
                response["reset_state_sha256"] == reset_state_sha256,
                "policy response reset state hash drifted",
            )
            server_latencies.append(float(response["policy_seconds"]))
            normalized_clip_fractions.append(float(response["normalized_clip_fraction"]))
            action_queue.extend(action.copy() for action in actions[:execution_horizon])
            policy_calls += 1

        raw_action = action_queue.popleft()
        action, clip_count = libero_env_action(raw_action)
        environment_clip_count += clip_count
        observation, _, done, _ = environment.step(action)
        simulator_done = simulator_done or bool(done)
        policy_steps += 1
        success = bool(environment.check_success())
        if success:
            action_queue.clear()

    successful_steps = policy_steps if success else None
    return {
        "action_clipped_channels": environment_clip_count,
        "action_clip_fraction": environment_clip_count / (policy_steps * 6) if policy_steps else 0.0,
        "action_continuous_channels": policy_steps * 6,
        "elapsed_seconds": time.perf_counter() - started,
        "environment_seed": environment_seed,
        "evaluation_seed": evaluation_seed,
        "execution_horizon": execution_horizon,
        "init_state_id": init_state_id if reset_source == "official" else None,
        "normalized_action_clip_fraction": (
            statistics.fmean(normalized_clip_fractions) if normalized_clip_fractions else 0.0
        ),
        "policy_budget": policy_budget,
        "policy_calls": policy_calls,
        "policy_latency_p50_seconds": percentile(policy_latencies, 50),
        "policy_latency_p95_seconds": percentile(policy_latencies, 95),
        "policy_latency_seconds": policy_latencies,
        "policy_steps": policy_steps,
        "reset_id": init_state_id,
        "reset_source": reset_source,
        "reset_state_sha256": reset_state_sha256,
        "server_latency_p50_seconds": percentile(server_latencies, 50),
        "server_latency_p95_seconds": percentile(server_latencies, 95),
        "server_latency_seconds": server_latencies,
        "settle_steps": settle_executed,
        "simulator_done": bool(simulator_done),
        "steps_to_success": successful_steps,
        "success": success,
        "suite": suite,
        "task_id": task_id,
        "task_name": task_name,
    }


def summarize_episodes(episodes: list[dict[str, Any]], *, execution_horizon: int) -> dict[str, Any]:
    require(episodes, "cannot summarize an empty rollout")
    tasks: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for episode in episodes:
        tasks.setdefault((episode["suite"], episode["task_id"]), []).append(episode)
    task_metrics: list[dict[str, Any]] = []
    for (suite, task_id), values in sorted(tasks.items()):
        successes = sum(bool(value["success"]) for value in values)
        task_metrics.append(
            {
                "episodes": len(values),
                "success_rate": successes / len(values),
                "successes": successes,
                "suite": suite,
                "task_id": task_id,
                "task_name": values[0]["task_name"],
                "wilson95": wilson95(successes, len(values)),
            }
        )

    suites: list[dict[str, Any]] = []
    for suite in SUITES:
        suite_tasks = [value for value in task_metrics if value["suite"] == suite]
        if not suite_tasks:
            continue
        suite_episodes = [value for value in episodes if value["suite"] == suite]
        successes = sum(bool(value["success"]) for value in suite_episodes)
        suites.append(
            {
                "episodes": len(suite_episodes),
                "pooled_success_rate": successes / len(suite_episodes),
                "pooled_wilson95": wilson95(successes, len(suite_episodes)),
                "suite": suite,
                "suite_task_macro_success": statistics.fmean(value["success_rate"] for value in suite_tasks),
                "successes": successes,
                "tasks": len(suite_tasks),
            }
        )

    successful_steps = [float(value["steps_to_success"]) for value in episodes if value["steps_to_success"] is not None]
    latency_values = [float(latency) for value in episodes for latency in value["policy_latency_seconds"]]
    server_latency_values = [float(latency) for value in episodes for latency in value["server_latency_seconds"]]
    clipped_channels = sum(int(value["action_clipped_channels"]) for value in episodes)
    continuous_channels = sum(int(value["action_continuous_channels"]) for value in episodes)
    policy_calls = sum(int(value["policy_calls"]) for value in episodes)
    complete_40_task_macro = len(task_metrics) == 40 and {
        (item["suite"], item["task_id"]) for item in task_metrics
    } == {(suite, task_id) for suite in SUITES for task_id in range(10)}
    successes = sum(bool(value["success"]) for value in episodes)
    return {
        "action_clip_fraction": clipped_channels / continuous_channels if continuous_channels else 0.0,
        "complete_40_task_macro": complete_40_task_macro,
        "episodes": len(episodes),
        "execution_horizon": execution_horizon,
        "normalized_action_clip_fraction": (
            sum(float(value["normalized_action_clip_fraction"]) * int(value["policy_calls"]) for value in episodes)
            / policy_calls
            if policy_calls
            else 0.0
        ),
        "overall_40_task_macro_success": (
            statistics.fmean(value["success_rate"] for value in task_metrics) if complete_40_task_macro else None
        ),
        "overall_pooled_success_rate": successes / len(episodes),
        "overall_pooled_wilson95": wilson95(successes, len(episodes)),
        "policy_calls": policy_calls,
        "policy_latency_p50_seconds": percentile(latency_values, 50),
        "policy_latency_p95_seconds": percentile(latency_values, 95),
        "server_latency_p50_seconds": percentile(server_latency_values, 50),
        "server_latency_p95_seconds": percentile(server_latency_values, 95),
        "steps_to_success_mean": statistics.fmean(successful_steps) if successful_steps else None,
        "steps_to_success_p50": percentile(successful_steps, 50),
        "steps_to_success_p95": percentile(successful_steps, 95),
        "suites": suites,
        "task_metrics": task_metrics,
    }


def bind_official_summary(
    summary: dict[str, Any],
    *,
    contamination: Mapping[str, Any],
    episode_matrix_sha256: str,
) -> dict[str, Any]:
    require(summary["complete_40_task_macro"] is True, "official summary does not contain all 40 tasks")
    require(summary["episodes"] == OFFICIAL_PRIMARY_EPISODES, "official summary denominator is not 1,999")
    task_counts = {(item["suite"], item["task_id"]): item["episodes"] for item in summary["task_metrics"]}
    expected_counts = {(suite, task_id): OFFICIAL_RESETS_PER_TASK for suite in SUITES for task_id in range(10)}
    expected_counts[("libero_goal", 7)] = OFFICIAL_RESETS_PER_TASK - 1
    require(task_counts == expected_counts, "official task denominators do not implement the contamination exclusion")
    result = dict(summary)
    result["reporting"] = {
        "contamination_ledger_sha256": contamination["ledger_sha256"],
        "denominator": OFFICIAL_PRIMARY_EPISODES,
        "episode_matrix_sha256": episode_matrix_sha256,
        "excluded_episode_count": 1,
        "excluded_episodes": contamination["excluded_episodes"],
        "full_official_episode_count": OFFICIAL_FULL_EPISODES,
        "label": contamination["report_label"],
        "non_blind_full_set_reported": False,
    }
    result["schema"] = OFFICIAL_SUMMARY_SCHEMA
    return result


def _construct_environment(task: Any) -> Any:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=256,
        camera_widths=256,
        camera_names=list(CAMERA_NAMES),
        control_freq=20,
        render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
    )


def _development_task_resets(
    bank: DevelopmentResetBank,
    *,
    suite_name: str,
    task_id: int,
    task: Any,
    official_states: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from libero.libero import get_libero_path

    identity = (suite_name, task_id)
    require(identity in bank.task_records, f"development reset bank has no task {suite_name}:{task_id}")
    record = bank.task_records[identity]
    require(record["task_name"] == task.name, "development reset task name changed")
    require(record["instruction"] == task.language, "development reset task instruction changed")
    bddl = record["bddl"]
    require(bddl.get("file") == task.bddl_file, "development reset BDDL filename changed")
    require(bddl.get("problem_folder") == task.problem_folder, "development reset BDDL folder changed")
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    require(sha256_file(bddl_path) == bddl.get("sha256"), "development reset BDDL SHA-256 changed")
    state_size, official_hashes = canonical_official_state_hashes(official_states)
    official = record["official_states"]
    require(official["state_size"] == state_size, "development reset official-state size changed")
    require(official["sha256"] == list(official_hashes), "development reset official-state hashes changed")
    require(
        official["root_sha256"] == sequence_root_sha256(official_hashes),
        "development reset official-state root changed",
    )
    states = bank.states_by_task[identity]
    entries = record["entries"]
    require(len(states) == len(entries) == bank.manifest["states_per_task"], "development reset rows changed")
    return states, entries


def run_rollouts(
    *,
    client: PolicyClient,
    train_seed: int,
    evaluation_seed: int,
    suites: tuple[str, ...],
    task_ids: tuple[int, ...],
    init_state_ids: tuple[int, ...],
    execution_horizon: int,
    episode_sink: Any,
    reset_source: str = "official",
    development_bank: DevelopmentResetBank | None = None,
    official_episode_identities: frozenset[tuple[str, int, int]] | None = None,
) -> list[dict[str, Any]]:
    from libero.libero import benchmark

    benchmark_map = benchmark.get_benchmark_dict()
    if official_episode_identities is not None:
        require(reset_source == "official", "an official episode matrix cannot select development resets")
    episodes: list[dict[str, Any]] = []
    for suite_name in suites:
        suite = benchmark_map[suite_name]()
        require(suite.n_tasks == 10, f"{suite_name} task count changed")
        for task_id in task_ids:
            task = suite.get_task(task_id)
            official_states = np.asarray(suite.get_task_init_states(task_id))
            require(len(official_states) == 50, f"{suite_name} task {task_id} fixed-state count changed")
            if reset_source == "official":
                require(development_bank is None, "official rollouts must not use a development bank")
                states = official_states
                entries: list[dict[str, Any] | None] = [None] * len(states)
            else:
                require(development_bank is not None, "clean-dev rollouts require a development bank")
                states, clean_entries = _development_task_resets(
                    development_bank,
                    suite_name=suite_name,
                    task_id=task_id,
                    task=task,
                    official_states=official_states,
                )
                entries = list(clean_entries)
            environment = _construct_environment(task)
            try:
                for init_state_id in init_state_ids:
                    require(init_state_id < len(states), f"reset id {init_state_id} is outside the selected bank")
                    if (
                        official_episode_identities is not None
                        and (suite_name, task_id, init_state_id) not in official_episode_identities
                    ):
                        continue
                    entry = entries[init_state_id]
                    episode = run_episode(
                        environment,
                        initial_state=states[init_state_id],
                        client=client,
                        train_seed=train_seed,
                        evaluation_seed=evaluation_seed,
                        suite=suite_name,
                        task_id=task_id,
                        task_name=task.name,
                        instruction=task.language,
                        init_state_id=init_state_id,
                        execution_horizon=execution_horizon,
                        policy_budget=POLICY_BUDGETS[suite_name],
                        reset_source=reset_source,
                        reset_state_sha256=None if entry is None else entry["state_sha256"],
                        environment_seed=ENVIRONMENT_SEED if entry is None else entry["sampler_seed"],
                        expected_settled_state_sha256=(None if entry is None else entry["settled_state_sha256"]),
                    )
                    episodes.append(episode)
                    episode_sink.write(json.dumps(episode, allow_nan=False, sort_keys=True) + "\n")
                    episode_sink.flush()
                    os.fsync(episode_sink.fileno())
                    print(json.dumps(episode, allow_nan=False, sort_keys=True), flush=True)
            finally:
                environment.close()
    if official_episode_identities is not None:
        observed = {(episode["suite"], episode["task_id"], episode["reset_id"]) for episode in episodes}
        require(
            observed == set(official_episode_identities),
            "official rollout did not consume the exact episode matrix",
        )
        require(
            len(episodes) == len(official_episode_identities),
            "official rollout emitted duplicate episode identities",
        )
    return episodes


def dry_run(
    client: PolicyClient,
    health: dict[str, Any],
    *,
    execution_horizon: int,
    evaluation_seed: int,
) -> dict[str, Any]:
    pixels = np.arange(math.prod(IMAGE_SHAPE), dtype=np.uint32).reshape(IMAGE_SHAPE)
    agentview = (pixels % 251).astype(np.uint8)
    wrist = ((pixels * 7 + 3) % 251).astype(np.uint8)
    actions, response = client.predict(
        suite="libero_spatial",
        task_id=0,
        reset_source="official",
        reset_id=0,
        reset_state_sha256=None,
        replan_id=0,
        execution_horizon=execution_horizon,
        train_seed=health["train_seed"],
        evaluation_seed=evaluation_seed,
        instruction="pick up the black bowl between the plate and the ramekin and place it on the plate",
        agentview_rgb=agentview,
        wrist_rgb=wrist,
        state=np.zeros(8, dtype=np.float32),
    )
    return {
        "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "evaluation_seed": response["evaluation_seed"],
        "execution_horizon": execution_horizon,
        "health": health,
        "inference_seed": response["inference_seed"],
        "inference_seed_behavior": response["inference_seed_behavior"],
        "nfe": response["nfe"],
        "objective": response["objective"],
        "policy_seconds": response["policy_seconds"],
        "reset_id": response["reset_id"],
        "reset_source": response["reset_source"],
        "reset_state_sha256": response["reset_state_sha256"],
        "sampler": response["sampler"],
        "status": "ok",
    }


def run_policy_warmups(
    client: PolicyClient,
    health: dict[str, Any],
    *,
    count: int,
    execution_horizon: int,
    evaluation_seed: int,
) -> list[dict[str, Any]]:
    require(count > 0, "rollout mode requires at least one discarded policy warm-up call")
    reports: list[dict[str, Any]] = []
    for warmup_index in range(count):
        report = dry_run(
            client,
            health,
            execution_horizon=execution_horizon,
            evaluation_seed=evaluation_seed,
        )
        report.pop("health")
        report["warmup_index"] = warmup_index
        reports.append(report)
    return reports


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "official-score"), default="development")
    parser.add_argument(
        "--socket",
        type=Path,
        default=Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")) / "run/libero-policy.sock",
    )
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="libero_spatial")
    parser.add_argument("--task-ids", default="0", help="sorted task selection such as 0,2-4 or all")
    parser.add_argument("--init-state-ids", default="0", help="selected reset indices such as 0,2-9 or all")
    parser.add_argument(
        "--evaluation-seed", type=int, required=True, help="policy-noise seed shared across comparisons"
    )
    parser.add_argument(
        "--reset-source",
        choices=("official", "clean-dev"),
        default="official",
        help="published fixed states or an authenticated non-official development bank",
    )
    parser.add_argument("--dev-state-bank", type=Path, help="required with --reset-source clean-dev")
    parser.add_argument("--execution-horizon", type=int, choices=(1, 4), required=True)
    parser.add_argument("--policy-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--policy-warmup-calls",
        type=int,
        default=2,
        help="synthetic policy calls discarded before rollout latency measurement",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="one synthetic IPC call; do not construct MuJoCo")
    parser.add_argument(
        "--allow-fake-policy", action="store_true", help="explicitly allow test-only fake server health"
    )
    parser.add_argument("--preregistration-manifest", type=Path)
    parser.add_argument("--preregistration-sha256")
    parser.add_argument("--cell-id")
    parser.add_argument(
        "--final-freeze-token",
        help="explicit string whose SHA-256 must match pre-registration; the token is never written",
    )
    return parser.parse_args(argv)


def validate_mode_arguments(args: argparse.Namespace) -> None:
    require(0 <= args.evaluation_seed < 2**63, "evaluation-seed must be in [0, 2^63)")
    require(
        math.isfinite(args.policy_timeout_seconds) and args.policy_timeout_seconds > 0,
        "policy timeout must be positive",
    )
    require(args.policy_warmup_calls >= 0, "policy-warmup-calls must be nonnegative")
    preregistration_values = (
        args.preregistration_manifest,
        args.preregistration_sha256,
        args.cell_id,
        args.final_freeze_token,
    )
    if args.mode == "development":
        require(
            all(value is None for value in preregistration_values),
            "development mode cannot receive a pre-registration",
        )
        if args.dry_run:
            require(args.output_dir is None, "development dry-run does not create rollout artifacts")
        else:
            require(args.reset_source == "clean-dev", "development rollout must use authenticated clean-dev resets")
            require(args.output_dir is not None, "development rollout requires --output-dir")
            require(args.policy_warmup_calls > 0, "development rollout requires at least one warm-up call")
        return
    require(not args.dry_run, "official-score cannot use --dry-run")
    require(not args.allow_fake_policy, "official-score cannot allow a fake policy")
    require(args.reset_source == "official", "official-score requires published official resets")
    require(args.dev_state_bank is None, "official-score cannot receive a development reset bank")
    require(args.suite == "all", "official-score requires --suite all")
    require(args.task_ids == "all", "official-score requires --task-ids all")
    require(args.init_state_ids == "all", "official-score requires --init-state-ids all")
    require(args.output_dir is not None, "official-score requires --output-dir")
    require(
        args.policy_warmup_calls == OFFICIAL_POLICY_WARMUP_CALLS,
        f"official-score requires exactly {OFFICIAL_POLICY_WARMUP_CALLS} policy warm-up calls",
    )
    require(args.preregistration_manifest is not None, "official-score requires --preregistration-manifest")
    _require_sha256(args.preregistration_sha256, "official-score --preregistration-sha256")
    require(isinstance(args.cell_id, str) and bool(args.cell_id), "official-score requires --cell-id")
    require(
        isinstance(args.final_freeze_token, str) and bool(args.final_freeze_token.strip()),
        "official-score requires an explicit --final-freeze-token",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_mode_arguments(args)
    project_root = Path(__file__).resolve().parents[1]
    evaluator_environment = validate_evaluator_process_environment(project_root)
    require(platform.python_version().startswith("3.12."), "LIBERO evaluator requires Python 3.12")

    official = args.mode == "official-score"
    contamination: dict[str, Any] | None = None
    official_episodes: list[dict[str, Any]] | None = None
    selected_cell: dict[str, Any] | None = None
    preregistration_sha256: str | None = None
    if official:
        contamination = load_contamination_contract(project_root)
        simulator_preflight = run_exact_simulator_preflight(project_root, construct_environment=True)
        simulator_attestation_sha256 = canonical_sha256(simulator_preflight)
        official_episodes = official_episode_matrix(contamination)
        assert args.preregistration_manifest is not None
        assert args.preregistration_sha256 is not None
        assert args.cell_id is not None
        assert args.final_freeze_token is not None
        _manifest, selected_cell, preregistration_sha256 = load_preregistration(
            args.preregistration_manifest.resolve(),
            cell_id=args.cell_id,
            execution_horizon=args.execution_horizon,
            evaluation_seed=args.evaluation_seed,
            final_freeze_token=args.final_freeze_token,
            preregistration_sha256=args.preregistration_sha256,
            simulator_attestation_sha256=simulator_attestation_sha256,
            contamination=contamination,
        )
    else:
        simulator_preflight = run_exact_simulator_preflight(
            project_root,
            construct_environment=not args.dry_run,
        )
        simulator_attestation_sha256 = canonical_sha256(simulator_preflight)

    if args.reset_source == "official":
        require(args.dev_state_bank is None, "official reset evaluation must not receive --dev-state-bank")
        development_bank = None
        reset_count = 50
    else:
        require(args.dev_state_bank is not None, "clean-dev reset evaluation requires --dev-state-bank")
        development_bank = load_development_reset_bank(args.dev_state_bank.resolve())
        reset_count = int(development_bank.manifest["states_per_task"])
    suites = SUITES if args.suite == "all" else (args.suite,)
    task_ids = parse_selection(args.task_ids, upper=10, name="task ids")
    init_state_ids = parse_selection(args.init_state_ids, upper=reset_count, name="reset ids")

    with PolicyClient(args.socket, timeout_seconds=args.policy_timeout_seconds) as client:
        health = client.health()
        checkpoint_identity = None
        if official:
            assert selected_cell is not None
            checkpoint_identity = validate_official_policy_health(
                health,
                selected_cell=selected_cell,
                execution_horizon=args.execution_horizon,
            )
        else:
            validate_policy_health(health, allow_fake_policy=args.allow_fake_policy)
        if args.dry_run:
            print(
                json.dumps(
                    dry_run(
                        client,
                        health,
                        execution_horizon=args.execution_horizon,
                        evaluation_seed=args.evaluation_seed,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return

        assert args.output_dir is not None
        policy_warmup = run_policy_warmups(
            client,
            health,
            count=args.policy_warmup_calls,
            execution_horizon=args.execution_horizon,
            evaluation_seed=args.evaluation_seed,
        )

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        bank_identity = None
        if development_bank is not None:
            bank_identity = {
                "base_seed": development_bank.manifest["base_seed"],
                "manifest_sha256": development_bank.manifest_sha256,
                "root_sha256": development_bank.manifest["root_sha256"],
                "schema": development_bank.manifest["schema"],
                "states_per_task": development_bank.manifest["states_per_task"],
            }
        run_manifest = {
            "created_utc": datetime.now(UTC).isoformat(),
            "evaluator_environment": evaluator_environment,
            "evaluation_seed": args.evaluation_seed,
            "execution_horizon": args.execution_horizon,
            "init_state_ids": list(init_state_ids),
            "policy_health": health,
            "policy_socket": str(args.socket),
            "policy_warmup": {
                "count": args.policy_warmup_calls,
                "included_in_episode_latency": False,
                "reports": policy_warmup,
            },
            "protocol": PROTOCOL,
            "reset_identity": {
                "bank": bank_identity,
                "id_field": "published_init_state_id" if args.reset_source == "official" else "clean_dev_reset_id",
                "source": args.reset_source,
                "state_sha256": None if args.reset_source == "official" else "per-episode-manifest-entry",
            },
            "reset_source": args.reset_source,
            "simulator_preflight": simulator_preflight,
            "simulator_attestation_sha256": simulator_attestation_sha256,
            "suites": list(suites),
            "task_ids": list(task_ids),
        }
        if official:
            assert contamination is not None
            assert official_episodes is not None
            assert selected_cell is not None
            assert preregistration_sha256 is not None
            assert args.preregistration_manifest is not None
            assert args.final_freeze_token is not None
            run_manifest.update(
                {
                    "cell": selected_cell,
                    "checkpoint": checkpoint_identity,
                    "contamination": contamination,
                    "episode_count": len(official_episodes),
                    "episode_matrix_sha256": canonical_sha256(official_episodes),
                    "final_freeze_token_sha256": hashlib.sha256(args.final_freeze_token.encode("utf-8")).hexdigest(),
                    "mode": "official-score",
                    "preregistration_manifest": str(args.preregistration_manifest.resolve()),
                    "preregistration_sha256": preregistration_sha256,
                    "schema": OFFICIAL_RUN_SCHEMA,
                }
            )
        else:
            run_manifest["mode"] = "development"
        (output_dir / "run.json").write_text(
            json.dumps(run_manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (output_dir / "episodes.jsonl").open("x", encoding="utf-8") as episode_sink:
            episodes = run_rollouts(
                client=client,
                train_seed=health["train_seed"],
                evaluation_seed=args.evaluation_seed,
                suites=suites,
                task_ids=task_ids,
                init_state_ids=init_state_ids,
                execution_horizon=args.execution_horizon,
                episode_sink=episode_sink,
                reset_source=args.reset_source,
                development_bank=development_bank,
                official_episode_identities=(
                    None
                    if official_episodes is None
                    else frozenset(
                        (episode["suite"], episode["task_id"], episode["reset_id"]) for episode in official_episodes
                    )
                ),
            )
        if official:
            assert official_episodes is not None
            observed_episode_order = [
                {"reset_id": episode["reset_id"], "suite": episode["suite"], "task_id": episode["task_id"]}
                for episode in episodes
            ]
            require(observed_episode_order == official_episodes, "official episode order differs from pre-registration")
        summary = summarize_episodes(episodes, execution_horizon=args.execution_horizon)
        if official:
            assert contamination is not None
            summary = bind_official_summary(
                summary,
                contamination=contamination,
                episode_matrix_sha256=canonical_sha256(official_episodes),
            )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
