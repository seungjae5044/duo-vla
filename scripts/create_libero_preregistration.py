#!/usr/bin/env python3
"""Create and validate one sealed 24-cell official LIBERO pre-registration."""

# ruff: noqa: E402 -- authenticate local import roots before importing project code.

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent


def _activate_project_source_root() -> Path:
    source_root = _SCRIPT_DIR.parent / "src"
    if source_root.resolve(strict=True) != source_root or not stat.S_ISDIR(os.lstat(source_root).st_mode):
        raise RuntimeError("project src import root must be a canonical real directory")
    identity_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")

    def same_identity(left: os.stat_result, right: os.stat_result) -> bool:
        return all(getattr(left, name) == getattr(right, name) for name in identity_fields)

    def walk(directory: int, prefix: tuple[str, ...]) -> None:
        before = os.fstat(directory)
        names = sorted(os.listdir(directory))
        for name in names:
            context = "/".join((*prefix, name))
            observed = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if stat.S_ISDIR(observed.st_mode):
                if name == "__pycache__":
                    continue
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory,
                )
                try:
                    if not same_identity(observed, os.fstat(child)):
                        raise RuntimeError(f"project source directory changed while opening: {context}")
                    walk(child, (*prefix, name))
                finally:
                    os.close(child)
            elif not (stat.S_ISREG(observed.st_mode) and name.endswith(".py")):
                raise RuntimeError(f"project source import entry is unsafe: {context}")
        if names != sorted(os.listdir(directory)) or not same_identity(before, os.fstat(directory)):
            raise RuntimeError(f"project source directory changed during inventory: {'/'.join(prefix) or '.'}")

    root_descriptor = os.open(
        source_root,
        os.O_RDONLY | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        if sorted(os.listdir(root_descriptor)) != ["duo_vla"]:
            raise RuntimeError("project src import root must contain only the real duo_vla package directory")
        package_descriptor = os.open(
            "duo_vla",
            os.O_RDONLY | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=root_descriptor,
        )
        try:
            walk(package_descriptor, ("duo_vla",))
        finally:
            os.close(package_descriptor)
    finally:
        os.close(root_descriptor)
    source_text = str(source_root)
    sys.path[:] = [source_text] + [
        entry for entry in sys.path if str(Path(entry or os.getcwd()).resolve()) != source_text
    ]
    return source_root


_PROJECT_SOURCE_ROOT = _activate_project_source_root()


def _load_local_module(name: str, path: Path) -> Any:
    expected = path.resolve(strict=True)
    existing = sys.modules.get(name)
    if existing is not None:
        module_file = getattr(existing, "__file__", None)
        origin = getattr(getattr(existing, "__spec__", None), "origin", None)
        if not isinstance(module_file, str) or not isinstance(origin, str):
            raise RuntimeError(f"existing local module {name} has no file origin")
        if Path(module_file).resolve() != expected or Path(origin).resolve() != expected:
            raise RuntimeError(f"existing local module {name} has an unexpected origin")
        return existing
    specification = importlib.util.spec_from_file_location(name, expected)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot construct import specification for {expected}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_load_local_module("evaluate_libero", _SCRIPT_DIR / "evaluate_libero.py")
_load_local_module("qualify_libero_expert_replay", _SCRIPT_DIR / "qualify_libero_expert_replay.py")

from evaluate_libero import (
    _CELL_CREATOR_FIELDS,
    AGGREGATION_PYTHON_VERSION,
    FINAL_CHECKPOINT_UPDATE,
    OFFICIAL_EXECUTION_HORIZONS,
    OFFICIAL_POLICY_WARMUP_CALLS,
    OFFICIAL_RESETS_PER_TASK,
    OFFICIAL_TRAIN_SEEDS,
    PREREGISTRATION_SCHEMA,
    PROTOCOL,
    SIMULATOR_ATTESTATION_SCHEMA,
    SUITES,
    _read_strict_json,
    canonical_sha256,
    capture_official_output_roots,
    derive_output_claim,
    load_contamination_contract,
    official_episode_matrix,
    publish_bytes_and_sha256_exclusive,
    validate_evaluator_process_environment,
    validate_preregistration_manifest,
)
from qualify_libero_expert_replay import load_qualification_report, qualification_identity


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def stable_regular_file_sha256(path: Path) -> str:
    """Hash one stable regular file without following its final path component."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"pre-registration source is not regular: {path}")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            while block := source.read(1024 * 1024):
                size += len(block)
                digest.update(block)
            after = os.fstat(source.fileno())
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(
            all(getattr(before, name) == getattr(after, name) for name in stable) and size == after.st_size,
            f"pre-registration source changed while being hashed: {path}",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True, help='strict JSON object with one "cells" list')
    parser.add_argument("--simulator-attestation", type=Path, required=True, help="strict preflight JSON report")
    parser.add_argument("--expert-replay-qualification", type=Path, required=True)
    parser.add_argument(
        "--expert-replay-qualification-sha256",
        required=True,
        help="independently recorded raw qualification report digest",
    )
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--final-freeze-token", required=True, help="hashed into the manifest; never written verbatim")
    parser.add_argument(
        "--official-output-root",
        type=Path,
        required=True,
        help="dedicated existing empty canonical directory for the 24 official run directories",
    )
    parser.add_argument(
        "--official-claim-root",
        type=Path,
        required=True,
        help="dedicated existing empty canonical directory for immutable attempt claims",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(0 <= args.evaluation_seed < 2**63, "evaluation seed must be in [0, 2^63)")
    require(bool(args.final_freeze_token.strip()), "final freeze token must be non-empty")
    project_root = Path(__file__).resolve().parents[1]
    validate_evaluator_process_environment(project_root)
    aggregator_path = project_root / "scripts/aggregate_libero_official.py"
    aggregator_sha256 = stable_regular_file_sha256(aggregator_path)
    creator_path = Path(__file__).resolve()
    creator_sha256 = stable_regular_file_sha256(creator_path)
    evaluator_path = project_root / "scripts/evaluate_libero.py"
    evaluator_sha256 = stable_regular_file_sha256(evaluator_path)
    bridge_path = project_root / "scripts/libero_bridge.py"
    bridge_sha256 = stable_regular_file_sha256(bridge_path)
    preflight_path = project_root / "scripts/preflight_libero_env.py"
    preflight_sha256 = stable_regular_file_sha256(preflight_path)
    cells_document, _ = _read_strict_json(args.cells.resolve(), name="LIBERO pre-registration cell document")
    require(set(cells_document) == {"cells"}, 'cell document must contain exactly one "cells" field')
    attestation, attestation_raw_sha256 = _read_strict_json(
        args.simulator_attestation.resolve(),
        name="LIBERO simulator attestation",
    )
    require(
        attestation.get("schema") == SIMULATOR_ATTESTATION_SCHEMA
        and attestation.get("status") == "ok"
        and attestation.get("environment_constructed") is True,
        "official pre-registration requires a successful full simulator attestation",
    )
    project_sources = attestation.get("project_sources")
    require(isinstance(project_sources, dict), "simulator attestation has no project source identity")
    require(
        project_sources.get("evaluator") == evaluator_sha256
        and project_sources.get("bridge") == bridge_sha256
        and project_sources.get("preflight") == preflight_sha256,
        "simulator attestation project sources differ from pre-registration sources",
    )
    simulator_attestation_sha256 = canonical_sha256(attestation)
    qualification, qualification_raw_sha256 = load_qualification_report(
        args.expert_replay_qualification.resolve(),
        expected_raw_sha256=args.expert_replay_qualification_sha256,
        project_root=project_root,
        simulator_attestation=attestation,
        simulator_attestation_raw_sha256=attestation_raw_sha256,
    )
    contamination = load_contamination_contract(project_root)
    episodes = official_episode_matrix(contamination)
    episode_matrix_sha256 = canonical_sha256(episodes)
    roots = capture_official_output_roots(
        args.official_output_root,
        args.official_claim_root,
        require_empty=True,
    )
    freeze_token_sha256 = hashlib.sha256(args.final_freeze_token.encode("utf-8")).hexdigest()
    input_cells = cells_document["cells"]
    require(isinstance(input_cells, list), 'cell document "cells" must be a list')
    cells: list[dict[str, object]] = []
    for index, value in enumerate(input_cells):
        require(isinstance(value, dict), f"input cell {index} must be an object")
        require(set(value) == _CELL_CREATOR_FIELDS, f"input cell {index} fields differ from the creator schema")
        selected_policy = {name: value[name] for name in ("inference_seed_behavior", "nfe", "objective", "sampler")}
        cells.append(
            {
                **value,
                "episode_matrix_sha256": episode_matrix_sha256,
                "output_claim": derive_output_claim(value["cell_id"], roots, freeze_token_sha256),
                "serving_policy_sha256": canonical_sha256(selected_policy),
            }
        )
    expert_replay_qualification = qualification_identity(
        qualification,
        report_raw_sha256=qualification_raw_sha256,
    )
    qualified_checkpoint_identity = {
        (
            cell["checkpoint"]["source_tree_sha256"],
            cell["checkpoint"]["dataset_tree_sha256"],
            cell["checkpoint"]["dataset_content_inventory_sha256"],
        )
        for cell in cells
    }
    require(
        qualified_checkpoint_identity
        == {
            (
                expert_replay_qualification["project_source_tree_sha256"],
                expert_replay_qualification["dataset_tree_metadata_sha256"],
                expert_replay_qualification["dataset_content_inventory_sha256"],
            )
        },
        "official checkpoints do not use the expert-replay-qualified source/data identity",
    )
    manifest = {
        "aggregation_python_version": AGGREGATION_PYTHON_VERSION,
        "aggregator_sha256": aggregator_sha256,
        "benchmark_protocol": PROTOCOL,
        "cells": cells,
        "contamination": contamination,
        "episode_count": len(episodes),
        "episode_matrix_sha256": episode_matrix_sha256,
        "episodes": episodes,
        "evaluation_seed": args.evaluation_seed,
        "expert_replay_qualification": expert_replay_qualification,
        "execution_horizons": list(OFFICIAL_EXECUTION_HORIZONS),
        "final_checkpoint_update": FINAL_CHECKPOINT_UPDATE,
        "final_freeze_token_sha256": freeze_token_sha256,
        "official_resets_per_task": OFFICIAL_RESETS_PER_TASK,
        "official_output_roots": roots,
        "policy_warmup_calls": OFFICIAL_POLICY_WARMUP_CALLS,
        "schema": PREREGISTRATION_SCHEMA,
        "simulator_attestation_sha256": simulator_attestation_sha256,
        "suites": list(SUITES),
        "task_ids": list(range(10)),
        "training_seeds": list(OFFICIAL_TRAIN_SEEDS),
    }
    validate_preregistration_manifest(
        manifest,
        contamination=contamination,
        simulator_attestation_sha256=simulator_attestation_sha256,
    )
    output = args.output.resolve()
    require(output.parent.is_dir(), "pre-registration output parent must already exist")
    payload = (json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

    def commit_guard() -> None:
        require(
            stable_regular_file_sha256(aggregator_path) == aggregator_sha256,
            "official aggregator changed while creating the pre-registration",
        )
        require(
            stable_regular_file_sha256(creator_path) == creator_sha256
            and stable_regular_file_sha256(evaluator_path) == evaluator_sha256,
            "pre-registration source changed while creating the manifest",
        )
        require(
            stable_regular_file_sha256(bridge_path) == bridge_sha256
            and stable_regular_file_sha256(preflight_path) == preflight_sha256,
            "pre-registration dependency changed while creating the manifest",
        )
        require(
            capture_official_output_roots(
                Path(roots["runs"]["path"]),
                Path(roots["claims"]["path"]),
                require_empty=True,
            )
            == roots,
            "official output root identity changed while creating the pre-registration",
        )

    digest = publish_bytes_and_sha256_exclusive(output, payload, commit_guard=commit_guard)
    print(json.dumps({"manifest": str(output), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
