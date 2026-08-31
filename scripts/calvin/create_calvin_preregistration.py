#!/usr/bin/env python3
# ruff: noqa: E402, I001, UP006, UP035, UP045
"""Create one sealed canonical 24-cell CALVIN ABC->D pre-registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_EVALUATOR_SOURCE_NAMES = ("evaluate_calvin.py", "calvin_bridge.py", "preflight.py")


def _source_file_identity(path: Path) -> Dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open pre-registration evaluator source: {path}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"pre-registration evaluator source is not regular: {path}")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
            after = os.fstat(source.fileno())
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields) or size != after.st_size:
            raise RuntimeError(f"pre-registration evaluator source changed while being hashed: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return {"bytes": size, "path": str(path.resolve()), "sha256": digest.hexdigest()}


def _capture_evaluator_source_identities(script_dir: Path = _SCRIPT_DIR) -> Dict[str, Dict[str, Any]]:
    root = script_dir.resolve()
    return {name: _source_file_identity(root / name) for name in _EVALUATOR_SOURCE_NAMES}


def _require_evaluator_sources_unchanged(expected: Mapping[str, Any]) -> None:
    if set(expected) != set(_EVALUATOR_SOURCE_NAMES):
        raise RuntimeError("pre-registration evaluator source snapshot names changed")
    observed = {}  # type: Dict[str, Dict[str, Any]]
    for name in _EVALUATOR_SOURCE_NAMES:
        identity = expected[name]
        if not isinstance(identity, Mapping) or set(identity) != {"bytes", "path", "sha256"}:
            raise RuntimeError(f"pre-registration evaluator source identity {name} is invalid")
        path = identity["path"]
        if not isinstance(path, str):
            raise RuntimeError(f"pre-registration evaluator source identity {name} path is invalid")
        observed[name] = _source_file_identity(Path(path))
    if observed != expected:
        raise RuntimeError("pre-registration evaluator sources changed after import-time snapshot")


def _require_imported_local_module(module: Any, source_name: str, expected: Mapping[str, Any]) -> None:
    identity = expected.get(source_name)
    if source_name not in _EVALUATOR_SOURCE_NAMES or not isinstance(identity, Mapping):
        raise RuntimeError(f"pre-registration dependency identity {source_name} is invalid")
    if set(identity) != {"bytes", "path", "sha256"}:
        raise RuntimeError(f"pre-registration dependency identity {source_name} fields are invalid")
    module_file = getattr(module, "__file__", None)
    module_spec = getattr(module, "__spec__", None)
    spec_origin = getattr(module_spec, "origin", None)
    if not isinstance(module_file, str) or Path(module_file).resolve() != Path(identity["path"]):
        raise RuntimeError(f"imported pre-registration dependency {source_name} has an unexpected __file__")
    if not isinstance(spec_origin, str) or Path(spec_origin).resolve() != Path(identity["path"]):
        raise RuntimeError(f"imported pre-registration dependency {source_name} has an unexpected spec origin")
    if _source_file_identity(Path(module_file)) != identity:
        raise RuntimeError(f"imported pre-registration dependency {source_name} differs from the pre-import snapshot")


_CREATOR_SOURCE_PATH = Path(__file__).resolve()
_IMPORT_CREATOR_SOURCE_IDENTITY = _source_file_identity(_CREATOR_SOURCE_PATH)
_IMPORT_EVALUATOR_SOURCE_IDENTITIES = _capture_evaluator_source_identities()


def _require_creator_source_unchanged() -> None:
    require_identity = _source_file_identity(_CREATOR_SOURCE_PATH)
    if require_identity != _IMPORT_CREATOR_SOURCE_IDENTITY:
        raise RuntimeError("pre-registration creator source changed after process startup")


import evaluate_calvin as _evaluate_calvin

_require_imported_local_module(
    _evaluate_calvin,
    "evaluate_calvin.py",
    _IMPORT_EVALUATOR_SOURCE_IDENTITIES,
)

from evaluate_calvin import (
    _CELL_FIELDS,
    _POLICY_CREATOR_FIELDS,
    EVALUATION_SEED,
    FINAL_CHECKPOINT_UPDATE,
    INFERENCE_SEED_DOMAIN,
    NUM_SEQUENCES,
    OFFICIAL_DIRECT_NFE,
    OFFICIAL_FLOW_NFES,
    OFFICIAL_TRAIN_SEEDS,
    PREREGISTRATION_SCHEMA,
    PROTOCOL,
    SEQUENCE_SHA256,
    SUBTASKS_PER_SEQUENCE,
    SUPPORTED_EXECUTION_HORIZONS,
    _read_strict_json,
    canonical_json_bytes,
    canonical_sha256,
    publish_bytes_and_sha256_exclusive,
    regenerate_official_sequences,
    selected_policy_contract,
    validate_preregistration_manifest,
)

import preflight as _preflight

_require_imported_local_module(
    _preflight,
    "preflight.py",
    _IMPORT_EVALUATOR_SOURCE_IDENTITIES,
)

from preflight import ATTESTATION_SCHEMA, PYTHON_VERSION, validate_official_dataset_identity

_require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_exact_keys(value: Dict[str, Any], expected: set, name: str) -> None:
    observed = set(value)
    require(
        observed == expected,
        f"{name} fields differ: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}",
    )


def _attestation_content_sha256(attestation: Dict[str, Any]) -> str:
    payload = {name: value for name, value in attestation.items() if name != "attestation_sha256"}
    return canonical_sha256(payload)


def _require_live_evaluator_sources(attestation: Dict[str, Any]) -> None:
    try:
        registered = attestation["runtime"]["sources"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("CALVIN attestation has no evaluator source identities") from exc
    require(
        canonical_json_bytes(_IMPORT_EVALUATOR_SOURCE_IDENTITIES) == canonical_json_bytes(registered),
        "preflight attestation sources differ from the creator import-time snapshot",
    )
    _require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)


def load_preflight_attestation(path: Path) -> Dict[str, Any]:
    """Load the exact wrapper emitted by ``calvin/preflight.py``."""

    report, _digest = _read_strict_json(path)
    _require_exact_keys(
        report,
        {"attestation", "dataset", "egl", "runtime_attestation", "sequences"},
        "CALVIN preflight report",
    )
    attestation = report["attestation"]
    require(isinstance(attestation, dict), "preflight report has no official runtime/data attestation")
    _require_exact_keys(
        attestation,
        {"attestation_sha256", "dataset", "runtime", "schema"},
        "CALVIN official attestation",
    )
    require(attestation["schema"] == ATTESTATION_SCHEMA, "CALVIN official attestation schema mismatch")
    digest = attestation["attestation_sha256"]
    require(
        isinstance(digest, str) and len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
        "CALVIN official attestation SHA-256 is invalid",
    )
    require(digest == _attestation_content_sha256(attestation), "CALVIN official attestation self-digest mismatch")
    validate_official_dataset_identity(attestation.get("dataset"))
    require(
        canonical_json_bytes(report["runtime_attestation"]) == canonical_json_bytes(attestation["runtime"]),
        "preflight runtime and official attestation runtime differ",
    )
    sequences = report["sequences"]
    require(isinstance(sequences, dict), "preflight sequence report must be an object")
    _require_exact_keys(
        sequences,
        {"count", "first_state", "first_tasks", "sha256"},
        "preflight sequence report",
    )
    require(
        sequences.get("count") == NUM_SEQUENCES and sequences.get("sha256") == SEQUENCE_SHA256,
        "preflight sequence report differs from the official sequence contract",
    )
    _require_live_evaluator_sources(attestation)
    return attestation


def _canonical_cell_sort_key(cell: Dict[str, Any]) -> tuple:
    policy = cell["policy"]
    objective_order = 0 if policy["objective"] == "rectified_flow" else 1
    return (policy["train_seed"], objective_order, policy["nfe"], cell["execution_horizon"])


def build_manifest(
    cells_document: Dict[str, Any],
    *,
    aggregator_sha256: str,
    attestation_sha256: str,
    final_freeze_token: str,
    sequences: List[Any],
) -> Dict[str, Any]:
    """Derive serving-policy identities and validate the complete manifest."""

    require(bool(final_freeze_token.strip()), "final freeze token must be non-empty")
    require(
        isinstance(aggregator_sha256, str)
        and len(aggregator_sha256) == 64
        and all(character in "0123456789abcdef" for character in aggregator_sha256),
        "aggregator SHA-256 is invalid",
    )
    require(
        isinstance(attestation_sha256, str)
        and len(attestation_sha256) == 64
        and all(character in "0123456789abcdef" for character in attestation_sha256),
        "runtime/data attestation SHA-256 is invalid",
    )
    _require_exact_keys(cells_document, {"cells"}, "CALVIN cell document")
    input_cells = cells_document["cells"]
    require(isinstance(input_cells, list), 'cell document "cells" must be a list')
    cells = []  # type: List[Dict[str, Any]]
    for index, candidate in enumerate(input_cells):
        require(isinstance(candidate, dict), f"input cell {index} must be an object")
        _require_exact_keys(candidate, _CELL_FIELDS, f"input cell {index}")
        policy = candidate["policy"]
        require(isinstance(policy, dict), f"input cell {index} policy must be an object")
        _require_exact_keys(policy, _POLICY_CREATOR_FIELDS, f"input cell {index} policy")
        contract = selected_policy_contract(policy["objective"], policy["nfe"])
        cells.append(
            {
                **candidate,
                "policy": {
                    **policy,
                    "identity_sha256": canonical_sha256(contract),
                },
            }
        )
    cells.sort(key=_canonical_cell_sort_key)
    manifest = {
        "aggregation_python_version": PYTHON_VERSION,
        "aggregator_sha256": aggregator_sha256,
        "benchmark_protocol": PROTOCOL,
        "cells": cells,
        "direct_nfe": OFFICIAL_DIRECT_NFE,
        "evaluation_seed": EVALUATION_SEED,
        "execution_horizons": list(SUPPORTED_EXECUTION_HORIZONS),
        "final_checkpoint_update": FINAL_CHECKPOINT_UPDATE,
        "final_freeze_token_sha256": hashlib.sha256(final_freeze_token.encode("utf-8")).hexdigest(),
        "flow_nfes": list(OFFICIAL_FLOW_NFES),
        "inference_seed_domain": INFERENCE_SEED_DOMAIN,
        "runtime_attestation_sha256": attestation_sha256,
        "schema": PREREGISTRATION_SCHEMA,
        "sequence_count": NUM_SEQUENCES,
        "sequence_sha256": SEQUENCE_SHA256,
        "sequences": sequences,
        "subtasks_per_sequence": SUBTASKS_PER_SEQUENCE,
        "training_seeds": list(OFFICIAL_TRAIN_SEEDS),
    }
    validate_preregistration_manifest(
        manifest,
        sequences,
        runtime_attestation_sha256=attestation_sha256,
    )
    return manifest


def _write_exclusive_json(
    path: Path,
    value: Dict[str, Any],
    *,
    commit_guard: Optional[Callable[[], None]] = None,
) -> str:
    payload = (json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
    return publish_bytes_and_sha256_exclusive(path, payload, commit_guard=commit_guard)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True, help='strict JSON object with one "cells" list')
    parser.add_argument("--runtime-attestation", type=Path, required=True, help="full CALVIN preflight JSON report")
    parser.add_argument("--evaluation-seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--final-freeze-token", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    _require_creator_source_unchanged()
    _require_evaluator_sources_unchanged(_IMPORT_EVALUATOR_SOURCE_IDENTITIES)
    args = parse_args()
    require(
        platform.python_version() == PYTHON_VERSION,
        f"CALVIN pre-registration creator requires Python {PYTHON_VERSION}, found {platform.python_version()}",
    )
    require(args.evaluation_seed == EVALUATION_SEED, "official CALVIN evaluation seed must equal zero")
    require(bool(args.final_freeze_token.strip()), "final freeze token must be non-empty")
    aggregator_path = _SCRIPT_DIR / "aggregate_calvin_official.py"
    aggregator_identity = _source_file_identity(aggregator_path)
    aggregator_sha256 = aggregator_identity["sha256"]
    cells_document, _digest = _read_strict_json(args.cells.resolve())
    attestation = load_preflight_attestation(args.runtime_attestation.resolve())
    generated = regenerate_official_sequences(EVALUATION_SEED)
    sequences = json.loads(canonical_json_bytes(generated).decode("ascii"))
    manifest = build_manifest(
        cells_document,
        aggregator_sha256=aggregator_sha256,
        attestation_sha256=attestation["attestation_sha256"],
        final_freeze_token=args.final_freeze_token,
        sequences=sequences,
    )
    output = args.output.resolve()

    def commit_guard() -> None:
        _require_creator_source_unchanged()
        require(
            _source_file_identity(aggregator_path) == aggregator_identity,
            "aggregate_calvin_official.py changed while creating the pre-registration",
        )
        _require_live_evaluator_sources(attestation)

    commit_guard()
    digest = _write_exclusive_json(output, manifest, commit_guard=commit_guard)
    print(json.dumps({"manifest": str(output), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
