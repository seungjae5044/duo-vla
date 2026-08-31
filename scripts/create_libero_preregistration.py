#!/usr/bin/env python3
"""Create and validate one sealed 24-cell official LIBERO pre-registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from evaluate_libero import (
    _CELL_FIELDS,
    FINAL_CHECKPOINT_UPDATE,
    OFFICIAL_EXECUTION_HORIZONS,
    OFFICIAL_RESETS_PER_TASK,
    OFFICIAL_TRAIN_SEEDS,
    PREREGISTRATION_SCHEMA,
    PROTOCOL,
    SUITES,
    _read_strict_json,
    canonical_sha256,
    load_contamination_contract,
    official_episode_matrix,
    validate_preregistration_manifest,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True, help='strict JSON object with one "cells" list')
    parser.add_argument("--simulator-attestation", type=Path, required=True, help="strict preflight JSON report")
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--final-freeze-token", required=True, help="hashed into the manifest; never written verbatim")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(0 <= args.evaluation_seed < 2**63, "evaluation seed must be in [0, 2^63)")
    require(bool(args.final_freeze_token.strip()), "final freeze token must be non-empty")
    project_root = Path(__file__).resolve().parents[1]
    cells_document, _ = _read_strict_json(args.cells.resolve(), name="LIBERO pre-registration cell document")
    require(set(cells_document) == {"cells"}, 'cell document must contain exactly one "cells" field')
    attestation, _ = _read_strict_json(
        args.simulator_attestation.resolve(),
        name="LIBERO simulator attestation",
    )
    require(
        attestation.get("schema") == "duo-vla-libero-simulator-attestation-v2"
        and attestation.get("status") == "ok"
        and attestation.get("environment_constructed") is True,
        "official pre-registration requires a successful full simulator attestation",
    )
    simulator_attestation_sha256 = canonical_sha256(attestation)
    contamination = load_contamination_contract(project_root)
    episodes = official_episode_matrix(contamination)
    episode_matrix_sha256 = canonical_sha256(episodes)
    input_cells = cells_document["cells"]
    require(isinstance(input_cells, list), 'cell document "cells" must be a list')
    cells: list[dict[str, object]] = []
    derived_fields = {"episode_matrix_sha256", "serving_policy_sha256"}
    for index, value in enumerate(input_cells):
        require(isinstance(value, dict), f"input cell {index} must be an object")
        expected = _CELL_FIELDS - derived_fields
        require(set(value) == expected, f"input cell {index} fields differ from the creator schema")
        selected_policy = {name: value[name] for name in ("inference_seed_behavior", "nfe", "objective", "sampler")}
        cells.append(
            {
                **value,
                "episode_matrix_sha256": episode_matrix_sha256,
                "serving_policy_sha256": canonical_sha256(selected_policy),
            }
        )
    manifest = {
        "benchmark_protocol": PROTOCOL,
        "cells": cells,
        "contamination": contamination,
        "episode_count": len(episodes),
        "episode_matrix_sha256": episode_matrix_sha256,
        "episodes": episodes,
        "evaluation_seed": args.evaluation_seed,
        "execution_horizons": list(OFFICIAL_EXECUTION_HORIZONS),
        "final_checkpoint_update": FINAL_CHECKPOINT_UPDATE,
        "final_freeze_token_sha256": hashlib.sha256(args.final_freeze_token.encode("utf-8")).hexdigest(),
        "official_resets_per_task": OFFICIAL_RESETS_PER_TASK,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256(payload).hexdigest()
    companion = output.with_suffix(output.suffix + ".sha256")
    with companion.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {output.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"manifest": str(output), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
