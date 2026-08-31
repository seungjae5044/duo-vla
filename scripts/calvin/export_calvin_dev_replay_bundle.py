#!/usr/bin/env python3
"""Export authenticated held-out CALVIN A/B/C replay data from archive-direct v4.

This is the only development-boundary process allowed to read episode members.
It runs under Python 3.11, uses the authenticated Phase-A reader, and publishes
a content-addressed bundle consumed by the Python-3.8 simulator environment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from duo_vla.data.calvin_archive import CalvinArchiveReader  # noqa: E402
from duo_vla.data.calvin_dev_states import (  # noqa: E402
    ACTION_SHAPE,
    DATASET_CRITICAL_FILES,
    MANIFEST_NAME,
    ROBOT_SHAPE,
    SCENE_SHAPE,
    BundledCalvinReplay,
    CalvinDevCandidate,
    CalvinDevStateError,
    CalvinResetFrame,
    authenticate_dev_inputs,
    build_replay_bundle,
    load_strict_json,
    load_validation_candidates,
    write_replay_bundle_exclusive,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CalvinDevStateError(message)


def _canonical_vector(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    source = np.asarray(value)
    require(source.shape == shape, f"{label} has shape {source.shape}, expected {shape}")
    require(np.issubdtype(source.dtype, np.floating), f"{label} must have floating dtype")
    result = np.ascontiguousarray(source, dtype=np.dtype("<f8")).copy(order="C")
    require(bool(np.isfinite(result).all()), f"{label} contains non-finite values")
    return result


def _decode_episode_member(raw: bytes, *, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            require(
                {"rel_actions", "robot_obs", "scene_obs"}.issubset(set(archive.files)),
                f"CALVIN member lacks replay arrays: {path}",
            )
            robot_obs = _canonical_vector(archive["robot_obs"], ROBOT_SHAPE, f"{path}:robot_obs")
            scene_obs = _canonical_vector(archive["scene_obs"], SCENE_SHAPE, f"{path}:scene_obs")
            action = _canonical_vector(archive["rel_actions"], ACTION_SHAPE, f"{path}:rel_actions")
    except (OSError, ValueError, EOFError) as exc:
        raise CalvinDevStateError(f"cannot decode authenticated CALVIN member: {path}") from exc
    require(robot_obs[14] in (-1.0, 1.0), f"CALVIN robot gripper state is invalid: {path}")
    require(action[6] in (-1.0, 1.0), f"CALVIN action gripper is invalid: {path}")
    require(bool((np.abs(action[:6]) <= 1.0 + 1e-6).all()), f"CALVIN action exceeds official units: {path}")
    return robot_obs, scene_obs, action


def collect_candidate_replays(
    reader: Any,
    candidates: tuple[CalvinDevCandidate, ...],
) -> tuple[BundledCalvinReplay, ...]:
    """Read each required member exactly through the authenticated reader API."""

    replays: list[BundledCalvinReplay] = []
    for candidate in candidates:
        actions: list[np.ndarray] = []
        member_identities: list[dict[str, Any]] = []
        start_robot: np.ndarray | None = None
        start_scene: np.ndarray | None = None
        for global_index in range(candidate.global_start, candidate.global_end_exclusive):
            path = f"training/episode_{global_index:07d}.npz"
            before = reader.member_record(path)
            raw = reader.read_member_bytes(path)
            after = reader.member_record(path)
            require(before == after, f"CALVIN member identity changed during replay export: {path}")
            require(
                before.path == path and before.global_index == global_index,
                f"CALVIN member path/index differs: {path}",
            )
            require(
                before.is_file and before.split == "training",
                f"CALVIN replay source is not a training file: {path}",
            )
            logical_sha256 = before.logical_sha256.hex()
            require(len(raw) == before.logical_bytes, f"CALVIN member logical byte count differs: {path}")
            require(hashlib.sha256(raw).hexdigest() == logical_sha256, f"CALVIN member logical SHA-256 differs: {path}")
            robot_obs, scene_obs, action = _decode_episode_member(raw, path=path)
            if start_robot is None:
                start_robot = robot_obs
                start_scene = scene_obs
            actions.append(action)
            member_identities.append(
                {
                    "global_index": global_index,
                    "logical_bytes": before.logical_bytes,
                    "logical_sha256": logical_sha256,
                    "path": path,
                }
            )
        require(start_robot is not None and start_scene is not None, "CALVIN annotation produced no replay members")
        replays.append(
            BundledCalvinReplay(
                candidate=candidate,
                frame=CalvinResetFrame(
                    robot_obs=start_robot,
                    scene_obs=start_scene,
                    source_frame_sha256=member_identities[0]["logical_sha256"],
                ),
                actions=np.stack(actions),
                member_identities=tuple(member_identities),
                record_sha256="",
            )
        )
    return tuple(replays)


def export_replay_bundle(
    *,
    data_root: Path,
    normalization: Path,
    source_root: Path,
    revision_file: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(sys.version_info[:2] == (3, 11), "CALVIN replay exporter requires Python 3.11")
    root = Path(data_root).resolve(strict=True)
    training_root = root / "task_ABC_D/training"
    inputs = authenticate_dev_inputs(training_root, normalization, source_root, revision_file)
    current_manifest = load_strict_json(root / MANIFEST_NAME)
    with CalvinArchiveReader.from_manifest(root) as reader:
        require(
            reader.authenticated_manifest == current_manifest,
            "Phase-A reader manifest differs from current v4 inputs",
        )
        for relative in DATASET_CRITICAL_FILES:
            require(
                reader.read_authenticated_metadata_bytes(relative) == inputs.metadata[relative],
                f"Phase-A reader metadata differs from current projected bytes: {relative}",
            )
        candidates = load_validation_candidates(inputs.metadata, inputs.split)
        replays = collect_candidate_replays(reader, candidates)
    current_inputs = authenticate_dev_inputs(training_root, normalization, source_root, revision_file)
    require(
        current_inputs == inputs,
        "CALVIN v4 inputs, projected metadata, simulator config, or development sources changed during export",
    )
    inputs = current_inputs
    source_identity = {
        "calvin_archive_source_sha256": inputs.identity["calvin_archive_source_sha256"],
        "dev_states_source_sha256": inputs.identity["dev_states_source_sha256"],
        "replay_exporter_source_sha256": inputs.identity["replay_exporter_source_sha256"],
        "schema": "duo-vla-calvin-dev-replay-export-source-v1",
    }
    manifest, artifacts = build_replay_bundle(replays, inputs, source_identity)
    write_replay_bundle_exclusive(output_dir, manifest, artifacts, inputs)
    return manifest


def parse_args() -> argparse.Namespace:
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=cache_root / "data/calvin")
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=cache_root / "simulators/calvin")
    parser.add_argument("--revision-file", type=Path, default=_PROJECT_ROOT / "scripts/calvin/revisions.env")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_replay_bundle(
        data_root=args.data_root,
        normalization=args.normalization,
        source_root=args.source_root,
        revision_file=args.revision_file,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "bundle": str(args.output_dir.resolve()),
                "records": len(manifest["records"]),
                "root_sha256": manifest["root_sha256"],
                "schema": manifest["schema"],
                "status": "ok",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
