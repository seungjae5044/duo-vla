"""Deterministic, dependency-light contracts for clean LIBERO development reset banks."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

DEV_STATE_SCHEMA = "duo-vla-libero-dev-states-v1"
SAMPLER_SEED_DOMAIN = "duo-vla-libero-dev-state-sampler-v1"
STATE_ENCODING = "contiguous-little-endian-float64"
CAPTURE_PHASE = "after-seeded-bddl-reset-before-settle"
VALIDATION_GATES = (
    "finite-full-state",
    "byte-exact-set-get-round-trip",
    "not-successful-before-settle",
    "finite-nonterminal-nonsuccessful-settle",
    "byte-reproducible-sampler-and-settle",
    "official-state-byte-exclusion",
    "bank-wide-byte-uniqueness",
)
STATE_DTYPE = np.dtype("<f8")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
OFFICIAL_STATES_PER_TASK = 50
SETTLE_STEPS = 10
OPEN_GRIPPER_NOOP = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

_TOP_LEVEL_KEYS = {
    "base_seed",
    "generator",
    "root_sha256",
    "schema",
    "simulator",
    "states_per_task",
    "tasks",
}
_TASK_KEYS = {
    "artifact",
    "bddl",
    "entries",
    "instruction",
    "official_states",
    "rejections",
    "suite",
    "task_id",
    "task_name",
}
_ENTRY_KEYS = {"attempt_id", "reset_id", "sampler_seed", "settled_state_sha256", "state_sha256"}
_ARTIFACT_KEYS = {"bytes", "dtype", "path", "sha256", "shape"}
_OFFICIAL_KEYS = {"count", "root_sha256", "sha256", "state_size"}
_REJECTION_KEYS = {"bank_duplicate", "initial_success", "official_match", "settle_success"}
_GENERATOR_KEYS = {
    "capture_phase",
    "max_attempts_per_task",
    "sampler_seed_domain",
    "settle_action",
    "settle_steps",
    "state_encoding",
    "validation_gates",
}


class DevStateValidationError(RuntimeError):
    """A sampled state or serialized bank violated the clean-development contract."""


class InitialSuccessError(DevStateValidationError):
    """The sampled reset already satisfies the task before settling."""


class SettleSuccessError(DevStateValidationError):
    """The sampled reset satisfies the task during mandatory settling."""


@dataclass(frozen=True, slots=True)
class CandidateValidation:
    state_size: int
    state_sha256: str
    settled_state_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevStateValidationError(message)


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(value)
    _require(
        observed == expected,
        f"{name} fields differ: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}",
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def canonical_state(state: Any, *, expected_size: int | None = None) -> np.ndarray:
    """Return an owned one-dimensional canonical ``<f8`` MuJoCo state."""

    source = np.asarray(state)
    _require(source.ndim == 1 and source.size > 0, "MuJoCo state must be a non-empty one-dimensional array")
    _require(source.dtype.kind in "fiu", "MuJoCo state must contain numeric values")
    if expected_size is not None:
        _require(type(expected_size) is int and expected_size > 0, "expected state size must be positive")
        _require(
            source.size == expected_size,
            f"MuJoCo state size changed: expected {expected_size}, got {source.size}",
        )
    result = np.ascontiguousarray(source, dtype=STATE_DTYPE)
    _require(result.dtype.str == "<f8", f"canonical state dtype changed: {result.dtype.str}")
    _require(bool(np.isfinite(result).all()), "MuJoCo state contains non-finite values")
    return result.copy(order="C")


def canonical_state_bytes(state: Any, *, expected_size: int | None = None) -> bytes:
    return canonical_state(state, expected_size=expected_size).tobytes(order="C")


def state_sha256(state: Any, *, expected_size: int | None = None) -> str:
    return hashlib.sha256(canonical_state_bytes(state, expected_size=expected_size)).hexdigest()


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    else:
        text = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return text.encode("utf-8")


def sequence_root_sha256(values: Sequence[str]) -> str:
    normalized = list(values)
    _require(all(_is_sha256(value) for value in normalized), "hash sequence contains an invalid SHA-256")
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def canonical_official_state_hashes(
    states: Sequence[Any] | np.ndarray,
    *,
    expected_count: int = OFFICIAL_STATES_PER_TASK,
) -> tuple[int, tuple[str, ...]]:
    values = list(states)
    _require(len(values) == expected_count, f"expected {expected_count} official states, got {len(values)}")
    first = canonical_state(values[0])
    state_size = int(first.size)
    hashes = [state_sha256(first)]
    hashes.extend(state_sha256(value, expected_size=state_size) for value in values[1:])
    _require(len(set(hashes)) == expected_count, "official state bank contains byte-identical states")
    return state_size, tuple(hashes)


def deterministic_sampler_seed(base_seed: int, suite: str, task_id: int, attempt_id: int) -> int:
    _require(type(base_seed) is int and 0 <= base_seed < 2**63, "base_seed must be an integer in [0, 2^63)")
    _require(suite in SUITES, f"unknown LIBERO suite {suite!r}")
    _require(type(task_id) is int and 0 <= task_id < 10, "task_id must be in [0, 10)")
    _require(type(attempt_id) is int and attempt_id >= 0, "attempt_id must be nonnegative")
    identity = [SAMPLER_SEED_DOMAIN, base_seed, suite, task_id, attempt_id]
    digest = hashlib.blake2b(canonical_json_bytes(identity), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def sample_reproducible_pre_settle_state(
    environment: Any,
    *,
    sampler_seed: int,
    expected_size: int,
) -> np.ndarray:
    """Seed and reset the BDDL sampler twice, requiring byte-identical full states."""

    _require(type(sampler_seed) is int and 0 <= sampler_seed < 2**32, "sampler_seed must be a uint32")

    def sample_once() -> np.ndarray:
        environment.seed(sampler_seed)
        environment.reset()
        return canonical_state(environment.get_sim_state(), expected_size=expected_size)

    first = sample_once()
    second = sample_once()
    _require(
        first.tobytes(order="C") == second.tobytes(order="C"),
        "seeded BDDL reset sampler did not reproduce the same full MuJoCo state",
    )
    return first


def _restore_and_settle(
    environment: Any,
    state: np.ndarray,
    *,
    sampler_seed: int,
    settle_steps: int,
) -> np.ndarray:
    environment.seed(sampler_seed)
    environment.reset()
    environment.set_init_state(state.copy())
    restored = canonical_state(environment.get_sim_state(), expected_size=int(state.size))
    _require(
        restored.tobytes(order="C") == state.tobytes(order="C"),
        "full MuJoCo state failed a byte-exact set/get round trip",
    )
    if bool(environment.check_success()):
        raise InitialSuccessError("sampled reset is successful before settling")

    for step_index in range(settle_steps):
        transition = environment.step(OPEN_GRIPPER_NOOP.copy())
        _require(isinstance(transition, tuple) and len(transition) == 4, "settle step did not return a 4-tuple")
        _, reward, done, _ = transition
        try:
            finite_reward = math.isfinite(float(reward))
        except (TypeError, ValueError) as exc:
            raise DevStateValidationError("settle reward is not numeric") from exc
        _require(finite_reward, "settle reward is non-finite")
        _require(not bool(done), f"simulator terminated during settle step {step_index}")
        canonical_state(environment.get_sim_state(), expected_size=int(state.size))
        if bool(environment.check_success()):
            raise SettleSuccessError(f"sampled reset became successful during settle step {step_index}")
    return canonical_state(environment.get_sim_state(), expected_size=int(state.size))


def validate_pre_settle_candidate(
    environment: Any,
    state: Any,
    *,
    sampler_seed: int,
    settle_steps: int = SETTLE_STEPS,
) -> CandidateValidation:
    """Require exact round-trip and deterministic, finite, non-successful settling."""

    _require(type(settle_steps) is int and settle_steps > 0, "settle_steps must be positive")
    canonical = canonical_state(state)
    first_settled = _restore_and_settle(
        environment,
        canonical,
        sampler_seed=sampler_seed,
        settle_steps=settle_steps,
    )
    second_settled = _restore_and_settle(
        environment,
        canonical,
        sampler_seed=sampler_seed,
        settle_steps=settle_steps,
    )
    _require(
        first_settled.tobytes(order="C") == second_settled.tobytes(order="C"),
        "mandatory settle trajectory is not byte reproducible",
    )
    return CandidateValidation(
        state_size=int(canonical.size),
        state_sha256=state_sha256(canonical),
        settled_state_sha256=state_sha256(first_settled),
    )


def deterministic_npy_bytes(states: Any) -> bytes:
    values = np.asarray(states)
    _require(values.ndim == 2 and min(values.shape) > 0, "state bank must have non-empty [states, state_size] shape")
    canonical = np.ascontiguousarray(values, dtype=STATE_DTYPE)
    _require(canonical.dtype.str == "<f8", "state bank dtype must be <f8")
    _require(bool(np.isfinite(canonical).all()), "state bank contains non-finite values")
    buffer = io.BytesIO()
    np.save(buffer, canonical, allow_pickle=False)
    return buffer.getvalue()


def manifest_root_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("root_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def finalize_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require("root_sha256" not in payload, "unfinalized manifest must not contain root_sha256")
    normalized = json.loads(canonical_json_bytes(dict(payload)))
    _require(normalized.get("schema") == DEV_STATE_SCHEMA, "development-state manifest schema mismatch")
    normalized["root_sha256"] = manifest_root_sha256(normalized)
    return normalized


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    _require(_is_sha256(manifest.get("root_sha256")), "manifest root_sha256 is invalid")
    _require(manifest_root_sha256(manifest) == manifest["root_sha256"], "manifest root SHA-256 mismatch")
    return canonical_json_bytes(dict(manifest), pretty=True)


def task_artifact_path(suite: str, task_id: int) -> str:
    _require(suite in SUITES, f"unknown LIBERO suite {suite!r}")
    _require(type(task_id) is int and 0 <= task_id < 10, "task_id must be in [0, 10)")
    return f"{suite}/task_{task_id:02d}.npy"


def artifact_record(path: str, data: bytes, shape: tuple[int, int]) -> dict[str, Any]:
    _validate_relative_artifact_path(path)
    _require(len(data) > 0, "artifact data must not be empty")
    _require(len(shape) == 2 and min(shape) > 0, "artifact shape must be non-empty and two-dimensional")
    return {
        "bytes": len(data),
        "dtype": "<f8",
        "path": path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "shape": list(shape),
    }


def _validate_relative_artifact_path(value: Any) -> str:
    _require(isinstance(value, str) and value, "artifact path must be non-empty text")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts, "artifact path must stay within the bank root")
    _require(path.suffix == ".npy", "state artifact must use the .npy suffix")
    return value


def validate_manifest_and_artifacts(
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
) -> dict[str, np.ndarray]:
    """Fail closed on every manifest, hash, dtype, exclusion, and uniqueness invariant."""

    _exact_keys(manifest, _TOP_LEVEL_KEYS, "development-state manifest")
    _require(manifest["schema"] == DEV_STATE_SCHEMA, "development-state manifest schema mismatch")
    _require(_is_sha256(manifest["root_sha256"]), "manifest root_sha256 is invalid")
    _require(manifest_root_sha256(manifest) == manifest["root_sha256"], "manifest root SHA-256 mismatch")
    _require(type(manifest["base_seed"]) is int and 0 <= manifest["base_seed"] < 2**63, "invalid base_seed")
    _require(type(manifest["states_per_task"]) is int and manifest["states_per_task"] > 0, "invalid states_per_task")
    _require(isinstance(manifest["simulator"], dict), "simulator identity must be an object")
    generator = manifest["generator"]
    _require(isinstance(generator, dict), "generator contract must be an object")
    _exact_keys(generator, _GENERATOR_KEYS, "generator contract")
    _require(generator["sampler_seed_domain"] == SAMPLER_SEED_DOMAIN, "sampler seed domain mismatch")
    _require(generator["state_encoding"] == STATE_ENCODING, "state encoding mismatch")
    _require(generator["capture_phase"] == CAPTURE_PHASE, "state capture phase mismatch")
    _require(generator["validation_gates"] == list(VALIDATION_GATES), "state validation gates mismatch")
    _require(generator["settle_steps"] == SETTLE_STEPS, "settle-step contract mismatch")
    _require(generator["settle_action"] == OPEN_GRIPPER_NOOP.tolist(), "settle action mismatch")
    _require(
        type(generator["max_attempts_per_task"]) is int
        and generator["max_attempts_per_task"] >= manifest["states_per_task"],
        "invalid max_attempts_per_task",
    )

    tasks = manifest["tasks"]
    _require(isinstance(tasks, list) and tasks, "manifest must contain at least one task bank")
    expected_paths: set[str] = set()
    global_state_hashes: set[str] = set()
    loaded: dict[str, np.ndarray] = {}
    task_identities: set[tuple[str, int]] = set()
    for task_index, task in enumerate(tasks):
        _require(isinstance(task, dict), f"task bank {task_index} must be an object")
        _exact_keys(task, _TASK_KEYS, f"task bank {task_index}")
        suite = task["suite"]
        task_id = task["task_id"]
        _require(suite in SUITES, f"task bank {task_index} has an unknown suite")
        _require(type(task_id) is int and 0 <= task_id < 10, f"task bank {task_index} has an invalid task_id")
        identity = (suite, task_id)
        _require(identity not in task_identities, f"duplicate task bank identity {identity}")
        task_identities.add(identity)

        official = task["official_states"]
        _require(isinstance(official, dict), f"task bank {task_index} official_states must be an object")
        _exact_keys(official, _OFFICIAL_KEYS, f"task bank {task_index} official_states")
        official_hashes = official["sha256"]
        _require(
            isinstance(official_hashes, list) and len(official_hashes) == official["count"] == OFFICIAL_STATES_PER_TASK,
            f"task bank {task_index} must authenticate 50 official states",
        )
        _require(len(set(official_hashes)) == OFFICIAL_STATES_PER_TASK, "official state hashes are not unique")
        _require(
            sequence_root_sha256(official_hashes) == official["root_sha256"],
            f"task bank {task_index} official-state root hash mismatch",
        )
        _require(type(official["state_size"]) is int and official["state_size"] > 0, "invalid official state size")

        artifact = task["artifact"]
        _require(isinstance(artifact, dict), f"task bank {task_index} artifact must be an object")
        _exact_keys(artifact, _ARTIFACT_KEYS, f"task bank {task_index} artifact")
        path = _validate_relative_artifact_path(artifact["path"])
        _require(path not in expected_paths, f"duplicate artifact path {path}")
        expected_paths.add(path)
        _require(path in artifacts, f"missing state artifact {path}")
        data = artifacts[path]
        _require(type(data) is bytes, f"state artifact {path} must be bytes")
        _require(len(data) == artifact["bytes"], f"state artifact {path} byte length mismatch")
        _require(hashlib.sha256(data).hexdigest() == artifact["sha256"], f"state artifact {path} SHA-256 mismatch")
        _require(artifact["dtype"] == "<f8", f"state artifact {path} dtype contract mismatch")
        try:
            array = np.load(io.BytesIO(data), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise DevStateValidationError(f"state artifact {path} is not a valid non-pickle NPY") from exc
        _require(array.dtype.str == "<f8", f"state artifact {path} is not canonical <f8")
        _require(list(array.shape) == artifact["shape"], f"state artifact {path} shape mismatch")
        _require(
            array.shape == (manifest["states_per_task"], official["state_size"]),
            f"state artifact {path} does not match the bank dimensions",
        )
        _require(bool(np.isfinite(array).all()), f"state artifact {path} contains non-finite values")

        entries = task["entries"]
        _require(
            isinstance(entries, list) and len(entries) == array.shape[0],
            "state entries do not match artifact rows",
        )
        for reset_id, (entry, state) in enumerate(zip(entries, array, strict=True)):
            _require(isinstance(entry, dict), f"state entry {path}:{reset_id} must be an object")
            _exact_keys(entry, _ENTRY_KEYS, f"state entry {path}:{reset_id}")
            _require(entry["reset_id"] == reset_id, f"state entry {path}:{reset_id} has a non-canonical reset_id")
            _require(type(entry["attempt_id"]) is int and entry["attempt_id"] >= 0, "invalid attempt_id")
            _require(type(entry["sampler_seed"]) is int and 0 <= entry["sampler_seed"] < 2**32, "invalid sampler_seed")
            observed_hash = state_sha256(state, expected_size=official["state_size"])
            _require(observed_hash == entry["state_sha256"], f"state entry {path}:{reset_id} SHA-256 mismatch")
            _require(_is_sha256(entry["settled_state_sha256"]), "settled state SHA-256 is invalid")
            _require(observed_hash not in official_hashes, f"state entry {path}:{reset_id} matches an official state")
            _require(observed_hash not in global_state_hashes, f"state entry {path}:{reset_id} duplicates the bank")
            global_state_hashes.add(observed_hash)

        rejections = task["rejections"]
        _require(isinstance(rejections, dict), "rejections must be an object")
        _exact_keys(rejections, _REJECTION_KEYS, f"task bank {task_index} rejections")
        _require(
            all(type(value) is int and value >= 0 for value in rejections.values()),
            f"task bank {task_index} has invalid rejection counts",
        )
        _require(isinstance(task["bddl"], dict), "BDDL identity must be an object")
        _require(isinstance(task["task_name"], str) and task["task_name"], "task_name must be non-empty text")
        _require(isinstance(task["instruction"], str) and task["instruction"], "instruction must be non-empty text")
        loaded[path] = np.ascontiguousarray(array, dtype=STATE_DTYPE)

    _require(set(artifacts) == expected_paths, "state artifact set differs from the manifest")
    return loaded


def write_bank_exclusive(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
) -> str:
    """Validate and exclusively materialize one deterministic bank directory."""

    validate_manifest_and_artifacts(manifest, artifacts)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    incomplete = root / "INCOMPLETE"
    incomplete.write_text("development reset bank write in progress\n", encoding="utf-8")
    for relative_path in sorted(artifacts):
        destination = root.joinpath(*PurePosixPath(relative_path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(artifacts[relative_path])
            handle.flush()
            os.fsync(handle.fileno())
    serialized_manifest = manifest_bytes(manifest)
    with (root / "manifest.json").open("xb") as handle:
        handle.write(serialized_manifest)
        handle.flush()
        os.fsync(handle.fileno())
    incomplete.unlink()
    return hashlib.sha256(serialized_manifest).hexdigest()


def load_bank(bank_dir: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(bank_dir)
    _require(not (root / "INCOMPLETE").exists(), "development reset bank is incomplete")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevStateValidationError("cannot load development reset manifest") from exc
    _require(isinstance(manifest, dict), "development reset manifest must be an object")
    artifacts: dict[str, bytes] = {}
    tasks = manifest.get("tasks")
    _require(isinstance(tasks, list), "development reset manifest tasks must be a list")
    for task in tasks:
        _require(isinstance(task, dict) and isinstance(task.get("artifact"), dict), "invalid task artifact record")
        relative_path = _validate_relative_artifact_path(task["artifact"].get("path"))
        try:
            artifacts[relative_path] = root.joinpath(*PurePosixPath(relative_path).parts).read_bytes()
        except OSError as exc:
            raise DevStateValidationError(f"cannot load state artifact {relative_path}") from exc
    return manifest, validate_manifest_and_artifacts(manifest, artifacts)
