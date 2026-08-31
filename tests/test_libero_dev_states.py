from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_libero_dev_states import generate_task_bank

from duo_vla.benchmarks.libero_dev_states import (
    CAPTURE_PHASE,
    DEV_STATE_SCHEMA,
    OPEN_GRIPPER_NOOP,
    SAMPLER_SEED_DOMAIN,
    SETTLE_STEPS,
    STATE_ENCODING,
    VALIDATION_GATES,
    DevStateValidationError,
    artifact_record,
    canonical_official_state_hashes,
    canonical_state,
    deterministic_npy_bytes,
    deterministic_sampler_seed,
    finalize_manifest,
    load_bank,
    manifest_bytes,
    sample_reproducible_pre_settle_state,
    sequence_root_sha256,
    state_sha256,
    validate_manifest_and_artifacts,
    validate_pre_settle_candidate,
    write_bank_exclusive,
)


class _FakeEnvironment:
    def __init__(
        self,
        *,
        scripted_states: dict[int, np.ndarray] | None = None,
        initial_success_seeds: set[int] | None = None,
        settle_success_seeds: set[int] | None = None,
        nondeterministic_reset: bool = False,
        corrupt_round_trip: bool = False,
    ) -> None:
        self.scripted_states = scripted_states or {}
        self.initial_success_seeds = initial_success_seeds or set()
        self.settle_success_seeds = settle_success_seeds or set()
        self.nondeterministic_reset = nondeterministic_reset
        self.corrupt_round_trip = corrupt_round_trip
        self.current_seed = 0
        self.reset_count = 0
        self.steps = 0
        self.state = np.zeros(4, dtype=np.float64)

    def seed(self, value: int) -> None:
        self.current_seed = value

    def reset(self) -> dict[str, Any]:
        self.reset_count += 1
        self.steps = 0
        default = np.random.default_rng(self.current_seed).normal(size=4)
        self.state = np.asarray(self.scripted_states.get(self.current_seed, default), dtype=np.float64).copy()
        if self.nondeterministic_reset:
            self.state[0] += self.reset_count
        return {}

    def get_sim_state(self) -> np.ndarray:
        return self.state.copy()

    def set_init_state(self, state: np.ndarray) -> dict[str, Any]:
        self.steps = 0
        self.state = np.asarray(state, dtype=np.float64).copy()
        if self.corrupt_round_trip:
            self.state[-1] += 1.0
        return {}

    def check_success(self) -> bool:
        if self.steps == 0:
            return self.current_seed in self.initial_success_seeds
        return self.current_seed in self.settle_success_seeds

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        assert np.array_equal(action, OPEN_GRIPPER_NOOP)
        self.steps += 1
        self.state = self.state + np.asarray([0.001, 0.002, 0.003, 0.004], dtype=np.float64)
        return {}, 0.0, False, {}


def _official_states() -> np.ndarray:
    return np.stack([np.asarray([1000.0 + index, 1.0, 2.0, 3.0]) for index in range(50)])


def _valid_bank() -> tuple[dict[str, Any], dict[str, bytes], np.ndarray]:
    official = _official_states()
    state_size, official_hashes = canonical_official_state_hashes(official)
    states = np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
    data = deterministic_npy_bytes(states)
    path = "libero_spatial/task_00.npy"
    entries = [
        {
            "attempt_id": reset_id,
            "reset_id": reset_id,
            "sampler_seed": reset_id + 10,
            "settled_state_sha256": state_sha256(state + 0.01),
            "state_sha256": state_sha256(state),
        }
        for reset_id, state in enumerate(states)
    ]
    payload = {
        "base_seed": 7,
        "generator": {
            "capture_phase": CAPTURE_PHASE,
            "max_attempts_per_task": 20,
            "sampler_seed_domain": SAMPLER_SEED_DOMAIN,
            "settle_action": OPEN_GRIPPER_NOOP.tolist(),
            "settle_steps": SETTLE_STEPS,
            "state_encoding": STATE_ENCODING,
            "validation_gates": list(VALIDATION_GATES),
        },
        "schema": DEV_STATE_SCHEMA,
        "simulator": {"source_revision": "pinned"},
        "states_per_task": len(states),
        "tasks": [
            {
                "artifact": artifact_record(path, data, states.shape),
                "bddl": {"file": "task.bddl", "problem_folder": "suite", "sha256": "0" * 64},
                "entries": entries,
                "instruction": "test instruction",
                "official_states": {
                    "count": 50,
                    "root_sha256": sequence_root_sha256(official_hashes),
                    "sha256": list(official_hashes),
                    "state_size": state_size,
                },
                "rejections": {
                    "bank_duplicate": 0,
                    "initial_success": 0,
                    "official_match": 0,
                    "settle_success": 0,
                },
                "suite": "libero_spatial",
                "task_id": 0,
                "task_name": "test task",
            }
        ],
    }
    return finalize_manifest(payload), {path: data}, states


def test_state_hash_is_canonical_little_endian_float64() -> None:
    values = np.asarray([1.25, -0.0, 3.5], dtype=np.float32)
    canonical = canonical_state(values)
    expected_bytes = np.asarray(values, dtype="<f8").tobytes(order="C")
    assert canonical.dtype.str == "<f8"
    assert canonical.flags.c_contiguous and canonical.flags.owndata
    assert canonical.tobytes(order="C") == expected_bytes
    assert state_sha256(values) == hashlib.sha256(expected_bytes).hexdigest()
    with pytest.raises(DevStateValidationError, match="non-finite"):
        canonical_state(np.asarray([0.0, np.nan]))
    with pytest.raises(DevStateValidationError, match="one-dimensional"):
        canonical_state(np.zeros((1, 2)))


def test_official_hashes_require_exactly_50_unique_full_states() -> None:
    size, hashes = canonical_official_state_hashes(_official_states())
    assert size == 4
    assert len(hashes) == len(set(hashes)) == 50
    duplicated = _official_states()
    duplicated[-1] = duplicated[0]
    with pytest.raises(DevStateValidationError, match="byte-identical"):
        canonical_official_state_hashes(duplicated)


def test_sampler_seed_is_stable_and_identity_sensitive() -> None:
    baseline = deterministic_sampler_seed(7, "libero_goal", 3, 9)
    assert baseline == deterministic_sampler_seed(7, "libero_goal", 3, 9)
    assert (
        len(
            {
                baseline,
                deterministic_sampler_seed(8, "libero_goal", 3, 9),
                deterministic_sampler_seed(7, "libero_spatial", 3, 9),
                deterministic_sampler_seed(7, "libero_goal", 4, 9),
                deterministic_sampler_seed(7, "libero_goal", 3, 10),
            }
        )
        == 5
    )


def test_sampler_round_trip_and_settle_are_byte_reproducible() -> None:
    environment = _FakeEnvironment()
    sampled = sample_reproducible_pre_settle_state(environment, sampler_seed=123, expected_size=4)
    validation = validate_pre_settle_candidate(environment, sampled, sampler_seed=123)
    assert validation.state_sha256 == state_sha256(sampled)
    expected_settled = sampled.copy()
    for _ in range(SETTLE_STEPS):
        expected_settled += np.asarray([0.001, 0.002, 0.003, 0.004])
    assert validation.settled_state_sha256 == state_sha256(expected_settled)

    with pytest.raises(DevStateValidationError, match="did not reproduce"):
        sample_reproducible_pre_settle_state(
            _FakeEnvironment(nondeterministic_reset=True),
            sampler_seed=123,
            expected_size=4,
        )
    with pytest.raises(DevStateValidationError, match="round trip"):
        validate_pre_settle_candidate(
            _FakeEnvironment(corrupt_round_trip=True),
            sampled,
            sampler_seed=123,
        )


def test_task_generation_excludes_official_duplicates_and_successful_resets() -> None:
    official_states = _official_states()
    state_size, official_hashes = canonical_official_state_hashes(official_states)
    seeds = [deterministic_sampler_seed(11, "libero_spatial", 0, attempt) for attempt in range(6)]
    accepted_first = np.asarray([1.0, 2.0, 3.0, 4.0])
    initial_success = np.asarray([5.0, 6.0, 7.0, 8.0])
    settle_success = np.asarray([9.0, 10.0, 11.0, 12.0])
    accepted_second = np.asarray([13.0, 14.0, 15.0, 16.0])
    environment = _FakeEnvironment(
        scripted_states={
            seeds[0]: official_states[0],
            seeds[1]: accepted_first,
            seeds[2]: accepted_first,
            seeds[3]: initial_success,
            seeds[4]: settle_success,
            seeds[5]: accepted_second,
        },
        initial_success_seeds={seeds[3]},
        settle_success_seeds={seeds[4]},
    )
    bank_hashes: set[str] = set()
    generated = generate_task_bank(
        environment,
        suite="libero_spatial",
        task_id=0,
        official_hashes=official_hashes,
        state_size=state_size,
        base_seed=11,
        states_per_task=2,
        max_attempts_per_task=6,
        bank_hashes=bank_hashes,
    )
    np.testing.assert_array_equal(generated.states, np.stack((accepted_first, accepted_second)))
    assert [entry["attempt_id"] for entry in generated.entries] == [1, 5]
    assert generated.rejections == {
        "bank_duplicate": 1,
        "initial_success": 1,
        "official_match": 1,
        "settle_success": 1,
    }
    assert bank_hashes == {state_sha256(accepted_first), state_sha256(accepted_second)}


def test_npy_manifest_and_root_hash_are_deterministic_and_loadable(tmp_path: Path) -> None:
    first_manifest, first_artifacts, states = _valid_bank()
    second_manifest, second_artifacts, _ = _valid_bank()
    assert first_manifest == second_manifest
    assert manifest_bytes(first_manifest) == manifest_bytes(second_manifest)
    assert first_artifacts == second_artifacts
    assert deterministic_npy_bytes(states) == deterministic_npy_bytes(states.copy())

    loaded_memory = validate_manifest_and_artifacts(first_manifest, first_artifacts)
    np.testing.assert_array_equal(loaded_memory["libero_spatial/task_00.npy"], states)
    output = tmp_path / "bank"
    manifest_digest = write_bank_exclusive(output, first_manifest, first_artifacts)
    assert manifest_digest == hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    loaded_manifest, loaded_disk = load_bank(output)
    assert loaded_manifest == first_manifest
    np.testing.assert_array_equal(loaded_disk["libero_spatial/task_00.npy"], states)
    assert not (output / "INCOMPLETE").exists()


def test_manifest_rejects_root_artifact_official_and_bank_duplicate_tampering() -> None:
    manifest, artifacts, states = _valid_bank()
    wrong_root = copy.deepcopy(manifest)
    wrong_root["base_seed"] += 1
    with pytest.raises(DevStateValidationError, match="root SHA-256"):
        validate_manifest_and_artifacts(wrong_root, artifacts)

    path = "libero_spatial/task_00.npy"
    tampered = bytearray(artifacts[path])
    tampered[-1] ^= 1
    with pytest.raises(DevStateValidationError, match="SHA-256 mismatch"):
        validate_manifest_and_artifacts(manifest, {path: bytes(tampered)})

    def replaced_states(replacement: np.ndarray) -> tuple[dict[str, Any], dict[str, bytes]]:
        changed = copy.deepcopy(manifest)
        data = deterministic_npy_bytes(replacement)
        changed["tasks"][0]["artifact"] = artifact_record(path, data, replacement.shape)
        for entry, state in zip(changed["tasks"][0]["entries"], replacement, strict=True):
            entry["state_sha256"] = state_sha256(state)
        changed.pop("root_sha256")
        return finalize_manifest(changed), {path: data}

    official = _official_states()
    official_match, official_artifacts = replaced_states(np.stack((official[0], states[1])))
    with pytest.raises(DevStateValidationError, match="matches an official"):
        validate_manifest_and_artifacts(official_match, official_artifacts)

    duplicate, duplicate_artifacts = replaced_states(np.stack((states[0], states[0])))
    with pytest.raises(DevStateValidationError, match="duplicates the bank"):
        validate_manifest_and_artifacts(duplicate, duplicate_artifacts)
