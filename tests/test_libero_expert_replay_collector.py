"""Focused tests for deterministic LIBERO expert-replay evidence primitives."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from duo_vla import libero_replay_evidence as EVIDENCE
from duo_vla.benchmarks.libero import make_libero_action_chunk
from duo_vla.data import libero_stats

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "collect_libero_expert_replay",
    SCRIPTS / "collect_libero_expert_replay.py",
)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)
BINDER_SPEC = importlib.util.spec_from_file_location(
    "bind_libero_expert_replay",
    SCRIPTS / "bind_libero_expert_replay.py",
)
assert BINDER_SPEC is not None and BINDER_SPEC.loader is not None
BINDER = importlib.util.module_from_spec(BINDER_SPEC)
BINDER_SPEC.loader.exec_module(BINDER)


def test_pinned_original_hdf5_inventory_has_exact_semantic_and_raw_identity() -> None:
    path = ROOT / "configs/libero_original_hdf5_inventory.json"
    inventory, records, raw_sha256 = EVIDENCE.load_original_hdf5_inventory(path)

    assert inventory["content_sha256"] == EVIDENCE.ORIGINAL_HDF5_CONTENT_SHA256
    assert raw_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(records) == inventory["file_count"] == EVIDENCE.ORIGINAL_HDF5_FILE_COUNT == 40
    assert sum(item["bytes"] for item in records) == EVIDENCE.ORIGINAL_HDF5_TOTAL_BYTES
    assert [(item["suite"], item["task_id"]) for item in records] == [
        (suite, task_id) for suite in EVIDENCE.SUITES for task_id in range(10)
    ]


def test_noop_filter_matches_previous_retained_action_contract() -> None:
    actions = np.zeros((6, EVIDENCE.ACTION_DIM), dtype=np.float64)
    actions[:, -1] = -1.0
    actions[1, -1] = 1.0
    actions[2, 0] = EVIDENCE.NOOP_THRESHOLD / 2
    actions[2, -1] = 1.0
    actions[3, 0] = EVIDENCE.NOOP_THRESHOLD
    actions[3, -1] = 1.0
    actions[4, -1] = -1.0
    actions[5, 1] = -0.5

    assert EVIDENCE.retained_action_indices(actions) == (3, 4, 5)
    assert EVIDENCE.is_noop(actions[0], None)
    assert not EVIDENCE.is_noop(actions[1], actions[0])
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="no retained transition"):
        EVIDENCE.retained_action_indices(np.zeros((2, EVIDENCE.ACTION_DIM), dtype=np.float32))


def test_action_and_observation_digests_are_canonical_and_order_sensitive() -> None:
    actions = np.arange(21, dtype=np.float64).reshape(3, 7) / 10.0
    assert EVIDENCE.action_sequence_sha256(actions) == EVIDENCE.action_sequence_sha256(actions.astype(np.float32))
    assert EVIDENCE.action_sequence_sha256(actions) != EVIDENCE.action_sequence_sha256(actions[::-1])

    agentview = np.arange(np.prod(EVIDENCE.IMAGE_SHAPE), dtype=np.uint8).reshape(EVIDENCE.IMAGE_SHAPE)
    wrist = np.bitwise_xor(agentview, np.uint8(0xFF))
    state = np.arange(EVIDENCE.STATE_DIM, dtype=np.float32)
    first = EVIDENCE.ObservationSequenceDigester()
    first.update(agentview, wrist, state)
    second = EVIDENCE.ObservationSequenceDigester()
    second.update(wrist, agentview, state)
    assert first.count == 1
    assert first.hexdigest() != second.hexdigest()


def test_camera_rotation_is_exact_contiguous_single_180_degree_transform() -> None:
    image = np.arange(np.prod(EVIDENCE.IMAGE_SHAPE), dtype=np.uint8).reshape(EVIDENCE.IMAGE_SHAPE)
    rotated = EVIDENCE.rotate_simulator_rgb_for_training(image)

    assert np.array_equal(rotated, image[::-1, ::-1])
    assert rotated.flags.c_contiguous
    assert all(stride > 0 for stride in rotated.strides)
    assert np.array_equal(EVIDENCE.rotate_simulator_rgb_for_training(rotated), image)
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="native uint8"):
        EVIDENCE.rotate_simulator_rgb_for_training(image.astype(np.float32))


def test_demo_names_are_numeric_contiguous_and_not_lexicographic() -> None:
    assert EVIDENCE.sorted_demo_names(["demo_2", "demo_0", "demo_1"]) == (
        "demo_0",
        "demo_1",
        "demo_2",
    )
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="not contiguous"):
        EVIDENCE.sorted_demo_names(["demo_0", "demo_2"])
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="invalid HDF5 demo name"):
        EVIDENCE.sorted_demo_names(["demo_final"])


def test_source_root_inventory_rejects_extra_files_and_links(tmp_path: Path) -> None:
    _inventory, records, _raw_sha256 = EVIDENCE.load_original_hdf5_inventory(
        ROOT / "configs/libero_original_hdf5_inventory.json"
    )
    for record in records:
        path = tmp_path / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    COLLECTOR._source_root_inventory(tmp_path, records)

    extra = tmp_path / "extra.hdf5"
    extra.write_bytes(b"")
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="inventory differs"):
        COLLECTOR._source_root_inventory(tmp_path, records)
    extra.unlink()

    extra_directory = tmp_path / "unexpected-directory"
    extra_directory.mkdir()
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="inventory differs"):
        COLLECTOR._source_root_inventory(tmp_path, records)
    extra_directory.rmdir()

    fifo = tmp_path / "unexpected.fifo"
    os.mkfifo(fifo)
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="non-regular entry"):
        COLLECTOR._source_root_inventory(tmp_path, records)
    fifo.unlink()

    link = tmp_path / "linked.hdf5"
    link.symlink_to(tmp_path / records[0]["path"])
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="symbolic link"):
        COLLECTOR._source_root_inventory(tmp_path, records)


def test_regular_file_identity_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"content")
    assert EVIDENCE.stable_regular_file_identity(source) == {
        "bytes": 7,
        "sha256": hashlib.sha256(b"content").hexdigest(),
    }

    symlink = tmp_path / "symlink.bin"
    symlink.symlink_to(source)
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="without following links"):
        EVIDENCE.stable_regular_file_identity(symlink)

    hardlink = tmp_path / "hardlink.bin"
    os.link(source, hardlink)
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="exactly one hard link"):
        EVIDENCE.stable_regular_file_identity(source)


def test_production_normalizers_and_dataset_sample_drive_real_terminal_gates(tmp_path: Path) -> None:
    normalization = {
        "action": {"q01": [-1.0] * 6, "q99": [1.0] * 6},
        "dataset": {"id": "HuggingFaceVLA/libero", "revision": libero_stats.LIBERO_DATASET_REVISION},
        "schema": libero_stats.LIBERO_STATS_SCHEMA,
        "state": {"q01": [-2.0] * 8, "q99": [2.0] * 8},
    }
    normalization["content_sha256"] = libero_stats._content_hash(normalization)
    artifact = tmp_path / "normalization.json"
    artifact.write_text(json.dumps(normalization), encoding="utf-8")
    actions = torch.tensor(
        [
            [0.2, -0.3, 0.4, -0.5, 0.6, -0.7, -1.0],
            [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 1.0],
            [0.3, -0.2, 0.1, -0.4, 0.5, -0.6, 1.0],
        ],
        dtype=torch.float32,
    )
    state = torch.linspace(-1.5, 1.5, 8)

    class Dataset:
        episodes = (SimpleNamespace(length=3, task="instruction"),)

        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int]] = []

        def sample(self, episode_index: int, frame_index: int, *, horizon: int):
            self.calls.append((episode_index, frame_index, horizon))
            return SimpleNamespace(
                action_chunk=make_libero_action_chunk(actions, frame_index, horizon=horizon),
                episode_index=episode_index,
                frame_index=frame_index,
                instruction="instruction",
                observation=SimpleNamespace(state=state),
            )

        def _read_episode_rows(self, _episode):
            return actions

        @staticmethod
        def _validated_episode_actions(rows):
            return rows

    dataset = Dataset()
    round_trip, boundary = BINDER._production_dataset_gates(dataset, 0, artifact, normalization)

    assert dataset.calls == [(0, 2, 8)]
    assert round_trip["gripper_sign_exact"] is True
    assert round_trip["max_abs_error"] <= 1e-6
    assert boundary == {
        "checked_terminal_anchors": 1,
        "cross_boundary_count": 0,
        "horizon": 8,
        "padding_mask_exact": True,
        "padding_value": "zeros",
    }


def _fake_observation(value: int = 0) -> dict[str, np.ndarray]:
    agentview = np.zeros(EVIDENCE.IMAGE_SHAPE, dtype=np.uint8)
    wrist = np.ones(EVIDENCE.IMAGE_SHAPE, dtype=np.uint8)
    agentview[0, 0, 0] = value
    return {
        "agentview_image": agentview,
        "fake_state": np.full(EVIDENCE.STATE_DIM, float(value), dtype=np.float32),
        "robot0_eye_in_hand_image": wrist,
    }


class _FakeReplayEnvironment:
    def __init__(self, done_results: list[bool] | None = None) -> None:
        self.events: list[str] = []
        self.raw_actions: list[list[float]] = []
        self.closed = False
        self.done_results = list(done_results or [True])
        self.settle_remaining = EVIDENCE.SETTLE_STEPS

    def seed(self, seed: int) -> None:
        self.events.append(f"seed:{seed}")

    def reset(self) -> None:
        self.events.append("reset")

    def set_init_state(self, _state: np.ndarray) -> dict[str, np.ndarray]:
        self.events.append("set_init_state")
        self.settle_remaining = EVIDENCE.SETTLE_STEPS
        return _fake_observation()

    def get_sim_state(self) -> np.ndarray:
        return np.asarray([1.0, 2.0], dtype=np.float64)

    def step(self, action):
        if self.settle_remaining > 0:
            self.settle_remaining -= 1
            self.events.append("settle_step")
            return _fake_observation(), 0.0, False, {}
        self.events.append("replay_step")
        self.raw_actions.append(np.asarray(action).copy())
        return _fake_observation(7), 0.0, self.done_results.pop(0), {}

    def close(self) -> None:
        self.closed = True


def test_replay_seeds_once_and_dispatches_source_precision_without_float32_roundtrip(monkeypatch) -> None:
    monkeypatch.setattr(COLLECTOR, "canonical_proprioceptive_state", lambda observation: observation["fake_state"])
    environment = _FakeReplayEnvironment()
    COLLECTOR._seed_environment(environment)
    raw_value = np.nextafter(np.float64(0.12345678912345678), np.float64(1.0))
    assert raw_value != np.float64(np.float32(raw_value))
    action = np.asarray([[raw_value, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    demo = {"actions": action, "states": np.asarray([[1.0, 2.0]], dtype=np.float64)}

    replay = COLLECTOR._replay_demo(
        environment,
        demo,
        suite="libero_spatial",
        task_id=0,
        source_episode_index=0,
    )

    assert environment.events.count("seed:0") == 1
    assert environment.events[:3] == ["seed:0", "reset", "set_init_state"]
    assert environment.raw_actions[0][0] == float(raw_value)
    assert environment.raw_actions[0][0] != float(np.float32(raw_value))
    assert replay["action_sequence_sha256"] == EVIDENCE.action_sequence_sha256(action.astype(np.float32))


def test_first_success_replay_consumes_demo_resets_in_order_without_reseeding(monkeypatch) -> None:
    monkeypatch.setattr(COLLECTOR, "canonical_proprioceptive_state", lambda observation: observation["fake_state"])
    environment = _FakeReplayEnvironment([False, True])
    COLLECTOR._seed_environment(environment)
    first_action = np.asarray([[0.25, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float64)
    second_action = np.asarray([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    states = np.asarray([[1.0, 2.0]], dtype=np.float64)
    data = {
        "demo_0": {"actions": first_action, "states": states},
        "demo_1": {"actions": second_action, "states": states},
    }
    scan = {
        "demonstrations": [
            {
                "action_sequence_sha256": EVIDENCE.action_sequence_sha256(first_action),
                "initial_state_sha256": EVIDENCE.initial_state_sha256(states[0]),
            },
            {
                "action_sequence_sha256": EVIDENCE.action_sequence_sha256(second_action),
                "initial_state_sha256": EVIDENCE.initial_state_sha256(states[0]),
            },
        ]
    }

    attempts, selected = COLLECTOR._replay_first_success(
        environment,
        data,
        ("demo_0", "demo_1"),
        scan,
        suite="libero_spatial",
        task_id=0,
    )

    assert environment.events.count("seed:0") == 1
    assert environment.events.count("reset") == 2
    assert [attempt["source_episode_index"] for attempt in attempts] == [0, 1]
    assert [attempt["success"] for attempt in attempts] == [False, True]
    assert selected["source_episode_index"] == 1


def test_reset_determinism_uses_two_isolated_freshly_seeded_environments(monkeypatch) -> None:
    environments = [_FakeReplayEnvironment(), _FakeReplayEnvironment()]
    monkeypatch.setattr(COLLECTOR, "_construct_environment", lambda _task: environments.pop(0))
    monkeypatch.setattr(COLLECTOR, "canonical_proprioceptive_state", lambda observation: observation["fake_state"])
    owned = environments.copy()

    result = COLLECTOR._validate_reset_determinism(object(), np.asarray([1.0, 2.0]))

    assert result["environment_seed"] == 0
    assert result["probe_environment_count"] == 2
    assert result["seed_calls_per_environment"] == 1
    assert result["settle_steps"] == EVIDENCE.SETTLE_STEPS
    assert all(environment.events.count("seed:0") == 1 for environment in owned)
    assert all(environment.events[0] == "seed:0" and environment.closed for environment in owned)


def test_pre_dispatch_integrity_controls_use_mutation_detected_semantics() -> None:
    selected = {
        "action_sequence_sha256": "a",
        "inverted_gripper_action_sequence_sha256": "b",
        "observation_sequence_sha256": "c",
        "swapped_observation_sequence_sha256": "d",
        "zero_action_sequence_sha256": "e",
    }
    assert COLLECTOR._pre_dispatch_integrity_controls(selected, "instruction") == {
        "inverted_gripper": {"mutation_detected": True},
        "mismatched_language": {"mutation_detected": True},
        "swapped_cameras": {"mutation_detected": True},
        "zero_action": {"mutation_detected": True},
    }


def _valid_simulator_task_document() -> dict[str, object]:
    digest = "a" * 64
    return {
        "attempts": [
            {
                "action_sequence_sha256": digest,
                "source_episode_index": 0,
                "step_count": 1,
                "success": True,
            }
        ],
        "collector_source_sha256": "b" * 64,
        "pre_dispatch_integrity_controls": {
            name: {"mutation_detected": True} for name in BINDER.PRE_DISPATCH_MUTATIONS
        },
        "reset_determinism": {
            "environment_seed": 0,
            "first_observation_sha256": digest,
            "first_simulator_state_sha256": digest,
            "probe_environment_count": 2,
            "second_observation_sha256": digest,
            "second_simulator_state_sha256": digest,
            "seed_calls_per_environment": 1,
            "settle_steps": 10,
        },
        "schema": EVIDENCE.SIMULATOR_TASK_SCHEMA,
        "selected": {
            "action_sequence_sha256": digest,
            "alignment_probe": {
                "post_action_observation_sha256": "c" * 64,
                "pre_action_observation_sha256": "d" * 64,
                "retained_transition_index": 0,
                "source_transition_index": 0,
            },
            "initial_state_sha256": digest,
            "inverted_gripper_action_sequence_sha256": "e" * 64,
            "observation_sequence_sha256": digest,
            "source_episode_index": 0,
            "step_count": 1,
            "success": True,
            "swapped_observation_sequence_sha256": "f" * 64,
            "trajectory_sha256": digest,
            "zero_action_sequence_sha256": "0" * 64,
        },
        "source_file": {
            "bytes": 1,
            "path": "libero_spatial/task_demo.hdf5",
            "sha256": digest,
            "suite": "libero_spatial",
            "task_id": 0,
            "task_name": "task",
        },
        "source_scan": {
            "demonstrations": [
                {
                    "action_sequence_sha256": digest,
                    "initial_state_sha256": digest,
                    "raw_transition_count": 1,
                    "retained_transition_count": 1,
                    "source_action_dtype": "<f8",
                    "source_episode_index": 0,
                    "source_state_sequence_sha256": digest,
                }
            ],
            "gripper_counts": {"1": 1},
            "raw_transition_count": 1,
            "retained_transition_count": 1,
        },
        "task": {
            "instruction": "instruction",
            "suite": "libero_spatial",
            "task_id": 0,
            "task_name": "task",
        },
    }


def test_binder_requires_exact_simulator_task_fields_and_mutation_shape() -> None:
    document = _valid_simulator_task_document()
    BINDER._validate_simulator_task(document, suite="libero_spatial", task_id=0)

    with_extra = dict(document)
    with_extra["unexpected"] = True
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="fields differ"):
        BINDER._validate_simulator_task(with_extra, suite="libero_spatial", task_id=0)

    old_control_shape = json.loads(json.dumps(document))
    old_control_shape["pre_dispatch_integrity_controls"]["zero_action"] = {"mutation_detected": False}
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="integrity mutation"):
        BINDER._validate_simulator_task(old_control_shape, suite="libero_spatial", task_id=0)


def test_hdf5_scan_hashes_retained_actions_and_rejects_soft_links(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    valid = tmp_path / "valid.hdf5"
    actions = np.zeros((3, 7), dtype=np.float64)
    actions[:, -1] = [-1.0, 1.0, 1.0]
    actions[2, 0] = 0.5
    states = np.arange(15, dtype=np.float64).reshape(3, 5)
    with h5py.File(valid, "w") as output:
        demo = output.create_group("data/demo_0")
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("states", data=states)

    report = COLLECTOR.scan_hdf5_task(valid)
    assert report["raw_transition_count"] == 3
    assert report["retained_transition_count"] == 1
    assert report["gripper_counts"] == {"-1": 1, "1": 2}
    assert report["demonstrations"][0]["action_sequence_sha256"] == EVIDENCE.action_sequence_sha256(actions[2:])

    linked = tmp_path / "linked.hdf5"
    with h5py.File(linked, "w") as output:
        output.create_group("target")
        output["data"] = h5py.SoftLink("/target")
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="soft/external link"):
        COLLECTOR.scan_hdf5_task(linked)


def test_collect_and_bind_launchers_are_closed_and_executable() -> None:
    for program in ("collect_libero_expert_replay.py", "bind_libero_expert_replay.py"):
        source = (ROOT / "scripts" / program).read_text(encoding="utf-8")
        assert "qualify_libero_expert_replay import" not in source
    assert BINDER.build_expected_inputs.__module__ == "duo_vla.libero_replay_evidence"

    for name, venv in (
        ("run_collect_libero_expert_replay.sh", "libero-eval"),
        ("run_bind_libero_expert_replay.sh", "train"),
        ("run_libero_preflight.sh", "libero-eval"),
    ):
        path = ROOT / "scripts" / name
        launcher = path.read_text(encoding="utf-8")
        assert launcher.startswith("#!/bin/bash -p\nset -euo pipefail\n")
        assert 'export PATH="/usr/bin:/bin"' in launcher
        assert '[[ "${name}" == LD_* ]] && unset "${name}"' in launcher
        assert "exec /usr/bin/env -i" in launcher
        assert "PYTHONPATH" not in launcher
        assert '"PYTHONSAFEPATH=1"' in launcher
        assert '"PYTHONDONTWRITEBYTECODE=1"' in launcher
        assert " -P -B -X pycache_prefix=/dev/null " in launcher
        assert f'readonly environment_path="${{cache_root}}/venvs/{venv}"' in launcher
        assert os.access(path, os.X_OK)
    bootstrap = (ROOT / "scripts/bootstrap_libero_env.sh").read_text(encoding="utf-8")
    assert '"${project_dir}/scripts/run_libero_preflight.sh"' in bootstrap


def test_inventory_rejects_semantic_content_tampering() -> None:
    path = ROOT / "configs/libero_original_hdf5_inventory.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["files"][0]["bytes"] += 1
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="semantic hash differs"):
        EVIDENCE.validate_original_hdf5_inventory(changed)


def test_shared_strict_json_rejects_exponent_overflow(tmp_path: Path) -> None:
    path = tmp_path / "overflow.json"
    path.write_text('{"value":1e999}', encoding="utf-8")
    with pytest.raises(EVIDENCE.ReplayEvidenceError, match="non-finite"):
        EVIDENCE.load_strict_json(path, name="overflow fixture")
    with path.open("rb") as source, pytest.raises(EVIDENCE.ReplayEvidenceError, match="non-finite"):
        EVIDENCE._read_descriptor_json(source.fileno(), name="overflow descriptor")


def test_project_import_root_rejects_top_level_qualifier_shadow_and_direct_pyc(tmp_path: Path) -> None:
    project = tmp_path / "project"
    package = project / "src/duo_vla"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    shadow = project / "src/qualify_libero_expert_replay.py"
    shadow.write_text("raise RuntimeError('shadow executed')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="only the real duo_vla"):
        BINDER._validate_project_source_root(project)
    shadow.unlink()

    rogue_pyc = package / "rogue.pyc"
    rogue_pyc.write_bytes(b"poison")
    with pytest.raises(RuntimeError, match="forbidden file"):
        BINDER._validate_project_source_root(project)


def test_checkout_source_precedes_compatible_site_packages_shadow_and_preloaded_shadow_fails(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout_package = checkout / "src/duo_vla"
    checkout_package.mkdir(parents=True)
    (checkout_package / "__init__.py").write_text("ORIGIN = 'checkout'\n", encoding="utf-8")
    shadow_site = tmp_path / "site-packages"
    shadow_package = shadow_site / "duo_vla"
    shadow_package.mkdir(parents=True)
    (shadow_package / "__init__.py").write_text("ORIGIN = 'shadow'\n", encoding="utf-8")
    code = """
import importlib
import importlib.util
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
checkout = pathlib.Path(sys.argv[2])
shadow_site = pathlib.Path(sys.argv[3])
spec = importlib.util.spec_from_file_location('binder_bootstrap', root / 'scripts/bind_libero_expert_replay.py')
binder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(binder)

def purge():
    for name in list(sys.modules):
        if name == 'duo_vla' or name.startswith('duo_vla.'):
            del sys.modules[name]
    importlib.invalidate_caches()

purge()
sys.path[:] = [str(shadow_site), str(checkout / 'src')]
import duo_vla
assert duo_vla.ORIGIN == 'shadow'

purge()
sys.path[:] = [str(shadow_site), str(checkout / 'src')]
source_root = binder._validate_project_source_root(checkout)
assert sys.path == [str(checkout / 'src'), str(shadow_site)]
import duo_vla
assert duo_vla.ORIGIN == 'checkout'
binder._validate_project_module_origins(source_root, {'duo_vla'})

purge()
sys.path[:] = [str(shadow_site), str(checkout / 'src')]
import duo_vla
assert duo_vla.ORIGIN == 'shadow'
binder._validate_project_source_root(checkout)
try:
    binder._validate_project_module_origins(source_root, {'duo_vla'})
except RuntimeError as exc:
    assert 'escapes authenticated source root' in str(exc)
else:
    raise AssertionError('preloaded shadow duo_vla was accepted')
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-P",
            "-B",
            "-X",
            "pycache_prefix=/dev/null",
            "-c",
            code,
            str(ROOT),
            str(checkout),
            str(shadow_site),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_safe_path_ignores_scripts_json_shadow_and_valid_adjacent_poisoned_pyc(tmp_path: Path) -> None:
    eval_python = Path("/root/.cache/duo-vla/venvs/libero-eval/bin/python")
    if not eval_python.is_file():
        pytest.skip("pinned libero-eval interpreter is unavailable")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    marker = tmp_path / "json-shadow-ran"
    (scripts / "json.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    runner = scripts / "runner.py"
    runner.write_text("import json\nprint(json.__file__)\n", encoding="utf-8")
    environment = {
        "PATH": f"{eval_python.parent}:/usr/bin:/bin",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [str(eval_python), "-P", "-B", "-X", "pycache_prefix=/dev/null", str(runner)],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert not marker.exists()
    assert str(scripts / "json.py") not in completed.stdout

    import_root = tmp_path / "poisoned-import-root"
    import_root.mkdir()
    source = import_root / "victim.py"
    source.write_text("VALUE = 'EVIL'\n", encoding="utf-8")
    source_stat = source.stat()
    subprocess.run(
        [
            str(eval_python),
            "-B",
            "-c",
            "import py_compile,sys; py_compile.compile(sys.argv[1], doraise=True)",
            str(source),
        ],
        check=True,
        env=environment,
    )
    source.write_text("VALUE = 'SAFE'\n", encoding="utf-8")
    os.utime(source, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
    poisoned = subprocess.run(
        [
            str(eval_python),
            "-P",
            "-B",
            "-X",
            "pycache_prefix=/dev/null",
            "-c",
            "import sys; sys.path.append(sys.argv[1]); import victim; print(victim.VALUE)",
            str(import_root),
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert poisoned.stdout.strip() == "SAFE"


def test_python_side_runtime_rejects_manual_invocation_without_explicit_p_flag() -> None:
    eval_python = Path("/root/.cache/duo-vla/venvs/libero-eval/bin/python")
    if not eval_python.is_file():
        pytest.skip("pinned libero-eval interpreter is unavailable")
    cache_root = Path("/root/.cache/duo-vla")
    environment = {
        "DUO_VLA_CACHE_ROOT": str(cache_root),
        "HF_HOME": "/root/.cache/huggingface",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LIBERO_CONFIG_PATH": str(cache_root / "simulators/libero/config"),
        "MKL_NUM_THREADS": "1",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "MUJOCO_GL": "egl",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": f"{eval_python.parent}:/usr/bin:/bin",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    code = (
        "import importlib.util, pathlib, sys; "
        "root=pathlib.Path(sys.argv[1]); "
        "spec=importlib.util.spec_from_file_location('safe_preflight', root/'scripts/preflight_libero_env.py'); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "module.activate_project_source_root(root); "
        "module.validate_process_environment(root, pathlib.Path(sys.argv[2]))"
    )
    completed = subprocess.run(
        [
            str(eval_python),
            "-B",
            "-X",
            "pycache_prefix=/dev/null",
            "-c",
            code,
            str(ROOT),
            str(cache_root),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert completed.returncode != 0
    assert "exact safe Python flags" in completed.stderr


def test_full_training_gripper_scan_uses_production_action_validator_for_every_episode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Physical:
        chunk_index = 0
        file_index = 0
        global_start = 0

    physical = Physical()
    lengths = [1] * 1692 + [273465 - 1692]
    episodes = []
    start = 0
    for episode_index, length in enumerate(lengths):
        episodes.append(SimpleNamespace(episode_index=episode_index, global_start=start, length=length))
        start += length

    class Table:
        def __init__(self, length: int = 273465) -> None:
            self.length = length

        def slice(self, _offset: int, length: int):
            return Table(length)

    class Parquet:
        @staticmethod
        def read_table(_path: Path, *, columns: list[str]):
            assert columns == ["action"]
            return Table()

    class Dataset:
        root = tmp_path
        data_files = (physical,)

        def __init__(self) -> None:
            self.episodes = tuple(episodes)
            self._data_file_by_episode = {episode.episode_index: physical for episode in episodes}
            self.validation_calls = 0
            self.validated_actions = 0

        def _validated_episode_actions(self, rows: Table) -> torch.Tensor:
            self.validation_calls += 1
            self.validated_actions += rows.length
            actions = torch.zeros((rows.length, 7), dtype=torch.float32)
            actions[:, -1] = -1.0 if self.validation_calls == 1 else 1.0
            return actions

    dataset = Dataset()
    monkeypatch.setattr(BINDER, "_pyarrow_parquet", lambda: Parquet())
    values, validated_episodes, validated_actions = BINDER._scan_training_grippers(dataset)
    assert values == [-1.0, 1.0]
    assert validated_episodes == dataset.validation_calls == 1693
    assert validated_actions == dataset.validated_actions == 273465
