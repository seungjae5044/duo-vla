from __future__ import annotations

import math
import stat
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_libero import (
    parse_selection,
    rotate_eval_rgb_180,
    run_episode,
    run_policy_warmups,
    summarize_episodes,
    validate_policy_health,
    wilson95,
)
from libero_bridge import (
    LIBERO_EXECUTION_GEOMETRY,
    SCHEMA,
    BridgeProtocolError,
    PolicyClient,
    decode_rgb,
    encode_rgb,
    libero_replan_seed,
    make_predict_request,
    serving_execution_geometry,
    validate_action_response,
    validate_execution_geometry,
    validate_request,
    validate_serving_execution_geometry,
    wait_for_socket,
)
from serve_libero_policy import run_fake_server, select_serving_policy_contract

from duo_vla.benchmarks.libero_dev_states import state_sha256


def _images() -> tuple[np.ndarray, np.ndarray]:
    pixels = np.arange(256 * 256 * 3, dtype=np.uint32).reshape(256, 256, 3)
    return (pixels % 251).astype(np.uint8), ((pixels * 3 + 17) % 251).astype(np.uint8)


@contextmanager
def _running_fake_server(socket_path: Path, *, train_seed: int) -> Iterator[threading.Thread]:
    """Always release a test server, including while another assertion is unwinding."""

    server = threading.Thread(
        target=run_fake_server,
        kwargs={"socket_path": socket_path, "train_seed": train_seed},
        daemon=True,
    )
    server.start()
    try:
        wait_for_socket(socket_path, timeout_seconds=2.0)
        yield server
    finally:
        if server.is_alive() and socket_path.exists():
            try:
                with PolicyClient(socket_path, timeout_seconds=1.0) as cleanup_client:
                    cleanup_client.health()
                    cleanup_client.shutdown()
            except Exception:
                # The daemon flag is the final guard if the test intentionally corrupts the connection.
                pass
        server.join(timeout=2.0)


def test_fake_server_cleanup_survives_an_assertion_failure(tmp_path: Path) -> None:
    socket_path = tmp_path / "cleanup.sock"
    server: threading.Thread | None = None
    with (
        pytest.raises(AssertionError, match="intentional"),
        _running_fake_server(socket_path, train_seed=5) as running,
    ):
        server = running
        raise AssertionError("intentional cleanup test")
    assert server is not None and not server.is_alive()
    assert not socket_path.exists()


def test_v3_crn_seed_depends_only_on_evaluation_reset_identity() -> None:
    assert tuple(signature(libero_replan_seed).parameters) == (
        "evaluation_seed",
        "suite",
        "task_id",
        "reset_source",
        "reset_id",
        "reset_state_sha256",
        "replan_id",
    )
    image, wrist = _images()
    common = {
        "agentview_rgb": image,
        "evaluation_seed": 91,
        "instruction": "test instruction",
        "replan_id": 1234,
        "request_id": "crn",
        "reset_id": 49,
        "reset_source": "official",
        "reset_state_sha256": None,
        "state": np.zeros(8, dtype=np.float32),
        "suite": "libero_goal",
        "task_id": 9,
        "wrist_rgb": wrist,
    }
    k1 = make_predict_request(**common, execution_horizon=1, train_seed=1)
    k4 = make_predict_request(**common, execution_horizon=4, train_seed=2)
    assert k1["inference_seed"] == k4["inference_seed"]
    assert k1["inference_seed"] == libero_replan_seed(91, "libero_goal", 9, "official", 49, None, 1234)

    changed_evaluation = make_predict_request(**{**common, "evaluation_seed": 92}, execution_horizon=1, train_seed=1)
    changed_reset = make_predict_request(**{**common, "reset_id": 48}, execution_horizon=1, train_seed=1)
    clean_dev = make_predict_request(
        **{
            **common,
            "reset_id": 0,
            "reset_source": "clean-dev",
            "reset_state_sha256": "a" * 64,
        },
        execution_horizon=1,
        train_seed=1,
    )
    assert (
        len(
            {
                k1["inference_seed"],
                changed_evaluation["inference_seed"],
                changed_reset["inference_seed"],
                clean_dev["inference_seed"],
            }
        )
        == 4
    )

    with pytest.raises(BridgeProtocolError, match=r"\[0, 2\^63\)"):
        libero_replan_seed(-1, "libero_spatial", 0, "official", 0, None, 0)


def test_lossless_rgb_encoding_rejects_tampering() -> None:
    image, _ = _images()
    encoded = encode_rgb(image)
    assert np.array_equal(decode_rgb(encoded, name="camera"), image)
    encoded["sha256"] = "0" * 64
    with pytest.raises(BridgeProtocolError, match="SHA-256 mismatch"):
        decode_rgb(encoded, name="camera")


def test_rotation_is_exactly_once_positive_stride_copy() -> None:
    image, _ = _images()
    rotated = rotate_eval_rgb_180(image)
    assert np.array_equal(rotated, image[::-1, ::-1])
    assert rotated.flags.c_contiguous
    rotated[0, 0, 0] ^= 1
    assert not np.shares_memory(rotated, image)


def test_fake_policy_persistent_socket_is_private_and_deterministic(tmp_path: Path) -> None:
    socket_path = tmp_path / "policy.sock"
    with _running_fake_server(socket_path, train_seed=17) as server:
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        agentview, wrist = _images()
        request = {
            "evaluation_seed": 101,
            "suite": "libero_goal",
            "task_id": 3,
            "reset_source": "official",
            "reset_id": 7,
            "reset_state_sha256": None,
            "replan_id": 2,
            "execution_horizon": 4,
            "train_seed": 17,
            "instruction": "a deterministic test instruction",
            "agentview_rgb": agentview,
            "wrist_rgb": wrist,
            "state": np.arange(8, dtype=np.float32),
        }
        with PolicyClient(socket_path, timeout_seconds=2.0) as client:
            health = client.health()
            first, first_response = client.predict(**request)
            second, second_response = client.predict(**request)
            assert health["mode"] == "fake"
            assert health["train_seed"] == 17
            assert health["schema"] == "duo-vla-libero-policy-ipc-v5"
            assert health["execution_geometry"] is None
            assert health["training_execution_geometry"] is None
            assert health["serving_execution_geometry"] is None
            assert health["latency_runtime_sha256"] is None
            assert health["serving_runtime_sha256"] is None
            assert health["objective"] == "test_fake"
            assert health["sampler"] == "seeded_test_normal"
            assert health["nfe"] == 0
            assert health["inference_seed_behavior"] == "episode_identity_test_generator"
            assert np.array_equal(first, second)
            assert first.shape == (8, 7)
            assert set(first[:, 6]) <= {-1.0, 1.0}
            assert first_response["inference_seed"] == second_response["inference_seed"]
            assert first_response["evaluation_seed"] == request["evaluation_seed"]
            assert first_response["reset_source"] == request["reset_source"]
            assert first_response["reset_id"] == request["reset_id"]
            assert first_response["reset_state_sha256"] is None
            assert first_response["objective"] == health["objective"]
            assert first_response["sampler"] == health["sampler"]
            assert first_response["nfe"] == health["nfe"]
            client.shutdown()
    assert not server.is_alive()
    assert not socket_path.exists()


def test_policy_warmups_are_recorded_but_separate_from_episode_metrics(tmp_path: Path) -> None:
    socket_path = tmp_path / "warmup.sock"
    with (
        _running_fake_server(socket_path, train_seed=17) as server,
        PolicyClient(socket_path, timeout_seconds=2.0) as client,
    ):
        health = client.health()
        k1_reports = run_policy_warmups(
            client,
            health,
            count=2,
            execution_horizon=1,
            evaluation_seed=101,
        )
        reports = run_policy_warmups(
            client,
            health,
            count=2,
            execution_horizon=4,
            evaluation_seed=101,
        )
        client.shutdown()

    assert not server.is_alive()
    assert [report["warmup_index"] for report in reports] == [0, 1]
    assert all("health" not in report for report in reports)
    assert reports[0]["actions_sha256"] == reports[1]["actions_sha256"]
    assert reports[0]["inference_seed"] == reports[1]["inference_seed"]
    assert reports[0]["actions_sha256"] == k1_reports[0]["actions_sha256"]
    assert reports[0]["inference_seed"] == k1_reports[0]["inference_seed"]
    with pytest.raises(RuntimeError, match="at least one"):
        run_policy_warmups(
            client,
            health,
            count=0,
            execution_horizon=4,
            evaluation_seed=101,
        )


def test_evaluator_requires_content_addressed_runtime_for_real_policy() -> None:
    contract = {
        "objective": "rectified_flow",
        "sampler": "euler_uniform",
        "nfe": 10,
        "inference_seed_behavior": "episode_identity_gaussian_noise",
    }
    health = {
        **contract,
        "checkpoint": {
            "execution_geometry": LIBERO_EXECUTION_GEOMETRY,
            "kind": "resumable-libero-training",
            "manifest_sha256": "a" * 64,
            "policy_contract": contract,
            "policy_contract_sha256": "b" * 64,
            "source_tree_sha256": "d" * 64,
            "train_seed": 0,
        },
        "dataset_revision": "86958911c0f959db2bbbdb107eb3e17c5f9c798e",
        "execution_geometry": LIBERO_EXECUTION_GEOMETRY,
        "latency_runtime_sha256": "e" * 64,
        "mode": "real",
        "model_revision": "f7f5b7f5fa82ffc52addd066915886d497f5517b",
        "normalization_content_sha256": "a972b5d95a8aaa8ae7582bafcbc071261979cb46c2a3515b4da7a7cf0156ac73",
        "prefix_cache_scope": "request",
        "serving_runtime_sha256": "c" * 64,
        "train_seed": 0,
    }

    validate_policy_health(health, allow_fake_policy=False)
    changed_scope = dict(health, prefix_cache_scope="episode")
    with pytest.raises(RuntimeError, match="prefix cache scope"):
        validate_policy_health(changed_scope, allow_fake_policy=False)
    changed_latency_runtime = dict(health, latency_runtime_sha256=None)
    with pytest.raises(RuntimeError, match="latency runtime hash"):
        validate_policy_health(changed_latency_runtime, allow_fake_policy=False)
    health["serving_runtime_sha256"] = None
    with pytest.raises(RuntimeError, match="runtime hash"):
        validate_policy_health(health, allow_fake_policy=False)


def test_v5_rejects_v3_request_schema() -> None:
    agentview, wrist = _images()
    request = make_predict_request(
        request_id="request",
        suite="libero_spatial",
        task_id=0,
        reset_source="official",
        reset_id=0,
        reset_state_sha256=None,
        replan_id=0,
        execution_horizon=1,
        evaluation_seed=0,
        train_seed=0,
        instruction="test instruction",
        agentview_rgb=agentview,
        wrist_rgb=wrist,
        state=np.zeros(8, dtype=np.float32),
    )
    request["schema"] = "duo-vla-libero-policy-ipc-v3"
    with pytest.raises(BridgeProtocolError, match="request schema mismatch"):
        validate_request(request)


def test_v5_execution_geometry_is_exact() -> None:
    assert validate_execution_geometry(LIBERO_EXECUTION_GEOMETRY) == LIBERO_EXECUTION_GEOMETRY
    changed = dict(LIBERO_EXECUTION_GEOMETRY, physical_batch_size=1)
    with pytest.raises(BridgeProtocolError, match="training physical batch"):
        validate_execution_geometry(changed)
    with pytest.raises(BridgeProtocolError, match="fields differ"):
        validate_execution_geometry({**LIBERO_EXECUTION_GEOMETRY, "extra": True})


def test_v5_reports_distinct_b64_v2_training_and_b8_serving_geometry() -> None:
    kernel_sha256 = "a" * 64
    training = {
        **LIBERO_EXECUTION_GEOMETRY,
        "expert_batch_isolation": "sample_isolated_grouped_mm_v2",
        "physical_batch_size": 64,
        "serving_batch_size": 8,
        "execution_profile": "duovla-single-gpu-tp1-fused-v2-train-b64-serve-b8-v1",
        "tensor_parallel_size": 1,
        "shared_weight_kernel_sha256": kernel_sha256,
    }
    assert validate_execution_geometry(training) == training
    serving = serving_execution_geometry(training)
    assert serving == {
        "experts_implementation": "grouped_mm",
        "expert_batch_isolation": "sample_isolated_grouped_mm_v2",
        "physical_batch_size": 8,
        "execution_profile": "duovla-single-gpu-tp1-fused-v2-serve-b8-v1",
        "tensor_parallel_size": 1,
        "shared_weight_kernel_sha256": kernel_sha256,
    }
    assert validate_serving_execution_geometry(serving, training_execution_geometry=training) == serving

    with pytest.raises(BridgeProtocolError, match="serving batch"):
        validate_execution_geometry({**training, "serving_batch_size": 64})
    with pytest.raises(BridgeProtocolError, match="serving execution geometry differs"):
        validate_serving_execution_geometry(
            {**serving, "physical_batch_size": 64},
            training_execution_geometry=training,
        )


def test_v5_prediction_response_rejects_contract_and_echo_drift() -> None:
    contract = {
        "objective": "direct_regression",
        "sampler": "single_forward",
        "nfe": 1,
        "inference_seed_behavior": "episode_identity_echo_only",
    }
    response = {
        "actions": np.column_stack((np.zeros((8, 6), dtype=np.float32), np.ones(8, dtype=np.float32))).tolist(),
        "evaluation_seed": 91,
        "inference_seed": 123,
        "normalized_clip_fraction": 0.0,
        "operation": "predict",
        "policy_seconds": 0.1,
        "request_id": "request",
        "reset_id": 7,
        "reset_source": "clean-dev",
        "reset_state_sha256": "a" * 64,
        "schema": SCHEMA,
        "status": "ok",
        **contract,
    }
    actions, _ = validate_action_response(
        response,
        request_id="request",
        expected_evaluation_seed=91,
        expected_inference_seed=123,
        expected_reset_source="clean-dev",
        expected_reset_id=7,
        expected_reset_state_sha256="a" * 64,
        expected_policy_contract=contract,
    )
    assert actions.shape == (8, 7)

    wrong_seed = dict(response, inference_seed=124)
    with pytest.raises(BridgeProtocolError, match="inference seed"):
        validate_action_response(
            wrong_seed,
            request_id="request",
            expected_evaluation_seed=91,
            expected_inference_seed=123,
            expected_reset_source="clean-dev",
            expected_reset_id=7,
            expected_reset_state_sha256="a" * 64,
            expected_policy_contract=contract,
        )
    wrong_nfe = dict(response, nfe=2)
    with pytest.raises(BridgeProtocolError, match="one single_forward"):
        validate_action_response(
            wrong_nfe,
            request_id="request",
            expected_evaluation_seed=91,
            expected_inference_seed=123,
            expected_reset_source="clean-dev",
            expected_reset_id=7,
            expected_reset_state_sha256="a" * 64,
            expected_policy_contract=contract,
        )

    for name, changed in (
        ("evaluation_seed", 92),
        ("reset_source", "official"),
        ("reset_id", 8),
        ("reset_state_sha256", "b" * 64),
    ):
        with pytest.raises(BridgeProtocolError, match=name):
            validate_action_response(
                {**response, name: changed},
                request_id="request",
                expected_evaluation_seed=91,
                expected_inference_seed=123,
                expected_reset_source="clean-dev",
                expected_reset_id=7,
                expected_reset_state_sha256="a" * 64,
                expected_policy_contract=contract,
            )


def test_serving_nfe_override_is_flow_only_and_does_not_mutate_checkpoint_contract() -> None:
    flow = {
        "objective": "rectified_flow",
        "sampler": "euler_uniform",
        "nfe": 10,
        "inference_seed_behavior": "episode_identity_gaussian_noise",
    }
    selected = select_serving_policy_contract(flow, flow_steps_override=5)
    assert selected["nfe"] == 5
    assert flow["nfe"] == 10

    direct = {
        "objective": "direct_regression",
        "sampler": "single_forward",
        "nfe": 1,
        "inference_seed_behavior": "episode_identity_echo_only",
    }
    assert select_serving_policy_contract(direct, flow_steps_override=None) == direct
    with pytest.raises(RuntimeError, match="forbidden"):
        select_serving_policy_contract(direct, flow_steps_override=1)


class _FakeEnvironment:
    def __init__(self, success_after_policy_steps: int) -> None:
        self.success_after_policy_steps = success_after_policy_steps
        self.total_steps = 0
        self.seed_value: int | None = None
        self.sim_state = np.zeros(1, dtype=np.float64)
        self.observation = self._observation()

    @staticmethod
    def _observation() -> dict[str, np.ndarray]:
        agentview, wrist = _images()
        return {
            "agentview_image": agentview,
            "robot0_eye_in_hand_image": wrist,
            "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray([0.02, -0.02], dtype=np.float32),
        }

    def seed(self, value: int) -> None:
        self.seed_value = value

    def reset(self) -> dict[str, np.ndarray]:
        self.total_steps = 0
        return self.observation

    def set_init_state(self, state: np.ndarray) -> dict[str, np.ndarray]:
        self.sim_state = np.asarray(state, dtype=np.float64).copy()
        return self.observation

    def get_sim_state(self) -> np.ndarray:
        return self.sim_state.copy()

    def step(self, _: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        self.total_steps += 1
        return self.observation, 0.0, False, {}

    def check_success(self) -> bool:
        return self.total_steps >= 10 + self.success_after_policy_steps


class _ImpossibleLatencyClient:
    def predict(self, **request: Any) -> tuple[np.ndarray, dict[str, Any]]:
        actions = np.zeros((8, 7), dtype=np.float32)
        return actions, {
            "evaluation_seed": request["evaluation_seed"],
            "normalized_clip_fraction": 0.0,
            "policy_seconds": 100.0,
            "reset_id": request["reset_id"],
            "reset_source": request["reset_source"],
            "reset_state_sha256": request["reset_state_sha256"],
        }


def test_episode_rejects_server_latency_above_client_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluate_libero

    monkeypatch.setattr(evaluate_libero, "libero_state", lambda _: np.zeros(8, dtype=np.float32))
    with pytest.raises(RuntimeError, match="at least the server latency"):
        run_episode(
            _FakeEnvironment(success_after_policy_steps=1),
            initial_state=np.zeros(1, dtype=np.float32),
            client=_ImpossibleLatencyClient(),
            train_seed=0,
            evaluation_seed=101,
            suite="libero_spatial",
            task_id=0,
            task_name="test_task",
            instruction="test instruction",
            init_state_id=0,
            execution_horizon=1,
            policy_budget=20,
        )


@pytest.mark.parametrize(("execution_horizon", "expected_calls"), [(1, 5), (4, 2)])
def test_episode_queueing_and_success_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_horizon: int,
    expected_calls: int,
) -> None:
    import evaluate_libero

    monkeypatch.setattr(evaluate_libero, "libero_state", lambda _: np.zeros(8, dtype=np.float32))
    socket_path = tmp_path / f"policy-{execution_horizon}.sock"
    environment = _FakeEnvironment(success_after_policy_steps=5)
    with (
        _running_fake_server(socket_path, train_seed=23) as server,
        PolicyClient(socket_path, timeout_seconds=2.0) as client,
    ):
        client.health()
        episode = run_episode(
            environment,
            initial_state=np.zeros(1, dtype=np.float32),
            client=client,
            train_seed=23,
            evaluation_seed=101,
            suite="libero_spatial",
            task_id=0,
            task_name="test_task",
            instruction="test instruction",
            init_state_id=0,
            execution_horizon=execution_horizon,
            policy_budget=20,
        )
        client.shutdown()
    assert not server.is_alive()
    assert not socket_path.exists()
    assert environment.seed_value == 7
    assert episode["settle_steps"] == 10
    assert episode["policy_steps"] == 5
    assert episode["steps_to_success"] == 5
    assert episode["policy_calls"] == expected_calls
    assert episode["success"] is True


def test_clean_dev_episode_authenticates_settle_and_echoes_reset_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evaluate_libero

    monkeypatch.setattr(evaluate_libero, "libero_state", lambda _: np.zeros(8, dtype=np.float32))
    socket_path = tmp_path / "clean-dev.sock"
    initial_state = np.asarray([1.25, -2.5], dtype=np.float64)
    initial_hash = state_sha256(initial_state)
    environment = _FakeEnvironment(success_after_policy_steps=1)
    with (
        _running_fake_server(socket_path, train_seed=23) as server,
        PolicyClient(socket_path, timeout_seconds=2.0) as client,
    ):
        client.health()
        episode = run_episode(
            environment,
            initial_state=initial_state,
            client=client,
            train_seed=23,
            evaluation_seed=101,
            suite="libero_spatial",
            task_id=0,
            task_name="test_task",
            instruction="test instruction",
            init_state_id=0,
            execution_horizon=1,
            policy_budget=20,
            reset_source="clean-dev",
            reset_state_sha256=initial_hash,
            environment_seed=1234,
            expected_settled_state_sha256=initial_hash,
        )
        client.shutdown()
    assert not server.is_alive()
    assert episode["reset_source"] == "clean-dev"
    assert episode["reset_state_sha256"] == initial_hash
    assert episode["environment_seed"] == 1234
    assert episode["init_state_id"] is None


def test_selection_wilson_and_summary_keep_k_separate() -> None:
    assert parse_selection("4,0,2-3,2", upper=10, name="tasks") == (0, 2, 3, 4)
    assert parse_selection("all", upper=3, name="tasks") == (0, 1, 2)
    interval = wilson95(5, 10)
    assert interval is not None and interval[0] < 0.5 < interval[1]
    episodes = []
    for task_id, success in enumerate((True, False)):
        episodes.append(
            {
                "action_clipped_channels": 0,
                "action_clip_fraction": 0.0,
                "action_continuous_channels": 30,
                "elapsed_seconds": 1.0,
                "execution_horizon": 4,
                "normalized_action_clip_fraction": 0.0,
                "policy_calls": 2,
                "policy_latency_seconds": [0.1, 0.2],
                "server_latency_seconds": [0.08, 0.18],
                "steps_to_success": 5 if success else None,
                "success": success,
                "suite": "libero_spatial",
                "task_id": task_id,
                "task_name": f"task_{task_id}",
            }
        )
    summary = summarize_episodes(episodes, execution_horizon=4)
    assert summary["execution_horizon"] == 4
    assert summary["suites"][0]["suite_task_macro_success"] == 0.5
    assert summary["overall_40_task_macro_success"] is None
    assert summary["policy_latency_p95_seconds"] == pytest.approx(0.2, abs=0.02)
    assert summary["episode_elapsed_seconds"] == 2.0
    assert summary["episode_throughput_per_hour"] == 3600.0
    assert summary["policy_call_throughput_per_second"] == 2.0
    assert math.isfinite(summary["steps_to_success_mean"])
