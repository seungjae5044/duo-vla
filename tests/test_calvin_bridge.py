"""Tests for the Python-3.8-compatible CALVIN simulator/policy boundary."""

# The bridge itself must run under Python 3.8.  Keep these annotations in the
# same dialect even though the repository linter targets Python 3.11.
# ruff: noqa: UP006, UP035

import copy
import inspect
import socket
import stat
import struct
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "calvin"))

from calvin_bridge import (
    ACTION_DIM,
    ACTION_HORIZON,
    GRIPPER_IMAGE_SHAPE,
    SCHEMA,
    SEQUENCE_SHA256,
    STATE_DIM,
    STATIC_IMAGE_SHAPE,
    BridgeProtocolError,
    BridgeTimeoutError,
    PolicyClient,
    calvin_replan_seed,
    decode_gripper_rgb,
    decode_static_rgb,
    make_control_request,
    make_health_response,
    make_predict_request,
    make_predict_response,
    make_success_response,
    receive_frame,
    send_frame,
    serve_unix_policy,
    validate_action_response,
    validate_health_response,
    validate_request,
    wait_for_socket,
)


def _calvin_identity() -> Dict[str, Any]:
    return {
        "archive_bytes": 555_309_812_705,
        "archive_sha256": "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74",
        "central_directory_sha256": "b4f79bda7f6b966b51aa419badd0f7db7a8972a7b58d6d342af60aceff0ea31b",
        "dataset_manifest_file_sha256": "1" * 64,
        "dataset_manifest_schema": "duo-vla-calvin-dataset-manifest-v4",
        "dataset_manifest_sha256": "2" * 64,
        "member_index": {
            "bytes": 456,
            "path": "task_ABC_D.members-v2.sqlite3",
            "schema": "duo-vla-calvin-member-index-v2",
            "sha256": "3" * 64,
        },
        "member_inventory_sha256": "4" * 64,
        "metadata_files": [
            "ep_start_end_ids.npy",
            "lang_annotations/auto_lang_ann.npy",
            "scene_info.npy",
            ".hydra/merged_config.yaml",
        ],
        "metadata_sha256": "5" * 64,
        "name": "task_ABC_D",
        "reader_schema": "duo-vla-calvin-archive-reader-v1",
        "split": "training",
        "storage_identity_sha256": "6" * 64,
        "storage_mode": "archive-direct",
    }


def _arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    static_pixels = np.arange(np.prod(STATIC_IMAGE_SHAPE), dtype=np.uint32).reshape(STATIC_IMAGE_SHAPE)
    gripper_pixels = np.arange(np.prod(GRIPPER_IMAGE_SHAPE), dtype=np.uint32).reshape(GRIPPER_IMAGE_SHAPE)
    rgb_static = (static_pixels % 251).astype(np.uint8)
    rgb_gripper = ((gripper_pixels * 7 + 19) % 253).astype(np.uint8)
    state = np.linspace(-0.5, 0.5, STATE_DIM, dtype=np.float32)
    state[-1] = -1.0
    return rgb_static, rgb_gripper, state


def _request(**changes: Any) -> Dict[str, Any]:
    rgb_static, rgb_gripper, state = _arrays()
    fields = {
        "evaluation_seed": 91,
        "execution_horizon": 4,
        "instruction": "rotate the blue block to the right",
        "replan_idx": 12,
        "request_id": "predict-request",
        "rgb_gripper": rgb_gripper,
        "rgb_static": rgb_static,
        "sequence_idx": 17,
        "sequence_sha256": SEQUENCE_SHA256,
        "state": state,
        "subtask_idx": 3,
        "subtask_name": "rotate_blue_block_right",
        "train_seed": 7,
    }
    fields.update(changes)
    return make_predict_request(**fields)


def _actions() -> np.ndarray:
    actions = np.linspace(-0.75, 0.75, ACTION_HORIZON * ACTION_DIM, dtype=np.float32).reshape(
        ACTION_HORIZON, ACTION_DIM
    )
    actions[:, 6] = np.asarray([-1.0, 1.0] * 4, dtype=np.float32)
    return actions


def _validate_response(response: Dict[str, Any], request: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    episode = request["episode"]
    return validate_action_response(
        response,
        request_id=request["request_id"],
        expected_evaluation_seed=request["evaluation_seed"],
        expected_inference_seed=request["inference_seed"],
        expected_sequence_sha256=episode["sequence_sha256"],
        expected_sequence_idx=episode["sequence_idx"],
        expected_subtask_idx=episode["subtask_idx"],
        expected_subtask_name=episode["subtask_name"],
        expected_replan_idx=episode["replan_idx"],
        expected_execution_horizon=episode["execution_horizon"],
        expected_train_seed=request["train_seed"],
    )


def test_calvin_v1_constants_and_seed_identity_are_pinned() -> None:
    assert SCHEMA == "duovla-calvin-policy-ipc-v4"
    assert STATIC_IMAGE_SHAPE == (200, 200, 3)
    assert GRIPPER_IMAGE_SHAPE == (84, 84, 3)
    assert ACTION_HORIZON == 8
    assert ACTION_DIM == 7
    assert STATE_DIM == 8
    assert tuple(inspect.signature(calvin_replan_seed).parameters) == (
        "evaluation_seed",
        "sequence_sha256",
        "sequence_idx",
        "subtask_idx",
        "subtask_name",
        "replan_idx",
    )

    seed = calvin_replan_seed(0, SEQUENCE_SHA256, 0, 0, "rotate_blue_block_right", 0)
    assert seed == 1878932505119255186
    assert seed != calvin_replan_seed(1, SEQUENCE_SHA256, 0, 0, "rotate_blue_block_right", 0)
    assert seed != calvin_replan_seed(0, "a" * 64, 0, 0, "rotate_blue_block_right", 0)
    assert seed != calvin_replan_seed(0, SEQUENCE_SHA256, 1, 0, "rotate_blue_block_right", 0)
    assert seed != calvin_replan_seed(0, SEQUENCE_SHA256, 0, 1, "rotate_blue_block_right", 0)
    assert seed != calvin_replan_seed(0, SEQUENCE_SHA256, 0, 0, "move_slider_right", 0)
    assert seed != calvin_replan_seed(0, SEQUENCE_SHA256, 0, 0, "rotate_blue_block_right", 1)


def test_inference_seed_excludes_train_seed_and_execution_horizon() -> None:
    k1 = _request(train_seed=1, execution_horizon=1)
    k4 = _request(train_seed=2, execution_horizon=4)

    assert k1["inference_seed"] == k4["inference_seed"]
    with pytest.raises(BridgeProtocolError, match="pinned official sequences"):
        _request(sequence_sha256="a" * 64)
    with pytest.raises(BridgeProtocolError, match=r"\[0, 2\^63\)"):
        calvin_replan_seed(-1, SEQUENCE_SHA256, 0, 0, "task", 0)


def test_unequal_rgb_images_round_trip_losslessly_into_owned_arrays() -> None:
    rgb_static, rgb_gripper, _state = _arrays()
    request = _request()
    static_wire = request["observation"]["rgb_static"]
    gripper_wire = request["observation"]["rgb_gripper"]

    assert static_wire["shape"] == [200, 200, 3]
    assert gripper_wire["shape"] == [84, 84, 3]
    decoded_static = decode_static_rgb(static_wire)
    decoded_gripper = decode_gripper_rgb(gripper_wire)
    assert np.array_equal(decoded_static, rgb_static)
    assert np.array_equal(decoded_gripper, rgb_gripper)
    assert decoded_static.flags.owndata and decoded_static.flags.c_contiguous and decoded_static.flags.writeable
    assert decoded_gripper.flags.owndata and decoded_gripper.flags.c_contiguous and decoded_gripper.flags.writeable
    decoded_static[0, 0, 0] ^= 1
    assert not np.array_equal(decoded_static, rgb_static)

    with pytest.raises(BridgeProtocolError, match="shape must be"):
        decode_static_rgb(gripper_wire)
    tampered = dict(static_wire, sha256="0" * 64)
    with pytest.raises(BridgeProtocolError, match="SHA-256 mismatch"):
        decode_static_rgb(tampered)
    extra = dict(static_wire, scene_obs="forbidden")
    with pytest.raises(BridgeProtocolError, match="fields differ"):
        decode_static_rgb(extra)


def test_predict_request_is_exact_and_scene_obs_cannot_cross_ipc() -> None:
    request = _request()
    validated = validate_request(request)
    observation = validated["observation"]

    assert set(observation) == {"rgb_static", "rgb_gripper", "state"}
    assert observation["rgb_static"].shape == STATIC_IMAGE_SHAPE
    assert observation["rgb_gripper"].shape == GRIPPER_IMAGE_SHAPE
    assert observation["state"].shape == (8,)
    assert observation["state"].dtype == np.float32
    assert observation["state"].flags.owndata

    nested_scene = copy.deepcopy(request)
    nested_scene["observation"]["scene_obs"] = [0.0] * 24
    with pytest.raises(BridgeProtocolError, match=r"extra=\['scene_obs'\]"):
        validate_request(nested_scene)

    top_level_scene = copy.deepcopy(request)
    top_level_scene["scene_obs"] = [0.0] * 24
    with pytest.raises(BridgeProtocolError, match=r"extra=\['scene_obs'\]"):
        validate_request(top_level_scene)

    bad_state = copy.deepcopy(request)
    bad_state["observation"]["state"][-1] = 0.0
    with pytest.raises(BridgeProtocolError, match="gripper"):
        validate_request(bad_state)
    nonfinite_state = copy.deepcopy(request)
    nonfinite_state["observation"]["state"][0] = float("nan")
    with pytest.raises(BridgeProtocolError, match="finite"):
        validate_request(nonfinite_state)


def test_request_construction_requires_float32_state_and_supported_k() -> None:
    _rgb_static, _rgb_gripper, state = _arrays()
    with pytest.raises(BridgeProtocolError, match="dtype float32"):
        _request(state=state.astype(np.float64))
    with pytest.raises(BridgeProtocolError, match=r"\{1, 4\}"):
        _request(execution_horizon=2)
    with pytest.raises(BridgeProtocolError, match=r"\{1, 4\}"):
        _request(execution_horizon=True)


def test_action_response_rejects_every_echo_change_and_extra_field() -> None:
    request = _request()
    response = make_predict_response(request, _actions(), policy_seconds=0.125)
    decoded, metadata = _validate_response(response, request)

    assert np.array_equal(decoded, _actions())
    assert decoded.dtype == np.float32 and decoded.shape == (8, 7)
    assert decoded.flags.owndata and decoded.flags.c_contiguous and decoded.flags.writeable
    assert metadata["sequence_sha256"] == SEQUENCE_SHA256

    changes = {
        "evaluation_seed": 92,
        "execution_horizon": 1,
        "inference_seed": response["inference_seed"] + 1,
        "replan_idx": response["replan_idx"] + 1,
        "sequence_idx": response["sequence_idx"] + 1,
        "sequence_sha256": "a" * 64,
        "subtask_idx": response["subtask_idx"] - 1,
        "subtask_name": "move_slider_right",
        "train_seed": response["train_seed"] + 1,
    }
    for name, changed in changes.items():
        with pytest.raises(BridgeProtocolError, match=name):
            _validate_response(dict(response, **{name: changed}), request)

    with pytest.raises(BridgeProtocolError, match="request_id"):
        _validate_response(dict(response, request_id="tampered"), request)
    with pytest.raises(BridgeProtocolError, match="extra"):
        _validate_response(dict(response, scene_obs=[]), request)
    with pytest.raises(BridgeProtocolError, match="changed echoed"):
        make_success_response(request, actions=response["actions"], inference_seed=0, policy_seconds=0.1)


def test_action_response_requires_finite_actions_and_binary_gripper() -> None:
    request = _request()
    response = make_predict_response(request, _actions(), policy_seconds=0.0)

    nonfinite = copy.deepcopy(response)
    nonfinite["actions"][2][3] = float("inf")
    with pytest.raises(BridgeProtocolError, match="finite"):
        _validate_response(nonfinite, request)

    nonbinary = copy.deepcopy(response)
    nonbinary["actions"][4][6] = 0.0
    with pytest.raises(BridgeProtocolError, match="gripper"):
        _validate_response(nonbinary, request)

    wrong_shape = copy.deepcopy(response)
    wrong_shape["actions"].pop()
    with pytest.raises(BridgeProtocolError, match="8 rows"):
        _validate_response(wrong_shape, request)

    with pytest.raises(BridgeProtocolError, match="dtype float32"):
        make_predict_response(request, _actions().astype(np.float64), policy_seconds=0.1)


def test_length_prefixed_frames_reject_duplicate_fields() -> None:
    sender, receiver = socket.socketpair()
    try:
        message = make_control_request("health", "frame")
        send_frame(sender, message)
        assert receive_frame(receiver) == message

        duplicate = b'{"schema":"a","schema":"b"}'
        sender.sendall(struct.pack(">Q", len(duplicate)) + duplicate)
        with pytest.raises(BridgeProtocolError, match="finite UTF-8 JSON"):
            receive_frame(receiver)
    finally:
        sender.close()
        receiver.close()


def test_socket_timeout_is_translated_to_bridge_timeout() -> None:
    sender, receiver = socket.socketpair()
    receiver.settimeout(0.01)
    try:
        with pytest.raises(BridgeTimeoutError, match="receiving"):
            receive_frame(receiver)
    finally:
        sender.close()
        receiver.close()


@contextmanager
def _running_server(socket_path: Path, train_seed: int) -> Iterator[Tuple[threading.Thread, List[Exception]]]:
    errors = []  # type: List[Exception]

    def dispatch(request: Dict[str, Any]) -> Dict[str, Any]:
        operation = request["operation"]
        if operation == "health":
            return make_health_response(
                request,
                train_seed,
                calvin_identity=None,
                mode="fake",
                objective="test_fake",
                sampler="seeded_test_normal",
                nfe=0,
                checkpoint_manifest_sha256=None,
                policy_contract_sha256=None,
                normalization_content_sha256=None,
                normalization_metadata_sha256=None,
                model_revision=None,
                serving_runtime_sha256=None,
                execution_geometry=None,
            )
        if operation == "shutdown":
            return make_success_response(request, stopped=True)
        observation = request["observation"]
        assert set(observation) == {"rgb_static", "rgb_gripper", "state"}
        assert observation["rgb_static"].flags.owndata
        assert observation["rgb_gripper"].flags.owndata
        assert observation["state"].flags.owndata
        rng = np.random.RandomState(request["inference_seed"] & 0xFFFFFFFF)
        actions = rng.normal(size=(ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
        actions[:, 6] = np.where(actions[:, 6] >= 0, 1.0, -1.0)
        return make_predict_response(request, actions, policy_seconds=0.001)

    def run() -> None:
        try:
            serve_unix_policy(socket_path, dispatch)
        except Exception as exc:  # pragma: no cover - reported by the parent assertion
            errors.append(exc)

    server = threading.Thread(target=run, daemon=True)
    server.start()
    wait_for_socket(socket_path, timeout_seconds=2.0)
    try:
        yield server, errors
    finally:
        if server.is_alive() and socket_path.exists():
            try:
                with PolicyClient(socket_path, timeout_seconds=1.0) as cleanup:
                    cleanup.shutdown()
            except Exception:
                pass
        server.join(timeout=2.0)


def test_private_persistent_unix_socket_health_predict_shutdown(tmp_path: Path) -> None:
    socket_path = tmp_path / "calvin-policy.sock"
    with _running_server(socket_path, train_seed=7) as (server, errors):
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        rgb_static, rgb_gripper, state = _arrays()
        fields = {
            "evaluation_seed": 91,
            "execution_horizon": 4,
            "instruction": "rotate the blue block to the right",
            "replan_idx": 12,
            "rgb_gripper": rgb_gripper,
            "rgb_static": rgb_static,
            "sequence_idx": 17,
            "sequence_sha256": SEQUENCE_SHA256,
            "state": state,
            "subtask_idx": 3,
            "subtask_name": "rotate_blue_block_right",
            "train_seed": 7,
        }
        with PolicyClient(socket_path, timeout_seconds=2.0) as client:
            health = client.health()
            assert health["schema"] == SCHEMA
            assert health["train_seed"] == 7
            assert health["mode"] == "fake"
            assert health["objective"] == "test_fake"
            assert health["checkpoint_manifest_sha256"] is None
            assert health["static_image_shape"] == [200, 200, 3]
            assert health["gripper_image_shape"] == [84, 84, 3]
            with pytest.raises(BridgeProtocolError, match="train_seed"):
                client.predict(**dict(fields, train_seed=8))
            first, first_metadata = client.predict(**fields)
            second, second_metadata = client.predict(**fields)
            assert np.array_equal(first, second)
            assert first.flags.owndata
            assert set(first[:, 6]) <= {-1.0, 1.0}
            assert first_metadata["inference_seed"] == second_metadata["inference_seed"]
            client.shutdown()
    assert not server.is_alive()
    assert not errors
    assert not socket_path.exists()


def test_health_v4_strictly_binds_the_nested_calvin_dataset_identity() -> None:
    request = make_control_request("health", "real-health-v4")
    identity = _calvin_identity()
    kwargs = {
        "calvin_identity": identity,
        "mode": "real",
        "objective": "rectified_flow",
        "sampler": "euler_uniform",
        "nfe": 5,
        "checkpoint_manifest_sha256": "7" * 64,
        "policy_contract_sha256": "8" * 64,
        "normalization_content_sha256": "9" * 64,
        "normalization_metadata_sha256": identity["metadata_sha256"],
        "model_revision": "f7f5b7f5fa82ffc52addd066915886d497f5517b",
        "serving_runtime_sha256": "a" * 64,
        "execution_geometry": {
            "expert_batch_isolation": "sample_isolated_grouped_mm_v1",
            "experts_implementation": "grouped_mm",
            "fixed_physical_prefix_width": 600,
            "physical_batch_size": 8,
            "prefix_geometry_content_sha256": "b" * 64,
        },
    }
    response = make_health_response(request, 7, **kwargs)
    assert validate_health_response(response, "real-health-v4")["calvin_identity"] == identity

    legacy = dict(response, schema="duovla-calvin-policy-ipc-v3")
    with pytest.raises(BridgeProtocolError, match="schema mismatch"):
        validate_health_response(legacy, "real-health-v4")
    for name in ("calvin_identity", "normalization_metadata_sha256"):
        missing = dict(response)
        missing.pop(name)
        with pytest.raises(BridgeProtocolError, match="fields differ"):
            validate_health_response(missing, "real-health-v4")
    extra = dict(response, dataset_identity_copy={})
    with pytest.raises(BridgeProtocolError, match="fields differ"):
        validate_health_response(extra, "real-health-v4")

    for name, changed in (
        ("archive_bytes", 123),
        ("archive_sha256", "c" * 64),
        ("central_directory_sha256", "d" * 64),
        ("dataset_manifest_schema", "duo-vla-calvin-dataset-manifest-v3"),
        ("reader_schema", "legacy-reader"),
        ("storage_mode", "verified-extraction"),
    ):
        tampered = copy.deepcopy(response)
        tampered["calvin_identity"][name] = changed
        with pytest.raises(BridgeProtocolError):
            validate_health_response(tampered, "real-health-v4")
    for name, changed in (("bytes", True), ("path", "other.sqlite3"), ("schema", "v1"), ("sha256", "bad")):
        tampered = copy.deepcopy(response)
        tampered["calvin_identity"]["member_index"][name] = changed
        with pytest.raises(BridgeProtocolError):
            validate_health_response(tampered, "real-health-v4")

    missing_nested = copy.deepcopy(response)
    missing_nested["calvin_identity"].pop("metadata_files")
    with pytest.raises(BridgeProtocolError, match="fields differ"):
        validate_health_response(missing_nested, "real-health-v4")
    extra_nested = copy.deepcopy(response)
    extra_nested["calvin_identity"]["unbound"] = True
    with pytest.raises(BridgeProtocolError, match="fields differ"):
        validate_health_response(extra_nested, "real-health-v4")

    bad_make = dict(kwargs, normalization_metadata_sha256="e" * 64)
    with pytest.raises(BridgeProtocolError, match="normalization metadata differs"):
        make_health_response(request, 7, **bad_make)
    metadata_drift = copy.deepcopy(response)
    metadata_drift["normalization_metadata_sha256"] = "e" * 64
    with pytest.raises(BridgeProtocolError, match="normalization metadata differs"):
        validate_health_response(metadata_drift, "real-health-v4")
