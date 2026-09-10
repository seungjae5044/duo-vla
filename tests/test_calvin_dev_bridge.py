"""Focused tests for the disjoint held-out A/B/C IPC protocol."""

from __future__ import annotations

import copy
import socket
import stat
import struct
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "calvin"))

from calvin_dev_bridge import (
    ACTION_DIM,
    ACTION_HORIZON,
    GRIPPER_IMAGE_SHAPE,
    PROTOCOL,
    SCHEMA,
    STATE_DIM,
    STATIC_IMAGE_SHAPE,
    DevBridgeProtocolError,
    DevPolicyClient,
    decode_gripper_rgb,
    decode_static_rgb,
    make_control_request,
    make_health_response,
    make_predict_request,
    make_predict_response,
    make_success_response,
    receive_frame,
    serve_unix_policy,
    validate_health_response,
    validate_predict_response,
    validate_request,
    wait_for_socket,
)

_BANK_SHA256 = "a" * 64
_RESET_SHA256 = "b" * 64
_SPLIT_SHA256 = "c" * 64
_DATASET_SHA256 = "d" * 64
_MANIFEST_FILE_SHA256 = "e" * 64
_MEMBER_INDEX_SHA256 = "f" * 64
_REPLAY_BUNDLE_SHA256 = "1" * 64
_STORAGE_IDENTITY_SHA256 = "2" * 64


def _calvin_identity() -> dict[str, Any]:
    return {
        "archive_bytes": 555_309_812_705,
        "archive_sha256": "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74",
        "central_directory_sha256": "b4f79bda7f6b966b51aa419badd0f7db7a8972a7b58d6d342af60aceff0ea31b",
        "dataset_manifest_file_sha256": _MANIFEST_FILE_SHA256,
        "dataset_manifest_schema": "duo-vla-calvin-dataset-manifest-v4",
        "dataset_manifest_sha256": _DATASET_SHA256,
        "member_index": {
            "bytes": 456,
            "path": "task_ABC_D.members-v2.sqlite3",
            "schema": "duo-vla-calvin-member-index-v2",
            "sha256": _MEMBER_INDEX_SHA256,
        },
        "member_inventory_sha256": "5" * 64,
        "metadata_files": [
            "ep_start_end_ids.npy",
            "lang_annotations/auto_lang_ann.npy",
            "scene_info.npy",
            ".hydra/merged_config.yaml",
        ],
        "metadata_sha256": "6" * 64,
        "name": "task_ABC_D",
        "reader_schema": "duo-vla-calvin-archive-reader-v1",
        "split": "training",
        "storage_identity_sha256": _STORAGE_IDENTITY_SHA256,
        "storage_mode": "archive-direct",
    }


def _health_kwargs() -> dict[str, Any]:
    return {
        "calvin_identity": _calvin_identity(),
        "replay_bundle_sha256": _REPLAY_BUNDLE_SHA256,
    }


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    static = (np.arange(np.prod(STATIC_IMAGE_SHAPE), dtype=np.uint32) % 251).astype(np.uint8)
    gripper = ((np.arange(np.prod(GRIPPER_IMAGE_SHAPE), dtype=np.uint32) * 7) % 253).astype(np.uint8)
    state = np.linspace(-0.5, 0.5, STATE_DIM, dtype=np.float32)
    state[-1] = -1.0
    return static.reshape(STATIC_IMAGE_SHAPE), gripper.reshape(GRIPPER_IMAGE_SHAPE), state


def _request(**changes: Any) -> dict[str, Any]:
    static, gripper, state = _arrays()
    fields: dict[str, Any] = {
        "annotation_index": 19,
        "episode_index": 7,
        "evaluation_seed": 91,
        "execution_horizon": 4,
        "global_start": 1234,
        "instruction": "open the drawer",
        "replan_idx": 2,
        "request_id": "heldout-predict",
        "reset_bank_sha256": _BANK_SHA256,
        "reset_id_sha256": _RESET_SHA256,
        "reset_index": 3,
        "rgb_gripper": gripper,
        "rgb_static": static,
        "scene": "calvin_scene_A",
        "state": state,
        "task": "open_drawer",
        "train_seed": 17,
    }
    fields.update(changes)
    return make_predict_request(**fields)


def _actions() -> np.ndarray:
    actions = np.linspace(-0.75, 0.75, ACTION_HORIZON * ACTION_DIM, dtype=np.float32).reshape(8, 7)
    actions[:, 6] = np.asarray([-1.0, 1.0] * 4, dtype=np.float32)
    return actions


def test_development_protocol_is_disjoint_and_seed_excludes_ablation_identity() -> None:
    assert SCHEMA == "duovla-calvin-dev-policy-ipc-v4"
    assert PROTOCOL == "duovla-calvin-heldout-abc-v1"

    k1 = _request(train_seed=1, execution_horizon=1)
    k4 = _request(train_seed=999, execution_horizon=4)
    assert k1["inference_seed"] == k4["inference_seed"]

    with pytest.raises(DevBridgeProtocolError, match="A/B/C"):
        _request(scene="calvin_scene_D")
    official_envelope = copy.deepcopy(k1)
    official_envelope["schema"] = "duovla-calvin-policy-ipc-v4"
    with pytest.raises(DevBridgeProtocolError, match="development request schema"):
        validate_request(official_envelope)


def test_request_has_exact_observation_keys_and_lossless_owned_images() -> None:
    static, gripper, _state = _arrays()
    request = _request()
    validated = validate_request(request)

    assert set(validated["observation"]) == {"rgb_static", "rgb_gripper", "state"}
    decoded_static = decode_static_rgb(request["observation"]["rgb_static"])
    decoded_gripper = decode_gripper_rgb(request["observation"]["rgb_gripper"])
    assert np.array_equal(decoded_static, static) and decoded_static.flags.owndata
    assert np.array_equal(decoded_gripper, gripper) and decoded_gripper.flags.owndata
    assert validated["observation"]["state"].dtype == np.float32
    assert validated["observation"]["state"].flags.owndata

    nested = copy.deepcopy(request)
    nested["observation"]["scene_obs"] = [0.0] * 24
    with pytest.raises(DevBridgeProtocolError, match=r"extra=\['scene_obs'\]"):
        validate_request(nested)
    top_level = copy.deepcopy(request)
    top_level["scene_obs"] = [0.0] * 24
    with pytest.raises(DevBridgeProtocolError, match=r"extra=\['scene_obs'\]"):
        validate_request(top_level)


def test_prediction_response_echoes_every_reset_identity_and_owns_actions() -> None:
    request = _request()
    response = make_predict_response(request, _actions(), policy_seconds=0.125)
    actions, metadata = validate_predict_response(response, request)
    assert np.array_equal(actions, _actions())
    assert actions.dtype == np.float32 and actions.flags.owndata and actions.flags.writeable
    for name in (
        "reset_bank_sha256",
        "reset_id_sha256",
        "reset_index",
        "scene",
        "episode_index",
        "annotation_index",
        "global_start",
        "task",
        "replan_idx",
    ):
        assert metadata[name] == request["development_episode"][name]

    changed = dict(response, reset_id_sha256="e" * 64)
    with pytest.raises(DevBridgeProtocolError, match="reset_id_sha256"):
        validate_predict_response(changed, request)


def test_wire_rejects_duplicate_and_nonfinite_json_fields() -> None:
    sender, receiver = socket.socketpair()
    try:
        duplicate = b'{"schema":"first","schema":"second"}'
        sender.sendall(struct.pack(">Q", len(duplicate)) + duplicate)
        with pytest.raises(DevBridgeProtocolError, match="duplicate JSON field"):
            receive_frame(receiver)

        nonfinite = b'{"value":NaN}'
        sender.sendall(struct.pack(">Q", len(nonfinite)) + nonfinite)
        with pytest.raises(DevBridgeProtocolError, match="non-finite JSON"):
            receive_frame(receiver)
    finally:
        sender.close()
        receiver.close()


@contextmanager
def _running_server(socket_path: Path) -> Iterator[tuple[threading.Thread, list[Exception]]]:
    errors: list[Exception] = []

    def dispatch(request: dict[str, Any]) -> dict[str, Any]:
        if request["operation"] == "health":
            return make_health_response(
                request,
                train_seed=17,
                **_health_kwargs(),
                reset_bank_sha256=_BANK_SHA256,
                split_sha256=_SPLIT_SHA256,
                reset_count=8,
                mode="fake",
                objective="test_fake",
                sampler="seeded_test_normal",
                nfe=0,
                checkpoint_manifest_sha256=None,
                policy_contract_sha256=None,
                normalization_content_sha256=None,
                normalization_metadata_sha256=None,
                model_revision=None,
                execution_geometry=None,
            )
        if request["operation"] == "shutdown":
            return make_success_response(request, stopped=True)
        observation = request["observation"]
        assert set(observation) == {"rgb_static", "rgb_gripper", "state"}
        rng = np.random.RandomState(request["inference_seed"] & 0xFFFFFFFF)
        actions = rng.normal(size=(ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
        actions[:, 6] = np.where(actions[:, 6] >= 0.0, 1.0, -1.0)
        return make_predict_response(request, actions, policy_seconds=0.001)

    def run() -> None:
        try:
            serve_unix_policy(socket_path, dispatch)
        except Exception as exc:  # pragma: no cover - asserted by parent thread
            errors.append(exc)

    server = threading.Thread(target=run, daemon=True)
    server.start()
    wait_for_socket(socket_path, timeout_seconds=2.0)
    try:
        yield server, errors
    finally:
        if server.is_alive() and socket_path.exists():
            try:
                with DevPolicyClient(socket_path, timeout_seconds=1.0) as client:
                    client.shutdown()
            except Exception:
                pass
        server.join(timeout=2.0)


def test_private_persistent_socket_health_predict_shutdown(tmp_path: Path) -> None:
    socket_path = tmp_path / "calvin-heldout-abc.sock"
    with _running_server(socket_path) as (server, errors):
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        request = _request()
        episode = request["development_episode"]
        observation = validate_request(request)["observation"]
        fields = {
            "annotation_index": episode["annotation_index"],
            "episode_index": episode["episode_index"],
            "evaluation_seed": request["evaluation_seed"],
            "execution_horizon": episode["execution_horizon"],
            "global_start": episode["global_start"],
            "instruction": request["instruction"],
            "replan_idx": episode["replan_idx"],
            "reset_bank_sha256": episode["reset_bank_sha256"],
            "reset_id_sha256": episode["reset_id_sha256"],
            "reset_index": episode["reset_index"],
            "rgb_gripper": observation["rgb_gripper"],
            "rgb_static": observation["rgb_static"],
            "scene": episode["scene"],
            "state": observation["state"],
            "task": episode["task"],
            "train_seed": request["train_seed"],
        }
        with DevPolicyClient(socket_path, timeout_seconds=2.0) as client:
            health = client.health()
            assert health["protocol"] == PROTOCOL
            assert health["allowed_scenes"] == ["calvin_scene_A", "calvin_scene_B", "calvin_scene_C"]
            assert health["calvin_identity"]["dataset_manifest_file_sha256"] == _MANIFEST_FILE_SHA256
            assert health["calvin_identity"]["member_index"]["sha256"] == _MEMBER_INDEX_SHA256
            assert health["replay_bundle_sha256"] == _REPLAY_BUNDLE_SHA256
            assert health["calvin_identity"]["storage_identity_sha256"] == _STORAGE_IDENTITY_SHA256
            first, first_metadata = client.predict(**fields)
            second, second_metadata = client.predict(**fields)
            assert np.array_equal(first, second)
            assert first_metadata["inference_seed"] == second_metadata["inference_seed"]
            client.shutdown()
    assert not server.is_alive()
    assert not errors
    assert not socket_path.exists()


def test_health_v4_rejects_legacy_missing_extra_and_storage_drift() -> None:
    request = make_control_request("health", "health-v4")
    response = make_health_response(
        request,
        train_seed=17,
        **_health_kwargs(),
        reset_bank_sha256=_BANK_SHA256,
        split_sha256=_SPLIT_SHA256,
        reset_count=8,
        mode="fake",
        objective="test_fake",
        sampler="seeded_test_normal",
        nfe=0,
        checkpoint_manifest_sha256=None,
        policy_contract_sha256=None,
        normalization_content_sha256=None,
        normalization_metadata_sha256=None,
        model_revision=None,
        execution_geometry=None,
    )
    assert validate_health_response(response, "health-v4")["calvin_identity"]["storage_mode"] == "archive-direct"

    legacy = dict(response, schema="duovla-calvin-dev-policy-ipc-v3")
    with pytest.raises(DevBridgeProtocolError, match="schema mismatch"):
        validate_health_response(legacy, "health-v4")
    missing = dict(response)
    missing.pop("replay_bundle_sha256")
    with pytest.raises(DevBridgeProtocolError, match=r"missing=.*replay_bundle_sha256"):
        validate_health_response(missing, "health-v4")
    extra = dict(response, unbound=True)
    with pytest.raises(DevBridgeProtocolError, match=r"extra=.*unbound"):
        validate_health_response(extra, "health-v4")

    for name, changed in (
        ("dataset_manifest_schema", "duo-vla-calvin-dataset-manifest-v3"),
        ("reader_schema", "legacy-reader"),
        ("storage_mode", "verified-extraction"),
        ("storage_identity_sha256", "not-a-hash"),
        ("archive_sha256", "3" * 64),
        ("central_directory_sha256", "4" * 64),
    ):
        tampered = copy.deepcopy(response)
        tampered["calvin_identity"][name] = changed
        with pytest.raises(DevBridgeProtocolError):
            validate_health_response(tampered, "health-v4")

    for name, changed in (
        ("schema", "duo-vla-calvin-member-index-v1"),
        ("path", "other.sqlite3"),
        ("bytes", True),
        ("sha256", "not-a-hash"),
    ):
        tampered = copy.deepcopy(response)
        tampered["calvin_identity"]["member_index"][name] = changed
        with pytest.raises(DevBridgeProtocolError):
            validate_health_response(tampered, "health-v4")

    for container, field in (("calvin_identity", "metadata_files"), ("member_index", "path")):
        tampered = copy.deepcopy(response)
        nested = (
            tampered["calvin_identity"] if container == "calvin_identity" else tampered["calvin_identity"][container]
        )
        nested.pop(field)
        with pytest.raises(DevBridgeProtocolError, match="fields differ"):
            validate_health_response(tampered, "health-v4")
        nested[field] = None
        nested["extra"] = True
        with pytest.raises(DevBridgeProtocolError, match="fields differ"):
            validate_health_response(tampered, "health-v4")


def test_real_health_binds_normalization_metadata_to_nested_calvin_identity() -> None:
    request = make_control_request("health", "real-health-v4")
    kwargs = {
        **_health_kwargs(),
        "reset_bank_sha256": _BANK_SHA256,
        "split_sha256": _SPLIT_SHA256,
        "reset_count": 8,
        "mode": "real",
        "objective": "rectified_flow",
        "sampler": "euler_uniform",
        "nfe": 5,
        "checkpoint_manifest_sha256": "7" * 64,
        "policy_contract_sha256": "8" * 64,
        "normalization_content_sha256": "9" * 64,
        "normalization_metadata_sha256": _calvin_identity()["metadata_sha256"],
        "model_revision": "f7f5b7f5fa82ffc52addd066915886d497f5517b",
        "execution_geometry": {
            "expert_batch_isolation": "sample_isolated_grouped_mm_v1",
            "experts_implementation": "grouped_mm",
            "fixed_physical_prefix_width": 600,
            "physical_batch_size": 8,
            "prefix_geometry_content_sha256": "0" * 64,
        },
    }
    response = make_health_response(request, train_seed=17, **kwargs)
    assert validate_health_response(response, "real-health-v4")["calvin_identity"] == _calvin_identity()

    single_gpu_kwargs = copy.deepcopy(kwargs)
    single_gpu_kwargs["execution_geometry"].update(
        execution_profile="duovla-single-gpu-tp1-v1",
        tensor_parallel_size=1,
    )
    single_gpu_response = make_health_response(request, train_seed=17, **single_gpu_kwargs)
    assert (
        validate_health_response(single_gpu_response, "real-health-v4")["execution_geometry"]
        == (single_gpu_kwargs["execution_geometry"])
    )

    for geometry in (
        dict(single_gpu_kwargs["execution_geometry"], execution_profile="duovla-tp2-v1"),
        dict(single_gpu_kwargs["execution_geometry"], tensor_parallel_size=3),
    ):
        invalid_topology = copy.deepcopy(kwargs)
        invalid_topology["execution_geometry"] = geometry
        with pytest.raises(DevBridgeProtocolError):
            make_health_response(request, train_seed=17, **invalid_topology)

    partial_topology = copy.deepcopy(single_gpu_kwargs)
    partial_topology["execution_geometry"].pop("tensor_parallel_size")
    with pytest.raises(DevBridgeProtocolError, match="execution geometry fields differ"):
        make_health_response(request, train_seed=17, **partial_topology)

    bad_make = dict(kwargs, normalization_metadata_sha256="a" * 64)
    with pytest.raises(DevBridgeProtocolError, match="normalization metadata differs"):
        make_health_response(request, train_seed=17, **bad_make)
    tampered = copy.deepcopy(response)
    tampered["normalization_metadata_sha256"] = "a" * 64
    with pytest.raises(DevBridgeProtocolError, match="normalization metadata differs"):
        validate_health_response(tampered, "real-health-v4")
