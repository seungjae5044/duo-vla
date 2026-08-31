#!/usr/bin/env python3
"""Lossless local IPC contract shared by the LIBERO simulator and policy processes."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import socket
import stat
import struct
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "duo-vla-libero-policy-ipc-v5"
PROTOCOL = "duovla-libero-v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
RESET_SOURCES = ("official", "clean-dev")
INFERENCE_SEED_DOMAIN = "duo-vla-libero-inference-seed-v3"
ACTION_HORIZON = 8
ACTION_DIM = 7
STATE_DIM = 8
IMAGE_SHAPE = (256, 256, 3)
SUPPORTED_EXECUTION_HORIZONS = (1, 4)
LIBERO_PREFIX_GEOMETRY_SHA256 = "cc907e22ccd5ae704767edba606233dede39989a119ac544764b47aaa4fbe634"
LIBERO_EXECUTION_GEOMETRY = {
    "experts_implementation": "grouped_mm",
    "expert_batch_isolation": "sample_isolated_grouped_mm_v1",
    "physical_batch_size": 8,
    "fixed_physical_prefix_width": 545,
    "prefix_geometry_content_sha256": LIBERO_PREFIX_GEOMETRY_SHA256,
}
MAX_FRAME_BYTES = 2 * 1024 * 1024
_HEADER = struct.Struct(">Q")
_WIRE_POLICY_KEYS = {"objective", "sampler", "nfe", "inference_seed_behavior"}


class BridgeError(RuntimeError):
    """Base exception for an invalid or failed policy exchange."""


class BridgeProtocolError(BridgeError):
    """The peer violated the pinned wire contract."""


class BridgeTimeoutError(BridgeError):
    """The policy did not complete before the configured deadline."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BridgeProtocolError(message)


def libero_replan_seed(
    evaluation_seed: int,
    suite: str,
    task_id: int,
    reset_source: str,
    reset_id: int,
    reset_state_sha256: str | None,
    replan_id: int,
) -> int:
    """Derive policy randomness exclusively from the immutable evaluation episode identity."""

    require(
        isinstance(evaluation_seed, int) and not isinstance(evaluation_seed, bool) and 0 <= evaluation_seed < 2**63,
        "evaluation_seed must be an integer in [0, 2^63)",
    )
    require(suite in SUITES, f"unknown LIBERO suite {suite!r}")
    require(type(task_id) is int and 0 <= task_id < 10, "task_id is out of range")
    _validate_reset_identity(reset_source, reset_id, reset_state_sha256)
    require(type(replan_id) is int and replan_id >= 0, "replan_id must be a nonnegative integer")
    parts = (
        INFERENCE_SEED_DOMAIN,
        evaluation_seed,
        suite,
        task_id,
        reset_source,
        reset_id,
        reset_state_sha256,
        replan_id,
    )
    canonical = json.dumps(parts, allow_nan=False, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.blake2b(canonical.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _validate_reset_identity(reset_source: Any, reset_id: Any, reset_state_sha256: Any) -> None:
    require(reset_source in RESET_SOURCES, f"unknown reset_source {reset_source!r}")
    require(type(reset_id) is int and reset_id >= 0, "reset_id must be a nonnegative integer")
    if reset_source == "official":
        require(reset_id < 50, "official reset_id must be in [0, 50)")
        require(reset_state_sha256 is None, "official reset_state_sha256 must be null")
        return
    require(
        isinstance(reset_state_sha256, str)
        and len(reset_state_sha256) == 64
        and all(character in "0123456789abcdef" for character in reset_state_sha256),
        "clean-dev reset_state_sha256 must be 64 lowercase hexadecimal characters",
    )


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(value)
    require(
        observed == expected,
        f"{name} fields differ: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}",
    )


def validate_wire_policy_contract(value: Mapping[str, Any], *, allow_fake: bool = False) -> dict[str, Any]:
    """Validate the policy identity repeated by every v5 health/prediction response."""

    contract = {name: value.get(name) for name in _WIRE_POLICY_KEYS}
    objective = contract["objective"]
    sampler = contract["sampler"]
    nfe = contract["nfe"]
    seed_behavior = contract["inference_seed_behavior"]
    require(type(nfe) is int and nfe >= 0, "policy nfe must be a nonnegative integer")
    if objective == "rectified_flow":
        require(sampler == "euler_uniform", "rectified_flow must use euler_uniform")
        require(nfe in {1, 5, 10}, "rectified_flow nfe must be one of {1, 5, 10}")
        require(
            seed_behavior == "episode_identity_gaussian_noise",
            "rectified_flow inference seed behavior mismatch",
        )
    elif objective == "direct_regression":
        require(sampler == "single_forward" and nfe == 1, "direct_regression must use one single_forward")
        require(
            seed_behavior == "episode_identity_echo_only",
            "direct_regression inference seed behavior mismatch",
        )
    elif objective == "test_fake" and allow_fake:
        require(sampler == "seeded_test_normal" and nfe == 0, "test_fake policy contract mismatch")
        require(
            seed_behavior == "episode_identity_test_generator",
            "test_fake inference seed behavior mismatch",
        )
    else:
        raise BridgeProtocolError(f"unsupported policy objective {objective!r}")
    return contract


def validate_execution_geometry(value: Any) -> dict[str, Any]:
    """Validate the exact fixed-B8 execution geometry carried by real health."""

    require(isinstance(value, Mapping), "real policy health has no execution geometry")
    _exact_keys(value, set(LIBERO_EXECUTION_GEOMETRY), "execution geometry")
    observed = dict(value)
    require(observed == LIBERO_EXECUTION_GEOMETRY, f"policy execution geometry differs: {observed}")
    return observed


def encode_rgb(image: np.ndarray) -> dict[str, Any]:
    values = np.asarray(image)
    require(values.shape == IMAGE_SHAPE, f"RGB image must have shape {IMAGE_SHAPE}, got {values.shape}")
    require(values.dtype == np.uint8, f"RGB image must have dtype uint8, got {values.dtype}")
    raw = np.ascontiguousarray(values).tobytes(order="C")
    return {
        "data": base64.b64encode(raw).decode("ascii"),
        "dtype": "uint8",
        "encoding": "base64",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(IMAGE_SHAPE),
    }


def decode_rgb(value: Any, *, name: str) -> np.ndarray:
    require(isinstance(value, dict), f"{name} must be an encoded RGB object")
    _exact_keys(value, {"data", "dtype", "encoding", "sha256", "shape"}, name)
    require(value["dtype"] == "uint8", f"{name} dtype must be uint8")
    require(value["encoding"] == "base64", f"{name} encoding must be base64")
    require(value["shape"] == list(IMAGE_SHAPE), f"{name} shape must be {IMAGE_SHAPE}")
    require(isinstance(value["data"], str), f"{name} data must be text")
    try:
        raw = base64.b64decode(value["data"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BridgeProtocolError(f"{name} contains invalid base64") from exc
    require(len(raw) == math.prod(IMAGE_SHAPE), f"{name} byte length is invalid")
    require(hashlib.sha256(raw).hexdigest() == value["sha256"], f"{name} SHA-256 mismatch")
    return np.frombuffer(raw, dtype=np.uint8).reshape(IMAGE_SHAPE).copy()


def _validate_state(value: Any) -> np.ndarray:
    require(isinstance(value, list) and len(value) == STATE_DIM, f"state must contain {STATE_DIM} values")
    state = np.asarray(value, dtype=np.float32)
    require(state.shape == (STATE_DIM,) and bool(np.isfinite(state).all()), "state must contain eight finite values")
    return state.copy()


def make_control_request(operation: str, *, request_id: str) -> dict[str, Any]:
    require(operation in {"health", "shutdown"}, "invalid control operation")
    require(isinstance(request_id, str) and 0 < len(request_id) <= 160, "request_id must be non-empty text")
    return {"operation": operation, "request_id": request_id, "schema": SCHEMA}


def make_predict_request(
    *,
    request_id: str,
    suite: str,
    task_id: int,
    reset_source: str,
    reset_id: int,
    reset_state_sha256: str | None,
    replan_id: int,
    execution_horizon: int,
    evaluation_seed: int,
    train_seed: int,
    instruction: str,
    agentview_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: np.ndarray,
) -> dict[str, Any]:
    inference_seed = libero_replan_seed(
        evaluation_seed,
        suite,
        task_id,
        reset_source,
        reset_id,
        reset_state_sha256,
        replan_id,
    )
    require(execution_horizon in SUPPORTED_EXECUTION_HORIZONS, "execution_horizon must be one of {1, 4}")
    require(type(train_seed) is int and 0 <= train_seed < 2**63, "train_seed must be an integer in [0, 2^63)")
    require(isinstance(request_id, str) and 0 < len(request_id) <= 160, "request_id must be non-empty text")
    require(isinstance(instruction, str) and 0 < len(instruction) <= 1000, "instruction must be non-empty text")
    state_values = _validate_state(np.asarray(state, dtype=np.float32).reshape(-1).tolist())
    return {
        "episode": {
            "execution_horizon": execution_horizon,
            "replan_id": replan_id,
            "reset_id": reset_id,
            "reset_source": reset_source,
            "reset_state_sha256": reset_state_sha256,
            "suite": suite,
            "task_id": task_id,
        },
        "evaluation_seed": evaluation_seed,
        "inference_seed": inference_seed,
        "instruction": instruction,
        "observation": {
            "agentview_rgb": encode_rgb(agentview_rgb),
            "state": state_values.tolist(),
            "wrist_rgb": encode_rgb(wrist_rgb),
        },
        "operation": "predict",
        "request_id": request_id,
        "schema": SCHEMA,
        "train_seed": train_seed,
    }


def validate_request(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "request must be an object")
    require(value.get("schema") == SCHEMA, "request schema mismatch")
    operation = value.get("operation")
    require(operation in {"health", "predict", "shutdown"}, "unknown request operation")
    request_id = value.get("request_id")
    require(isinstance(request_id, str) and 0 < len(request_id) <= 160, "invalid request_id")
    if operation != "predict":
        _exact_keys(value, {"operation", "request_id", "schema"}, "control request")
        return dict(value)

    _exact_keys(
        value,
        {
            "episode",
            "evaluation_seed",
            "inference_seed",
            "instruction",
            "observation",
            "operation",
            "request_id",
            "schema",
            "train_seed",
        },
        "predict request",
    )
    episode = value["episode"]
    observation = value["observation"]
    require(isinstance(episode, dict), "episode must be an object")
    require(isinstance(observation, dict), "observation must be an object")
    _exact_keys(
        episode,
        {
            "execution_horizon",
            "replan_id",
            "reset_id",
            "reset_source",
            "reset_state_sha256",
            "suite",
            "task_id",
        },
        "episode",
    )
    _exact_keys(observation, {"agentview_rgb", "state", "wrist_rgb"}, "observation")
    execution_horizon = episode["execution_horizon"]
    require(execution_horizon in SUPPORTED_EXECUTION_HORIZONS, "execution_horizon must be one of {1, 4}")
    expected_seed = libero_replan_seed(
        value["evaluation_seed"],
        episode["suite"],
        episode["task_id"],
        episode["reset_source"],
        episode["reset_id"],
        episode["reset_state_sha256"],
        episode["replan_id"],
    )
    require(
        type(value["train_seed"]) is int and 0 <= value["train_seed"] < 2**63,
        "train_seed must be an integer in [0, 2^63)",
    )
    require(
        type(value["inference_seed"]) is int and value["inference_seed"] == expected_seed,
        "inference seed does not match the episode identity",
    )
    require(
        isinstance(value["instruction"], str) and 0 < len(value["instruction"]) <= 1000,
        "instruction must be non-empty text",
    )
    validated = dict(value)
    validated["observation"] = {
        "agentview_rgb": decode_rgb(observation["agentview_rgb"], name="agentview_rgb"),
        "state": _validate_state(observation["state"]),
        "wrist_rgb": decode_rgb(observation["wrist_rgb"], name="wrist_rgb"),
    }
    return validated


def make_success_response(request: Mapping[str, Any], **payload: Any) -> dict[str, Any]:
    reserved = {"operation", "request_id", "schema", "status"}
    require(not reserved.intersection(payload), "success response payload overrides its envelope")
    response = {
        "operation": request["operation"],
        "request_id": request["request_id"],
        "schema": SCHEMA,
        "status": "ok",
    }
    if request["operation"] == "predict":
        episode = request["episode"]
        echoed = {
            "evaluation_seed": request["evaluation_seed"],
            "inference_seed": request["inference_seed"],
            "reset_id": episode["reset_id"],
            "reset_source": episode["reset_source"],
            "reset_state_sha256": episode["reset_state_sha256"],
        }
        for name, expected in echoed.items():
            require(name not in payload or payload[name] == expected, f"prediction payload changed echoed {name}")
        response.update(echoed)
    response.update(payload)
    return response


def make_error_response(request: Mapping[str, Any] | None, exc: Exception) -> dict[str, Any]:
    return {
        "error": {"message": str(exc), "type": type(exc).__name__},
        "operation": request.get("operation", "invalid") if request else "invalid",
        "request_id": request.get("request_id", "unknown") if request else "unknown",
        "schema": SCHEMA,
        "status": "error",
    }


def validate_action_response(
    value: Any,
    *,
    request_id: str,
    expected_evaluation_seed: int,
    expected_inference_seed: int,
    expected_reset_source: str,
    expected_reset_id: int,
    expected_reset_state_sha256: str | None,
    expected_policy_contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    _validate_response_envelope(value, request_id=request_id, operation="predict")
    require(isinstance(value, dict), "response must be an object")
    _exact_keys(
        value,
        {
            "actions",
            "evaluation_seed",
            "inference_seed",
            "inference_seed_behavior",
            "nfe",
            "normalized_clip_fraction",
            "objective",
            "operation",
            "policy_seconds",
            "request_id",
            "reset_id",
            "reset_source",
            "reset_state_sha256",
            "sampler",
            "schema",
            "status",
        },
        "prediction response",
    )
    observed_policy_contract = validate_wire_policy_contract(
        value,
        allow_fake=expected_policy_contract.get("objective") == "test_fake",
    )
    require(
        observed_policy_contract == dict(expected_policy_contract),
        "prediction policy contract changed after health",
    )
    require(
        type(value.get("inference_seed")) is int and value["inference_seed"] == expected_inference_seed,
        "prediction inference seed does not match the request",
    )
    require(
        type(value.get("evaluation_seed")) is int and value["evaluation_seed"] == expected_evaluation_seed,
        "prediction evaluation_seed does not match the request",
    )
    require(value.get("reset_source") == expected_reset_source, "prediction reset_source does not match the request")
    require(
        type(value.get("reset_id")) is int and value["reset_id"] == expected_reset_id,
        "prediction reset_id does not match the request",
    )
    require(
        value.get("reset_state_sha256") == expected_reset_state_sha256,
        "prediction reset_state_sha256 does not match the request",
    )
    actions = np.asarray(value.get("actions"), dtype=np.float32)
    require(
        actions.shape == (ACTION_HORIZON, ACTION_DIM), f"policy actions must have shape {(ACTION_HORIZON, ACTION_DIM)}"
    )
    require(bool(np.isfinite(actions).all()), "policy actions contain non-finite values")
    require(bool(np.isin(actions[:, 6], (-1.0, 1.0)).all()), "policy gripper actions must be exactly {-1, +1}")
    policy_seconds = value.get("policy_seconds")
    require(
        isinstance(policy_seconds, (int, float)) and math.isfinite(policy_seconds) and policy_seconds >= 0,
        "invalid policy_seconds",
    )
    clip_fraction = value.get("normalized_clip_fraction")
    require(isinstance(clip_fraction, (int, float)) and 0.0 <= clip_fraction <= 1.0, "invalid normalized_clip_fraction")
    return actions.copy(), dict(value)


def _validate_response_envelope(value: Any, *, request_id: str, operation: str) -> None:
    require(isinstance(value, dict), "response must be an object")
    require(value.get("schema") == SCHEMA, "response schema mismatch")
    require(value.get("request_id") == request_id, "response request_id mismatch")
    require(value.get("operation") == operation, "response operation mismatch")
    status = value.get("status")
    require(status in {"ok", "error"}, "response status is invalid")
    if status == "error":
        error = value.get("error")
        _exact_keys(value, {"error", "operation", "request_id", "schema", "status"}, "error response")
        require(isinstance(error, dict), "error response payload must be an object")
        _exact_keys(error, {"message", "type"}, "error response payload")
        require(isinstance(error["message"], str) and error["message"], "error response message must be text")
        require(isinstance(error["type"], str) and error["type"], "error response type must be text")
        message = error["message"]
        raise BridgeError(f"policy server rejected {operation}: {message}")


def send_frame(peer: socket.socket, value: Mapping[str, Any]) -> None:
    try:
        payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BridgeProtocolError(f"message is not finite JSON: {exc}") from exc
    require(len(payload) <= MAX_FRAME_BYTES, f"message exceeds {MAX_FRAME_BYTES} byte limit")
    try:
        peer.sendall(_HEADER.pack(len(payload)) + payload)
    except TimeoutError as exc:
        raise BridgeTimeoutError("timed out while sending policy message") from exc


def _receive_exact(peer: socket.socket, size: int, *, allow_clean_eof: bool = False) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = peer.recv(remaining)
        except TimeoutError as exc:
            raise BridgeTimeoutError("timed out while receiving policy message") from exc
        if not chunk:
            if allow_clean_eof and not chunks:
                return None
            raise BridgeProtocolError("policy connection closed within a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_frame(peer: socket.socket) -> dict[str, Any] | None:
    header = _receive_exact(peer, _HEADER.size, allow_clean_eof=True)
    if header is None:
        return None
    (size,) = _HEADER.unpack(header)
    require(0 < size <= MAX_FRAME_BYTES, f"invalid policy frame length {size}")
    payload = _receive_exact(peer, size)
    assert payload is not None
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeProtocolError("policy frame is not valid UTF-8 JSON") from exc
    require(isinstance(value, dict), "policy frame root must be an object")
    return value


class PolicyClient:
    """One persistent, timeout-bound client connection for a rollout process."""

    def __init__(self, socket_path: str | Path, *, timeout_seconds: float = 300.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.socket_path = Path(socket_path)
        self.timeout_seconds = timeout_seconds
        self._peer: socket.socket | None = None
        self._counter = 0
        self._policy_contract: dict[str, Any] | None = None

    def connect(self) -> PolicyClient:
        if self._peer is not None:
            raise RuntimeError("policy client is already connected")
        status = self.socket_path.lstat()
        require(stat.S_ISSOCK(status.st_mode), f"policy endpoint is not a Unix socket: {self.socket_path}")
        require(status.st_uid == os.getuid(), f"policy socket is owned by another user: {self.socket_path}")
        require(stat.S_IMODE(status.st_mode) & 0o077 == 0, f"policy socket is not private: {self.socket_path}")
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        peer.settimeout(self.timeout_seconds)
        try:
            peer.connect(str(self.socket_path))
        except TimeoutError as exc:
            peer.close()
            raise BridgeTimeoutError(f"timed out connecting to {self.socket_path}") from exc
        except OSError:
            peer.close()
            raise
        self._peer = peer
        return self

    def close(self) -> None:
        if self._peer is not None:
            self._peer.close()
            self._peer = None
        self._policy_contract = None

    def __enter__(self) -> PolicyClient:
        return self.connect()

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request_id(self, operation: str) -> str:
        self._counter += 1
        return f"{os.getpid()}-{self._counter}-{operation}"

    def _exchange(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self._peer is None:
            raise RuntimeError("policy client is not connected")
        send_frame(self._peer, request)
        response = receive_frame(self._peer)
        if response is None:
            raise BridgeProtocolError("policy server closed without a response")
        return response

    def health(self) -> dict[str, Any]:
        request_id = self._request_id("health")
        response = self._exchange(make_control_request("health", request_id=request_id))
        _validate_response_envelope(response, request_id=request_id, operation="health")
        _exact_keys(
            response,
            {
                "action_dim",
                "action_horizon",
                "checkpoint",
                "dataset_revision",
                "execution_geometry",
                "inference_seed_behavior",
                "latency_runtime_sha256",
                "mode",
                "model_revision",
                "nfe",
                "normalization_content_sha256",
                "objective",
                "operation",
                "prefix_cache_scope",
                "protocol",
                "request_id",
                "sampler",
                "schema",
                "serving_runtime_sha256",
                "state_dim",
                "status",
                "train_seed",
            },
            "health response",
        )
        require(response.get("protocol") == PROTOCOL, "policy protocol mismatch")
        require(response.get("action_horizon") == ACTION_HORIZON, "policy action horizon mismatch")
        require(response.get("action_dim") == ACTION_DIM, "policy action dimension mismatch")
        require(response.get("state_dim") == STATE_DIM, "policy state dimension mismatch")
        require(response.get("prefix_cache_scope") == "request", "policy prefix cache scope mismatch")
        require(
            type(response.get("train_seed")) is int and 0 <= response["train_seed"] < 2**63,
            "policy health response has an invalid train_seed",
        )
        require(response.get("mode") in {"fake", "real"}, "policy health response has an invalid mode")
        mode = response["mode"]
        runtime_sha256 = response.get("serving_runtime_sha256")
        latency_runtime_sha256 = response.get("latency_runtime_sha256")
        if mode == "real":
            require(response["train_seed"] in {0, 1, 2}, "real policy health has a noncanonical train_seed")
            checkpoint = response.get("checkpoint")
            require(isinstance(checkpoint, dict), "real policy health has no checkpoint")
            require(checkpoint.get("kind") == "resumable-libero-training", "real policy checkpoint kind is invalid")
            require(checkpoint.get("train_seed") == response["train_seed"], "checkpoint and health train_seed disagree")
            for name in (
                "manifest_sha256",
                "policy_contract_sha256",
                "source_tree_sha256",
                "training_execution_environment_sha256",
            ):
                sha256 = checkpoint.get(name)
                require(
                    isinstance(sha256, str)
                    and len(sha256) == 64
                    and all(character in "0123456789abcdef" for character in sha256),
                    f"real policy checkpoint {name} is invalid",
                )
            require(isinstance(checkpoint.get("training_execution_environment"), dict), "checkpoint runtime is invalid")
            model_revision = response.get("model_revision")
            require(
                isinstance(model_revision, str)
                and len(model_revision) == 40
                and all(character in "0123456789abcdef" for character in model_revision),
                "real policy health has an invalid model revision",
            )
            normalization_sha256 = response.get("normalization_content_sha256")
            require(
                isinstance(normalization_sha256, str)
                and len(normalization_sha256) == 64
                and all(character in "0123456789abcdef" for character in normalization_sha256),
                "real policy health has an invalid normalization SHA-256",
            )
            require(
                isinstance(runtime_sha256, str)
                and len(runtime_sha256) == 64
                and all(character in "0123456789abcdef" for character in runtime_sha256),
                "real policy health has an invalid serving runtime SHA-256",
            )
            require(
                isinstance(latency_runtime_sha256, str)
                and len(latency_runtime_sha256) == 64
                and all(character in "0123456789abcdef" for character in latency_runtime_sha256),
                "real policy health has an invalid latency runtime SHA-256",
            )
            execution_geometry = validate_execution_geometry(response.get("execution_geometry"))
            require(
                checkpoint.get("execution_geometry") == execution_geometry,
                "checkpoint and health execution geometry disagree",
            )
        else:
            for name in (
                "checkpoint",
                "execution_geometry",
                "latency_runtime_sha256",
                "model_revision",
                "normalization_content_sha256",
                "serving_runtime_sha256",
            ):
                require(response.get(name) is None, f"fake policy health cannot claim {name}")
        self._policy_contract = validate_wire_policy_contract(
            response,
            allow_fake=response.get("mode") == "fake",
        )
        return response

    def predict(self, **request_fields: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if self._policy_contract is None:
            raise RuntimeError("call health() before predict() to bind the v5 policy contract")
        request_id = self._request_id("predict")
        request = make_predict_request(request_id=request_id, **request_fields)
        response = self._exchange(request)
        episode = request["episode"]
        return validate_action_response(
            response,
            request_id=request_id,
            expected_evaluation_seed=request["evaluation_seed"],
            expected_inference_seed=request["inference_seed"],
            expected_reset_source=episode["reset_source"],
            expected_reset_id=episode["reset_id"],
            expected_reset_state_sha256=episode["reset_state_sha256"],
            expected_policy_contract=self._policy_contract,
        )

    def shutdown(self) -> dict[str, Any]:
        request_id = self._request_id("shutdown")
        response = self._exchange(make_control_request("shutdown", request_id=request_id))
        _validate_response_envelope(response, request_id=request_id, operation="shutdown")
        _exact_keys(response, {"operation", "request_id", "schema", "status", "stopped"}, "shutdown response")
        require(response["stopped"] is True, "shutdown response did not confirm termination")
        return response


def _prepare_socket(socket_path: Path) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        status = socket_path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(status.st_mode):
        raise FileExistsError(f"refusing to replace non-socket path: {socket_path}")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        probe.connect(str(socket_path))
    except OSError:
        socket_path.unlink()
    else:
        raise FileExistsError(f"policy socket already has a live listener: {socket_path}")
    finally:
        probe.close()


def serve_unix_policy(
    socket_path: str | Path,
    dispatch: Callable[[dict[str, Any]], Mapping[str, Any]],
    *,
    accept_timeout_seconds: float = 0.5,
    ready: Callable[[], None] | None = None,
) -> None:
    """Serve sequential persistent connections until ``dispatch`` handles shutdown."""

    path = Path(socket_path)
    _prepare_socket(path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.settimeout(accept_timeout_seconds)
    listener.bind(str(path))
    socket_inode = path.lstat().st_ino
    os.chmod(path, 0o600)
    listener.listen(1)
    if ready is not None:
        ready()
    should_stop = False
    try:
        while not should_stop:
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            with connection:
                connection.settimeout(None)
                while not should_stop:
                    request: dict[str, Any] | None = None
                    try:
                        wire_request = receive_frame(connection)
                        if wire_request is None:
                            break
                        request = validate_request(wire_request)
                        response = dict(dispatch(request))
                    except Exception as exc:
                        response = make_error_response(request, exc)
                    try:
                        send_frame(connection, response)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    should_stop = request is not None and request.get("operation") == "shutdown"
    finally:
        listener.close()
        try:
            if path.lstat().st_ino == socket_inode and stat.S_ISSOCK(path.lstat().st_mode):
                path.unlink()
        except FileNotFoundError:
            pass


def wait_for_socket(socket_path: str | Path, *, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.1)
        try:
            probe.connect(str(socket_path))
        except OSError as exc:
            last_error = exc
            time.sleep(0.02)
        else:
            probe.close()
            return
        finally:
            probe.close()
    raise TimeoutError(f"policy socket did not become ready: {socket_path}: {last_error}")
