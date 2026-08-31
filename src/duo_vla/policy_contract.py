"""Canonical, fail-closed contracts for trainable action-policy objectives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

POLICY_CONTRACT_SCHEMA = "duo-vla-policy-contract-v1"
RECTIFIED_FLOW = "rectified_flow"
DIRECT_REGRESSION = "direct_regression"

_POLICY_CONFIG_KEYS = {"objective", "sampler", "nfe", "inference_seed_behavior"}
_SAMPLING_CONFIG_KEYS = {
    "flow_path",
    "timestep_distribution",
    "integrator",
    "clip_intermediate_actions",
    "clip_final_normalized_actions",
}


@dataclass(frozen=True, slots=True)
class PolicyContract:
    """Complete semantics needed to train, resume, and serve one policy."""

    schema: ClassVar[str] = POLICY_CONTRACT_SCHEMA

    objective: str
    training_input: str
    training_target: str
    training_timestep: str
    sampler: str
    nfe: int
    inference_seed_behavior: str
    action_horizon: int
    action_dim: int
    clip_intermediate_actions: bool
    clip_final_normalized_actions: bool

    def __post_init__(self) -> None:
        if type(self.nfe) is not int or self.nfe <= 0:
            raise ValueError("policy contract nfe must be a positive integer")
        for name in ("action_horizon", "action_dim"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"policy contract {name} must be a positive integer")
        if type(self.clip_intermediate_actions) is not bool or type(self.clip_final_normalized_actions) is not bool:
            raise ValueError("policy contract clipping fields must be booleans")
        expected = _expected_semantics(self.objective)
        observed = {
            "training_input": self.training_input,
            "training_target": self.training_target,
            "training_timestep": self.training_timestep,
            "sampler": self.sampler,
            "inference_seed_behavior": self.inference_seed_behavior,
        }
        if observed != expected:
            raise ValueError(f"unsupported {self.objective!r} policy semantics: {observed}")
        if self.objective == DIRECT_REGRESSION and self.nfe != 1:
            raise ValueError("direct_regression must use exactly one function evaluation")
        if self.clip_intermediate_actions:
            raise ValueError("intermediate action clipping is unsupported")
        if not self.clip_final_normalized_actions:
            raise ValueError("final normalized action clipping must be enabled")

    def to_dict(self) -> dict[str, str | int | bool]:
        return {
            "schema": self.schema,
            "objective": self.objective,
            "training_input": self.training_input,
            "training_target": self.training_target,
            "training_timestep": self.training_timestep,
            "sampler": self.sampler,
            "nfe": self.nfe,
            "inference_seed_behavior": self.inference_seed_behavior,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "clip_intermediate_actions": self.clip_intermediate_actions,
            "clip_final_normalized_actions": self.clip_final_normalized_actions,
        }

    def wire_dict(self) -> dict[str, str | int]:
        """Return the objective identity repeated in IPC health and predictions."""

        return {
            "objective": self.objective,
            "sampler": self.sampler,
            "nfe": self.nfe,
            "inference_seed_behavior": self.inference_seed_behavior,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PolicyContract:
        if not isinstance(value, Mapping):
            raise TypeError("policy contract must be a mapping")
        expected_keys = {
            "schema",
            "objective",
            "training_input",
            "training_target",
            "training_timestep",
            "sampler",
            "nfe",
            "inference_seed_behavior",
            "action_horizon",
            "action_dim",
            "clip_intermediate_actions",
            "clip_final_normalized_actions",
        }
        if set(value) != expected_keys:
            raise ValueError("policy contract keys do not exactly match the schema")
        if value["schema"] != cls.schema:
            raise ValueError("unsupported policy contract schema")
        text_fields = (
            "objective",
            "training_input",
            "training_target",
            "training_timestep",
            "sampler",
            "inference_seed_behavior",
        )
        if any(not isinstance(value[name], str) for name in text_fields):
            raise ValueError("policy contract text fields must be strings")
        return cls(
            objective=value["objective"],
            training_input=value["training_input"],
            training_target=value["training_target"],
            training_timestep=value["training_timestep"],
            sampler=value["sampler"],
            nfe=value["nfe"],
            inference_seed_behavior=value["inference_seed_behavior"],
            action_horizon=value["action_horizon"],
            action_dim=value["action_dim"],
            clip_intermediate_actions=value["clip_intermediate_actions"],
            clip_final_normalized_actions=value["clip_final_normalized_actions"],
        )


def policy_contract_from_config(config: Mapping[str, Any]) -> PolicyContract:
    """Resolve and cross-check the strict ``[policy]`` and ``[sampling]`` sections."""

    if not isinstance(config, Mapping):
        raise TypeError("resolved configuration must be a mapping")
    policy = _require_mapping(config, "policy")
    action = _require_mapping(config, "action")
    sampling = _require_mapping(config, "sampling")
    _require_exact_keys(policy, _POLICY_CONFIG_KEYS, "policy configuration")
    _require_exact_keys(sampling, _SAMPLING_CONFIG_KEYS, "sampling configuration")

    objective = policy["objective"]
    if not isinstance(objective, str):
        raise ValueError("policy.objective must be text")
    semantics = _expected_semantics(objective)
    for key in ("sampler", "inference_seed_behavior"):
        if policy[key] != semantics[key]:
            raise ValueError(f"policy.{key} is incompatible with objective {objective!r}")
    nfe = policy["nfe"]
    if type(nfe) is not int or nfe <= 0:
        raise ValueError("policy.nfe must be a positive integer")
    flow_steps = action.get("flow_steps")
    if type(flow_steps) is not int or flow_steps != nfe:
        raise ValueError("action.flow_steps must exactly equal policy.nfe")

    expected_sampling = _expected_sampling(objective)
    if dict(sampling) != expected_sampling:
        raise ValueError(f"sampling configuration is incompatible with objective {objective!r}")

    horizon = action.get("horizon")
    action_dim = action.get("dimension")
    if type(horizon) is not int or horizon <= 0 or type(action_dim) is not int or action_dim <= 0:
        raise ValueError("action horizon and dimension must be positive integers")
    return PolicyContract(
        objective=objective,
        training_input=semantics["training_input"],
        training_target=semantics["training_target"],
        training_timestep=semantics["training_timestep"],
        sampler=semantics["sampler"],
        nfe=nfe,
        inference_seed_behavior=semantics["inference_seed_behavior"],
        action_horizon=horizon,
        action_dim=action_dim,
        clip_intermediate_actions=sampling["clip_intermediate_actions"],
        clip_final_normalized_actions=sampling["clip_final_normalized_actions"],
    )


def validate_manifest_policy_contract(
    manifest: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
) -> PolicyContract:
    """Require a checkpoint manifest to repeat the config-derived contract exactly."""

    expected = policy_contract_from_config(resolved_config)
    recorded = PolicyContract.from_dict(manifest.get("policy_contract"))
    if recorded != expected:
        raise ValueError("checkpoint policy_contract differs from its resolved configuration")
    return recorded


def _expected_semantics(objective: str) -> dict[str, str]:
    if objective == RECTIFIED_FLOW:
        return {
            "training_input": "linear_noise_to_clean",
            "training_target": "velocity_clean_minus_noise",
            "training_timestep": "uniform_per_chunk",
            "sampler": "euler_uniform",
            "inference_seed_behavior": "episode_identity_gaussian_noise",
        }
    if objective == DIRECT_REGRESSION:
        return {
            "training_input": "zero_action_canvas",
            "training_target": "clean_action",
            "training_timestep": "fixed_one",
            "sampler": "single_forward",
            "inference_seed_behavior": "episode_identity_echo_only",
        }
    raise ValueError(f"unsupported policy objective: {objective!r}")


def _expected_sampling(objective: str) -> dict[str, str | bool]:
    common: dict[str, str | bool] = {
        "clip_intermediate_actions": False,
        "clip_final_normalized_actions": True,
    }
    if objective == RECTIFIED_FLOW:
        return {
            "flow_path": "linear",
            "timestep_distribution": "uniform",
            "integrator": "euler",
            **common,
        }
    if objective == DIRECT_REGRESSION:
        return {
            "flow_path": "none",
            "timestep_distribution": "fixed_one",
            "integrator": "single_forward",
            **common,
        }
    raise ValueError(f"unsupported policy objective: {objective!r}")


def _require_mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"resolved configuration has no [{name}] table")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
