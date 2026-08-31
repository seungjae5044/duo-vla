"""Deterministic update planning and exact masked-loss accounting for trainers."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

TRAINER_STATE_SCHEMA = "duo-vla-trainer-state-v1"
_MAX_COUNTER = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class MicrobatchPlan:
    """Stateless random seeds for one optimizer update's microstep."""

    update: int
    microstep: int
    data_seed: int
    flow_seed: int

    def __post_init__(self) -> None:
        _validate_counter(self.update, "update")
        _validate_counter(self.microstep, "microstep")
        _validate_counter(self.data_seed, "data_seed")
        _validate_counter(self.flow_seed, "flow_seed")


def make_microbatch_plan(seed: int, update: int, microstep: int) -> MicrobatchPlan:
    """Derive domain-separated data and flow seeds without consuming mutable RNG state."""

    _validate_counter(seed, "seed")
    _validate_counter(update, "update")
    _validate_counter(microstep, "microstep")
    encoded = struct.pack(">QQQ", seed, update, microstep)
    return MicrobatchPlan(
        update=update,
        microstep=microstep,
        data_seed=_derive_seed(encoded, domain=b"duovla-data-v1"),
        flow_seed=_derive_seed(encoded, domain=b"duovla-flow-v1"),
    )


def make_update_plan(
    seed: int,
    update: int,
    *,
    gradient_accumulation_steps: int,
) -> tuple[MicrobatchPlan, ...]:
    """Return every deterministic microbatch plan for one optimizer update."""

    _validate_counter(seed, "seed")
    _validate_counter(update, "update")
    if (
        isinstance(gradient_accumulation_steps, bool)
        or not isinstance(gradient_accumulation_steps, int)
        or gradient_accumulation_steps <= 0
    ):
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    return tuple(make_microbatch_plan(seed, update, microstep) for microstep in range(gradient_accumulation_steps))


@dataclass(frozen=True, slots=True)
class MaskedSSE:
    """A differentiable FP32 SSE numerator paired with its exact scalar denominator."""

    squared_error_sum: Tensor
    element_count: int

    def __post_init__(self) -> None:
        if self.squared_error_sum.shape != () or self.squared_error_sum.dtype != torch.float32:
            raise ValueError("squared_error_sum must be a scalar float32 tensor")
        if isinstance(self.element_count, bool) or not isinstance(self.element_count, int) or self.element_count <= 0:
            raise ValueError("element_count must be a positive integer")

    @property
    def mean(self) -> Tensor:
        return self.squared_error_sum / self.element_count

    def loss_for_total(self, total_element_count: int) -> Tensor:
        """Scale this numerator for backward within a larger accumulated update."""

        if (
            isinstance(total_element_count, bool)
            or not isinstance(total_element_count, int)
            or total_element_count < self.element_count
        ):
            raise ValueError("total_element_count must be an integer covering this component")
        return self.squared_error_sum / total_element_count


def masked_element_count(valid_mask: Tensor, *, action_dim: int) -> int:
    """Count scalar action targets selected by a boolean ``[batch, horizon]`` mask."""

    if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must have bool shape [batch, horizon]")
    if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
        raise ValueError("action_dim must be a positive integer")
    valid_positions = int(valid_mask.sum().item())
    if valid_positions == 0:
        raise ValueError("valid_mask contains no valid action positions")
    return valid_positions * action_dim


def masked_sse(prediction: Tensor, target: Tensor, valid_mask: Tensor) -> MaskedSSE:
    """Return an FP32 masked SSE that composes exactly across microbatches."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [batch, horizon, action_dim]")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating tensors")
    if valid_mask.shape != prediction.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must have bool shape [batch, horizon]")
    valid = valid_mask.to(device=prediction.device)
    element_count = masked_element_count(valid, action_dim=prediction.shape[-1])
    squared_error = (prediction.float() - target.to(device=prediction.device).float()).square()
    return MaskedSSE(squared_error[valid].sum(), element_count)


def combine_masked_sse(components: Sequence[MaskedSSE]) -> MaskedSSE:
    """Combine microbatch numerators and denominators without mean-of-means bias."""

    values = tuple(components)
    if not values:
        raise ValueError("components must not be empty")
    device = values[0].squared_error_sum.device
    if any(component.squared_error_sum.device != device for component in values):
        raise ValueError("all masked SSE components must use the same device")
    return MaskedSSE(
        torch.stack([component.squared_error_sum for component in values]).sum(),
        sum(component.element_count for component in values),
    )


@dataclass(frozen=True, slots=True)
class TrainerState:
    """Update-boundary progress; ``next_update`` is the next optimizer step to execute."""

    schema: ClassVar[str] = TRAINER_STATE_SCHEMA

    next_update: int = 0
    examples_seen: int = 0

    def __post_init__(self) -> None:
        _validate_counter(self.next_update, "next_update")
        _validate_counter(self.examples_seen, "examples_seen")

    def advance(self, *, examples: int) -> TrainerState:
        if isinstance(examples, bool) or not isinstance(examples, int) or examples <= 0:
            raise ValueError("examples must be a positive integer")
        return TrainerState(next_update=self.next_update + 1, examples_seen=self.examples_seen + examples)

    def to_dict(self) -> dict[str, str | int]:
        return {
            "schema": self.schema,
            "next_update": self.next_update,
            "examples_seen": self.examples_seen,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TrainerState:
        if not isinstance(value, Mapping):
            raise TypeError("trainer state must be a mapping")
        expected_keys = {"schema", "next_update", "examples_seen"}
        if set(value) != expected_keys:
            raise ValueError("trainer state keys do not exactly match the schema")
        if value["schema"] != cls.schema:
            raise ValueError("unsupported trainer state schema")
        next_update = _validate_counter(value["next_update"], "next_update")
        examples_seen = _validate_counter(value["examples_seen"], "examples_seen")
        return cls(next_update=next_update, examples_seen=examples_seen)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> TrainerState:
        if not isinstance(value, str):
            raise TypeError("serialized trainer state must be a string")
        try:
            decoded = json.loads(value, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as exc:
            raise ValueError("trainer state is not valid JSON") from exc
        return cls.from_dict(decoded)


def _derive_seed(encoded_plan: bytes, *, domain: bytes) -> int:
    digest = hashlib.blake2b(encoded_plan, digest_size=8, person=domain).digest()
    return int.from_bytes(digest, "big") & _MAX_COUNTER


def _validate_counter(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_COUNTER:
        raise ValueError(f"{name} must be an integer in [0, {_MAX_COUNTER}]")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
