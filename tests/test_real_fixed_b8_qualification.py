from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify_real_fixed_b8_sample_isolation.py"
SPEC = importlib.util.spec_from_file_location("real_fixed_b8_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


def _instruction(conversation: list[dict[str, Any]]) -> str:
    return next(item["text"] for item in conversation[0]["content"] if item["type"] == "text")


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(padding_side="left")
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, conversations: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        width = kwargs["processor_kwargs"]["max_length"]
        batch = len(conversations)
        input_ids = torch.zeros((batch, width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        mm_token_type_ids = torch.zeros_like(input_ids)
        images: list[torch.Tensor] = []
        for row, value in enumerate(conversations):
            instruction = _instruction(value)
            if instruction == GATE.TARGET.instruction:
                length = GATE.EXPECTED_TARGET_VALID_PREFIX_LENGTH
                image_seed = 17
            else:
                length = 520 + row
                image_seed = 31 + row
            attention_mask[row, -length:] = 1
            input_ids[row, -length:] = torch.arange(1, length + 1)
            mm_token_type_ids[row, -length:] = 1
            images.extend(
                (
                    torch.full((2, 3), image_seed, dtype=torch.float32),
                    torch.full((2, 3), image_seed + 1, dtype=torch.float32),
                )
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": torch.stack(images),
            "image_position_ids": torch.zeros((2 * batch, 2, 2), dtype=torch.long),
        }


def test_case_plans_keep_exact_b8_and_move_the_same_target() -> None:
    replicated, mixed, moved = GATE.case_plans()

    assert replicated.require_replicated_rows
    assert replicated.samples == (GATE.TARGET,) * 8
    assert mixed.target_row == 0
    assert moved.target_row == GATE.MOVED_TARGET_ROW
    assert moved.samples[moved.target_row] == GATE.TARGET
    assert sorted(sample.identifier for sample in mixed.samples) == sorted(
        sample.identifier for sample in moved.samples
    )

    mixed_embeddings = GATE.deterministic_action_embeddings(mixed.samples, hidden_size=16)
    moved_embeddings = GATE.deterministic_action_embeddings(moved.samples, hidden_size=16)
    replicated_embeddings = GATE.deterministic_action_embeddings(replicated.samples, hidden_size=16)
    assert torch.equal(mixed_embeddings[mixed.target_row], moved_embeddings[moved.target_row])
    assert torch.equal(mixed_embeddings[mixed.target_row], replicated_embeddings[0])
    assert all(torch.equal(replicated_embeddings[0], replicated_embeddings[row]) for row in range(1, 8))


def test_processor_path_pins_two_images_b8_width_and_no_truncation() -> None:
    processor = _FakeProcessor()
    plan = GATE.case_plans()[0]

    values, geometry = GATE._processor_inputs(
        processor,
        plan,
        prefix_width=GATE.DEFAULT_PREFIX_WIDTH,
        device=torch.device("cpu"),
    )

    assert processor.calls == [
        {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            "processor_kwargs": {
                "padding": "max_length",
                "max_length": 545,
                "truncation": False,
            },
        }
    ]
    assert values["input_ids"].shape == (8, 545)
    assert values["pixel_values"].shape[0] == 16
    assert geometry["target_valid_prefix_length"] == 544
    assert geometry["valid_prefix_lengths"] == [544] * 8


def test_registration_snapshot_detects_parameter_replacement_but_not_method_patch() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    expected = GATE.registration_snapshot(model)
    model[0].forward = lambda inputs: inputs  # type: ignore[method-assign]

    assert GATE.assert_registration_unchanged(expected, model, context="method patch") == expected

    model[0].weight = nn.Parameter(model[0].weight.detach().clone())
    with pytest.raises(RuntimeError, match="changed parameter identities"):
        GATE.assert_registration_unchanged(expected, model, context="replacement")


def test_case_comparison_requires_bitwise_prefix_output_loss_and_gradient() -> None:
    reference = GATE.CaseArtifacts(
        target_input_sha256="input",
        target_output=torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        target_gradient=torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
        prefix_digest_by_rank=("a", "b"),
        prefix_layer_digests_by_rank=(("l0", "l1"), ("r0", "r1")),
        encoder_layer_zero_component_digests_by_rank=(({"layer": "x"}), ({"layer": "y"})),
        loss_bytes=b"loss",
    )
    matching = GATE.CaseArtifacts(
        target_input_sha256="input",
        target_output=reference.target_output.clone(),
        target_gradient=reference.target_gradient.clone(),
        prefix_digest_by_rank=("a", "b"),
        prefix_layer_digests_by_rank=(("l0", "l1"), ("r0", "r1")),
        encoder_layer_zero_component_digests_by_rank=(({"layer": "x"}), ({"layer": "y"})),
        loss_bytes=b"loss",
    )

    assert all(GATE.assert_case_matches(reference, matching, name="matching").values())

    mismatched = GATE.CaseArtifacts(
        target_input_sha256="input",
        target_output=torch.tensor([1.0, 3.0], dtype=torch.bfloat16),
        target_gradient=matching.target_gradient,
        prefix_digest_by_rank=matching.prefix_digest_by_rank,
        prefix_layer_digests_by_rank=matching.prefix_layer_digests_by_rank,
        encoder_layer_zero_component_digests_by_rank=matching.encoder_layer_zero_component_digests_by_rank,
        loss_bytes=matching.loss_bytes,
    )
    with pytest.raises(RuntimeError, match="target parity failed"):
        GATE.assert_case_matches(reference, mismatched, name="mismatched")


def test_canonical_report_digest_and_exclusive_publication(tmp_path: Path) -> None:
    unfinalized = {"schema": GATE.REPORT_SCHEMA, "status": "ok", "value": 7}
    report = GATE.finalized_report(unfinalized)

    assert report["report_sha256"] == hashlib.sha256(GATE.canonical_json_bytes(unfinalized)).hexdigest()
    output = tmp_path / "qualification.json"
    assert GATE.write_canonical_json_exclusive(output, report) == output
    assert output.read_bytes() == GATE.canonical_json_bytes(report)
    with pytest.raises(FileExistsError):
        GATE.write_canonical_json_exclusive(output, report)
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_replicated_tensor_gate_rejects_nonfinite_values_without_distributed_runtime() -> None:
    value = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    assert GATE.assert_replicated_tensor("finite", value) == GATE.tensor_sha256(value)
    assert GATE.assert_replicated_tensor("scalar", torch.tensor(1.25)) == GATE.tensor_sha256(torch.tensor(1.25))
    with pytest.raises(RuntimeError, match="non-finite"):
        GATE.assert_replicated_tensor("bad", torch.tensor([float("nan")]))
