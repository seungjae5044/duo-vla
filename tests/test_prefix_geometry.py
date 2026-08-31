from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import duo_vla.prefix_geometry as prefix_geometry_module
from duo_vla.prefix_geometry import (
    CameraGeometry,
    SnapshotTreeIdentity,
    apply_fixed_prefix_chat_template,
    build_prefix_geometry_contract,
    create_prefix_geometry_contract,
    load_prefix_geometry_contract,
    measure_unbounded_prefix_valid_lengths,
    prefix_geometry_artifact_bytes,
    prefix_geometry_content_sha256,
    save_prefix_geometry_contract,
    validate_prefix_geometry_contract,
)


def _identity(repository_id: str = "example/model") -> SnapshotTreeIdentity:
    return SnapshotTreeIdentity(
        repository_id=repository_id,
        revision="a" * 40,
        tree_metadata_sha256="b" * 64,
        content_inventory_sha256="c" * 64,
        files_verified=7,
        total_bytes=1234,
    )


def _cameras() -> tuple[CameraGeometry, ...]:
    return (
        CameraGeometry("third_person", 200, 200),
        CameraGeometry("wrist", 84, 84),
    )


def _contract(*, width: int = 16) -> dict[str, Any]:
    return build_prefix_geometry_contract(
        model_identity=_identity(),
        processor_identity=_identity(),
        ordered_cameras=_cameras(),
        instruction_lengths={"move left": 11, "pick block": 12, "open drawer": 11},
        fixed_physical_prefix_width=width,
        padding_side="left",
    )


def _conversation_instruction(conversations: Any) -> list[str]:
    batch = conversations
    if batch and isinstance(batch[0], dict):
        batch = [batch]
    return [
        next(item["text"] for item in conversation[0]["content"] if item["type"] == "text") for conversation in batch
    ]


class _FakeProcessor:
    def __init__(
        self,
        lengths: dict[str, int],
        *,
        padding_side: str = "left",
        silently_truncate: bool = False,
        images_per_prefix: int = 2,
    ) -> None:
        self.lengths = lengths
        self.tokenizer = SimpleNamespace(padding_side=padding_side)
        self.silently_truncate = silently_truncate
        self.images_per_prefix = images_per_prefix
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, conversations: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        instructions = _conversation_instruction(conversations)
        lengths = [self.lengths[instruction] for instruction in instructions]
        processor_kwargs = kwargs["processor_kwargs"]
        if processor_kwargs["padding"] == "max_length":
            requested = processor_kwargs["max_length"]
            width = requested if self.silently_truncate else max(requested, max(lengths))
            if self.silently_truncate:
                lengths = [min(length, requested) for length in lengths]
        else:
            width = max(lengths)
        input_ids = torch.zeros((len(lengths), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, length in enumerate(lengths):
            if self.tokenizer.padding_side == "left":
                attention_mask[row, -length:] = 1
                input_ids[row, -length:] = torch.arange(1, length + 1)
            else:
                attention_mask[row, :length] = 1
                input_ids[row, :length] = torch.arange(1, length + 1)
        image_batch = len(lengths) * self.images_per_prefix
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": torch.zeros_like(input_ids),
            "pixel_values": torch.zeros((image_batch, 3, 4, 4)),
            "image_position_ids": torch.zeros((image_batch, 4, 2), dtype=torch.long),
        }


def test_contract_binds_identities_ordered_cameras_inventory_and_lengths() -> None:
    contract = _contract()

    assert contract["model"] == _identity().to_dict()
    assert [camera["name"] for camera in contract["ordered_cameras"]] == ["third_person", "wrist"]
    inventory = contract["instruction_inventory"]
    assert inventory["count"] == 3
    assert [record["instruction"] for record in inventory["records"]] == [
        "move left",
        "open drawer",
        "pick block",
    ]
    assert contract["geometry"] == {
        "fixed_physical_prefix_width": 16,
        "maximum_valid_prefix_length": 12,
        "valid_prefix_length_histogram": [
            {"count": 2, "valid_prefix_length": 11},
            {"count": 1, "valid_prefix_length": 12},
        ],
    }
    assert contract["tokenization"]["required_output_fields"] == [
        "attention_mask",
        "image_position_ids",
        "input_ids",
        "mm_token_type_ids",
        "pixel_values",
    ]
    assert validate_prefix_geometry_contract(contract) == contract


def test_builder_rejects_duplicates_and_width_without_a_padding_sentinel() -> None:
    common = {
        "model_identity": _identity(),
        "processor_identity": _identity(),
        "ordered_cameras": _cameras(),
        "padding_side": "left",
    }
    with pytest.raises(ValueError, match="contains duplicates"):
        build_prefix_geometry_contract(
            **common,
            instruction_lengths=(("same", 4), ("same", 4)),
            fixed_physical_prefix_width=8,
        )
    with pytest.raises(ValueError, match="padding sentinel"):
        build_prefix_geometry_contract(
            **common,
            instruction_lengths={"too long": 9},
            fixed_physical_prefix_width=8,
        )
    with pytest.raises(ValueError, match="padding sentinel"):
        build_prefix_geometry_contract(
            **common,
            instruction_lengths={"exactly full": 9},
            fixed_physical_prefix_width=9,
        )


def test_measurement_uses_unbounded_and_exact_fixed_width_without_truncation() -> None:
    processor = _FakeProcessor({"short": 7, "longer": 9})

    contract = create_prefix_geometry_contract(
        processor,
        model_identity=_identity(),
        processor_identity=_identity(),
        ordered_cameras=_cameras(),
        instructions=("longer", "short"),
        fixed_physical_prefix_width=12,
        padding_side="left",
    )

    assert [record["valid_prefix_length"] for record in contract["instruction_inventory"]["records"]] == [9, 7]
    assert len(processor.calls) == 4
    for unbounded in processor.calls[:2]:
        assert unbounded["processor_kwargs"] == {"padding": False, "truncation": False}
    for fixed in processor.calls[2:]:
        assert fixed["processor_kwargs"] == {
            "padding": "max_length",
            "max_length": 12,
            "truncation": False,
        }


def test_unbounded_measurement_discovers_width_without_fixed_padding() -> None:
    processor = _FakeProcessor({"short": 7, "longer": 9})

    lengths = measure_unbounded_prefix_valid_lengths(
        processor,
        instructions=("longer", "short"),
        ordered_cameras=_cameras(),
        padding_side="left",
    )

    assert lengths == {"longer": 9, "short": 7}
    assert len(processor.calls) == 2
    assert all(call["processor_kwargs"] == {"padding": False, "truncation": False} for call in processor.calls)


def test_runtime_helper_rejects_transformers_overlength_result() -> None:
    processor = _FakeProcessor({"too long": 13})
    conversation = [{"role": "user", "content": [{"type": "text", "text": "too long"}]}]

    with pytest.raises(ValueError, match="physical prefix width is 13, expected exactly 12"):
        apply_fixed_prefix_chat_template(
            processor,
            conversation,
            fixed_physical_prefix_width=12,
            padding_side="left",
            expected_batch_size=1,
            images_per_prefix=2,
        )

    assert processor.calls[-1]["processor_kwargs"] == {
        "padding": "max_length",
        "max_length": 12,
        "truncation": False,
    }


def test_measurement_detects_processor_that_silently_truncates() -> None:
    processor = _FakeProcessor({"too long": 13}, silently_truncate=True)

    with pytest.raises(ValueError, match="truncated or changed"):
        create_prefix_geometry_contract(
            processor,
            model_identity=_identity(),
            processor_identity=_identity(),
            ordered_cameras=_cameras(),
            instructions=("too long",),
            fixed_physical_prefix_width=12,
            padding_side="left",
        )


def test_runtime_helper_validates_padding_side_and_mask_contiguity() -> None:
    processor = _FakeProcessor({"one": 5, "two": 7}, padding_side="right")
    conversations = [
        [{"role": "user", "content": [{"type": "text", "text": instruction}]}] for instruction in ("one", "two")
    ]
    output = apply_fixed_prefix_chat_template(
        processor,
        conversations,
        fixed_physical_prefix_width=8,
        padding_side="right",
        expected_batch_size=2,
        images_per_prefix=2,
    )
    assert output["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ]

    with pytest.raises(ValueError, match="padding_side differs"):
        apply_fixed_prefix_chat_template(
            processor,
            conversations,
            fixed_physical_prefix_width=8,
            padding_side="left",
            expected_batch_size=2,
            images_per_prefix=2,
        )


def test_runtime_helper_requires_prefix_and_image_aligned_processor_fields() -> None:
    processor = _FakeProcessor({"one": 5})
    original = processor.apply_chat_template

    def missing_mm_token_type_ids(conversations: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        output = original(conversations, **kwargs)
        del output["mm_token_type_ids"]
        return output

    processor.apply_chat_template = missing_mm_token_type_ids  # type: ignore[method-assign]
    conversation = [{"role": "user", "content": [{"type": "text", "text": "one"}]}]
    with pytest.raises(ValueError, match=r"missing required fields.*mm_token_type_ids"):
        apply_fixed_prefix_chat_template(
            processor,
            conversation,
            fixed_physical_prefix_width=8,
            padding_side="left",
            expected_batch_size=1,
            images_per_prefix=2,
        )

    processor = _FakeProcessor({"one": 5}, images_per_prefix=1)
    with pytest.raises(ValueError, match="pixel_values first axis"):
        apply_fixed_prefix_chat_template(
            processor,
            conversation,
            fixed_physical_prefix_width=8,
            padding_side="left",
            expected_batch_size=1,
            images_per_prefix=2,
        )


def test_schema_extras_and_derived_field_tampering_are_rejected() -> None:
    extra = copy.deepcopy(_contract())
    extra["geometry"]["unrecognized"] = 1
    extra["content_sha256"] = prefix_geometry_content_sha256(extra)
    with pytest.raises(ValueError, match="fields differ"):
        validate_prefix_geometry_contract(extra)

    wrong_histogram = copy.deepcopy(_contract())
    wrong_histogram["geometry"]["valid_prefix_length_histogram"][0]["count"] = 3
    wrong_histogram["content_sha256"] = prefix_geometry_content_sha256(wrong_histogram)
    with pytest.raises(ValueError, match="histogram does not match"):
        validate_prefix_geometry_contract(wrong_histogram)

    bool_substitution = copy.deepcopy(_contract())
    bool_substitution["tokenization"]["tokenize"] = 1
    bool_substitution["content_sha256"] = prefix_geometry_content_sha256(bool_substitution)
    with pytest.raises(ValueError, match="constants differ"):
        validate_prefix_geometry_contract(bool_substitution)


def test_immutable_save_and_authenticated_load_reject_valid_substitution(tmp_path: Path) -> None:
    contract = _contract()
    path = tmp_path / "prefix-geometry.json"
    digest = save_prefix_geometry_contract(path, contract)

    loaded = load_prefix_geometry_contract(
        path,
        expected_content_sha256=digest,
        expected_model_identity=_identity(),
        expected_processor_identity=_identity(),
        expected_ordered_cameras=_cameras(),
        expected_instructions=("pick block", "move left", "open drawer"),
        expected_fixed_physical_prefix_width=16,
    )
    assert loaded == contract
    assert path.stat().st_nlink == 1
    assert path.read_bytes() == prefix_geometry_artifact_bytes(contract)
    with pytest.raises(FileExistsError, match="already exists"):
        save_prefix_geometry_contract(path, contract)

    substitute = _contract(width=20)
    path.write_bytes(prefix_geometry_artifact_bytes(substitute))
    with pytest.raises(ValueError, match="externally pinned"):
        load_prefix_geometry_contract(path, expected_content_sha256=digest)


def test_save_rejects_destination_substitution_and_preserves_foreign_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    path = tmp_path / "prefix-geometry.json"
    foreign = b"foreign destination\n"
    real_link = prefix_geometry_module.os.link

    def substitute_destination(*args: Any, **kwargs: Any) -> None:
        real_link(*args, **kwargs)
        path.unlink()
        path.write_bytes(foreign)

    monkeypatch.setattr(prefix_geometry_module.os, "link", substitute_destination)

    with pytest.raises(ValueError, match="not the linked source inode"):
        save_prefix_geometry_contract(path, contract)
    assert path.read_bytes() == foreign


def test_save_rejects_parent_replacement_without_deleting_published_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    publication = tmp_path / "publication"
    publication.mkdir()
    moved_publication = tmp_path / "publication-original"
    path = publication / "prefix-geometry.json"
    real_link = prefix_geometry_module.os.link

    def substitute_parent(*args: Any, **kwargs: Any) -> None:
        real_link(*args, **kwargs)
        publication.rename(moved_publication)
        publication.mkdir()

    monkeypatch.setattr(prefix_geometry_module.os, "link", substitute_parent)

    with pytest.raises(ValueError, match="parent directory changed"):
        save_prefix_geometry_contract(path, contract)
    assert not path.exists()
    assert (moved_publication / path.name).read_bytes() == prefix_geometry_artifact_bytes(contract)


def test_loader_rejects_noncanonical_duplicate_json_and_symlinks(tmp_path: Path) -> None:
    contract = _contract()
    canonical = tmp_path / "canonical.json"
    digest = save_prefix_geometry_contract(canonical, contract)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical JSON"):
        load_prefix_geometry_contract(noncanonical, expected_content_sha256=digest)

    duplicate = tmp_path / "duplicate.json"
    raw = prefix_geometry_artifact_bytes(contract).decode("utf-8")
    duplicate.write_text(
        raw.replace('{\n  "content_sha256"', '{\n  "schema": "duplicate",\n  "content_sha256"', 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="strict JSON"):
        load_prefix_geometry_contract(duplicate, expected_content_sha256=digest)

    leaf_link = tmp_path / "leaf-link.json"
    leaf_link.symlink_to(canonical)
    with pytest.raises(ValueError, match="without symlinks"):
        load_prefix_geometry_contract(leaf_link, expected_content_sha256=digest)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    parent_artifact = real_parent / "artifact.json"
    save_prefix_geometry_contract(parent_artifact, contract)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="without symlinks"):
        load_prefix_geometry_contract(linked_parent / "artifact.json", expected_content_sha256=digest)
