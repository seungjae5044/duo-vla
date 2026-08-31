from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from duo_vla.checkpointing import (
    interface_state_dict,
    load_checkpoint_manifest,
    load_interface_state_dict,
    load_lora_checkpoint,
    save_trainable_checkpoint,
)


class FakeAdapter:
    def save_pretrained(
        self,
        path: str,
        *,
        safe_serialization: bool,
        is_main_process: bool,
        save_embedding_layers: bool,
    ) -> None:
        assert safe_serialization and is_main_process and not save_embedding_layers
        output = Path(path)
        output.mkdir(parents=True)
        (output / "adapter_config.json").write_text(json.dumps({"r": 2}), encoding="utf-8")
        save_file({"layer.lora_A.weight": torch.ones(2, 3)}, output / "adapter_model.safetensors")


class IncompleteAdapter(FakeAdapter):
    def save_pretrained(
        self,
        path: str,
        *,
        safe_serialization: bool,
        is_main_process: bool,
        save_embedding_layers: bool,
    ) -> None:
        output = Path(path)
        output.mkdir(parents=True)
        (output / "adapter_config.json").write_text(json.dumps({"r": 2}), encoding="utf-8")


class FailingAdapter(FakeAdapter):
    def save_pretrained(
        self,
        path: str,
        *,
        safe_serialization: bool,
        is_main_process: bool,
        save_embedding_layers: bool,
    ) -> None:
        raise OSError("simulated adapter write failure")


def test_interface_state_roundtrip_is_strict(tmp_path: Path) -> None:
    source = {"projector": nn.Linear(3, 4), "velocity_head": nn.Linear(4, 2)}
    state = interface_state_dict(source)
    assert set(state) == {
        "projector.bias",
        "projector.weight",
        "velocity_head.bias",
        "velocity_head.weight",
    }
    path = tmp_path / "interface.safetensors"
    save_file(state, path)
    target = {"projector": nn.Linear(3, 4), "velocity_head": nn.Linear(4, 2)}
    load_interface_state_dict(path, target)
    for key, expected in state.items():
        module_name, parameter_name = key.split(".", maxsplit=1)
        torch.testing.assert_close(target[module_name].state_dict()[parameter_name], expected)


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("missing", "tensor keys differ"),
        ("unexpected", "tensor keys differ"),
        ("shape", "tensor shapes differ"),
        ("dtype", "dtype torch.float32"),
        ("nonfinite", "non-finite values"),
    ),
)
def test_interface_loader_authenticates_all_tensors_before_applying_state(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    source = {"projector": nn.Linear(3, 4), "velocity_head": nn.Linear(4, 2)}
    state = {name: value.clone() for name, value in interface_state_dict(source).items()}
    if corruption == "missing":
        state.pop("velocity_head.bias")
    elif corruption == "unexpected":
        state["projector.unexpected"] = torch.zeros(1)
    elif corruption == "shape":
        state["velocity_head.weight"] = state["velocity_head.weight"][:, :-1].contiguous()
    elif corruption == "dtype":
        state["projector.bias"] = state["projector.bias"].to(torch.bfloat16)
    elif corruption == "nonfinite":
        state["projector.bias"][0] = torch.nan
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(corruption)

    path = tmp_path / f"interface-{corruption}.safetensors"
    save_file(state, path)
    target = {"projector": nn.Linear(3, 4), "velocity_head": nn.Linear(4, 2)}
    original = {name: value.clone() for name, value in interface_state_dict(target).items()}

    with pytest.raises(ValueError, match=message):
        load_interface_state_dict(path, target)

    # A failure in a later module must not leave an earlier module partially loaded.
    assert all(torch.equal(value, original[name]) for name, value in interface_state_dict(target).items())


def test_trainable_checkpoint_has_hash_verified_manifest(tmp_path: Path) -> None:
    output = tmp_path / "checkpoint"
    stats = tmp_path / "normalization.json"
    stats.write_text('{"schema":"stats-v1"}\n', encoding="utf-8")
    manifest = save_trainable_checkpoint(
        output,
        adapted_model=FakeAdapter(),
        interface_modules={"projector": nn.Linear(3, 4)},
        manifest={"step": 7, "model_revision": "abc"},
        additional_artifacts={"normalization": stats},
    )
    assert manifest is not None
    assert not (output / "INCOMPLETE").exists()
    loaded = load_checkpoint_manifest(output)
    assert loaded["step"] == 7
    assert loaded["model_revision"] == "abc"
    assert set(loaded["artifacts"]) == {"interface", "lora_config", "lora_weights", "normalization"}
    assert (output / loaded["artifacts"]["normalization"]["path"]).read_text(encoding="utf-8") == stats.read_text(
        encoding="utf-8"
    )

    weights = output / loaded["artifacts"]["interface"]["path"]
    weights.write_bytes(weights.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="wrong size"):
        load_checkpoint_manifest(output)


def test_checkpoint_refuses_overwrite_and_incomplete_load(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="cannot create"):
        save_trainable_checkpoint(
            existing,
            adapted_model=FakeAdapter(),
            interface_modules={"projector": nn.Linear(1, 1)},
            manifest={},
        )
    (existing / "INCOMPLETE").write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        load_checkpoint_manifest(existing)


@pytest.mark.parametrize("malicious_path", ["../outside.bin", "/tmp/outside.bin"])
def test_checkpoint_manifest_rejects_artifacts_outside_checkpoint(
    tmp_path: Path,
    malicious_path: str,
) -> None:
    output = tmp_path / "checkpoint"
    manifest = save_trainable_checkpoint(
        output,
        adapted_model=FakeAdapter(),
        interface_modules={"projector": nn.Linear(1, 1)},
        manifest={},
    )
    assert manifest is not None
    manifest["artifacts"]["interface"]["path"] = malicious_path
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="contained relative path"):
        load_checkpoint_manifest(output)


def test_lora_loader_rejects_verified_decoy_paths_before_fixed_peft_consumer(tmp_path: Path) -> None:
    output = tmp_path / "checkpoint"
    manifest = save_trainable_checkpoint(
        output,
        adapted_model=FakeAdapter(),
        interface_modules={"projector": nn.Linear(1, 1)},
        manifest={},
    )
    assert manifest is not None
    decoy = output / "decoy-adapter-config.json"
    decoy.write_bytes((output / "lora/adapter_config.json").read_bytes())
    manifest["artifacts"]["lora_config"]["path"] = decoy.name
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed PEFT consumer"):
        load_lora_checkpoint(output, nn.Linear(1, 1))


def test_checkpoint_finalization_failure_retains_incomplete_marker(tmp_path: Path) -> None:
    output = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError, match="checkpoint finalization failed"):
        save_trainable_checkpoint(
            output,
            adapted_model=IncompleteAdapter(),
            interface_modules={"projector": nn.Linear(1, 1)},
            manifest={},
        )
    assert (output / "INCOMPLETE").is_file()


def test_checkpoint_rejects_nonfinite_manifest_values(tmp_path: Path) -> None:
    output = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError, match="checkpoint finalization failed"):
        save_trainable_checkpoint(
            output,
            adapted_model=FakeAdapter(),
            interface_modules={"projector": nn.Linear(1, 1)},
            manifest={"loss": float("nan")},
        )
    assert (output / "INCOMPLETE").is_file()


def test_checkpoint_propagates_adapter_serialization_failure(tmp_path: Path) -> None:
    output = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError, match="LoRA checkpoint serialization failed"):
        save_trainable_checkpoint(
            output,
            adapted_model=FailingAdapter(),
            interface_modules={"projector": nn.Linear(1, 1)},
            manifest={},
        )
    assert (output / "INCOMPLETE").is_file()
