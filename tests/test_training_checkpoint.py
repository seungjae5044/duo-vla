from __future__ import annotations

import hashlib
import io
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import duo_vla.training_checkpoint as training_checkpoint
from duo_vla.training import TrainerState
from duo_vla.training_checkpoint import (
    capture_rng_state,
    load_training_rank_state,
    optimizer_parameter_inventory,
    optimizer_parameter_inventory_sha256,
    optimizer_parameter_schema_sha256,
    restore_rng_state,
    save_training_rank_state,
    validate_training_progress,
)


def _optimizer_and_scheduler():
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: 1.0 / (update + 1))
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (parameter.square().sum()).backward()
        optimizer.step()
        scheduler.step()
    return parameter, optimizer, scheduler


def test_cpu_rng_capture_restore_is_exact() -> None:
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    restore_rng_state(state)

    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_training_rank_state_roundtrip_and_hash_guard(tmp_path: Path) -> None:
    parameter, optimizer, scheduler = _optimizer_and_scheduler()
    path = tmp_path / "rank.pt"
    named_parameters = [("parameter", parameter)]
    contract = {
        "config_sha256": "abc",
        "normalization_sha256": "def",
        "optimizer_parameter_schema_sha256": optimizer_parameter_schema_sha256(optimizer, named_parameters),
    }
    digest = save_training_rank_state(
        path,
        rank=0,
        world_size=2,
        trainer_state=TrainerState(next_update=3, examples_seen=192),
        optimizer=optimizer,
        named_parameters=named_parameters,
        scheduler=scheduler,
        run_contract=contract,
    )
    target_parameter, target_optimizer, target_scheduler = _optimizer_and_scheduler()

    restored = load_training_rank_state(
        path,
        expected_sha256=digest,
        rank=0,
        world_size=2,
        optimizer=target_optimizer,
        named_parameters=[("parameter", target_parameter)],
        scheduler=target_scheduler,
        run_contract=contract,
    )

    assert restored == TrainerState(next_update=3, examples_seen=192)
    assert target_scheduler.state_dict() == scheduler.state_dict()
    source_state = next(iter(optimizer.state.values()))
    target_state = next(iter(target_optimizer.state.values()))
    for key in source_state:
        torch.testing.assert_close(target_state[key], source_state[key])

    with pytest.raises(ValueError, match="hash mismatch"):
        load_training_rank_state(
            path,
            expected_sha256="0" * 64,
            rank=0,
            world_size=2,
            optimizer=target_optimizer,
            named_parameters=[("parameter", target_parameter)],
            scheduler=target_scheduler,
            run_contract=contract,
        )


def test_optimizer_parameter_schema_tracks_positional_mapping() -> None:
    first = nn.Parameter(torch.ones(2))
    second = nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([{"name": "trainable", "params": [first, second], "lr": 1e-3}])
    expected = optimizer_parameter_schema_sha256(
        optimizer,
        [("first", first), ("second", second)],
    )
    assert (
        optimizer_parameter_schema_sha256(
            optimizer,
            [("second", second), ("first", first)],
        )
        == expected
    )

    reordered = torch.optim.AdamW([{"name": "trainable", "params": [second, first], "lr": 1e-3}])
    assert (
        optimizer_parameter_schema_sha256(
            reordered,
            [("first", first), ("second", second)],
        )
        != expected
    )
    inventory = optimizer_parameter_inventory(optimizer, [("first", first), ("second", second)])
    assert optimizer_parameter_inventory_sha256(inventory) == expected


def test_rank_state_deserialization_uses_authenticated_bytes_and_rejects_same_byte_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter, optimizer, scheduler = _optimizer_and_scheduler()
    named_parameters = [("parameter", parameter)]
    contract = {"optimizer_parameter_schema_sha256": optimizer_parameter_schema_sha256(optimizer, named_parameters)}
    path = tmp_path / "rank.pt"
    digest = save_training_rank_state(
        path,
        rank=0,
        world_size=1,
        trainer_state=TrainerState(next_update=3, examples_seen=192),
        optimizer=optimizer,
        named_parameters=named_parameters,
        scheduler=scheduler,
        run_contract=contract,
    )
    replacement_parameter, replacement_optimizer, replacement_scheduler = _optimizer_and_scheduler()
    real_load = training_checkpoint.torch.load
    displaced = tmp_path / "authenticated-original.pt"

    def replace_path(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO)
        raw = source.getvalue()
        path.rename(displaced)
        path.write_bytes(raw)
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(training_checkpoint.torch, "load", replace_path)
    with pytest.raises(ValueError, match="identity changed during deserialization"):
        load_training_rank_state(
            path,
            expected_sha256=digest,
            rank=0,
            world_size=1,
            optimizer=replacement_optimizer,
            named_parameters=[("parameter", replacement_parameter)],
            scheduler=replacement_scheduler,
            run_contract=contract,
        )


def test_rank_state_deserialization_rejects_in_place_mutation_after_byte_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter, optimizer, scheduler = _optimizer_and_scheduler()
    named_parameters = [("parameter", parameter)]
    contract = {"optimizer_parameter_schema_sha256": optimizer_parameter_schema_sha256(optimizer, named_parameters)}
    path = tmp_path / "rank.pt"
    digest = save_training_rank_state(
        path,
        rank=0,
        world_size=1,
        trainer_state=TrainerState(next_update=3, examples_seen=192),
        optimizer=optimizer,
        named_parameters=named_parameters,
        scheduler=scheduler,
        run_contract=contract,
    )
    target_parameter, target_optimizer, target_scheduler = _optimizer_and_scheduler()
    real_load = training_checkpoint.torch.load

    def mutate_path(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO)
        with path.open("ab") as handle:
            handle.write(b"mutation")
            handle.flush()
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(training_checkpoint.torch, "load", mutate_path)
    with pytest.raises(ValueError, match="identity changed during deserialization"):
        load_training_rank_state(
            path,
            expected_sha256=digest,
            rank=0,
            world_size=1,
            optimizer=target_optimizer,
            named_parameters=[("parameter", target_parameter)],
            scheduler=target_scheduler,
            run_contract=contract,
        )


@pytest.mark.parametrize("substitution", ("symlink", "hardlink"))
def test_rank_state_resume_rejects_link_substitution(tmp_path: Path, substitution: str) -> None:
    parameter, optimizer, scheduler = _optimizer_and_scheduler()
    named_parameters = [("parameter", parameter)]
    contract = {"optimizer_parameter_schema_sha256": optimizer_parameter_schema_sha256(optimizer, named_parameters)}
    original = tmp_path / "original.pt"
    digest = save_training_rank_state(
        original,
        rank=0,
        world_size=1,
        trainer_state=TrainerState(next_update=3, examples_seen=192),
        optimizer=optimizer,
        named_parameters=named_parameters,
        scheduler=scheduler,
        run_contract=contract,
    )
    candidate = tmp_path / "candidate.pt"
    if substitution == "symlink":
        candidate.symlink_to(original.name)
    else:
        candidate.hardlink_to(original)
    target_parameter, target_optimizer, target_scheduler = _optimizer_and_scheduler()

    with pytest.raises((ValueError, OSError), match=r"regular non-symlink|exactly one link"):
        load_training_rank_state(
            candidate,
            expected_sha256=digest,
            rank=0,
            world_size=1,
            optimizer=target_optimizer,
            named_parameters=[("parameter", target_parameter)],
            scheduler=target_scheduler,
            run_contract=contract,
        )


def test_rank_state_rejects_stale_schema_and_unbound_optimizer_inventory(tmp_path: Path) -> None:
    first = nn.Parameter(torch.ones(2))
    second = nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([{"name": "trainable", "params": [first, second], "lr": 1e-3}])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _update: 1.0)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (first.square().sum() + second.square().sum()).backward()
        optimizer.step()
        scheduler.step()
    named_parameters = [("first", first), ("second", second)]
    contract = {"optimizer_parameter_schema_sha256": optimizer_parameter_schema_sha256(optimizer, named_parameters)}
    path = tmp_path / "rank.pt"
    save_training_rank_state(
        path,
        rank=0,
        world_size=1,
        trainer_state=TrainerState(next_update=2, examples_seen=128),
        optimizer=optimizer,
        named_parameters=named_parameters,
        scheduler=scheduler,
        run_contract=contract,
    )
    canonical_payload = torch.load(path, map_location="cpu", weights_only=True)

    for mutation, message in (("stale", "unsupported"), ("hash", "inventory hash"), ("truncated", "cover")):
        payload = {**canonical_payload}
        expected_contract = dict(contract)
        if mutation == "stale":
            payload["schema"] = "duo-vla-training-rank-state-v1"
        elif mutation == "hash":
            expected_contract["optimizer_parameter_schema_sha256"] = "a" * 64
            payload["run_contract"] = dict(expected_contract)
        else:
            payload["optimizer"] = {
                **payload["optimizer"],
                "state": {0: payload["optimizer"]["state"][0]},
            }
        torch.save(payload, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        target_first = nn.Parameter(torch.ones(2))
        target_second = nn.Parameter(torch.ones(3))
        target_optimizer = torch.optim.AdamW(
            [{"name": "trainable", "params": [target_first, target_second], "lr": 1e-3}]
        )
        target_scheduler = torch.optim.lr_scheduler.LambdaLR(target_optimizer, lambda _update: 1.0)
        with pytest.raises(ValueError, match=message):
            load_training_rank_state(
                path,
                expected_sha256=digest,
                rank=0,
                world_size=1,
                optimizer=target_optimizer,
                named_parameters=[("first", target_first), ("second", target_second)],
                scheduler=target_scheduler,
                run_contract=expected_contract,
            )


def test_training_progress_requires_aligned_optimizer_scheduler_and_examples() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([{"name": "only", "params": [parameter], "lr": 1e-3}])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: 1.0 / (update + 1))
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().sum().backward()
        optimizer.step()
        scheduler.step()
    trainer_state = TrainerState(next_update=3, examples_seen=192)

    validate_training_progress(
        trainer_state,
        optimizer=optimizer,
        scheduler=scheduler,
        examples_per_update=64,
        expected_learning_rates=[optimizer.param_groups[0]["lr"]],
    )

    with pytest.raises(ValueError, match="examples_seen"):
        validate_training_progress(
            TrainerState(next_update=3, examples_seen=191),
            optimizer=optimizer,
            scheduler=scheduler,
            examples_per_update=64,
            expected_learning_rates=[optimizer.param_groups[0]["lr"]],
        )
    with pytest.raises(ValueError, match="learning rate"):
        validate_training_progress(
            trainer_state,
            optimizer=optimizer,
            scheduler=scheduler,
            examples_per_update=64,
            expected_learning_rates=[1.0],
        )
