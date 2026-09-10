from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_libero as TRAIN

from duo_vla.dp2_fork import DP2_FORK_SCHEMA, DP2_RNG_DERIVATION, EXPECTED_TRAINING_GPU_UUIDS
from duo_vla.run_config import load_resolved_toml
from duo_vla.training import masked_sse

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dp2_config() -> dict[str, object]:
    return load_resolved_toml(PROJECT_ROOT / "configs/libero_dp2_fused_v2_b32.toml")


def test_dp2_contract_pins_replica_topology_and_global_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _dp2_config()
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)

    TRAIN._validate_and_build_interface_config(config)
    geometry = TRAIN._execution_geometry(config)

    assert config["optimization"]["physical_batch_size"] == 32
    assert config["optimization"]["microbatch_size"] == 32
    assert config["optimization"]["gradient_accumulation_steps"] == 1
    assert config["optimization"]["global_batch_size"] == 64
    assert config["optimization"]["serving_batch_size"] == 8
    assert config["training"]["max_cached_files"] == TRAIN.DP2_MAX_CACHED_FILES
    assert geometry["execution_profile"] == TRAIN.DP2_EXECUTION_PROFILE
    assert geometry["strategy"] == "data_parallel"
    assert geometry["world_size"] == 2
    assert geometry["data_parallel_size"] == 2
    assert geometry["tensor_parallel_size"] == 1
    assert geometry["rank_physical_batch_size"] == 32
    assert geometry["canonical_plan_partition"] == "contiguous-b8-chunks-by-rank"
    assert geometry["gradient_reduction"] == "sum_globally_normalized_sse_gradients"


def test_dp2_runtime_uses_tp1_venv_despite_two_visible_devices() -> None:
    config = _dp2_config()

    assert (
        TRAIN._training_environment_name(
            expected_world_size=2,
            resolved_config=config,
        )
        == "train-single-gpu"
    )
    assert TRAIN._training_environment_name(expected_world_size=2, resolved_config=None) == "train"
    assert TRAIN._training_environment_name(expected_world_size=1, resolved_config=None) == "train-single-gpu"


def test_dp2_fork_cli_binds_cache_377_and_rejects_contract_overrides() -> None:
    values = {
        "task": None,
        "seed": None,
        "total_updates": None,
        "warmup_updates": None,
        "microbatch_size": None,
        "gradient_accumulation_steps": None,
        "validation_interval": None,
        "validation_samples": None,
        "checkpoint_interval": None,
        "permanent_checkpoint_interval": None,
        "log_interval": None,
        "max_cached_files": TRAIN.DP2_MAX_CACHED_FILES,
    }

    TRAIN._validate_dp2_fork_cli(SimpleNamespace(**values))
    with pytest.raises(ValueError, match="max_cached_files"):
        TRAIN._validate_dp2_fork_cli(SimpleNamespace(**{**values, "max_cached_files": 128}))
    with pytest.raises(ValueError, match="seed"):
        TRAIN._validate_dp2_fork_cli(SimpleNamespace(**{**values, "seed": 1}))


def test_dp2_restore_mode_forbids_fresh_or_ambiguous_dp2_start() -> None:
    TRAIN._validate_dp2_restore_mode(
        is_data_parallel=True,
        has_fork_manifest=True,
        has_resume_checkpoint=False,
    )
    TRAIN._validate_dp2_restore_mode(
        is_data_parallel=True,
        has_fork_manifest=False,
        has_resume_checkpoint=True,
    )
    with pytest.raises(ValueError, match="requires exactly one"):
        TRAIN._validate_dp2_restore_mode(
            is_data_parallel=True,
            has_fork_manifest=False,
            has_resume_checkpoint=False,
        )
    with pytest.raises(ValueError, match="requires exactly one"):
        TRAIN._validate_dp2_restore_mode(
            is_data_parallel=True,
            has_fork_manifest=True,
            has_resume_checkpoint=True,
        )


def test_dp2_fork_digest_argument_requires_external_anchor_and_forbids_resume() -> None:
    fork_path = Path("/frozen/fork.json")
    digest = "a" * 64
    assert (
        TRAIN._validate_dp2_fork_digest_argument(
            fork_from=fork_path,
            resume=None,
            expected_sha256=[digest],
        )
        == digest
    )
    with pytest.raises(ValueError, match="requires one preregistered"):
        TRAIN._validate_dp2_fork_digest_argument(
            fork_from=fork_path,
            resume=None,
            expected_sha256=None,
        )
    with pytest.raises(ValueError, match="valid only with"):
        TRAIN._validate_dp2_fork_digest_argument(
            fork_from=None,
            resume=Path("/run/checkpoint"),
            expected_sha256=[digest],
        )
    with pytest.raises(ValueError, match="requires one preregistered"):
        TRAIN._validate_dp2_fork_digest_argument(
            fork_from=fork_path,
            resume=None,
            expected_sha256=[digest, digest],
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("distributed", "gradient_reduction"), "mean", "unsupported data-parallel topology"),
        (("distributed", "world_size"), 1, "unsupported data-parallel topology"),
        (("optimization", "physical_batch_size"), 16, "rank_physical_batch_size"),
        (("optimization", "gradient_accumulation_steps"), 2, "must equal"),
    ),
)
def test_dp2_contract_rejects_semantic_drift(
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    config = _dp2_config()
    config[path[0]][path[1]] = value
    if path == ("optimization", "physical_batch_size"):
        config["optimization"]["microbatch_size"] = value
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)

    with pytest.raises(ValueError, match=message):
        TRAIN._validate_and_build_interface_config(config)


def test_dp2_rank_partition_is_disjoint_contiguous_and_preserves_all_eight_plans() -> None:
    expected = TRAIN.make_update_plan(
        71,
        19,
        gradient_accumulation_steps=TRAIN.CANONICAL_MICROSTEPS_PER_UPDATE,
    )
    rank_groups = tuple(
        TRAIN._rank_canonical_update_plan_groups(
            71,
            19,
            rank=rank,
            data_parallel_size=2,
            physical_batch_size=32,
        )
        for rank in range(2)
    )
    rank_plans = tuple(tuple(plan for group in groups for plan in group) for groups in rank_groups)

    assert rank_plans[0] == expected[:4]
    assert rank_plans[1] == expected[4:]
    assert rank_plans[0] + rank_plans[1] == expected
    assert all(len(groups) == 1 and len(groups[0]) == 4 for groups in rank_groups)
    assert not set(rank_plans[0]) & set(rank_plans[1])


def test_dp2_validation_partitions_global_2048_into_two_rank_local_1024_streams() -> None:
    plans = tuple(TRAIN.make_microbatch_plan(97, 0, microstep) for microstep in range(2048 // 8))
    local = tuple(TRAIN._partition_canonical_plans_by_rank(plans, rank=rank, data_parallel_size=2) for rank in range(2))
    groups = tuple(TRAIN._group_canonical_plans(rank_plans, physical_batch_size=32) for rank_plans in local)

    assert len(local[0]) == len(local[1]) == 128
    assert len(groups[0]) == len(groups[1]) == 32
    assert all(len(group) == 4 for rank_groups in groups for group in rank_groups)
    assert local[0] + local[1] == plans
    assert [plan.microstep for plan in local[0]] == list(range(128))
    assert [plan.microstep for plan in local[1]] == list(range(128, 256))


def test_dp2_flow_rng_partition_reassembles_the_canonical_global_stream() -> None:
    config = load_resolved_toml(PROJECT_ROOT / "configs/libero.toml")
    contract = TRAIN.policy_contract_from_config(config)
    plans = TRAIN.make_update_plan(31, 7, gradient_accumulation_steps=8)
    clean = torch.linspace(-1.0, 1.0, 64 * 8 * 7, dtype=torch.float32).reshape(64, 8, 7)
    expected = TRAIN._canonical_training_pair(clean, contract, plans)
    rank_pairs = []
    for rank in range(2):
        local_plans = TRAIN._partition_canonical_plans_by_rank(plans, rank=rank, data_parallel_size=2)
        rank_pairs.append(
            TRAIN._canonical_training_pair(
                clean[rank * 32 : (rank + 1) * 32],
                contract,
                local_plans,
            )
        )

    assert torch.equal(torch.cat([pair.input_actions for pair in rank_pairs]), expected.input_actions)
    assert torch.equal(torch.cat([pair.timesteps for pair in rank_pairs]), expected.timesteps)
    assert torch.equal(torch.cat([pair.target for pair in rank_pairs]), expected.target)


def test_dp2_sum_of_globally_normalized_local_gradients_matches_global_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = torch.linspace(-1.0, 1.0, 64 * 2 * 3, dtype=torch.float32).reshape(64, 2, 3)
    targets = torch.flip(values, dims=(0,)) * 0.25
    valid = torch.ones((64, 2), dtype=torch.bool)
    valid[::5, 1] = False

    reference_weight = torch.tensor(0.75, requires_grad=True)
    reference = masked_sse(values * reference_weight, targets, valid)
    reference.mean.backward()
    assert reference_weight.grad is not None

    local_gradients = []
    local_counts = []
    for rank in range(2):
        weight = torch.tensor(0.75, requires_grad=True)
        local = slice(rank * 32, (rank + 1) * 32)
        component = masked_sse(values[local] * weight, targets[local], valid[local])
        local_counts.append(component.element_count)
        component.loss_for_total(reference.element_count).backward()
        assert weight.grad is not None
        local_gradients.append(weight.grad.detach().clone())

    reduced_weight = torch.nn.Parameter(torch.tensor(0.75))
    reduced_weight.grad = local_gradients[0].clone()

    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)

    def all_reduce(value: torch.Tensor, *, op: object) -> None:
        assert op == TRAIN.dist.ReduceOp.SUM
        value.add_(local_gradients[1].reshape_as(value))

    monkeypatch.setattr(TRAIN.dist, "all_reduce", all_reduce)
    TRAIN._sum_data_parallel_gradients_([("weight", reduced_weight)])

    assert sum(local_counts) == reference.element_count
    torch.testing.assert_close(reduced_weight.grad, reference_weight.grad, atol=2e-6, rtol=2e-6)


def test_dp2_initial_local_gradients_need_not_match_before_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([3.0])
    partition = SimpleNamespace(replicated=(("lora", parameter),), sharded=())

    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)

    def gather(output: list[object], value: object) -> None:
        output[:] = [copy.deepcopy(value), copy.deepcopy(value)]

    monkeypatch.setattr(TRAIN.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        TRAIN,
        "assert_replicated_tensor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("pre-reduction comparison")),
    )

    TRAIN._assert_initial_gradients_present(partition, [], data_parallel=True)
    with pytest.raises(AssertionError, match="pre-reduction"):
        TRAIN._assert_initial_gradients_present(partition, [], data_parallel=False)


def test_dp2_fork_metadata_round_trips_and_rejects_cross_link_drift() -> None:
    hashes = {
        name: f"{index:064x}"
        for index, name in enumerate(
            sorted(
                    {
                        "fork_manifest_sha256",
                        "fork_parent_config_sha256",
                        "fork_parent_manifest_sha256",
                        "fork_parent_optimizer_parameter_schema_sha256",
                        "fork_parent_resolved_config_sha256",
                        "fork_parent_source_tree_sha256",
                        "fork_parent_training_rank_state_sha256",
                        "fork_parent_venv_root_sha256",
                        "fork_semantic_recipe_sha256",
                        "fork_static_resolved_toml_sha256",
                }
            ),
            start=1,
        )
    }
    contract = {
        **hashes,
        "fork_parent_checkpoint_device": "12",
        "fork_parent_checkpoint_inode": "34",
        "fork_parent_checkpoint_path": "/runs/parent/checkpoints/update-001000",
        "fork_parent_run_uuid": "00000000-0000-0000-0000-000000000001",
        "fork_parent_update": "1000",
        "fork_rng_derivation": DP2_RNG_DERIVATION,
        "fork_schema": DP2_FORK_SCHEMA,
        "serving_tensor_parallel_size": "1",
        "serving_world_size": "1",
        "training_gpu_uuids": ",".join(EXPECTED_TRAINING_GPU_UUIDS),
        "training_tensor_parallel_size": "1",
        "training_world_size": "2",
    }
    lineage = {
        "fork_manifest_sha256": contract["fork_manifest_sha256"],
        "parent_checkpoint": contract["fork_parent_checkpoint_path"],
        "parent_checkpoint_device": 12,
        "parent_checkpoint_inode": 34,
        "parent_config_sha256": contract["fork_parent_config_sha256"],
        "parent_manifest_sha256": contract["fork_parent_manifest_sha256"],
        "parent_optimizer_parameter_schema_sha256": contract["fork_parent_optimizer_parameter_schema_sha256"],
        "parent_resolved_config_sha256": contract["fork_parent_resolved_config_sha256"],
        "parent_run_uuid": contract["fork_parent_run_uuid"],
        "parent_source_tree_sha256": contract["fork_parent_source_tree_sha256"],
        "parent_update": 1000,
        "semantic_recipe_sha256": contract["fork_semantic_recipe_sha256"],
    }
    manifest = {
        **contract,
        "canonical_plan_partition": "contiguous-b8-chunks-by-rank",
        "cuda_allocator_peak_memory_bytes_by_rank": [10, 20],
        "cuda_allocator_peak_memory_bytes_max": 20,
        "data_parallel_size": 2,
        "expert_batch_isolation": "sample_isolated_grouped_mm_v2",
        "execution_profile": TRAIN.DP2_EXECUTION_PROFILE,
        "fork_lineage": lineage,
        "gradient_reduction": "sum_globally_normalized_sse_gradients",
        "max_cached_files": 377,
        "optimizer_parameter_schema_sha256": contract["fork_parent_optimizer_parameter_schema_sha256"],
        "physical_batch_size": 32,
        "rank_physical_batch_size": 32,
        "run_seed": 0,
        "run_uuid": "00000000-0000-0000-0000-000000000002",
        "serving_batch_size": 8,
        "strategy": "data_parallel",
        "tensor_parallel_size": 1,
        "world_size": 2,
    }

    assert TRAIN.validate_dp2_checkpoint_lineage(manifest) == (contract, lineage)
    corrupted = copy.deepcopy(manifest)
    corrupted["fork_lineage"]["parent_manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="cross-links"):
        TRAIN.validate_dp2_checkpoint_lineage(corrupted)

    missing = copy.deepcopy(manifest)
    del missing["fork_lineage"]
    with pytest.raises(ValueError, match="lineage"):
        TRAIN.validate_dp2_checkpoint_lineage(missing)


def test_dp2_peak_memory_is_gathered_in_rank_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(TRAIN.torch.cuda, "max_memory_allocated", lambda _device: 2 * 2**30)

    def gather(output: list[object], value: object) -> None:
        output[:] = [value, int(3.5 * 2**30)]

    monkeypatch.setattr(TRAIN.dist, "all_gather_object", gather)

    assert TRAIN._data_parallel_peak_memory_bytes(torch.device("cuda", 0)) == [2 * 2**30, int(3.5 * 2**30)]
    assert TRAIN._data_parallel_peak_memory_gib(torch.device("cuda", 0)) == [2.0, 3.5]


def test_dp2_core_uses_explicit_sum_and_never_wraps_ddp() -> None:
    source = (PROJECT_ROOT / "scripts/train_libero.py").read_text(encoding="utf-8")
    server_source = (PROJECT_ROOT / "scripts/serve_libero_policy.py").read_text(encoding="utf-8")

    assert "_sum_data_parallel_gradients_(optimizer_named_parameters)" in source
    assert "component.loss_for_total(total_elements).backward()" in source
    assert "DistributedDataParallel" not in source
    assert "ReduceOp.AVG" not in source
    assert '"fork_lineage": fork_lineage' in source
    assert "_DP2_FORK_ONLY_RUN_CONTRACT_FIELDS" in source
    assert "restore_mode = parser.add_mutually_exclusive_group()" in source
    assert "validate_dp2_checkpoint_lineage(resume_manifest)" in source
    assert "validate_dp2_checkpoint_lineage(manifest)" in server_source
    assert 'fork_lineage["semantic_recipe_sha256"]' in source
    assert 'fork_lineage["semantic_recipe_sha256"]' in server_source
    assert '"cuda_allocator_peak_memory_bytes_by_rank"' in source
    assert '"cuda_allocator_peak_memory_bytes_max"' in source
