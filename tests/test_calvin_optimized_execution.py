from __future__ import annotations

import ast
import copy
import importlib.util
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from duo_vla import calvin_execution as execution
from duo_vla.data.calvin import CalvinAnchor
from duo_vla.data.calvin_batching import coalesce_calvin_batches, collate_calvin_samples
from duo_vla.gpu_preflight import require_idle_training_gpus
from duo_vla.objectives import make_seeded_policy_training_pair
from duo_vla.policy_contract import policy_contract_from_config
from duo_vla.run_config import canonical_config_sha256, load_resolved_toml, save_resolved_config
from duo_vla.runtime_integrity import static_environment_identity
from duo_vla.training import make_update_plan, masked_element_count, masked_sse

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


TRAIN = module("optimized_calvin_trainer_test", "scripts/train_calvin.py")
SERVER = module("optimized_calvin_server_test", "scripts/calvin/serve_policy.py")
DEV_BRIDGE = module("optimized_calvin_dev_bridge_test", "scripts/calvin/calvin_dev_bridge.py")
EVAL = module("optimized_calvin_evaluator_test", "scripts/calvin/evaluate_calvin.py")


def config(batch=32, *, dp=False, direct=False):
    stem = "calvin_abc_to_d" + ("_direct" if direct else "")
    suffix = "_dp2_fused_v2_b32" if dp else f"_fused_v2_b{batch}"
    return load_resolved_toml(ROOT / "configs" / f"{stem}{suffix}.toml")


@pytest.mark.parametrize("batch,dp", [(8, False), (16, False), (32, False), (64, False), (32, True)])
@pytest.mark.parametrize("direct", [False, True])
def test_all_optimized_recipes_connect_trainer_serving_and_python38_boundaries(monkeypatch, batch, dp, direct):
    recipe = config(batch, dp=dp, direct=direct)
    topology = execution.execution_from_config(recipe)
    monkeypatch.setattr(TRAIN.dist, "get_world_size", lambda: topology.world_size)
    interface = TRAIN._validate_and_build_interface_config(recipe)
    assert interface.state_dim == 8 and interface.action_horizon == 8
    assert topology.physical_batch * topology.accumulation * topology.data_parallel_size == 64
    geometry = execution.execution_geometry(recipe)
    recipe["execution_geometry"] = geometry
    assert SERVER._execution_geometry_from_config(recipe) == geometry
    assert SERVER._validate_execution_geometry(geometry) == geometry
    SERVER._calvin_bridge._validate_execution_geometry(geometry, allow_none=False)
    DEV_BRIDGE._validate_execution_geometry(geometry, allow_none=False)
    assert EVAL._validate_execution_geometry(geometry, "test") == geometry
    assert geometry["serving_batch_size"] == 8 and geometry["tensor_parallel_size"] == 1
    objective = "direct_regression" if direct else "rectified_flow"
    assert load_resolved_toml(ROOT / "configs" / execution.canonical_recipe_name(recipe, objective)) == {
        k: v for k, v in recipe.items() if k != "execution_geometry"
    }
    assert recipe["benchmark"]["train_environments"] == ["A", "B", "C"]
    assert recipe["benchmark"]["evaluation_environment"] == "D"
    assert recipe["benchmark"]["execution_horizons"] == [1, 4]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("optimization", "global_batch_size", 128),
        ("optimization", "microbatch_size", 8),
        ("optimization", "gradient_accumulation_steps", 2),
        ("optimization", "serving_batch_size", 1),
        ("model", "tensor_parallel_size", 2),
        ("model", "tensor_parallel_size", True),
        ("distributed", "gradient_reduction", "mean"),
        ("distributed", "canonical_plan_partition", "random"),
        ("distributed", "tensor_parallel_size", True),
    ],
)
def test_rejects_optimized_recipe_drift(section, key, value):
    recipe = config(dp=True)
    recipe[section][key] = value
    with pytest.raises(ValueError):
        execution.execution_from_config(recipe, world_size=2)


@pytest.mark.parametrize(
    "key,value",
    [
        ("physical_batch_size", 16),
        ("data_parallel_size", True),
        ("serving_batch_size", 1),
        ("global_batch_size", 128),
        ("tensor_parallel_size", 2),
        ("execution_profile", "duovla-single-gpu-tp1-v1"),
    ],
)
def test_wire_geometry_rejects_training_serving_topology_spoofing(key, value):
    geometry = execution.execution_geometry(config(dp=True))
    geometry[key] = value
    for validate in (
        SERVER._validate_execution_geometry,
        lambda x: SERVER._calvin_bridge._validate_execution_geometry(x, allow_none=False),
        lambda x: DEV_BRIDGE._validate_execution_geometry(x, allow_none=False),
        lambda x: EVAL._validate_execution_geometry(x, "test"),
    ):
        with pytest.raises((RuntimeError, ValueError)):
            validate(geometry)


@pytest.mark.parametrize("batch", [8, 16, 32, 64])
def test_batch_coalescing_preserves_eight_canonical_rng_plans(batch):
    plans = make_update_plan(1, 91, gradient_accumulation_steps=8)
    groups = execution.group_plans(plans, physical_batch=batch)
    assert tuple(p for group in groups for p in group) == plans
    assert len(groups) == 64 // batch


@pytest.mark.parametrize("direct", [False, True])
def test_dp_partition_and_objective_rng_reconstruct_original_b8_stream(direct):
    plans = make_update_plan(2, 75, gradient_accumulation_steps=8)
    ranked = [execution.group_plans(plans, physical_batch=32, rank=rank, data_parallel_size=2)[0] for rank in range(2)]
    assert ranked[0] + ranked[1] == plans and set(ranked[0]).isdisjoint(ranked[1])
    clean = torch.linspace(-1, 1, 64 * 8 * 7).reshape(64, 8, 7)
    contract = policy_contract_from_config(config(dp=True, direct=direct))
    canonical = [
        make_seeded_policy_training_pair(clean[i * 8 : (i + 1) * 8], contract, seed=p.flow_seed)
        for i, p in enumerate(plans)
    ]
    coalesced = [execution.canonical_training_pair(clean[r * 32 : (r + 1) * 32], contract, ranked[r]) for r in range(2)]
    for key in ("input_actions", "timesteps", "target"):
        assert torch.equal(
            torch.cat([getattr(p, key) for p in canonical]), torch.cat([getattr(p, key) for p in coalesced])
        )


def test_rank_wide_io_deduplicates_reads_without_dropping_repeated_samples(monkeypatch):
    helpers = module("calvin_batch_test_helpers", "tests/test_calvin_batching.py")
    anchors = tuple(CalvinAnchor(annotation_index=i, global_index=i, task="task") for i in range(8))
    samples = tuple(helpers._sample(i, i, i + 1) for i in range(8))
    calls = []
    dataset = SimpleNamespace(sample_many=lambda values: calls.append(values) or samples)
    monkeypatch.setattr(TRAIN, "_fixed_distinct_anchors", lambda *a, **kw: anchors)
    plans = make_update_plan(0, 0, gradient_accumulation_steps=8)
    groups = execution.group_plans(plans, physical_batch=64)
    batches = TRAIN._materialize_plan_batches(dataset, object(), groups, state_normalizer=helpers._state_normalizer())
    assert calls == [anchors]
    assert len(batches) == 1 and batches[0].batch_size == 64
    baseline = collate_calvin_samples(samples, state_normalizer=helpers._state_normalizer())
    for name in ("states", "clean_actions", "action_valid_mask"):
        assert torch.equal(getattr(batches[0], name), torch.cat([getattr(baseline, name)] * 8))
    # Cross-plan duplicates are intentional. Within-plan duplicates still fail.
    with pytest.raises(ValueError, match="distinct"):
        collate_calvin_samples((samples[0], samples[0]), state_normalizer=helpers._state_normalizer())
    with pytest.raises(ValueError):
        coalesce_calvin_batches((baseline,), physical_batch_size=32)


def test_sample_isolation_backend_selection_is_explicit():
    install, verify = execution.expert_backend_functions(execution.FUSED_BACKEND)
    assert install.__name__ == "install_sample_isolated_grouped_mm_experts_v2"
    assert verify.__name__ == "verify_sample_isolated_grouped_mm_experts_v2"
    with pytest.raises(ValueError):
        execution.expert_backend_functions("batched_without_isolation")


def _dp_cpu_worker(rank, rendezvous):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        torch.manual_seed(101)
        reference = torch.nn.Linear(3, 7)
        replica = copy.deepcopy(reference)
        inputs = torch.randn(64, 8, 3)
        targets = torch.randn(64, 8, 7)
        valid = torch.ones(64, 8, dtype=torch.bool)
        valid[:32, 2:] = False  # Unequal valid counts catch per-rank mean/DDP-average mistakes.
        local = slice(rank * 32, (rank + 1) * 32)
        count = execution.reduce_scalar(
            masked_element_count(valid[local], action_dim=7), device=torch.device("cpu"), dtype=torch.int64
        )
        assert count == masked_element_count(valid, action_dim=7)
        masked_sse(reference(inputs), targets, valid).mean.backward()
        masked_sse(replica(inputs[local]), targets[local], valid[local]).loss_for_total(count).backward()
        execution.sum_gradients_(tuple(replica.named_parameters()))
        for expected, actual in zip(reference.parameters(), replica.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-6, atol=1e-7)
        expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001)
        optimizer = torch.optim.AdamW(replica.parameters(), lr=0.001)
        expected_optimizer.step()
        optimizer.step()
        for expected, actual in zip(reference.parameters(), replica.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
        gathered = [None, None]
        dist.all_gather_object(gathered, {k: v.detach().tolist() for k, v in replica.state_dict().items()})
        assert gathered[0] == gathered[1]
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_real_two_process_cpu_reduction_matches_global_batch_optimizer_step(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.spawn(_dp_cpu_worker, args=(str(tmp_path / "gloo-init"),), nprocs=2, join=True)


@pytest.mark.parametrize("busy,a6000", [(False, False), (True, False), (False, True)])
def test_gpu_guard_is_read_only_and_refuses_busy_or_forbidden_devices(monkeypatch, busy, a6000):
    import duo_vla.gpu_preflight as guard

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert "--id=0,1" in command and kwargs["timeout"] == 15
        if any("--query-gpu=" in arg for arg in command):
            name = "NVIDIA RTX A6000" if a6000 else "NVIDIA RTX PRO 6000 Blackwell"
            return SimpleNamespace(stdout=f"0, GPU-one, {name}, 98304\n1, GPU-two, {name}, 98304\n")
        return SimpleNamespace(stdout="GPU-one, 123\n" if busy else "")

    monkeypatch.setattr(guard.subprocess, "run", run)
    if busy or a6000:
        with pytest.raises(RuntimeError, match=r"busy|prohibited"):
            require_idle_training_gpus((0, 1))
    else:
        assert require_idle_training_gpus((0, 1))["idle"] is True
    before = len(calls)
    with pytest.raises(ValueError, match="prohibited"):
        require_idle_training_gpus((2, 3))
    assert len(calls) == before


def test_source_inventory_and_python38_protocol_boundaries_remain_consistent():
    assert TRAIN._source_tree_sha256(ROOT, single_gpu=True) == SERVER._source_tree_sha256(ROOT, single_gpu=True)
    for relative in execution.OPTIMIZED_SOURCE_PATHS:
        assert relative in TRAIN._CALVIN_SINGLE_GPU_SOURCE_EXPLICIT_RELATIVE_PATHS
    for name in ("calvin_bridge.py", "calvin_dev_bridge.py", "evaluate_calvin.py"):
        ast.parse((ROOT / "scripts/calvin" / name).read_text(), feature_version=(3, 8))
    source = (ROOT / "scripts/train_calvin.py").read_text()
    assert "replica_mode" in source and "sum_gradients_(optimizer_named_parameters)" in source
    assert "gradient_accumulation_steps=8" in source and "advance(examples=64)" in source
    assert "DistributedDataParallel(" not in source


def test_launchers_never_select_a6000_or_implicitly_preempt_training():
    for relative in (
        "scripts/run_calvin_train_single_gpu.sh",
        "scripts/run_calvin_train_dp2.sh",
        "scripts/calvin/run_policy_server_single_gpu.sh",
        "scripts/calvin/run_policy_server_dev_single_gpu.sh",
    ):
        source = (ROOT / relative).read_text()
        assert "require_idle_training_gpus" in source
        assert "CUDA_VISIBLE_DEVICES=0" in source and "CUDA_VISIBLE_DEVICES=2" not in source
        assert "pkill" not in source and "kill -" not in source


def _optimized_checkpoint(tmp_path, monkeypatch, *, dp):
    """Exercise real identity/config validation; only artifact bytes are synthetic."""
    import duo_vla.backbones.loading
    import duo_vla.checkpointing
    import duo_vla.data.calvin_stats
    import duo_vla.run_config

    helpers = module("optimized_checkpoint_fixture_helpers", "tests/test_calvin_policy_server.py")
    checkpoint, manifest, recipe, stats = helpers._checkpoint_fixture(tmp_path)
    synthetic_benchmark = copy.deepcopy(recipe["benchmark"])
    recipe.update(config(64 if not dp else 32, dp=dp))
    recipe["benchmark"] = synthetic_benchmark
    recipe["execution_geometry"] = execution.execution_geometry(recipe)
    recipe["source_tree_sha256"] = SERVER._source_tree_sha256(ROOT, single_gpu=True)
    environment = recipe["execution_environment"]
    runtime = environment["authenticated_runtime"]
    world = 2 if dp else 1
    runtime.update(
        lock_path="envs/train-single-gpu/uv.lock",
        lock_sha256=SERVER.SINGLE_GPU_TRAIN_LOCK_SHA256,
        packages=SERVER.SINGLE_GPU_EXPECTED_TRAIN_PACKAGES,
    )
    runtime["environment"]["DUO_VLA_TRAIN_VENV"] = "/root/.cache/duo-vla/venvs/train-single-gpu"
    runtime["environment"]["CUDA_VISIBLE_DEVICES"] = "0,1" if dp else "0"
    runtime["static_environment_sha256"] = static_environment_identity(runtime["environment"])["sha256"]
    runtime["module_origins"] = {
        key: value.replace("/venvs/train/", "/venvs/train-single-gpu/")
        for key, value in runtime["module_origins"].items()
    }
    runtime["sys_path"] = [value.replace("/venvs/train/", "/venvs/train-single-gpu/") for value in runtime["sys_path"]]
    runtime["torchrun"].update(local_world_size=world, role_world_size=world, world_size=world)
    environment.update(
        cuda_runtime="12.9", world_size=world, gpu_names=["Test GPU"] * world, gpu_capability=[[12, 0]] * world
    )
    for key in ("torch", "transformers", "peft"):
        environment[key] = SERVER.SINGLE_GPU_EXPECTED_TRAIN_PACKAGES[key]
    manifest.update(recipe["execution_geometry"])
    manifest.update(
        execution_geometry=recipe["execution_geometry"],
        source_tree_sha256=recipe["source_tree_sha256"],
        execution_environment_sha256=canonical_config_sha256(environment),
        config_sha256=save_resolved_config(checkpoint / "artifacts/resolved_config.json", recipe),
    )
    canonical_name = execution.canonical_recipe_name(recipe, "rectified_flow")
    frozen_canonical = copy.deepcopy(recipe)
    real_load = duo_vla.run_config.load_resolved_toml
    monkeypatch.setattr(
        duo_vla.run_config,
        "load_resolved_toml",
        lambda path: copy.deepcopy(frozen_canonical) if Path(path).name == canonical_name else real_load(path),
    )
    monkeypatch.setattr(duo_vla.checkpointing, "load_checkpoint_manifest", lambda *a, **kw: manifest)
    monkeypatch.setattr(duo_vla.backbones.loading, "validate_decoder_attention_lora_weights", lambda *a, **kw: {})
    monkeypatch.setattr(duo_vla.data.calvin_stats, "load_calvin_state_normalizer", lambda *a, **kw: (object(), stats))
    monkeypatch.setenv("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla")
    monkeypatch.setenv("HF_HOME", "/root/.cache/huggingface")
    monkeypatch.setattr(SERVER, "content_address_train_venv", lambda _root: helpers._train_venv_identity())
    monkeypatch.setattr(
        SERVER,
        "_committed_checkpoint_record",
        lambda *a, **kw: SimpleNamespace(
            update=30_000, manifest_sha256=SERVER.sha256_file(checkpoint / "manifest.json")
        ),
    )
    return helpers, checkpoint, manifest, recipe, stats


@pytest.mark.parametrize("dp", [False, True])
def test_complete_optimized_checkpoint_resolves_without_confusing_training_and_serving_world_size(
    tmp_path,
    monkeypatch,
    dp,
):
    helpers, checkpoint, manifest, recipe, _stats = _optimized_checkpoint(tmp_path, monkeypatch, dp=dp)
    _, _, seed, report, resolved, contract, _ = SERVER.resolve_checkpoint(
        checkpoint,
        training_root=tmp_path / "task_ABC_D/training",
        project_root=ROOT,
        train_seed_override=None,
        model_snapshot_report=helpers._model_snapshot_report(),
        authenticated_generation=object(),
    )
    assert seed == 1 and resolved == recipe and contract["objective"] == "rectified_flow"
    assert report["training_execution_environment"]["world_size"] == (2 if dp else 1)
    assert resolved["model"]["tensor_parallel_size"] == 1
    assert resolved["execution_geometry"]["serving_batch_size"] == 8
    for key in execution.EXTENDED_GEOMETRY_FIELDS | {"tensor_parallel_size", "physical_batch_size"}:
        changed = {**manifest, key: str(manifest[key])}
        with pytest.raises(RuntimeError, match=key):
            SERVER._load_checkpoint_prefix_geometry(
                checkpoint, changed, recipe, model_snapshot_report=helpers._model_snapshot_report()
            )
    # Both objects share the runtime payload, as a loaded manifest/config would.
    manifest["execution_environment"]["cuda_runtime"] = "12.6"
    manifest["execution_environment_sha256"] = canonical_config_sha256(manifest["execution_environment"])
    with pytest.raises(RuntimeError, match="canonical training runtime"):
        SERVER._validate_checkpoint_training_environment(manifest, recipe, project_root=ROOT, run_seed=1)


def test_intermediate_dp_checkpoint_resolves_only_through_abc_development_path(tmp_path, monkeypatch):
    from duo_vla.data.calvin_dev_states import AuthenticatedCalvinDevInputs

    helpers, checkpoint, manifest, recipe, stats = _optimized_checkpoint(tmp_path, monkeypatch, dp=True)
    dev = module("optimized_dev_checkpoint_resolver", "scripts/calvin/serve_policy_dev.py")
    dev_helpers = module("optimized_dev_checkpoint_helpers", "tests/test_calvin_dev_policy_server.py")
    dev_helpers._install_v4_storage_identity(manifest, recipe, stats)
    pilot = checkpoint.with_name("update-000120")
    checkpoint.rename(pilot)
    recipe["optimization"].update(total_updates=500, warmup_updates=50)
    manifest.update(dev_helpers._progress_manifest(total=500, update=120, complete=False))
    manifest["config_sha256"] = save_resolved_config(pilot / "artifacts/resolved_config.json", recipe)
    monkeypatch.setattr(dev.official_policy, "content_address_train_venv", lambda _root: helpers._train_venv_identity())
    monkeypatch.setattr(
        dev,
        "_committed_checkpoint_record",
        lambda *a, **kw: SimpleNamespace(update=120, manifest_sha256=SERVER.sha256_file(pilot / "manifest.json")),
    )
    authenticated_inputs = AuthenticatedCalvinDevInputs(
        identity={},
        member_index=dict(stats["dataset"]["member_index"]),
        split=dict(stats["split"]),
        stats=stats,
        training_root=str((tmp_path / "task_ABC_D/training").resolve()),
    )
    _, _, _, report, resolved, _, _ = dev.resolve_development_checkpoint(
        pilot,
        training_root=tmp_path / "task_ABC_D/training",
        project_root=ROOT,
        authenticated_inputs=authenticated_inputs,
        train_seed_override=1,
        model_snapshot_report=helpers._model_snapshot_report(),
    )
    assert report["checkpoint_progress"]["selected_update"] == 120
    assert report["benchmark_status"] == dev.DEVELOPMENT_STATUS
    assert resolved["execution_geometry"]["data_parallel_size"] == 2
    with pytest.raises(RuntimeError, match="configuration mismatches"):
        SERVER.resolve_checkpoint(
            pilot,
            training_root=tmp_path / "task_ABC_D/training",
            project_root=ROOT,
            train_seed_override=None,
            model_snapshot_report=helpers._model_snapshot_report(),
            authenticated_generation=object(),
        )


def test_resume_requires_exact_typed_training_and_serving_topology():
    geometry = execution.execution_geometry(config(dp=True))
    contract = {key: str(value) for key, value in geometry.items()}
    TRAIN._validate_resume_manifest_run_contract(geometry, contract)
    for key in execution.EXTENDED_GEOMETRY_FIELDS | {"physical_batch_size", "tensor_parallel_size"}:
        for wrong in (str(geometry[key]), True, geometry[key] + 1):
            changed = {**geometry, key: wrong}
            with pytest.raises(ValueError, match=key):
                TRAIN._validate_resume_manifest_run_contract(changed, contract)
