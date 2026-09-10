# CALVIN ABC training / D serving: opt-in fused execution

Status: implementation and CPU regression coverage, **not yet GPU-qualified**.
No CALVIN GPU training or D rollout was started for this change. Existing LIBERO
training is left running; A6000 devices are prohibited.

## Execution profiles

The optimized profiles reuse LIBERO's `sample_isolated_grouped_mm_v2` backend.
Its routing groups are `(sample_id, expert_id)`: samples share a launch, not a
routing/reduction domain. The legacy B8/v1 recipes remain available.

| Training profile | GPUs | Physical batch per rank | Accumulation | Global batch | Physical forwards per rank/update | Serving batch |
| --- | --- | --- | --- | --- | --- | --- |
| Legacy B8 | 1 (TP1) or 2 (TP2) | 8 | 8 | 64 | 8 | 8 |
| Fused B8 | 1 (TP1) | 8 | 8 | 64 | 8 | 8 |
| Fused B16 | 1 (TP1) | 16 | 4 | 64 | 4 | 8 |
| Fused B32 | 1 (TP1) | 32 | 2 | 64 | 2 | 8 |
| Fused B64 | 1 (TP1) | 64 | 1 | 64 | 1 | 8 |
| Fused DP2 B32 | 2 independent TP1 replicas | 32 | 1 | 64 | 1 | 8 on one GPU |

For rectified flow, use `configs/calvin_abc_to_d_fused_v2_b{8,16,32,64}.toml`
or `configs/calvin_abc_to_d_dp2_fused_v2_b32.toml`. Direct-regression counterparts
insert `_direct` after `calvin_abc_to_d`.

B32/rank is the explicit DP2 candidate, not a measured maximum. B64 single-GPU
support is also not a claim that it fits this full model. Measure peak memory
and throughput before selecting a profile. There is no automatic batch-size
fallback: silently changing geometry would break the recorded execution contract.

## What is preserved and what changes

- Prefix/vision encoding stays frozen and native BF16, outside the decoder's
  autocast region. Decoder attention LoRA and action-interface trainables and
  optimizer states stay FP32. No prefix tuning or model quantization is added.
- Every global update still starts with the same eight canonical B8 data/noise
  plans. Coalescing preserves their order and draws flow/direct noise separately
  for each original plan. DP2 assigns the first four plans to rank 0 and the
  remaining four to rank 1; it does not reseed a new B32 distribution.
- DP2 sums valid-action element counts across ranks. Each local SSE is divided
  by that global count, then FP32 trainable gradients are **summed**, not averaged.
  This preserves the objective when ranks have different amounts of valid action
  padding. Gradient clipping happens after reduction. The reported loss is global.
- Archive reads are deduplicated across the rank's update and passed together to
  the existing ordered `sample_many` reader. Repeated samples across canonical
  plans are expanded back; they are not dropped from the training distribution.
  The optimized decoded-frame LRU defaults to 4,096 frames per rank (roughly
  580 MB for the two uint8 camera images alone), configurable with
  `--max-cached-frames`. ZIP authentication and dataset identity checks remain.
- Checkpoints explicitly distinguish training batch/DP size from serving B8/TP1.
  D serving loads the consolidated adapter/interface from a DP2 checkpoint onto
  one GPU and installs fused v2 at B8. Singleton replication/output checks and
  request-specific noise semantics remain. The existing once-per-request prefix
  cache is reused across denoising steps; it is not reused across observations.

Eight per-sample expert dispatches become one fused dispatch at B8; coalescing
also reduces physical forward calls. These are structural reductions, **not an
8x wall-clock speedup claim**. Prefix compute, archive I/O, gradient exchange,
validation, checkpointing, and simulator time remain in the end-to-end cost.

ABC training and held-out ABC validation are unchanged. D is not introduced into
training, normalization fitting, or checkpoint selection. Action chunk H=8,
state/action adapters, NFE settings, K={1,4}, and the official evaluation matrix
are unchanged. Official D serving still requires the complete 30,000-update
checkpoint; intermediate/pilot checkpoints use the separate ABC development path.

## Launching after GPU qualification

Example only; replace the artifact paths and choose a new run directory:

```bash
DUO_VLA_CACHE_ROOT=/hdd2/hyunbin/vla/cache \
HF_HOME=/hdd2/hyunbin/vla/huggingface \
./scripts/run_calvin_train_dp2.sh \
  /ABS/PATH/task_ABC_D/training \
  /ABS/PATH/calvin_stats.json \
  /ABS/PATH/NEW_CALVIN_RUN \
  --config configs/calvin_abc_to_d_dp2_fused_v2_b32.toml \
  --prefix-geometry-artifact /ABS/PATH/prefix_geometry.json \
  --seed 0
```

The launcher uses `train-single-gpu` (CUDA 12.9), two processes, and physical
GPUs 0,1. The single-GPU training launcher uses GPU 0 with an explicit fused
single-GPU config. These launchers query availability before starting CUDA,
reject occupied GPUs, require at least 80 GiB/device, and reject A6000s. They
never kill existing processes or fall back to another device. This check is a
point-in-time guard, not an exclusive GPU reservation.

Use the existing `scripts/calvin/run_policy_server_single_gpu.sh` for the final
checkpoint and `run_policy_server_dev_single_gpu.sh` for ABC development. They
have the same GPU 0 availability guard; `--help` remains CPU-only. Example D
policy endpoint (the authenticated D evaluator is launched separately):

```bash
DUO_VLA_CACHE_ROOT=/hdd2/hyunbin/vla/cache \
HF_HOME=/hdd2/hyunbin/vla/huggingface \
./scripts/calvin/run_policy_server_single_gpu.sh \
  /ABS/PATH/NEW_CALVIN_RUN/checkpoints/update-030000 \
  --dataset-root /ABS/PATH/task_ABC_D \
  --socket /ABS/PATH/calvin-policy.sock
```

Checkpoint cadence/retention are not changed by these recipes: the canonical
defaults remain saves every 1,000 updates and permanent retention every 5,000.
Only resume with the same authenticated source, config, runtime, and topology.
This change does not migrate existing TP2/v1 optimizer or RNG state into DP2/v2.
Keep existing training workspaces immutable.

## Verification and remaining GPU gates

`tests/test_calvin_optimized_execution.py` covers all ten recipes, cross-batch
noise streams, duplicate I/O expansion, typed resume geometry, complete and
intermediate checkpoint resolution, Python 3.8 protocol compatibility, and
GPU-launch safety using mocks. Its real two-process CPU/Gloo test compares the
reduced gradients and AdamW step against a global B64 reference with deliberately
unequal per-rank valid-action counts. It does not load the full model or simulate D.

Before a long training run, on authorized idle GPUs 0,1:

1. Compare fixed-anchor v1/v2 full-model outputs and gradients, and singleton
   serving behavior; mathematical isolation does not imply bitwise equivalence
   across different batch geometries/kernels.
2. Measure B8/B16/B32/B64 peak memory, steady-state update time, archive I/O time,
   and DP2 B32 throughput. Select the largest safe, useful batch, not merely the
   largest configuration file.
3. Exercise real two-rank checkpoint save/resume and compare optimizer, scheduler,
   RNG, and next-update outcomes against an uninterrupted run.
4. Serve the resulting checkpoint through TP1/B8 and measure latency and ABC
   development rollouts before final official D evaluation.

The existing `compare_calvin_training_reproducibility.py` is a legacy TP2
qualification tool, not evidence that these new profiles are GPU-qualified.
