# Experimental Spatial prefix-encoder LoRA continuation

`scripts/train_libero_encoder_lora.py` is an explicit, separate continuation of the
Spatial 15,000-update checkpoint. It does not change the qualified frozen-prefix
recipes or silently enable encoder training in existing launchers.

The authorized stage is 1,000 **additional optimizer updates**, ending at 16,000.
Decoder LoRA, the action interface, and new prefix language-encoder LoRA all use a
constant learning rate of `5e-5`, without a scheduler or warmup. Decoder/interface
AdamW moments and counters are inherited; encoder moments start fresh. Base model,
vision tower, and multimodal projector weights remain frozen. Action horizon 8,
prefix geometry, Spatial-only task-uniform sampling, train/validation split,
normalization, and flow objective remain inherited from the parent.

The encoder gets independent rank-16, alpha-32, zero-B adapters on its attention
projections. Its last-layer q/o projections are excluded because their only effect
is on the discarded final encoder hidden state, not the prefix KV used by the
action loss. All 113 installed projections receive gradients. Encoder and decoder
base tensors remain tied, but their adapter parameters are independent. Encoder
adapters use a separate state namespace to avoid inclusion in decoder PEFT files.
The unchanged encoder base weights allow this stage's prefix adaptation to be
removed by unloading the encoder adapters. Text-generation quality **with adapters
active is not established**; frozen base weights alone do not guarantee it.

## Memory and numerical checks

The native differentiable prefix fits batch 8 (~88.9 GiB) but batch 16 ran out of
memory on the 96-GB GPU. The separately opted-in `--checkpoint-prefix` path uses
per-layer recomputation with a private, single-layer KV cache inside each closure;
recomputation cannot append duplicate entries to the shared conditioning cache.
It invokes layer modules so the sample-isolated fused-MoE hooks remain balanced.
Under `no_grad`, validation/inference uses the original native encoder forward.

Tiny-model eager/SDPA tests check output, gradient, frozen-vision, cache-length,
serialization isolation, and strict restoration behavior. A real-model batch-8
probe additionally compares all 470 trainable gradient tensors against the native
path before and after one optimizer update. Both comparisons had zero maximum
absolute error, including the nonzero-B second update. Larger supported batches
must be measured before selecting the final run; batch 64 also needs the measured
`PYTORCH_ALLOC_CONF=expandable_segments:True` allocator variant on this host.

## Execution and checkpoints

Use an immutable copy of `src/`, `scripts/`, and `configs/` outside the repository
and the pinned `train-single-gpu` environment. The runner permits only physical
GPU 0/1, rejects A6000, and requires two full data-parallel replicas for a training
stage. Capacity probes can use GPU 0 or 1 individually. Check GPU ownership before
launching; the script does not terminate other jobs or select devices automatically.

Each update consumes disjoint contiguous canonical B8 chunks across ranks, with
global batch size `2 * --batch-size` and no gradient accumulation. Gradients are
normalized by the global valid action-element count, then explicitly SUM-reduced
(no additional division by two). Initial trainables and inherited optimizer states,
the first reduced gradients/update, and every saved checkpoint are checked for
byte-identical replicas.

The default checkpoint/validation interval is 250 additional updates. Validation
uses the same 2,048 held-out samples as the parent. `--stop-after N` performs a clean
stop after N stage updates and saves a checkpoint; `--resume PATH` continues the
same 1,000-update budget, recipe, data stream, optimizer counters, and RNG states.
Capacity probes (`--probe-steps`) never contribute to that budget or save model
checkpoints. SIGTERM/SIGINT request a checkpointed stop at an update boundary.

The output contains `recipe.json`, `encoder_config.json`, `metrics.jsonl`,
`progress.json`, `latest_checkpoint.json`, and hash-verified checkpoints. Every
checkpoint includes decoder PEFT, action interface, encoder adapter safetensors,
both optimizer/RNG states, and the effective continuation recipe. The inherited
`resolved_config.json` is explicitly labeled as a **base recipe only**; the effective
settings are in `artifacts/recipe.json`. Frozen-prefix checkpoint loaders fail
closed on encoder-adapted checkpoints. A serving consumer must explicitly opt in
and install/load the separate encoder state before using such a policy; the old
rollout launcher must not silently ignore those weights.

## Post-training rollout

`scripts/supervise_encoder_rollout.py --training-pid PID --output PATH` watches an
already-live torchrun process using a Linux pidfd; it never starts or restarts
training. After that exact process exits, it requires a successful training
supervisor exit, exactly 1,000 sequential stage metrics with the fixed LR/global
batch, and authenticated checkpoints at 15,250/15,500/15,750/16,000. Only then does
it check that GPUs 0/1 are idle and start two independent TP1 inference replicas.
Cancelling this evaluation supervisor does not cancel the training job.

`scripts/rollout_libero_encoder_lora.py` loads both decoder PEFT and the separate
encoder adapters. It verifies critical policy source hashes against the training
recipe, uses native prefix prefill outside autocast (matching training's FP32
router/adapter math), and BF16 autocast only for the action decoder. It does not
install activation checkpointing for inference. Each server qualifies repeat,
batch permutation, and singleton-padded action outputs before any scored steps.

The fixed requested evaluation is NFE=4, K=8, Spatial tasks 0–9, published reset IDs
0–19: 200 unique episodes, split into 100 per GPU using alternating reset IDs.
Evaluation seed 0, environment seed 7, ten settling steps and a 220-action budget
match the preceding Spatial grid. EGL render-device IDs are resolved from PCI
addresses, not assumed to equal CUDA indices. Simulator CUDA compute is disabled.

Reset physics is recorded over the mandatory ten settling steps without extra
steps, state filtering or replacements. Displacement above 0.1 m or linear speed
above 3 m/s produces a diagnostic warning, not an automatic policy failure or
exclusion. The combined result verifies both exact episode matrices, NFE/K,
checkpoint and encoder-weight hashes, per-shard inference parity and reset-audit
coverage. It writes `result.json`, a combined `episodes.jsonl`, per-GPU logs and
progress. Existing frozen-prefix serving launchers still reject these checkpoints.

These results are experimental, not an official LIBERO benchmark result.

## Completed run (2026-09-10 UTC)

The continuation completed all 1,000 additional updates on GPUs 0/1 with global
batch 128 (64 per replica), prefix recomputation and constant `5e-5` learning rates.
Checkpoints at 15,250, 15,500, 15,750 and 16,000 were authenticated; both final
optimizer replicas matched exactly. Inherited parameter counters reached 16,000
and encoder adapter counters reached 1,000. Vision/base/projector weights remained
outside the trainable parameter inventory.

The final minibatch training loss was `0.1569599485`; final validation loss was
`0.1924395026`, versus the parent's `0.1894425398` on the same held-out sample set.
This does not demonstrate a validation-loss improvement.

The final checkpoint's NFE=4, K=8 evaluation completed **85/200 successes (42.5%)**,
with Wilson 95% interval 35.85–49.43%. Wall time from server launch was 1,110.75 s
(18 min 31 s), including model loading and qualification. Each task had 20 unique
episodes; success counts for task IDs 0 through 9 were respectively
`17, 1, 12, 13, 10, 6, 13, 7, 5, 1`. Both inference replicas passed repeat,
permutation and singleton checks with zero maximum absolute action error.

Reset audits covered all 200 episodes and recorded 102 displacement warnings,
without excluding or replacing any episode. Maximum observed displacement was
0.16282 m, maximum linear speed was 1.36082 m/s, and maximum speed after the ten
settling steps was 0.05922 m/s. A warning alone does not establish a physics bug.
This run does not isolate the causal effect of encoder LoRA from the additional
training; a matched frozen-prefix continuation/control would be needed for that.

Artifacts are external to Git, relative to `/hdd2/hyunbin/vla/cache`:

- Training: `runs/libero-spatial15k-prefix-lora-dp2-b128-lr5e-5-1k-recompute-v1`.
- Final checkpoint: the training run's `checkpoints/update-016000`, manifest SHA256
  `92ef7fb967ef056f44d9205d4bb3cdd53d9ef51f023353c67f182f609024ca5e`.
- Evaluation: `reports/spatial-prefix-lora-final-nfe4-k8-200-v2`, including
  `result.json`, `protocol.json`, combined `episodes.jsonl`, and per-GPU reset audits.

Final checks reconciled both shards against all 200 expected task/reset pairs,
verified the policy/encoder checkpoint identities and all 102 recorded evaluation
source hashes, and confirmed that all owned training/evaluation processes exited.
