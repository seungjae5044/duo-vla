# Training and reproduction

This document describes the implemented training path. Commands below launch the committed recipe; they do not imply
that the full LIBERO or CALVIN ABC→D benchmark has been completed. In particular, a 500-update CALVIN run is a
development pilot, not the predeclared 30,000-update benchmark training budget.

## Method

Each example contains a third-person RGB image, a wrist RGB image, a raw language instruction, current proprioceptive
state, and an action chunk `A_1` with horizon `H=8` and action dimension `Da=7`. One scalar `t ~ Uniform(0,1)` and one
iid Gaussian tensor `epsilon` are sampled per chunk:

```text
A_t = (1 - t) * epsilon + t * A_1
V*  = A_1 - epsilon
loss = masked_mean((Vhat - V*)**2)
```

The loss is accumulated in FP32. Collated clean-action tails and projected suffix embeddings are zeroed. Raw noisy
actions and flow targets may still contain noise at padded positions. Padded suffix rows may be computed internally as
queries, but they are never attention keys and cannot influence a valid query; their final decoder hidden states are
zeroed, and they are excluded from the loss and execution. Inference starts from Gaussian noise and uses uniform-step
Euler integration from `t=0` to `t=1`; the default is 10 function evaluations. Only the final normalized chunk is
clipped to `[-1,1]`.

The continuous action input at horizon position `j` is the sum of:

```text
Linear(A_t[j]) + timestep_mlp(sinusoidal_256(t))
               + state_mlp(state) + horizon_embedding[j] + action_type_embedding
```

DiffusionGemma processes the two images and language with its native frozen multimodal prefix prefill, which produces a
layer-wise KV cache rather than a conventional encoder output for cross-attention. Action positions are a fully
bidirectional suffix that can attend to valid prefix keys and every valid action position. The frozen prefix cache is
computed once per training microbatch and once per new rollout observation; rollout reuses it across Euler steps.

Trainable state is limited to:

- decoder action-suffix self-attention LoRA with rank 16, alpha 32, and dropout 0;
- all existing `q_proj`, `k_proj`, `v_proj`, and `o_proj` targets (115 projections in the pinned revision);
- action, state, and timestep projections, horizon and action-type embeddings, and the velocity output head.

The vision tower, prefix/text computation, token embeddings, vocabulary head, and every pretrained DiffusionGemma base
tensor are frozen. The backbone and forward activations use BF16; LoRA, the continuous interface, optimizer state, loss,
metrics, and normalization statistics remain FP32.

The direct-regression control is selected with a `*_direct*.toml` config. It keeps the same data, masks, frozen
backbone, LoRA topology, and action-interface inventory. It supplies an all-zero action canvas with `t=1`, predicts the
clean normalized chunk `A_1`, and performs one decoder forward. `Wa` remains present for checkpoint-shape parity, but
its weight is inactive because its input is zero; its bias remains trainable.

## Canonical optimization recipe

| Field | Value |
| --- | --- |
| Optimizer | AdamW |
| LoRA / interface peak learning rate | `1e-4` / `1e-3` |
| Betas / epsilon / weight decay | `(0.9, 0.95)` / `1e-8` / `1e-10` |
| Physical batch / accumulation / global batch | `8` / `8` / `64` |
| Gradient clipping | TP-aware global norm `1.0` |
| Schedule | 1,000-update linear warm-up, then cosine decay to `0.1x` peak |
| Default budget | 30,000 optimizer updates |
| Validation / checkpoint interval | 1,000 updates |
| Validation samples | `2,048` |
| EMA / image augmentation | disabled / none |
| Training seeds | `0`, `1`, `2` |

Tensor parallel size 2 shards one model and does not multiply the sample count. The fixed physical batch is part of the
model's grouped-MoE execution contract, so production training fails closed if batch or accumulation settings change
it. The resolved config, source tree, data/model identities, normalization and prefix artifacts, optimizer
parameter inventory, RNG state, and checkpoint lineage are authenticated for resume and serving.

## Data and normalization

LIBERO uses the pinned `HuggingFaceVLA/libero` revision in `configs/libero.toml`. It holds out complete demonstrations
within each task using split seed 1729, then fits 1st/99th-percentile statistics on training episodes only. All eight
state dimensions and the first six OSC_POSE action channels use the percentile map to `[-1,1]`; the binary gripper
channel stays in `{-1,+1}`. Sampling is uniform over task, then episode, then anchor frame.

CALVIN trains only on `task_ABC_D/training` scenes A/B/C. It holds out complete play episodes within each scene using
split seed 1729 and requires every annotated task on both sides. The first seven state values use training-only
1st/99th-percentile statistics and the previous gripper state stays binary. The first six `rel_actions` are already the
official scaled and clipped controller representation, so they use an identity transform; applying another percentile
transform is incorrect. Sampling is uniform over task, then annotation, then timestep. Scene D trajectories never
enter training, normalization, tuning, or checkpoint selection.

Both benchmarks require a separately generated prefix-geometry artifact. It authenticates the exact model/processor,
ordered cameras, complete instruction inventory, and fixed sentinel-padded prefix width. Training compares the
artifact's semantic hash and width with the pins in its benchmark config and fails closed on a mismatch.

## Environment and model

From the repository root, choose cache roots on a filesystem large enough for the model and datasets:

```bash
export DUO_VLA_CACHE_ROOT=/root/.cache/duo-vla
export HF_HOME=/root/.cache/huggingface

./scripts/bootstrap_train_env.sh
./scripts/download_model.sh
```

The model download is pinned to DiffusionGemma revision
`f7f5b7f5fa82ffc52addd066915886d497f5517b`. Access to the upstream gated repository may require authenticating the
Hugging Face CLI first. Training and evaluation use separate environments; follow
[environment.md](environment.md), [libero_simulator_setup.md](libero_simulator_setup.md), and
[calvin_bootstrap.md](calvin_bootstrap.md) for their pinned setup and preflight commands.

The examples below define reusable paths:

```bash
TRAIN_PY="$DUO_VLA_CACHE_ROOT/venvs/train/bin/python"
MODEL_REV=f7f5b7f5fa82ffc52addd066915886d497f5517b
MODEL_SNAPSHOT="$HF_HOME/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/$MODEL_REV"
mkdir -p "$DUO_VLA_CACHE_ROOT/contracts/normalization"
mkdir -p "$DUO_VLA_CACHE_ROOT/contracts/prefix-geometry"
mkdir -p "$DUO_VLA_CACHE_ROOT/runs"
```

Use new artifact and run paths for each invocation. Prefix-geometry publication is exclusive, and the training
launchers will not silently replace an incompatible existing run.

## LIBERO artifacts and training

Download the exact dataset snapshot and bootstrap the simulator source used to authenticate the instruction inventory:

```bash
./scripts/download_libero.sh
./scripts/bootstrap_libero_env.sh

LIBERO_REV=86958911c0f959db2bbbdb107eb3e17c5f9c798e
LIBERO_SNAPSHOT="$HF_HOME/hub/datasets--HuggingFaceVLA--libero/snapshots/$LIBERO_REV"
LIBERO_NORMALIZATION="$DUO_VLA_CACHE_ROOT/contracts/normalization/libero-v1.json"
LIBERO_PREFIX="$DUO_VLA_CACHE_ROOT/contracts/prefix-geometry/libero-v2.json"

PYTHONPATH=src "$TRAIN_PY" -m duo_vla.data.libero_stats \
  "$LIBERO_SNAPSHOT" "$LIBERO_NORMALIZATION" \
  --split-seed 1729 --validation-fraction 0.1

PYTHONPATH=src "$TRAIN_PY" scripts/create_libero_prefix_geometry.py \
  --dataset-snapshot "$LIBERO_SNAPSHOT" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --libero-source "$DUO_VLA_CACHE_ROOT/simulators/libero/source" \
  --output "$LIBERO_PREFIX"
```

Launch the full rectified-flow recipe for seed 0:

```bash
./scripts/run_libero_train.sh \
  "$LIBERO_SNAPSHOT" \
  "$LIBERO_NORMALIZATION" \
  "$DUO_VLA_CACHE_ROOT/runs/libero-flow-seed-0" \
  --config configs/libero.toml \
  --prefix-geometry-artifact "$LIBERO_PREFIX" \
  --seed 0
```

For the matched direct baseline, use a new output directory and replace the config with
`configs/libero_direct_regression.toml`. A single-task run may additionally pass the exact canonical instruction through
`--task`; it is a development run, not the 40-task benchmark checkpoint.

## CALVIN artifacts and training

Prepare the pinned simulator and archive-direct ABC→D dataset. The dataset archive is very large; read
[calvin_bootstrap.md](calvin_bootstrap.md) before starting the download.

```bash
./scripts/calvin/checkout.sh
./scripts/calvin/bootstrap_env.sh
./scripts/calvin/download_dataset.sh archive-direct

CALVIN_TRAINING_ROOT="$DUO_VLA_CACHE_ROOT/data/calvin/task_ABC_D/training"
CALVIN_NORMALIZATION="$DUO_VLA_CACHE_ROOT/contracts/normalization/calvin-abc-to-d-v4.json"
CALVIN_PREFIX="$DUO_VLA_CACHE_ROOT/contracts/prefix-geometry/calvin-abc-to-d-v1.json"

PYTHONPATH=src "$TRAIN_PY" -m duo_vla.data.calvin_stats \
  "$CALVIN_TRAINING_ROOT" "$CALVIN_NORMALIZATION" \
  --split-seed 1729 --validation-fraction 0.1 \
  --archive-sha256 c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74 \
  --max-cached-frames 1

PYTHONPATH=src "$TRAIN_PY" scripts/create_calvin_prefix_geometry.py \
  --training-root "$CALVIN_TRAINING_ROOT" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --calvin-source "$DUO_VLA_CACHE_ROOT/simulators/calvin" \
  --output "$CALVIN_PREFIX"
```

Launch the full rectified-flow recipe for seed 0:

```bash
./scripts/run_calvin_train.sh \
  "$CALVIN_TRAINING_ROOT" \
  "$CALVIN_NORMALIZATION" \
  "$DUO_VLA_CACHE_ROOT/runs/calvin-flow-seed-0" \
  --config configs/calvin_abc_to_d.toml \
  --prefix-geometry-artifact "$CALVIN_PREFIX" \
  --seed 0
```

The exact reduced 500-update CALVIN flow pilot uses the same recipe and artifacts, changing only the declared update
budget:

```bash
./scripts/run_calvin_train.sh \
  "$CALVIN_TRAINING_ROOT" \
  "$CALVIN_NORMALIZATION" \
  "$DUO_VLA_CACHE_ROOT/runs/calvin-flow-500-pilot-seed-0" \
  --config configs/calvin_abc_to_d.toml \
  --prefix-geometry-artifact "$CALVIN_PREFIX" \
  --seed 0 \
  --total-updates 500
```

Do not pass a separate warm-up override for this pilot. When `--total-updates 500` is supplied, the trainer derives
`warmup_updates=min(1000, total_updates // 10)=50`, while retaining the fixed batch, optimizer, learning rates, data
split, `validation_samples=2048`, validation rule, and all artifact checks. The configured final update is validated and
checkpointed even though the normal interval is 1,000. For a matched direct pilot, use a new output directory and
`--config configs/calvin_abc_to_d_direct.toml`; no other recipe field changes.

A planned segmented invocation can stop cleanly at an authenticated boundary with `--stop-after-updates N`. That clean
stop transactionally commits a resumable checkpoint, but it does not change the configured total-update budget, mark
the run complete, or introduce an off-schedule validation pass. Resume the same run contract and output directory with
`--resume checkpoints/update-NNNNNN`; changing the objective, update budget, seed, normalization, prefix artifact, or
source identity is rejected. An unplanned interruption does not imply that a new checkpoint exists: resume is possible
only from the latest transactionally committed checkpoint.

The 500-update pilot may be evaluated only as held-out A/B/C development evidence using
[calvin_development_evaluation.md](calvin_development_evaluation.md). It must not be reported as an official ABC→D
result. Official D evaluation requires completed, frozen checkpoints and the pre-registration workflow in
[calvin_protocol.md](calvin_protocol.md).

### Recorded 500-update CALVIN development pilot

The completed seed-0 pilot used the command above without additional optimization overrides: rectified-flow objective,
physical batch 8, eight accumulation steps, global batch 64, AdamW, LoRA/interface peak learning rates `1e-4`/`1e-3`,
50-update derived warm-up, cosine decay to `0.1x`, and 2,048 fixed validation samples. It completed exactly 500
optimizer updates and 32,000 training chunks; update 500 was validated and transactionally committed with
`complete=true`.

| Item | Measured value |
| --- | ---: |
| update-1 / update-500 train loss | `1.1704777883675022` / `0.3088843029670865` |
| update-500 validation loss | `0.3035793172255102` |
| update-500 checkpoint manifest file SHA-256 | `7e6a7c2c0f759ec2dbcb2d5854ab7258414a0eb60dc41d21c97200f0b40c94aa` |
| metric rows / training chunks | `500` / `32,000` |
| held-out A/B/C rollout, `K=4`, `NFE=10` | `2/12` |
| scene A / B / C successes | `1/4`, `0/4`, `1/4` |

The rollout used evaluation seed `20260829` and the preselected 12-reset smoke view of an authenticated 100-record,
34-task A/B/C development bank. Its real-policy summary has `plumbing_only=false` and `gate_passed=true`; the summary
file SHA-256 is `f922cc78f7e9032f719735c690e44d17a8bcc569425707d40b68590c7539198d`. It measures independent
annotated-subtask success, not the official sequential `SR_1` through `SR_5` or `AvgLen` metrics. Environment D was not
used for training, tuning, checkpoint selection, or any learned-policy rollout.
