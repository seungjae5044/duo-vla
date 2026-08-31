# LIBERO policy bridge and evaluator

The evaluator keeps incompatible dependencies in separate processes. `hf-libero`, robosuite, MuJoCo, and EGL run in
the pinned `libero-eval` environment. DiffusionGemma, the checkpoint LoRA, the action interface, and normalization run
once in the pinned `train` environment. The processes exchange lossless RGB/state requests and copied float32 action
chunks over a private Unix socket.

The implementation follows `duovla-libero-v1`: two cameras in `[agentview, wrist]` order, one 180-degree evaluation
rotation, the 8D state, an 8x7 action chunk, and independent `K=1` or `K=4` runs. IPC schema v4 binds every connection
to a reported objective, sampler, actual function-evaluation count (NFE), inference-seed behavior, exact fixed-B8
execution geometry, and the SHA-256 of the strict deterministic serving runtime. Fake health carries null checkpoint,
model, normalization, execution-geometry, and runtime identities. Every evaluator
invocation must provide a separately predeclared `--evaluation-seed`. Inference seeds derive only from that value,
suite, task, reset source/identity, and replan index; training seed, checkpoint, objective, NFE, and K do not enter the
hash. This supplies common random numbers across comparison cells. A policy timeout, identity/seed mismatch, contract
drift, v3 request, or malformed response fails the rollout; it never inserts a no-op in place of a prediction.

## Exact preflight

Run the policy-side preflight before loading model weights. It identifies the CUDA/GPU runtime but does not construct
DiffusionGemma:

```bash
./scripts/run_libero_policy_server.sh \
  /path/to/checkpoint \
  --preflight-only
```

The gate verifies the train lockfile SHA-256, exact package versions, all checkpoint artifact sizes and hashes, the
pinned model/dataset revisions, normalization content SHA-256, the complete local model/processor snapshot, and the
saved resolved-config hash. It authenticates the checkpoint-copied LIBERO prefix artifact against the config-pinned
semantic SHA-256 and `P=545`, and requires the manifest to repeat B=8, grouped-MM, and sample-isolation identities. The
extra sentinel position keeps every request on the explicit SDPA mask path. It
derives the policy objective only from that verified config and requires the checkpoint manifest to repeat the same
canonical policy contract and hash. It also rejects checkpoints whose recorded training runtime did not enable strict
PyTorch/cuDNN determinism, disable TF32, and pin the cuBLAS workspace. Legacy B=1/B=32 or non-isolated checkpoints are intentionally rejected;
`--train-seed` cannot be used to guess their objective or sampler.

The serving-runtime digest binds the canonical process environment, Python/PyTorch/CUDA/cuDNN/NCCL and GPU
identities, package/lock and launcher/bridge/server hashes, the live source-tree hash checked against the checkpoint,
the training execution-environment hash, the explicit cuBLAS/default-linalg backend selections, and the exact
grouped-MM/sample-isolated B=8, P=545 prefix geometry. Launchers scrub inherited CUBLAS/CUDA/CUDNN/NCCL/PYTORCH/TORCH
algorithm overrides; the runtime rejects any noncanonical variable that survives process construction.

The evaluator independently authenticates the simulator before contacting the server. Development dry-runs use
`--imports-only`; rollout development and official score mode run the full EGL construction/reset smoke. The v2
attestation binds the current evaluator lock, clean Git source/tree, module origins, installed distribution RECORD
files, complete asset tree, evaluator/bridge/preflight/launcher sources, all 40 BDDL/language identities, and the raw
bytes of all 2,000 published initial states. Official pre-registration binds the canonical SHA-256 of this full report.
The evaluator, preflight, and pre-registration launchers execute Python through `env -i`: nothing is inherited except
the explicitly selected cache-root value, and the child receives an exact closed allowlist containing the pinned venv
`PATH`, repository-only `PYTHONPATH`, locale, EGL selection, thread counts, and Hugging Face cache. This also removes
unregistered `LIBGL`/Mesa/EGL/GBM/NVIDIA rendering overrides rather than trying to enumerate them. Direct execution
with any extra variable fails.

## Start one persistent real policy

From the repository root, start the TP=2 server. It loads DiffusionGemma and the checkpoint only once and then serves
sequential rollout requests until explicitly stopped:

```bash
./scripts/run_libero_policy_server.sh \
  /path/to/checkpoint \
  --socket /root/.cache/duo-vla/run/libero-policy.sock
```

The launcher uses `torchrun --standalone --nproc-per-node=2`; both ranks participate in every prefix encoding and
every action-suffix forward. One request is replicated into a fixed physical B=8, and the server rejects the result
unless all eight normalized action rows are bitwise identical. A direct checkpoint makes one forward. A flow
checkpoint defaults to its authenticated
NFE and may be evaluated at another committed ablation point with `--flow-steps 1`, `5`, or `10`; direct checkpoints
reject this option. Rank 0 owns a mode-`0600` Unix socket. If the path exists as a live socket the launcher refuses to
replace it; it removes only a verified stale socket. Images and state never leave the host.

Before constructing MuJoCo, make one synthetic request through the real bridge:

```bash
./scripts/run_libero_eval.sh \
  --socket /root/.cache/duo-vla/run/libero-policy.sock \
  --evaluation-seed 123 \
  --execution-horizon 1 \
  --dry-run
```

## Sealed canonical evaluation only

Development rollout mode now rejects published resets and requires an authenticated `--reset-source clean-dev` bank.
Official states are reachable only through `--mode official-score`, which requires an externally frozen manifest,
its independently recorded raw content SHA-256, a matching final-freeze token, and one canonical cell ID.

First create the full simulator attestation without executing policy inference:

```bash
./scripts/run_libero_preflight.sh \
  --output-json /read-only/freeze/libero-simulator-attestation.json
```

Prepare a strict `{"cells": [...]}` document containing exactly 24 cells: three training seeds, flow NFE 1/5/10 plus
direct NFE 1, and K 1/4. Every cell records the final update-30,000 checkpoint manifest/source SHA, checkpoint policy
contract SHA, actual selected serving-policy SHA, deterministic serving-runtime SHA, and fixed B8/P545 geometry. NFE
and K cells for one seed/objective must reuse the same final checkpoint. Then create the sealed manifest:

```json
{
  "cells": [{
    "cell_id": "seed-0-flow-nfe-10-k-4",
    "checkpoint": {
      "manifest_sha256": "<64hex>",
      "source_tree_sha256": "<64hex>",
      "update": 30000
    },
    "execution_geometry": {
      "experts_implementation": "grouped_mm",
      "expert_batch_isolation": "sample_isolated_grouped_mm_v1",
      "physical_batch_size": 8,
      "fixed_physical_prefix_width": 545,
      "prefix_geometry_content_sha256": "cc907e22ccd5ae704767edba606233dede39989a119ac544764b47aaa4fbe634"
    },
    "execution_horizon": 4,
    "inference_seed_behavior": "episode_identity_gaussian_noise",
    "nfe": 10,
    "objective": "rectified_flow",
    "policy_contract_sha256": "<checkpoint policy contract 64hex>",
    "policy_warmup_calls": 2,
    "sampler": "euler_uniform",
    "serving_runtime_sha256": "<policy preflight 64hex>",
    "train_seed": 0
  }]
}
```

The shown list must be expanded to the exact 24 cells; the creator rejects a template or partial matrix. It derives
and inserts the common `episode_matrix_sha256` and each actual selected `serving_policy_sha256` from the authenticated
episode inventory and the explicit wire-contract fields.

```bash
./scripts/run_create_libero_preregistration.sh \
  --cells /read-only/freeze/libero-cells.json \
  --simulator-attestation /read-only/freeze/libero-simulator-attestation.json \
  --evaluation-seed 123 \
  --final-freeze-token '<externally-held-token>' \
  --output /read-only/freeze/libero-preregistration.json
```

The generated raw manifest SHA-256 and the token must be stored outside the evaluation output. Computing the expected
SHA from the candidate manifest at score time defeats the seal. One official cell is then invoked as:

```bash
./scripts/run_libero_eval.sh \
  --mode official-score \
  --socket /root/.cache/duo-vla/run/libero-policy.sock \
  --suite all \
  --task-ids all \
  --init-state-ids all \
  --evaluation-seed 123 \
  --execution-horizon 4 \
  --policy-warmup-calls 2 \
  --preregistration-manifest /read-only/freeze/libero-preregistration.json \
  --preregistration-sha256 '<externally-recorded-64hex-digest>' \
  --cell-id seed-0-flow-nfe-10-k-4 \
  --final-freeze-token '<externally-held-token>' \
  --output-dir /root/.cache/duo-vla/evaluations/seed-0-flow-nfe-10-k-4
```

Official mode regenerates the exact suite/task/reset inventory and evaluates 1,999 episodes in published order. It
starts from all 50 resets for all 40 tasks, then always excludes only `libero_goal` task 7/reset 0 according to the
content-pinned `evaluation_contamination.json`. The same exclusion is applied to every objective/NFE/K/training-seed
cell. Goal task 7 therefore has denominator 49; every other task has 50. `summary.json` explicitly records the 1,999
primary denominator, the excluded identity, the 2,000 non-blind full-set size, and that no full-set result was emitted.
There is no CLI path for an official 2,000-episode score.

Before the first prediction, official mode also proves that the selected checkpoint is update 30,000, is the current
authenticated run-journal tip, and matches the pre-registered manifest, source, objective, selected NFE, K, train seed,
policy contract, serving runtime, execution geometry, and simulator attestation.

Every output directory is created exclusively and contains:

- `run.json`: policy/checkpoint identity, exact simulator preflight report, selections, K, and the independent
  evaluation seed/reset identity, plus every discarded synthetic policy warm-up call;
- `episodes.jsonl`: flushed and fsynced after every episode;
- `summary.json`: task success counts/rates and Wilson intervals, suite task-macro success, pooled suite Wilson
  intervals, steps-to-success, policy calls, end-to-end and server p50/p95 latency, and action clip fractions.

`overall_40_task_macro_success` remains null unless all 40 suite/task identities are present. Success is checked after
every action, remaining queued actions are discarded immediately, and the ten open-gripper settling steps do not
consume the policy budget. Prefix and action caches have request/episode scope rather than leaking across resets. IPC
health must report `prefix_cache_scope == "request"`; any other value is rejected before a prediction.
Before the first episode, rollout mode requires at least one synthetic policy warm-up call and defaults to two. These
calls are recorded under `policy_warmup` in `run.json` and are never included in episode p50/p95 latency. Override the
count only as a predeclared factor with `--policy-warmup-calls`; use the same value for every compared policy.

## CPU-only fake bridge check

The fake server is explicitly test-only and is rejected by the evaluator unless `--allow-fake-policy` is supplied:

```bash
./scripts/run_libero_policy_server.sh \
  --fake-policy \
  --train-seed 0 \
  --socket /root/.cache/duo-vla/run/libero-policy-test.sock

./scripts/run_libero_eval.sh \
  --socket /root/.cache/duo-vla/run/libero-policy-test.sock \
  --evaluation-seed 123 \
  --execution-horizon 4 \
  --dry-run \
  --allow-fake-policy
```

The focused tests additionally exercise persistent framing, byte-exact camera transport, seed parity with the core
benchmark contract, objective/sampler/NFE drift rejection, flow-only serving overrides, private socket permissions,
K-dependent queueing, early-success discard, Wilson intervals, and metric aggregation:

```bash
/root/.cache/duo-vla/venvs/train/bin/pytest -q tests/test_libero_bridge.py
```
