# LIBERO policy bridge and evaluator

The evaluator keeps incompatible dependencies in separate processes. `hf-libero`, robosuite, MuJoCo, and EGL run in
the pinned `libero-eval` environment. DiffusionGemma, the checkpoint LoRA, the action interface, and normalization run
once in the pinned `train` environment. The processes exchange lossless RGB/state requests and copied float32 action
chunks over a private Unix socket.

The implementation follows `duovla-libero-v1`: two cameras in `[agentview, wrist]` order, one 180-degree evaluation
rotation, the 8D state, an 8x7 action chunk, and independent `K=1` or `K=4` runs. IPC schema v5 binds every connection
to a reported objective, sampler, actual function-evaluation count (NFE), inference-seed behavior, exact fixed-B8
execution geometry, the SHA-256 of the strict deterministic serving runtime, and a latency-comparability runtime
SHA-256. Fake health carries null checkpoint, model, normalization, execution-geometry, and runtime identities. Every evaluator
invocation must provide a separately predeclared `--evaluation-seed`. Inference seeds derive only from that value,
suite, task, reset source/identity, and replan index; training seed, checkpoint, objective, NFE, and K do not enter the
hash. This supplies common random numbers across comparison cells. A policy timeout, identity/seed mismatch, contract
drift, legacy-schema request, or malformed response fails the rollout; it never inserts a no-op in place of a prediction.

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

The serving-runtime digest binds the closed canonical process environment, the exact train-venv tree,
Python/framework/driver binary identities, runtime-library and device-mapping identities, package/lock and
launcher/bridge/server hashes, the live source-tree hash checked against the checkpoint, the training
execution-environment hash, deterministic backend selections, and the exact grouped-MM/sample-isolated B=8, P=545
prefix geometry. The real-policy launcher uses `env -i`, so caller thread, allocator, locale, library, and Python
injection overrides do not reach the process.
The policy preflight additionally emits `latency_runtime_sha256`. It covers the same serving software, deterministic
settings, execution geometry, accelerator capabilities/identities, and serving environment, but deliberately omits
only the checkpoint's training-execution-environment SHA. That training provenance can differ by seed without changing
the machine performing inference. Every latency-reporting cell must share the one pre-registered latency-runtime hash;
the full per-checkpoint serving-runtime hash remains authenticated independently.

The evaluator independently authenticates the simulator before contacting the server. Development dry-runs use
`--imports-only`; rollout development and official score mode run the full EGL construction/reset smoke. The v3
attestation binds the current evaluator lock, clean Git source/tree, module origins, installed distribution RECORD
files, complete evaluator-venv and resolved base-Python identities, complete asset tree,
evaluator/bridge/preflight/launcher sources, all 40 BDDL/language identities, and the raw bytes of all 2,000 published
initial states. Official pre-registration binds the canonical SHA-256 of this full report.
The evaluator, preflight, and pre-registration launchers execute Python through `env -i`: nothing is inherited except
the explicitly selected cache-root value, and the child receives an exact closed allowlist containing the pinned venv
`PATH`, locale, EGL selection, thread counts, and Hugging Face cache. Python runs with safe-path/no-bytecode flags, an
impossible `/dev/null` bytecode-cache prefix, no `PYTHONPATH`, and an exact import path. This also removes
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
Official states are reachable only through the sealed `official-score` or `official-spatial-score` modes. Both require
an externally frozen manifest, its independently recorded raw content SHA-256, a matching final-freeze token, and one
canonical cell ID. The two modes use different schemas and cannot consume each other's manifests.

### Single-cell official Spatial score

`official-spatial-score` is a deliberately narrow report, separate from the unchanged 24-cell `official-score`
protocol. It accepts exactly one update-30,000 checkpoint/policy cell, one training seed, and one execution horizon.
It evaluates all ten `libero_spatial` tasks at all 50 published official resets, for exactly 500 rows in canonical
task/reset order. It does not load or apply the Goal contamination exclusion because no Goal task is in scope.

The manifest, run journal, and summary use the dedicated schemas
`duo-vla-libero-official-spatial-preregistration-v1`, `duo-vla-libero-official-spatial-run-v1`, and
`duo-vla-libero-official-spatial-summary-v1`. Their reporting scope is fixed to the label
`libero_spatial_official_500_single_checkpoint_single_policy_cell_not_full_g6_not_three_seed_aggregate` and contains
explicit false-valued permissions for full-G6 and three-seed-aggregate claims. This result is therefore only a
single-checkpoint, single-policy-cell Spatial score. It is not a full G6 result, not a three-seed aggregate, and is not
accepted by the 24-cell official aggregator. A 5,000-update development checkpoint is rejected.

Prepare the normal strict cell document, but include exactly one fully selected final cell. Create separate empty run
and claim roots, then seal the Spatial manifest:

```bash
mkdir /sealed/libero-spatial-runs /sealed/libero-spatial-claims
./scripts/run_create_libero_preregistration.sh \
  --mode official-spatial-score \
  --cells /read-only/freeze/libero-spatial-cell.json \
  --simulator-attestation /read-only/freeze/libero-simulator-attestation.json \
  --expert-replay-qualification /read-only/freeze/libero-expert-replay-qualification/qualification.json \
  --expert-replay-qualification-sha256 '<externally-recorded-report-64hex>' \
  --evaluation-seed 123 \
  --final-freeze-token '<externally-held-token>' \
  --official-output-root /sealed/libero-spatial-runs \
  --official-claim-root /sealed/libero-spatial-claims \
  --output /read-only/freeze/libero-spatial-preregistration.json
```

Run the one pre-registered cell with the same seed, horizon, cell ID, and canonical derived output path:

```bash
./scripts/run_libero_eval.sh \
  --mode official-spatial-score \
  --socket /root/.cache/duo-vla/run/libero-policy.sock \
  --suite libero_spatial \
  --task-ids all \
  --init-state-ids all \
  --evaluation-seed 123 \
  --execution-horizon 4 \
  --policy-warmup-calls 2 \
  --preregistration-manifest /read-only/freeze/libero-spatial-preregistration.json \
  --preregistration-sha256 '<externally-recorded-64hex-digest>' \
  --cell-id seed-0-flow-nfe-10-k-4 \
  --final-freeze-token '<externally-held-token>' \
  --output-dir /sealed/libero-spatial-runs/seed-0-flow-nfe-10-k-4
```

The exclusive claim-before-simulator behavior, environment/attestation checks, policy-health binding, warm-up policy,
and crash-safe journal are the same strict mechanisms used by `official-score`; only the sealed scope and its result
labels differ.

First create the full simulator attestation without executing policy inference:

```bash
./scripts/run_libero_preflight.sh \
  --output-json /read-only/freeze/libero-simulator-attestation.json
```

Next produce and validate the regenerated 40-task expert replay evidence as specified in
[libero_expert_replay_qualification.md](libero_expert_replay_qualification.md). The qualification output and its raw
SHA-256 are mandatory freeze inputs. A pre-registration cannot be created from a simulator attestation alone.

Prepare a strict `{"cells": [...]}` document containing exactly 24 cells: three training seeds, flow NFE 1/5/10 plus
direct NFE 1, and K 1/4. Every cell records the final update-30,000 checkpoint manifest/source SHA, checkpoint policy
contract SHA, actual selected serving-policy SHA, full deterministic serving-runtime SHA, latency-runtime SHA, and
fixed B8/P545 geometry. NFE and K cells for one seed/objective must reuse the same final checkpoint; the latency-runtime
SHA must be identical across all seeds, objectives, NFEs, and K values. Then create the sealed manifest:

```json
{
  "cells": [{
    "cell_id": "seed-0-flow-nfe-10-k-4",
    "checkpoint": {
      "dataset_content_inventory_sha256": "63fd7a951ebb397a33c43cad4a7c48c7c6911bd8d1481ff99b07da5f7890782c",
      "dataset_tree_sha256": "d9c14b4aff28bcc56f341b171c6a5a3b10510d4bd0378662891c5156d245add8",
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
    "latency_runtime_sha256": "<policy preflight latency runtime 64hex>",
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
and inserts the common `episode_matrix_sha256`, each actual selected `serving_policy_sha256`, and each cell's sole
canonical run/claim paths. The two pre-existing empty roots are recorded by absolute path plus device/inode; they must
be distinct real directories that do not contain one another.

```bash
mkdir /sealed/libero-official-runs /sealed/libero-official-claims
./scripts/run_create_libero_preregistration.sh \
  --cells /read-only/freeze/libero-cells.json \
  --simulator-attestation /read-only/freeze/libero-simulator-attestation.json \
  --expert-replay-qualification /read-only/freeze/libero-expert-replay-qualification/qualification.json \
  --expert-replay-qualification-sha256 '<externally-recorded-report-64hex>' \
  --evaluation-seed 123 \
  --final-freeze-token '<externally-held-token>' \
  --official-output-root /sealed/libero-official-runs \
  --official-claim-root /sealed/libero-official-claims \
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
  --output-dir /sealed/libero-official-runs/seed-0-flow-nfe-10-k-4
```

`--output-dir` is not a selectable destination: it must exactly name the path derived in the frozen cell. Before full
simulator construction or the first policy connection, the evaluator durably publishes the exclusive external claim
`/sealed/libero-official-claims/<cell-id>.json` and its SHA-256 sidecar. It then creates the canonical run directory
and running journal. A collision rejects the attempt, and every later setup or rollout failure leaves either the
immutable claim alone or the claim plus a durable failed journal.

Official mode regenerates the exact suite/task/reset inventory and evaluates 1,999 episodes in published order. It
starts from all 50 resets for all 40 tasks, then always excludes only `libero_goal` task 7/reset 0 according to the
content-pinned `evaluation_contamination.json`. The same exclusion is applied to every objective/NFE/K/training-seed
cell. Goal task 7 therefore has denominator 49; every other task has 50. `summary.json` explicitly records the 1,999
primary denominator, the excluded identity, the 2,000 non-blind full-set size, and that no full-set result was emitted.
There is no CLI path for an official 2,000-episode score.

Before the first prediction, official mode also proves that the selected checkpoint is update 30,000, is the current
authenticated run-journal tip, and matches the pre-registered manifest, source, objective, selected NFE, K, train seed,
policy contract, serving runtime, execution geometry, and simulator attestation. The qualification validator-runtime
object is preserved in the pre-registration and run journal and must exactly match the process/evaluator-venv identity
bound to the simulator attestation; aggregation repeats the same equality checks.

Every output directory is created exclusively and contains:

- `run.json`: policy/checkpoint identity, exact simulator preflight report, selections, K, and the independent
  evaluation seed/reset identity, plus every discarded synthetic policy warm-up call;
- `episodes.jsonl`: flushed and fsynced after every episode;
- `summary.json`: task success counts/rates and Wilson intervals, suite task-macro success, pooled suite Wilson
  intervals, steps-to-success, policy calls, episode-only end-to-end and server p50/p95 latency, episode throughput,
  policy-call throughput, and action clip fractions.
- `completion.json` and `completion.json.sha256`: the final commit marker binding the cell, pre-registration, external
  claim, exact 1,999-row episode stream, summary, and final complete `run.json`. Without this pair the attempt is not a
  completed official result.

## Authenticate and aggregate the complete matrix

The pre-registration also seals the raw SHA-256 of `aggregate_libero_official.py` and its exact Python version. After
all 24 cells finish, create one strict inventory containing every pre-registered cell exactly once. Record each digest
from the raw bytes independently; do not calculate an expected digest from a file after deciding to accept it:

```json
{
  "preregistration_sha256": "<externally-recorded-preregistration-64hex>",
  "runs": [{
    "cell_id": "seed-0-flow-nfe-10-k-4",
    "claim_json_sha256": "<raw-claim-64hex>",
    "completion_json_sha256": "<raw-completion-64hex>",
    "episodes_jsonl_sha256": "<raw-episodes-64hex>",
    "run_json_sha256": "<raw-run-64hex>",
    "summary_json_sha256": "<raw-summary-64hex>"
  }],
  "schema": "duo-vla-libero-official-run-inventory-v2"
}
```

The inventory deliberately contains no path field: the aggregator derives every run and claim path only from the
frozen pre-registration. The list must contain all 24 unique cells. At aggregation time both roots must have exact
inventories—24 canonical real run directories and only the 24 claim/sidecar pairs—with no alternate, omitted,
symlinked, hard-linked, or extra attempt artifacts. Seal the inventory's raw bytes outside the run directories, then
invoke:

```bash
./scripts/run_aggregate_libero_official.sh \
  --preregistration-manifest /read-only/freeze/libero-preregistration.json \
  --preregistration-sha256 '<externally-recorded-preregistration-64hex>' \
  --run-inventory /read-only/freeze/libero-run-inventory.json \
  --run-inventory-sha256 '<externally-recorded-inventory-64hex>' \
  --output /read-only/results/libero-official-matrix.json
```

The verifier rejects missing, duplicated, or off-matrix cells; authenticates every raw run, episode stream, and
summary; binds checkpoint, source-tree, policy-contract, runtime, simulator, cell, freeze-token, and pre-registration
identities; and validates every episode against the frozen 1,999-row order and denominator. It then recomputes each
cell summary from `episodes.jsonl`. The final report contains all per-cell summaries and eight separate comparisons:
flow NFE 1/5/10 and direct NFE 1, each at K=1 and K=4. Every comparison reports the three seed values, their mean, and
sample standard deviation for success, episode-only client/server p50/p95, and throughput. All three seeds must share
the pre-registered latency-runtime identity. K values are never pooled. Publication is exclusive and crash-safe; the JSON commit marker
is accompanied by `libero-official-matrix.json.sha256` and neither target is overwritten.

`overall_40_task_macro_success` remains null unless all 40 suite/task identities are present. Success is checked after
every action, remaining queued actions are discarded immediately, and the ten open-gripper settling steps do not
consume the policy budget. Prefix and action caches have request/episode scope rather than leaking across resets. IPC
health must report `prefix_cache_scope == "request"`; any other value is rejected before a prediction.
Before the first episode, rollout mode requires at least one synthetic policy warm-up call and defaults to two. These
calls are recorded under `policy_warmup` in `run.json` and are never included in episode p50/p95 latency. Override the
count only for development. Official pre-registration fixes exactly two calls, reports that count in every summary and
the final matrix, requires the repeated outputs to be deterministic, and requires identical output/seed identities
between K=1 and K=4 for the same seed/objective/NFE. Both repeated calls use reserved `replan_id=520`; their
K-independent response identities are validated and journaled immediately before scoring begins. Each persisted
client round-trip latency must be at least its paired server latency. Throughput is computed only from the sum of
persisted episode elapsed times: episodes/hour and policy calls/second; simulator construction, warm-ups, and
aggregation time are excluded.

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
