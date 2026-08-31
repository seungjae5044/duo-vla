# CALVIN protocol: `duovla-calvin-abc-to-d-v1`

## Split and input contract

Train only on `task_ABC_D/training` play data from environments A, B, and C. Evaluate zero-shot in environment D using
the simulator configuration under `task_ABC_D/validation`. D trajectory files must not affect training, normalization,
checkpoint selection, or hyperparameters. Create an internal validation split by holding out whole A/B/C episodes.

The official ABC→D ZIP is 555,309,812,705 bytes and must be verified against its published SHA-256. Production uses the
v4 archive-direct layout: the retained ZIP, an exact-schema v2 SQLite member index, and only six projected metadata
files. The transactional preparer streams every central/local header and member payload, publishes a manifest with
`storage.mode=archive-direct` and reader v1, and never extracts episode NPZ files. Normalization and training
authenticate the full retained archive, recomputed member inventory, member index, current critical metadata, and each
selected frame before unpickling or reading samples. The older extracted v3 layout remains parity-only and cannot be
promoted into an official runtime/data attestation. Record the
CALVIN parent, `calvin_env`, and TACTO submodule commits. References:
[CALVIN paper](https://arxiv.org/abs/2112.03227),
[dataset README](https://github.com/mees/calvin/blob/main/dataset/README.md).

Training and rollout schemas differ:

| Meaning | training NPZ | rollout observation |
| --- | --- | --- |
| static RGB | `rgb_static` | `rgb_obs.rgb_static` |
| wrist RGB | `rgb_gripper` | `rgb_obs.rgb_gripper` |
| robot state | `robot_obs` | `robot_obs` |

Use raw uint8 HWC arrays. Current pinned data/config uses 200×200 static and 84×84 wrist images. The model state is the
official `robot_no_joints` selection `robot_obs[0:7] + robot_obs[14:15]`: TCP position (3), TCP Euler orientation (3),
gripper width (1), and previous binary gripper state (1). Normalize the first seven values with A/B/C statistics and
retain the last value as exact `{-1,+1}`. The 24D `scene_obs` is exclusively for reset/task oracle and must never enter a
model request. Source: [official proprio config](https://github.com/mees/calvin/blob/main/calvin_models/conf/datamodule/proprioception_dims/robot_no_joints.yaml).

The official debug NPZs store `robot_obs`, `scene_obs`, and `rel_actions` as float64 despite older prose describing
float32 arrays. The reader accepts only finite floating source arrays, then casts model state and action tensors to
float32 at the collation boundary. It must not reject an authenticated archive solely because these source arrays are
float64, and it must never forward `scene_obs` while doing that cast.

## Action semantics and training chunks

Target `rel_actions[7]` is already the environment's scaled command:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

The first three values are position delta multiplied by 50; rotation values are Euler delta multiplied by 20; both are
clipped to `[-1,1]`. The environment applies 0.02 and 0.05 on receipt. Do not convert the model output to meters/radians
before `env.step`. In the pinned implementation deltas accumulate on the controller's world-axis target pose
(`use_target_pose=True`), despite older prose describing a gripper-frame action. Gripper is `-1` close and `+1` open;
the pinned official wrapper maps an exact zero prediction to `-1` with a strict `> 0` threshold. This is the opposite
semantic polarity from LIBERO.

The simulator mutates the first six array elements in place while scaling. The adapter must return a fresh owned
`float32[7]` copy; passing a row view from an action queue corrupts the stored chunk. Source:
[pinned robot implementation](https://github.com/mees/calvin_env/blob/1431a46bd36bde5903fb6345e68b5ccc30def666/calvin_env/robot/robot.py).

Use raw instruction strings and half-open `[start,end)` intervals from
`training/lang_annotations/auto_lang_ann.npy`; ignore its precomputed language embeddings. This differs from
`ep_start_end_ids.npy`, whose episode end is inclusive. At anchor `i`, pair the images/state/language at `i` with
`rel_actions[i:i+8]`. Stop at both annotation and episode end. The raw chunk helper pads continuous values with zero
and repeats the final gripper command, preserving a valid action-shaped record. Collation then zeroes every channel at
padded positions; attention and loss mask those positions completely. Statistics count each A/B/C timestep once, not
once per overlapping annotation. Start-up asserts that `scene_info.npy` contains only A/B/C and the evaluation merged
config instantiates D. Sources: [official loader](https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/datasets/disk_dataset.py),
[annotation visualizer](https://github.com/mees/calvin/blob/main/scripts/visualize_dataset.py).

CALVIN production training uses a fixed physical batch of eight and eight gradient-accumulation steps for global
batch 64. The configuration must explicitly pin `grouped_mm`, `sample_isolated_grouped_mm_v1`, B=8, the fixed prefix
width, and the semantic SHA-256 of a canonical prefix-geometry artifact. That artifact must cover the authenticated
A/B/C training instructions and every official D evaluation phrase, with ordered raw cameras `rgb_static` then
`rgb_gripper`. The generator sets P to the measured maximum valid prefix length plus one; the mandatory padding
sentinel keeps replicated serving and mixed training batches on the same explicit-mask SDPA path. Training requires
`--prefix-geometry-artifact`; it copies the artifact into every checkpoint and binds
it to resolved config, resume rank state, manifest, serving preflight, runtime identity, health, and official
pre-registration. No provisional CALVIN width or hash is accepted before the full instruction inventory is measured.

## Official long-horizon evaluation

Generate the official 1,000 deterministic sequences with seed 0. Each sequence starts from one fixed neutral robot
state and deterministic scene state, then attempts five feasible subtasks without resetting the environment. Each
subtask has at most 360 environment actions; test the official task oracle after every step, advance immediately on
success, and terminate the whole sequence at the first failed subtask.

The current official evaluator calls `model.reset()` at each subtask even though the environment persists. Duo-VLA
therefore clears its action queue, prefix cache, and model-side episode state at each subtask and regenerates from the
current observation/new language. Discard queued actions immediately on success. Evaluation instructions are the first
fixed phrase for each task in `new_playtable_validation.yaml`, not training annotations.

At 30 Hz, `H=8` spans 0.267 simulated seconds; `K=1` replans at 30 Hz and `K=4` at 7.5 Hz. Headless evaluation has no
real-time deadline, so separately report wall-clock policy latency/throughput. Derive flow RNG from the frozen domain,
evaluation seed 0, canonical sequence-list SHA-256, sequence index, subtask index/name, and replan index. Training seed,
checkpoint, objective, NFE, and K are deliberately absent, providing common random numbers across comparison cells and
making evaluator sharding irrelevant.

Sources: [official evaluator](https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py),
[sequence generator](https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/multistep_sequences.py),
[validation language](https://github.com/mees/calvin/blob/main/calvin_models/conf/annotations/new_playtable_validation.yaml).

If `R_i` is the number of consecutive successes in sequence `i`, report all of:

```text
SR_k = mean(R_i >= k), k=1..5
AvgLen = mean(R_i) = SR_1 + ... + SR_5
```

The 1,000 D sequences form one indivisible final test. Do not policy-score a 10/100-sequence subset before the recipe is
frozen. Per-task summaries from attempted tasks are conditional on reaching that task and are secondary, not the
primary score.

## Runtime isolation and gates

The official simulator is a Python 3.8-era stack (including its pinned Hydra/PyTorch ecosystem), while the model uses
modern Transformers/PyTorch. Run it as a separate EGL process and exchange lossless images/state/language and copied
actions over local IPC. Source-only preflight, infrastructure checks, and scoring all enter through the same
`run_official_evaluator.sh` `env -i` Python 3.8.20 contract; its fixed CUDA/EGL ordinal and headless renderer settings
were qualified by the real source-only EGL smoke and have no fallback. This is the sole supported launcher and an
ordinary direct Python invocation fails closed, though it is not an OS-level prohibition on manually reconstructing
the identical closed environment. The repository's evaluator, v8 pre-registration
creator, and official aggregator CLIs implement the frozen protocol contract. The remaining operational work is obtaining sufficient external storage, downloading and
authenticating ABC_D, and executing the real runs; it does not require replacing the supplied CLI contract.

Before freezing the policy, D access is infrastructure-only: source/config import, canonical sequence hashing,
environment construction/reset, schema/camera/frequency checks, hand-written action scaling, and dummy IPC are allowed,
but no learned checkpoint may act on D observations and no reward/oracle/video/trajectory outcome may be inspected.
Policy-scored rollout gates use held-out A/B/C resets. After pre-registering every checkpoint and objective/NFE/K/seed
cell, run the full 1,000 D sequences for each cell without using intermediate scores to alter any remaining run.
The sealed matrix is exactly 24 cells: three training seeds; flow NFE 1/5/10 or direct NFE 1; and K 1/4. Final
aggregation accepts exactly one content-addressed completed run per cell, reauthenticates and recomputes all per-cell
metrics, then reports eight comparison rows as three-seed mean and sample standard deviation with the `n-1` denominator.

The official-score gate is additionally bound to a canonical runtime/data attestation. Before parsing YAML or
constructing Hydra/OmegaConf, the oracle, or the environment, it verifies clean exact-revision CALVIN, `calvin_env`,
and TACTO checkouts; exact module origins and all pinned package versions; Python/platform/runtime identity; fixed raw
hashes for the validation-language and oracle YAML; the dataset v4 archive-direct manifest, exact v2 member-index
schema/hash/metadata and every canonical row, the pinned live central directory/ZIP64 trailer, and current D projected
metadata bytes; and raw hashes of the evaluator, IPC bridge, and preflight sources. The v8 frozen pre-registration
contains this attestation hash, the raw official-aggregator hash, and the exact Python 3.8.20 aggregation runtime; its
own raw file hash is supplied independently on the command line. Aggregation snapshots all four local source files
before importing evaluator code, rejects end-of-run drift, and requires each run-attested source identity to match that
snapshot. Local imports must also match the pre-import path, spec origin, and raw identity. YAML and the merged
validation config are parsed directly from the stable authenticated bytes rather than reopening their paths. The
aggregator reloads the authenticated validation-language YAML and accepts only the exact first phrase for every
recorded subtask instruction. Journal and exclusive payload/sidecar commit guards reopen targets without following
symlinks and bind the created inode, exact bytes, and link count before and after the guard; target-entry substitution
therefore fails the publication, and a completed journal is downgraded to failed. All 24 cells freeze one common
evaluation-only `serving_runtime_sha256` (runtime schema v5) and one exact execution-geometry object, which the evaluator
compares with live policy health before any prediction. The serving runtime binds current software, deterministic
settings, driver, and physical devices; the selected checkpoint separately retains its authenticated training-runtime
identity.
Policy IPC v4 additionally carries the exact 15-field normalization `dataset` identity as nested `calvin_identity`;
official scoring and final aggregation compare it byte-for-byte with the runtime/data attestation and reject any
flat normalization-metadata mismatch.
Any mismatch is a benchmark-blocking failure, not a warning.

Official scoring explicitly selects the pre-registered warm-up count (two by default and at least one). The v8
pre-registration binds dedicated canonical run/claim roots by absolute path and device/inode and derives every cell's
only valid run directory and external claim path. Before constructing a policy client, scoring exclusively creates the
run directory and immutable claim/sidecar. A collision permanently consumes the attempt even if its run directory is
later deleted. The evaluator durably records partial setup and warm-up progress, and
uses reserved replan indices beginning at 360 so warm-up requests cannot collide with scored request identities.
Each request intent is committed before dispatch and its completed report is committed afterward. Synthetic warm-up
outputs are discarded and excluded from episode latency. Aggregation requires their action hashes
to be bitwise identical across the paired K=1/K=4 cells for each seed/objective/NFE and recomputes episode-only latency
and throughput from the raw sequence records.

A successful cell ends only when an exclusive completion JSON/sidecar binds the claim, final run, episodes, summary,
pre-registration, cell, and 1,000-record count. Inventory v3 contains raw hashes but no caller-selected paths.
Aggregation derives all paths from pre-registration, requires exact 24-directory and 48-file root inventories, rejects
running/failed/missing-marker attempts and linked or extra paths, and publishes matrix-summary v5.

Required gates include checksum/split/schema assertions, one-command scaling and gripper polarity, no input mutation,
offline-vs-rollout observation parity, half-open annotation chunking, absence of `scene_obs` from RPC, fixed sequence
hash, dummy-policy smoke, fixed-batch overfit, held-out A/B/C rollout, then the indivisible final D runs.

The held-out checkpoint-development gate has its own reset-bank, simulator, IPC,
and metric contract. See [Held-out CALVIN A/B/C development rollout](calvin_development_evaluation.md).
