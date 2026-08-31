# LIBERO protocol: `duovla-libero-v1`

## Track definition and data provenance

The primary experiment is one multitask checkpoint over all 40 tasks in `libero_spatial`, `libero_object`,
`libero_goal`, and `libero_10`. It is not the original continual-learning track and is not reported as such.

Training uses `HuggingFaceVLA/libero` pinned at commit
`86958911c0f959db2bbbdb107eb3e17c5f9c798e`. The observed repository contains 40 tasks, 1,693 episodes, 273,465
frames, and about 32.5 GiB of files. Startup verifies these values rather than trusting `main`.

The pinned revision has internally inconsistent episode file pointers: 1,690 of 1,693 `data/file_index` values do not
identify the physical parquet containing the episode's declared global range. The reader therefore indexes all 377
physical parquet files by their contiguous `index` spans, requires exact coverage of `[0, 273465)`, resolves each whole
episode by `[dataset_from_index, dataset_to_index)`, and verifies the per-row `episode_index`. It never treats the stale
episode file pointer as authoritative. This validation is part of the dataset gate rather than a silent repair.

The source choice resolves a material alignment problem: the official raw HDF5 writer records observations after
`env.step(action)`, so pairing columns on the same row can produce `(o_(t+1), a_t)`. The OpenVLA regeneration path
records the pre-action observation, drops no-ops, and retains successful trajectories. Raw official HDF5 therefore is
not accepted directly; it must first be replayed/regenerated and pass the alignment gate.

References: [official HDF5 writer](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py),
[OpenVLA regeneration](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py),
[OpenPI conversion](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py).

## Canonical sample

| Duo field | Dataset field | Contract |
| --- | --- | --- |
| third-person RGB | `observation.images.image` | regenerated `agentview`, uint8 256×256 |
| wrist RGB | `observation.images.image2` | regenerated `eye_in_hand`, uint8 256×256 |
| state | `observation.state` | float 8D |
| action | `action` | float 7D OSC_POSE input |
| language | episode task | raw canonical instruction |

State order is EEF position (3), quaternion converted to axis-angle (3), and two continuous gripper joint positions.
The pinned robosuite quaternion convention/converter is used during rollout. State gripper joints are continuous state
and remain part of percentile normalization.

Actions are normalized OSC_POSE controller inputs, not measured physical deltas or joint velocities:

```text
[dx, dy, dz, drx, dry, drz, gripper]
```

Do not finite-difference observations or apply another controller scale. The first six dataset channels must lie in
`[-1,1]`. Gripper labels must be exactly `-1` (open) or `+1` (close); a 0/1 conversion or second sign flip is an error.
See the [robosuite 1.4 OSC config](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.0/robosuite/controllers/config/osc_pose.json)
and [Panda gripper convention](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.0/robosuite/models/grippers/panda_gripper.py).

Dataset images already use the regenerated 180° convention and are not transformed again. Raw evaluation images from
both cameras are rotated exactly once with `image[::-1, ::-1]`, copied to positive-stride contiguous arrays, then sent
to the DiffusionGemma processor in `[agentview, wrist]` order.

## Chunking, sampling, and statistics

At episode frame `i`, use observation/state/language at `i` and actions `a_i ... a_(i+7)`. A chunk never crosses an
episode/task boundary. Preserve the partial final chunk, zero-fill missing slots, and mask them from attention keys and
loss. The default sampler is `uniform task -> uniform episode -> uniform anchor`; frame-uniform sampling is an
ablation because it overweights long tasks.

First hold out 10% of complete episodes within each task using the committed seed/hash algorithm. Compute one set of
1st/99th percentile statistics over only the remaining episodes of the pinned 40-task training set. Normalize all
eight state values and the first six action values. Exclude action gripper, padded positions, and validation episodes.
Recompute statistics rather than importing LeRobot metadata, and save revision, exact episode set, frame/task counts,
inactive dimensions, algorithm/schema version, and checksum.

The committed split seed is `1729`. On the pinned revision it yields 1,525 training episodes / 246,015 frames and 168
validation episodes / 27,450 frames. The exact linear-quantile artifact has content SHA-256
`a972b5d95a8aaa8ae7582bafcbc071261979cb46c2a3515b4da7a7cf0156ac73`; its training/validation episode-list hashes are
`f2098187483947c7f37ac8cd1732d02354a839074cb0586af85a013d2b84a802` and
`dd61379f3743ea0c6dfd2238c21a4f88c832a92f465d0df7085f0f2fb5797065`. The measured bounds are:

```text
state q01  = [-0.39863038, -0.26797378, 0.03906375, 1.50411502,
              -2.73305418, -1.08140895, 0.00174251, -0.04002758]
state q99  = [ 0.13486923,  0.33424789, 1.26880903, 3.27946862,
               2.40883372,  0.59483355, 0.04031170, -0.00180910]
action q01 = [-0.70446426, -0.80089283, -0.93750000,
              -0.11464286, -0.16285715, -0.22392857]
action q99 = [ 0.93750000,  0.86785716,  0.93750000,
               0.13178572,  0.19285715,  0.33321428]
```

Hub metadata says 10 FPS while the reference rollout simulator executes at 20 Hz. The training contract is “eight
consecutive regenerated transitions”; it does not relabel the chunk as 0.8 seconds. In rollout, executing all eight
actions would span at most 0.4 seconds.

## Final rollout

Only checkpoints carrying a verified resolved config plus the identical manifest `policy_contract` are eligible. The
rectified-flow policy uses seeded Gaussian initialization and uniform Euler sampling; direct regression uses zero
action queries at fixed `t=1` and exactly one forward. Strict IPC schema v5 repeats objective, sampler, actual NFE,
seed behavior, evaluation seed, and reset identity in health/prediction as applicable, and binds real-policy health to
the content-addressed deterministic serving runtime plus a latency-comparability identity that excludes only
checkpoint-specific training-runtime provenance. A flow checkpoint may be served
at NFE 1, 5, or 10 without rewriting its authenticated default contract; a direct checkpoint cannot accept an NFE
override.

- environment: pinned `hf-libero`/robosuite 1.4, relative OSC_POSE, EGL;
- render: 256×256 agentview and wrist; no history or random crop;
- tasks: 10 per suite; all 50 official fixed initial states in their published order;
- seed: environment 7; deterministic model seed derived from the predeclared evaluation seed, suite, task, initial
  state, and replan index, never from the training seed;
- reset: clear action queue, policy state, and prefix cache at every episode;
- settle: 10 open-gripper no-op steps, excluded from the policy budget;
- budgets: Spatial 220, Object 280, Goal 300, Long (`libero_10`) 520 policy steps;
- success: check `env.check_success()` after every action and immediately discard the remaining chunk on success;
- execution: report `K=1` and `K=4` separately; regenerate when the queue empties.
- latency: exactly two discarded deterministic warm-up calls, one with K=1 and one with K=4, at reserved
  `replan_id=520` in official mode; validate their K-independent response identity immediately before scoring; client round-trip must be no smaller
  than paired server time; aggregate episode-only client/server p50/p95, episodes/hour, and policy calls/second only
  across cells with the identical latency-runtime identity; never pool K.
- attempt lifecycle: pre-register dedicated real run/claim roots and derive one path pair per cell. Durably publish the
  external claim before simulator construction or policy connection, journal every later failure, and accept a run
  only when its exclusive completion marker binds the claim, pre-registration, 1,999 episodes, summary, and final run.
  Aggregation derives paths only from the pre-registration and requires exact root inventories.

This is the OpenPI-style reference rather than LeRobot's alternate 280-step Spatial budget. Source:
[OpenPI evaluator](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py),
[LeRobot wrapper](https://github.com/huggingface/lerobot/blob/main/src/lerobot/envs/libero.py).

Do not run smoke or development policies on the published 50 states. The pinned LeRobot data does not retain full
MuJoCo states, so demonstration resets cannot be reconstructed from it. Instead, materialize a separate hashed set of
task-valid resets by seeding and resetting the BDDL training environment. Reject any generated state whose serialized
bytes match one of the 50 published states, and use only this frozen pool for G4, wrapper checks, and intervention
rollouts. Once the recipe is frozen, the final comparable run uses all 50 published states per task (2,000 episodes per
factor cell).

An earlier pilot already exposed Goal task 7 / state 0 for `K=1` and `K=4`. The immutable contamination ledger therefore
excludes this cell symmetrically from the primary clean-holdout comparison, leaving 1,999 episodes and 49 states for
that task. An all-2,000 result may be reported separately as non-blind full-set comparability. Report task success
counts, suite macro success, overall 40-task macro success, steps-to-success, policy calls, warmed p50/p95 latency, and
action clip fraction. A seed-wise Wilson 95% interval is only a descriptive interval over that checkpoint's binary
episode outcomes; report all three training seeds and their mean/standard deviation separately. Never average `K=1`
and `K=4` into one number.

## Environment approval gates

Before model training: schema/revision counts, exact gripper set, controller impulse directions, normalization round trip,
episode-boundary chunk fixture, pre-action alignment replay, camera transform parity, and deterministic fixed-state reset
must pass. Replay exactly one canonically selected, first-success expert demonstration for every task and require 40/40
success. Pre-dispatch integrity mutations (zero action, mismatched language, swapped cameras, and inverted gripper)
must produce distinct provenance hashes; these checks are not simulator rollouts or success-rate measurements.

These requirements are enforced by the strict evidence/report validator described in
[libero_expert_replay_qualification.md](libero_expert_replay_qualification.md). The official pre-registration embeds
the report's semantic source/config/data/runtime and raw-evidence identity, and its checkpoint cells must carry the
same qualified dataset tree/content, full train-venv v2 identity (including its base-Python identity), validator-runtime
identity, and source-tree hashes. The evaluator run journal and aggregator require exact equality with those frozen
objects and with the simulator-attestation v3 evaluator runtime. The two-stage simulator collector
and parquet binder are implemented, but the pinned 31.47-GiB original-HDF5 corpus is not materialized and no real
40-task report exists, so this gate remains open and final benchmark readiness must not be claimed.
