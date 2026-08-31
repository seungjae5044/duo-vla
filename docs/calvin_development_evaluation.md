# Held-out CALVIN A/B/C development rollout

This is a checkpoint-development gate over held-out episodes from
`task_ABC_D/training`. It is not the official ABC→D benchmark. It constructs
only `calvin_scene_A`, `calvin_scene_B`, and `calvin_scene_C`; it does not import
the official sequence generator, instantiate scene D, execute an action in D,
or report official `SR_k`/`AvgLen` metrics.

## Authentication boundary

Development replay now has a two-stage, version-separated boundary. The
Python 3.11 exporter is the only development process that may open the
authenticated archive or decode an episode member. The pinned Python 3.8 bank
generator and evaluator never import the archive reader and never open an
archive or `episode_*.npz` path. They consume only authenticated metadata bytes,
simulator configuration bytes, and the exported replay bundle.

All stages fail closed unless these identities agree:

- the source directory is exactly `task_ABC_D/training`;
- the archive-direct v4 manifest names the published 555,309,812,705-byte
  archive and binds its raw manifest hash, content hash, full archive SHA-256,
  ZIP64 central-directory identity, complete member inventory, exact projected
  metadata inventory, reader schema, and storage-identity hash;
- the v2 SQLite member index has its manifest-bound path, byte count, SHA-256,
  exact schema/index layout, row count, metadata, and integrity. Bundle member
  path, logical byte count, logical SHA-256, and global index are rechecked
  against this current index without reopening member data;
- the v4 normalization artifact binds that manifest/storage generation and the current training
  metadata, uses the official identity-scaled relative-action transform, and
  contains the seed-1729, 10% whole-episode split;
- recomputing the scene-grouped split from the authenticated episode and
  annotation metadata gives the exact recorded train/validation partition and
  preserves every task on both sides;
- the CALVIN, `calvin_env`, and TACTO git revisions and the exact A/B/C scene
  YAML and task-oracle YAML hashes match the pinned contract.

Production development tooling accepts v4/v2 archive-direct storage only;
legacy v3 extraction/normalization artifacts are rejected. Validation critical
files remain opaque archive identity material. The development code never
parses validation trajectories, scene D reset states, official sequences, D
rewards, or D oracle outcomes.

The complete archive is not present in the repository development environment,
so repository tests use small fake fixtures and monkeypatch only the expensive
published-archive verification boundary. They are preflight tests, not measured
robot-policy results.

## Replay bundle and deterministic reset bank

First export the held-out candidates under the canonical Python 3.11 training
runtime. The exporter uses only public `CalvinArchiveReader` methods, reads the
entire half-open relative-action sequence for every held-out candidate, and
records each source member's original path, logical bytes, and logical SHA-256
together with the start `robot_obs`/`scene_obs`. It reauthenticates the v4
generation and source files before exclusive publication:

```bash
/root/.cache/duo-vla/venvs/train/bin/python3.11 \
  scripts/calvin/export_calvin_dev_replay_bundle.py \
  --data-root /root/.cache/duo-vla/data/calvin \
  --normalization /path/to/calvin-normalization-v4.json \
  --source-root /root/.cache/duo-vla/simulators/calvin \
  --output-dir /path/to/calvin-heldout-abc-replay-bundle
```

The bundle has an exact file inventory and a root SHA-256 over its input,
source, record, and per-artifact identities. Candidate omission/reordering,
array or member-identity tampering, current manifest/index/storage drift, and
source-code drift are rejected. Publication is exclusive and never overwrites
an existing path. The exporter is development-only and is not a runtime
dependency of production CALVIN training, so it is not part of the production
training source-tree hash.

Then replay that bundle and create the reset bank in the pinned Python 3.8
CALVIN environment:

```bash
/root/.cache/duo-vla/venvs/calvin-eval/bin/python3.8 \
  scripts/calvin/generate_calvin_dev_states.py \
  --training-root /root/.cache/duo-vla/data/calvin/task_ABC_D/training \
  --normalization /path/to/calvin-normalization-v4.json \
  --source-root /root/.cache/duo-vla/simulators/calvin \
  --replay-bundle /path/to/calvin-heldout-abc-replay-bundle \
  --output-dir /path/to/calvin-heldout-abc-bank
```

For each scene/task pair, candidates are restricted to normalization-validation
episodes and ordered by a domain-separated hash of the base seed and immutable
annotation identity. A candidate is retained only when a fresh environment can
be reset from its source `robot_obs[15]` and `scene_obs[24]`, is not already
successful, and its recorded half-open annotation actions satisfy the pinned
task oracle. The bank contains one replay-authenticated reset per observed
scene/task. Its float64 arrays, record identities, rejection log, balanced
four-distinct-task-per-scene smoke view, and input identities are covered by a
root SHA-256. Output creation is exclusive and will not overwrite a bank.

Do not use the pinned `get_env(..., scene=...)` override here: that version
discards the result of `OmegaConf.merge`. The generator directly replaces
`config.scene` with the authenticated A/B/C YAML, verifies the table asset and
movable-object order, and then instantiates a new environment. The pinned
`Robot.reset` also ignores the previous gripper-command field, so reset first
sets `environment.robot.gripper_action = int(robot_obs[14])`, then supplies
owned copies of both reset arrays and validates the restored joints, gripper,
scene state, and camera schemas.

## Development IPC

`scripts/calvin/calvin_dev_bridge.py` defines the separate
`duovla-calvin-dev-policy-ipc-v4` wire schema and
`duovla-calvin-heldout-abc-v1` protocol. An official
`duovla-calvin-policy-ipc-v4` endpoint is deliberately incompatible.

The private Unix-socket protocol accepts A/B/C only. Its exact health schema
contains one nested `calvin_identity` object, byte-for-byte equal to the v4
normalization artifact's dataset identity. It binds the pinned archive byte
count/hash and ZIP64 central-directory hash, raw/content/schema manifest
identities, v2 index path/bytes/schema/hash, member inventory, exact metadata
file inventory/hash, reader schema, and storage identity/mode. The remaining
health fields bind the replay bundle, reset bank, split, normalization, and
checkpoint artifacts; normalization metadata must equal the nested dataset
metadata hash. Legacy v3 health envelopes and missing, extra, or drifted
identity fields fail closed. Every prediction is bound to the reset, episode,
annotation, task, and replan identities. Images are lossless uint8 arrays with
per-field SHA-256; state and actions are exact float32 contracts. Observation
keys are exact, so reset-only `scene_obs` cannot cross the model boundary. The
inference seed is derived from evaluation seed, bank/reset identity, task, and
replan index; it intentionally excludes train seed, objective, NFE, and
execution horizon so A/B comparisons share noise.

The development evaluator requires a policy process that serves this dev
schema. It will reject the official endpoint rather than silently sharing its
schema:

First authenticate a real development checkpoint without loading CUDA:

```bash
./scripts/calvin/run_policy_server_dev.sh \
  /path/to/training-run/checkpoints/update-001000 \
  --normalization /path/to/calvin-normalization-v4.json \
  --reset-bank /path/to/calvin-heldout-abc-bank \
  --preflight-only
```

Then start the persistent TP=2 endpoint on its dedicated
socket:

```bash
./scripts/calvin/run_policy_server_dev.sh \
  /path/to/training-run/checkpoints/update-001000 \
  --normalization /path/to/calvin-normalization-v4.json \
  --reset-bank /path/to/calvin-heldout-abc-bank \
  --socket /path/to/calvin-heldout-abc.sock
```

The launcher removes caller-controlled Python, CUDA, library, locale, and
threading overrides before selecting the pinned Python 3.11 train environment.
Real serving always uses `torchrun --nproc-per-node=2`; the reused policy core
loads DiffusionGemma with tensor parallel size two. Every singleton request is
replicated into a fixed physical batch of eight, including its prefix, state,
validity mask, and one cloned seeded noise row. Encoder and decoder grouped-MM
experts execute each row independently; the server requires all eight action
outputs to be bitwise identical and returns row zero. Rectified-flow noise is
derived only from the request's deterministic development inference seed.

The server authenticates the full v4 archive-direct/v2-member-index chain,
normalization and split, pinned clean simulator sources and A/B/C scene/oracle
YAMLs, reset bank, model snapshot, lock and installed packages, training source
tree/runtime, resolved config, LoRA/interface artifacts, and checkpoint policy
contract. Health reports bind the reset-bank root, reset count, split, dataset
manifest, normalization content/metadata, checkpoint manifest, model revision,
selected policy contract, and exact execution geometry (backend, isolation,
B, P, and prefix-artifact hash). Every prediction is additionally checked against
the exact bank record at its `reset_index`, including reset, episode,
annotation, task, scene, start-frame, and raw-instruction identities.

For development only, the selected checkpoint may be either a completed run or
an intermediate checkpoint. It must be the latest transactionally committed
tip in its training run's `run_journal.json`; standalone/copied checkpoint
directories and post-checkpoint/pre-journal crash remnants are rejected. An
intermediate checkpoint must have `complete: false`, while a checkpoint at its
configured final update must have `complete: true`. A reduced-update pilot may
set `optimization.total_updates` to an integer in `[1, 30000]`. The trainer
then derives `optimization.warmup_updates = min(1000, total_updates // 10)`,
and the development resolver requires exactly that value; warmup is not an
independent tuning override. Thus the canonical 30,000-update run retains
1,000 warmup updates, while a 500-update pilot requires 50. All other recipe,
artifact, data, source, model, and runtime checks remain unchanged.

For socket plumbing without model loading, start the same authenticated bank
boundary with a fake policy:

```bash
./scripts/calvin/run_policy_server_dev.sh \
  --fake-policy \
  --train-seed 1 \
  --normalization /path/to/calvin-normalization-v4.json \
  --reset-bank /path/to/calvin-heldout-abc-bank \
  --socket /path/to/calvin-heldout-abc.sock
```

Fake health cannot claim checkpoint, model, policy-contract, or normalization
identity or execution geometry, and the evaluator requires `--allow-fake-policy`. Fake results always
remain plumbing-only.

With either endpoint running, invoke the Python 3.8 simulator-side evaluator:

```bash
/root/.cache/duo-vla/venvs/calvin-eval/bin/python3.8 \
  scripts/calvin/evaluate_calvin_dev.py \
  --training-root /root/.cache/duo-vla/data/calvin/task_ABC_D/training \
  --normalization /path/to/calvin-normalization-v4.json \
  --source-root /root/.cache/duo-vla/simulators/calvin \
  --reset-bank /path/to/calvin-heldout-abc-bank \
  --socket /path/to/calvin-heldout-abc.sock \
  --execution-horizon 4 \
  --output-dir /path/to/dev-run
```

`serve_policy_dev.py` always identifies itself as
`heldout_abc_development_only_not_official_calvin_abc_to_d` and sets official
benchmark metrics as disallowed in its preflight/readiness reports. It never
loads official sequence definitions, creates scene D, executes a D action, or
queries a D task oracle. Its socket cannot be used by `calvin_bridge` v4, and
held-out A/B/C success rates must never be renamed or reported as official
CALVIN ABC→D scores.

Each bank reset gets a newly constructed environment. The evaluator sends the
raw annotation instruction and policy state `robot_obs[:7] + robot_obs[14:15]`,
executes copied official-unit actions, asks the task oracle after every action,
and immediately clears queued actions on success. `K` may be 1 or 4 and each
reset has at most 360 actions. Results use only
`heldout_abc_subtask_success(_rate)` names and are reported per independent
reset and per A/B/C scene. `--allow-fake-policy` exists solely for IPC plumbing;
its summary always has `plumbing_only: true` and `gate_passed: false`, even if
the fake actions happen to trigger the fixture oracle.

Focused non-simulator checks are:

```bash
.venv/bin/pytest -q \
  tests/test_calvin_dev_states.py \
  tests/test_calvin_dev_replay_bundle_export.py \
  tests/test_calvin_dev_state_generator.py \
  tests/test_calvin_dev_bridge.py \
  tests/test_calvin_dev_evaluation.py \
  tests/test_calvin_dev_policy_server.py
```
