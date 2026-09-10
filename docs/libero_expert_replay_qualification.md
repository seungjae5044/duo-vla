# LIBERO 40-task expert-replay qualification

Final LIBERO training is benchmark-ready only after a real
`duo-vla-libero-expert-replay-qualification-v1` report passes. A simulator preflight alone is insufficient: this
qualification authenticates the original demonstrations, reproduces the upstream regeneration procedure, binds each
selected replay to the exact training parquet episode, and exercises the production normalization and chunking code.

## Pinned source corpus

The original input is pinned by [the committed inventory](../configs/libero_original_hdf5_inventory.json):

- repository: `yifengzhu-hf/LIBERO-datasets`;
- revision: `f13aa24a3da8c43c7225569f28c562979fa0e35a`;
- 40 task HDF5 files, exactly 33,784,856,577 bytes;
- semantic inventory SHA-256: `c4976da211c1895914d3d07c956ef48a6a9175367a22ff03ff43112064ee14e1`;
- raw inventory-file SHA-256: `3b5f9b164434c91c41a3ff1e9681d91ab856969699c40fd7247ba93ed909cd7c`.

The source root must contain exactly those 40 paths. Files must be regular, independently materialized files with one
hard link; symlinks, extra files, HDF5 soft/external links, virtual datasets, and external dataset storage are rejected.
The corpus is about 31.47 GiB, so provisioning it is a deliberate storage operation and is not performed by bootstrap
or qualification scripts.

## Two-stage evidence generation

The two stages deliberately run in different closed environments.

All replay launchers start Python with safe-path and no-bytecode mode, an impossible `/dev/null` bytecode-cache
prefix, and no `PYTHONPATH`. Before importing project modules, each entry point rejects linked, special, sourceless
bytecode, or native-extension entries in the project package tree. The exact invocation flags, import path, and closed
environment are authenticated again immediately before publication.

### 1. Simulator replay

`collect_libero_expert_replay.py` runs in the pinned simulator environment. It authenticates every HDF5 byte, scans
every source action/state sequence, and follows the upstream regeneration semantics:

- construct one replay environment per task and seed it with `0` exactly once;
- reset and replay source demos in numeric order without reseeding between demos;
- execute original HDF5 action values at their source precision, while using canonical float32 actions only for the
  stored-training digest;
- apply the upstream no-op filter against the previous retained action;
- record the pre-action observation, rotate both simulator cameras 180 degrees exactly once, and preserve the
  `[agentview, wrist]` order;
- authenticate the pinned 1,693-episode source/parquet alignment, then select the first source-order replay that is
  both present in that regenerated training snapshot and successful in the live simulator; current-only successes
  that upstream regeneration excluded cannot be substituted for training evidence.

Reset-repeatability and controller impulse probes use separate freshly seeded environments, so they cannot consume the
replay environment's RNG state. The four mutation checks—zero actions, mismatched language, swapped cameras, and
inverted gripper—are pre-dispatch integrity checks. They prove that provenance hashes distinguish those mutations;
they are not policy rollouts and do not claim a simulator success rate.

Run stage one only after recording the inventory and simulator-attestation hashes outside the output directory:

```bash
./scripts/run_collect_libero_expert_replay.sh \
  --source-root /materialized/libero-original-hdf5 \
  --inventory ./configs/libero_original_hdf5_inventory.json \
  --inventory-sha256 3b5f9b164434c91c41a3ff1e9681d91ab856969699c40fd7247ba93ed909cd7c \
  --simulator-attestation /read-only/freeze/libero-simulator-attestation.json \
  --simulator-attestation-sha256 '<externally-recorded-raw-64hex>' \
  --output-dir /evidence/libero-expert-replay
```

The output directory must not exist. Record the printed `simulator-stage.json` SHA-256 before stage two. A failed or
interrupted directory is evidence of that attempt and is never resumed or overwritten; start again with a new path.

### 2. Training-parquet binding

`bind_libero_expert_replay.py` runs in the closed training environment. It authenticates the complete pinned
Hugging Face snapshot and the complete train virtual environment before reading data. It then:

- finds exactly one parquet episode for each `(instruction, float32 action-sequence hash, length)` tuple;
- requires 40 unique episode indices and the exact 40 task-index inventory;
- sends every one of the 1,693 episodes through the production action validator, covering all 273,465 actions before
  accepting the exact gripper set;
- compares every decoded pre-action RGB/state frame against the simulator sequence. The exact action digest and exact
  frame count select the episode; both cameras are compared as 8x8 RGB block means with bounded absolute error,
  per-frame and sequence-mean correlation floors, while the full eight-dimensional state has bounded error and must
  be closer at the declared frame than at either adjacent frame. Full-resolution simulator and parquet RGB hashes are
  retained separately as provenance because EGL/platform rendering is not bitwise portable;
- runs normalization through `load_libero_normalizers` and the production normalizer classes;
- obtains a terminal sample through `LiberoParquetDataset.sample(..., horizon=8)` and requires one valid action,
  seven zero-filled positions, and the exact boolean validity mask;
- reauthenticates the full snapshot, train virtual environment, sources, and all stage-one inputs immediately before
  publishing `evidence.json` last.

Run stage two against the same stage-one directory:

```bash
./scripts/run_bind_libero_expert_replay.sh \
  --simulator-stage /evidence/libero-expert-replay/simulator-stage.json \
  --simulator-stage-sha256 '<externally-recorded-stage-one-64hex>' \
  --inventory ./configs/libero_original_hdf5_inventory.json \
  --inventory-sha256 3b5f9b164434c91c41a3ff1e9681d91ab856969699c40fd7247ba93ed909cd7c \
  --simulator-attestation /read-only/freeze/libero-simulator-attestation.json \
  --simulator-attestation-sha256 '<externally-recorded-raw-64hex>' \
  --dataset-tree-metadata /root/.cache/huggingface/hub/datasets--HuggingFaceVLA--libero/trees/86958911c0f959db2bbbdb107eb3e17c5f9c798e.json \
  --dataset-tree-metadata-sha256 d9c14b4aff28bcc56f341b171c6a5a3b10510d4bd0378662891c5156d245add8 \
  --normalization-artifact /root/.cache/duo-vla/data/libero/normalization-v1.json \
  --snapshot-root /root/.cache/huggingface/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e
```

Record the printed evidence-manifest SHA-256 externally. The committed bundle contains exactly 83 authenticated raw
records: 40 simulator task records, 40 parquet bindings, both stage records, and the stage-one commit record. The
manifest and its checksum companion are the only additional files; alternate, linked, failed, or unreferenced paths
are rejected.

## Qualification gates

The manifest contains exactly one successful regenerated demo per canonical task and these nine ordered gates:

1. pinned schema, revision, and row/file/task counts;
2. the exact `{-1,+1}` gripper set in source and training data;
3. all positive and negative OSC_POSE translation/rotation impulse directions plus gripper polarity;
4. production state/action normalization round trip;
5. production terminal zero-padding/mask with no cross-episode leakage;
6. pre-action `(observation_t, action_t)` alignment: exact action/length identity, bounded RGB/state alignment, and a
   state temporal-margin check rejecting either adjacent-frame pairing;
7. single 180-degree camera-transform parity, positive-stride `uint8[256,256,3]`, and camera order;
8. isolated deterministic regeneration-reset probes for all 40 tasks;
9. pre-dispatch provenance mutation detection for zero actions, mismatched language, swapped cameras, and inverted
   gripper.

The validator does not trust `passed=true`. It reopens every JSON record, requires exact schemas and paths, checks the
stage-one commit chain, recomputes source-scan/attempt/selection/trajectory relationships, and cross-checks the stage
results against the published gate results.

## Validate and publish

Qualification independently rehashes the live training snapshot and train virtual environment before and after raw
bundle validation. The train-venv v2 identity includes a no-follow content inventory of the resolved external base
Python installation, executable, standard library, startup hooks, and the venv-to-base symlink chain. The simulator
attestation v3 likewise contains the complete evaluator venv and base-Python identity. This reads tens of GiB and is
intentionally not a quick metadata-only check.

```bash
./scripts/run_qualify_libero_expert_replay.sh \
  --evidence-manifest /evidence/libero-expert-replay/evidence.json \
  --evidence-manifest-sha256 '<externally-recorded-evidence-64hex>' \
  --simulator-attestation /read-only/freeze/libero-simulator-attestation.json \
  --simulator-attestation-sha256 '<externally-recorded-raw-64hex>' \
  --dataset-tree-metadata /root/.cache/huggingface/hub/datasets--HuggingFaceVLA--libero/trees/86958911c0f959db2bbbdb107eb3e17c5f9c798e.json \
  --dataset-tree-metadata-sha256 d9c14b4aff28bcc56f341b171c6a5a3b10510d4bd0378662891c5156d245add8 \
  --snapshot-root /root/.cache/huggingface/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e \
  --normalization-artifact /root/.cache/duo-vla/data/libero/normalization-v1.json \
  --normalization-artifact-sha256 8a0428184e4db8463f3986e9a8c4f912e1027815669f4b4880c4bd3e8d3bcf29 \
  --original-hdf5-inventory ./configs/libero_original_hdf5_inventory.json \
  --original-hdf5-inventory-sha256 3b5f9b164434c91c41a3ff1e9681d91ab856969699c40fd7247ba93ed909cd7c \
  --output-dir /read-only/freeze/libero-expert-replay-qualification
```

The qualification output directory must not exist. `qualification.json` is published last as the commit marker. The
report freezes source/config hashes, both dataset identities and counts, normalization, original-HDF5 inventory,
simulator runtime/task inventory, the full train-venv identity, all evidence hashes, and exact 40/40 counts. A separate
validator-runtime identity binds the live qualifier process, evaluator venv, installed distributions, module origins,
site-packages inventory, and authenticated project sources; it is recomputed immediately before the exclusive commit.
Official pre-registration and checkpoint authentication must preserve those identities.

## Current status

The GPU-0 / TP=1 qualification is closed for the current implementation. The pinned 40-file HDF5 corpus was fully
authenticated, all 40 simulator replays were bound to 40 unique episodes in the 1,693-episode parquet snapshot, and
the independent qualifier reported 40/40 successful tasks:

- simulator attestation raw SHA-256: `86b114584f9874dc72c7ac9f61de8b2d23f2700b5206e65d4e578e48ad8bcca1`;
- simulator-stage SHA-256: `fb73b6b0198db6e4e3e21347c0eb7cac5f7b5b28fc0311085bc4dd6f3c8a26c9`;
- evidence-manifest SHA-256: `d01b916fa5266b803d60c15cbdb9ebcf409c24535ecadff17fd4af1222ec312b`;
- qualification-report SHA-256: `9687438e5b4de76e20d3287e8873c2477542b24a39a7056294ea79c726392300`.

The report is stored at
`/hdd2/hyunbin/vla/cache/reports/g6/libero-expert-replay-qualification-tp1-v2/qualification.json`. This development
qualification is not the later external freeze token and does not authorize official benchmark calls.
