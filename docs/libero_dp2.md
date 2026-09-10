# LIBERO DP2 fork and launch contract

This profile is a new two-rank data-parallel run, not an in-place continuation of the TP1 run:

- profile: `duovla-dp2-tp1-fused-v2-train-b32-serve-b8-v1`
- global batch: 64; each rank executes one physical B32 forward made from four consecutive canonical B8 chunks
- gradients: local SSE divided by the all-reduced valid-element count, followed by SUM all-reduction
- training: DP world 2, full TP1 model replica on each rank, physical GPUs 0 and 1
- serving: one TP1 process at B8; it does not require either training GPU identity
- software: both ranks use the existing authenticated `train-single-gpu` venv (Torch 2.13.0+cu129/CUDA 12.9)
- data cache: `training.max_cached_files=377`, sealed again as `--max-cached-files 377`
- stream seed: 0, inherited from the authenticated parent; mutable per-rank RNG is domain-separated only after the fork

## Freeze the fork

First preregister the absolute parent update-1000 checkpoint path, its raw `manifest.json` SHA-256, parent source SHA-256,
parent config SHA-256, parent run UUID, a fresh child run UUID, the child output path, and the fork-document path. The
parent must be the authorized tip in its own `run_journal.json`. The child output path must not yet exist.

Run the creator only after the TP1 checkpoint exists, substituting the preregistered values:

```bash
/hdd2/hyunbin/vla/cache/venvs/train-single-gpu/bin/python -P -B -X pycache_prefix=/dev/null \
  /hdd2/hyunbin/vla/cache/workspaces/duo-vla-dp2-v1/scripts/create_libero_dp2_fork.py \
  /ABS/PARENT_RUN/checkpoints/update-001000 \
  /ABS/NEW_DP2_RUN \
  --expected-parent-manifest-sha256 PARENT_MANIFEST_SHA256 \
  --expected-parent-source-tree-sha256 PARENT_SOURCE_SHA256 \
  --expected-parent-run-uuid PARENT_RUN_UUID \
  --child-run-uuid NEW_CHILD_RUN_UUID \
  --output /ABS/FROZEN_FORK/fork.json
```

The creator writes `fork.json.sha256` first and `fork.json` last as the commit marker. Record the printed
`fork_manifest_sha256` in immutable preregistration evidence. The trainer accepts only the canonical, regular,
single-link JSON and adjacent sidecar when both match that independently supplied digest. The document binds parent checkpoint device/inode/path,
manifest/config/source/run UUID/update, exact trainable and rank-state artifact hashes, optimizer schema, the shared
train-venv content identity, child source inventory, static resolved TOML, output/run UUID, GPU UUIDs, DP/serving
topologies, cache size, and deterministic rank-RNG derivation.

## Start and resume

Start only when both physical GPUs are intentionally available. The launcher independently verifies both UUIDs and
refuses to run when either GPU has an existing compute PID:

```bash
DUO_VLA_CACHE_ROOT=/hdd2/hyunbin/vla/cache \
HF_HOME=/hdd2/hyunbin/vla/huggingface \
/hdd2/hyunbin/vla/cache/workspaces/duo-vla-dp2-v1/scripts/run_libero_train_dp2.sh \
  /ABS/LIBERO_SNAPSHOT \
  /ABS/NORMALIZATION.json \
  /ABS/NEW_DP2_RUN \
  --prefix-geometry-artifact /ABS/PREFIX_GEOMETRY.json \
  --fork-from /ABS/FROZEN_FORK/fork.json \
  --expected-fork-manifest-sha256 PREREGISTERED_FORK_SHA256
```

Fork mode requires `--expected-fork-manifest-sha256` exactly once. It is forbidden during ordinary resume. Do not pass
`--seed`, `--config`, batch/accumulation options, or `--max-cached-files`; these are sealed by the launcher.
`--stop-after-updates` remains an operational stop boundary and does not change the run contract.

An ordinary DP2 resume uses the same launcher and config but selects a checkpoint inside the child output:

```bash
DUO_VLA_CACHE_ROOT=/hdd2/hyunbin/vla/cache \
HF_HOME=/hdd2/hyunbin/vla/huggingface \
/hdd2/hyunbin/vla/cache/workspaces/duo-vla-dp2-v1/scripts/run_libero_train_dp2.sh \
  /ABS/LIBERO_SNAPSHOT \
  /ABS/NORMALIZATION.json \
  /ABS/NEW_DP2_RUN \
  --prefix-geometry-artifact /ABS/PREFIX_GEOMETRY.json \
  --resume checkpoints/update-NNNNNN
```

The first DP2 checkpoint is a new-run journal root (`parent_manifest_sha256=null`). Its strict `fork_lineage` records the
authenticated TP1 ancestry. Later checkpoints use the ordinary local DP2 parent chain and preserve the same fork
lineage. Ordinary resume requires two rank-state artifacts and never falls back to the TP1 rank state.

## Transition qualification evidence

Preserve an immutable copy of the parent update-1000 checkpoint/run before any TP1 continuation, and preserve the full
DP2 update-1001 checkpoint before resuming it. Normal retention may retire these non-permanent checkpoints. The sealed
comparison uses reference metrics updates 1–1001, DP2 metrics updates 1001–1100, baseline timing 901–1000, warm DP2
timing 1002–1100, and the separate DP2 update-1100 checkpoint for allocator memory:

```bash
/hdd2/hyunbin/vla/cache/venvs/train-single-gpu/bin/python \
  scripts/analyze_libero_dp2_transition.py \
  --parent-checkpoint /IMMUTABLE/PARENT/update-001000 \
  --reference-checkpoint /IMMUTABLE/REFERENCE/update-001001 \
  --candidate-checkpoint /IMMUTABLE/DP2/update-001001 \
  --candidate-performance-checkpoint /ABS/DP2_RUN/checkpoints/update-001100 \
  --reference-metrics /IMMUTABLE/REFERENCE/metrics.jsonl \
  --candidate-metrics /ABS/DP2_RUN/metrics.jsonl \
  --output /ABS/EVIDENCE/dp2-transition-analysis.json
```

The analyzer authenticates both metrics files by path, byte count, and SHA-256; cross-links checkpoint `last_metrics`,
run/config/source/fork lineage and normalized semantic recipe; and reads rank-local/max allocator peaks only from the
update-1100 checkpoint manifest.

No full GPU run is performed by this plumbing change. Before qualification, verify two-rank model/optimizer equality,
global valid-element loss normalization, first-step parity from update 1000 to 1001, memory headroom on each GPU, checkpoint
save/resume, and single-GPU B8 serving from the resulting DP2 checkpoint.
