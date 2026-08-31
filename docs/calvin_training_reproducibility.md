# CALVIN strict training reproducibility qualification

`scripts/compare_calvin_training_reproducibility.py` is a CPU-only,
fail-closed qualification for the canonical two-update CALVIN smoke sequence.
It compares an update-2 journal tip produced by stopping after update one and
resuming against a separately-created uninterrupted update-2 journal tip.

The smoke retains the production optimization schedule: `total_updates=30000`
and `warmup_updates=1000`. It changes only the checkpoint cadence to one so the
uninterrupted two-update invocation materializes the same update-1 comparison
point as the interrupted run. The trainer records both
`training.checkpoint_interval=1` and `training.permanent_checkpoint_interval=1`
in the common resolved config. Do not use `--total-updates 2`.

The resolved configuration must otherwise have the exact inventory and static
values obtained from `configs/calvin_abc_to_d.toml`. The only smoke overrides
are both checkpoint intervals, the selected canonical run seed, `run.task=null`,
and the canonical `run.max_cached_frames=512`. Dataset, model, prefix geometry,
training-runtime, execution-geometry, and model-tree tables are dynamic only
where the trainer records authenticated identities; their complete schemas and
cross-field hashes are validated.

The caller must pin both raw `run_journal.json` SHA-256 values, both canonical
run UUIDs, and the common semantic resolved-config SHA-256. The comparator then
authenticates the complete update-1 -> update-2 manifest lineage and every
manifest-listed checkpoint artifact before loading any rank state. It requires
exactly TP=2 and exactly the two canonical checkpoint directories; it never
loads the model, CALVIN data, or a CUDA tensor.

Rank-state files must be non-symlink regular files with one link. The comparator
reads and hashes each through one stable `O_NOFOLLOW` descriptor and gives
`torch.load` an in-memory copy of those exact authenticated bytes; it never
rehashes one pathname and then reopens that pathname for deserialization. The
full production run-contract field inventory and the AdamW/LambdaLR state
schemas are validated before logical comparison. Rank-state schema v2 also
stores the ordered, named optimizer parameter inventory. Its digest must equal
`optimizer_parameter_schema_sha256`; group order, the pinned model's exact 230
LoRA A/B parameters across all 115 decoder-attention targets and their global
shapes, and the exact 14 action-interface parameters at hidden size 2816 are
checked. Optimizer parameter IDs and moment coverage/shapes must then match
that inventory. Rank-state schema v1 checkpoints are rejected and must be
regenerated. TP optimizer moments deserialize as CPU `DTensor`
objects; the comparator also binds their global mesh/placement metadata while
comparing and hashing the authenticated per-rank local bytes, without creating
a process group or initializing CUDA.

The comparison permits only these semantic exclusions:

- Training metrics: `update_seconds`.
- Logical per-rank update-1 and update-2 state: `run_contract.run_uuid`.
  Optimizer, scheduler, CPU/CUDA RNG snapshots, trainer progress, and every
  other run contract field remain exact at both updates.
- Checkpoint manifests: the independently authenticated `run_uuid`, parent
  manifest lineage, the two rank-state byte hashes, and
  `last_metrics.update_seconds`. No other field is normalized.

The update-2 LoRA adapter, LoRA config, action interface, and copied resolved
config are compared as raw byte streams. Root `resolved_config.json` is also
byte-exact, and every checkpoint copy must equal its own root copy.

## Exact smoke launch sequence

Use the same config, seed, data/normalization/prefix artifacts, and
`--checkpoint-interval 1` for all three invocations. The flow example below
uses seed zero; repeat the qualification independently for every required
objective/seed pair. `--stop-after-updates` is invocation-local and therefore
does not alter the 30,000-update run contract.

```bash
# Interrupted run, invocation 1: update 0 -> 1.
bash scripts/run_calvin_train.sh \
  /path/to/task_ABC_D/training \
  /path/to/calvin-abc-to-d-normalization.json \
  /path/to/interrupted-resumed-run \
  --config configs/calvin_abc_to_d.toml \
  --prefix-geometry-artifact /path/to/calvin-prefix-geometry.json \
  --seed 0 \
  --checkpoint-interval 1 \
  --stop-after-updates 1

# Interrupted run, invocation 2: authenticate update 1, resume, then update 2.
bash scripts/run_calvin_train.sh \
  /path/to/task_ABC_D/training \
  /path/to/calvin-abc-to-d-normalization.json \
  /path/to/interrupted-resumed-run \
  --config configs/calvin_abc_to_d.toml \
  --prefix-geometry-artifact /path/to/calvin-prefix-geometry.json \
  --seed 0 \
  --checkpoint-interval 1 \
  --resume checkpoints/update-000001 \
  --stop-after-updates 1

# Independent uninterrupted run: one process executes updates 1 and 2.
bash scripts/run_calvin_train.sh \
  /path/to/task_ABC_D/training \
  /path/to/calvin-abc-to-d-normalization.json \
  /path/to/uninterrupted-run \
  --config configs/calvin_abc_to_d.toml \
  --prefix-geometry-artifact /path/to/calvin-prefix-geometry.json \
  --seed 0 \
  --checkpoint-interval 1 \
  --stop-after-updates 2
```

Both resolved configs must have `run.task=null`; a single-task smoke is not an
ABC-to-D training reproducibility qualification.

The artifacts authenticate distinct run UUIDs, config, update lineage, metrics,
and checkpoint contents. The process boundary itself is not encoded in the
current run journal, so the statement that the first run used two invocations
and the second used one remains an externally asserted operator fact. Preserve
the launch transcript alongside the externally pinned journal hashes; this
qualification does not turn that transcript into a cryptographic receipt.

## Closed CPU runtime and source identity

Run only by executing `scripts/calvin/run_compare_training_reproducibility.sh`
directly. Its kernel shebang clears the incoming environment before Bash starts,
uses `/bin/bash --noprofile --norc`, and fixes the canonical cache root to
`/root/.cache/duo-vla`; neither `DUO_VLA_CACHE_ROOT` nor `CALVIN_TRAIN_VENV` is
an override for this qualification. The launcher starts the canonical train
Python with `-I -S -B`, validates its exact
CPython 3.11.15 executable and `pyvenv.cfg`, and does not execute `site` or any
`.pth` file. It manually adds only the project `src` and canonical train-venv
site-packages directories, verifies that `_virtualenv`,
`_cuda_bindings_redirector`, and `_distutils_hack` were not preloaded, clears
GPU visibility, and executes the exact comparator byte string it authenticated.
The bootstrap injects a process-local global capability which the comparator
consumes before importing torch or project modules. Supplying the former public
proof environment variables to `python scripts/compare_calvin_training_reproducibility.py`
therefore fails before argument parsing. This boundary is a direct-invocation
fail-close contract, not a claim of cryptographic unforgeability against a
hostile caller that writes its own Python bootstrap. Invoking the file as
`bash scripts/calvin/run_compare_training_reproducibility.sh` bypasses the
kernel shebang and is therefore explicitly unsupported; the body fails closed
when it detects that entry mode, but it cannot retroactively prevent a hostile
`BASH_ENV` hook that Bash already executed. Use the executable path directly.

This runtime is deliberately separate from CALVIN's Python 3.8.20 simulator
environment; the comparator is not Python-3.8-compatible and does not need to
be. A passing report records the isolated/no-site/safe-path/ignore-environment/
no-bytecode flags, the bootstrap module inventory, exact venv identity, and
complete `sys.path`.

Before importing torch or the six directly used production modules, the
comparator snapshots itself, the launcher, those modules, and the same
production source tree hashed by `train_calvin.py`. It verifies the snapshot at
main startup, comparison completion, immediately before exclusive report
publication, and again after the complete unnamed report inode is durably
staged but before its atomic final-name commit. The same publisher guards the
canonical runtime identity at both staging boundaries. Qualification files are
separately identified and are not added to the production training source-tree
hash.

The CALVIN trainer, policy server, and comparator use the same versioned v2
source-tree framing: magic/schema bytes, exact file count, then 64-bit
length-prefixed UTF-8 relative paths and length-prefixed file bytes. Required
explicit files may not be omitted, and symlinks or non-regular entries fail.
Each source file is read from a stable nofollow descriptor. The filesystem-root
to project-root directory chain, every in-tree directory, and every explicit
file ancestor are opened and reauthenticated through retained
`O_DIRECTORY|O_NOFOLLOW` descriptors, so a symlink or directory replacement at
any level fails closed. This replaces the former ambiguous path-plus-content
concatenation; LIBERO source-hash contracts are unchanged.

`src/duo_vla/**` is discovered recursively, so new CALVIN backend modules are
included automatically. Production files outside that tree are an explicit
audit hook: every new trainer/server/data-preparation script or configuration
must be added to `_CALVIN_SOURCE_EXPLICIT_RELATIVE_PATHS` in `train_calvin.py`
and `serve_policy.py`, and to `_TRAINING_SOURCE_EXPLICIT_RELATIVE_PATHS` in the
comparator. Parity tests require those three inventories and their resulting
digests to remain identical. A new archive-direct preparation script therefore
must update the inventory before its output can support a canonical claim.

Run through the closed CPU launcher:

```bash
scripts/calvin/run_compare_training_reproducibility.sh \
  --interrupted-resumed-run /path/to/resumed-run \
  --interrupted-resumed-journal-sha256 <64-lowercase-hex> \
  --interrupted-resumed-run-uuid <canonical-uuid> \
  --uninterrupted-run /path/to/uninterrupted-run \
  --uninterrupted-journal-sha256 <64-lowercase-hex> \
  --uninterrupted-run-uuid <different-canonical-uuid> \
  --expected-config-sha256 <64-lowercase-hex> \
  --output /new/path/calvin-update2-reproducibility.json
```

The output path must not exist and must be outside both run roots. The output
filesystem must support Linux `O_TMPFILE` and linking the pinned unnamed inode
through `/proc/self/fd`. A passing report is canonical JSON and carries
`report_sha256`, computed over the report with that field removed. The
publisher writes, syncs, and byte-verifies an unnamed inode while the final
name is absent, performs its final source/runtime guard, then creates the final
name in one atomic no-replace link and syncs the retained parent-directory
descriptor. Readers can therefore observe either no final name or the complete
report bytes, never a partially written report. Parent-directory and final-name
inode/device/link-count identities are rechecked around the commit.

If an attack or I/O failure is detected after the atomic link, the publisher
never performs a racy stat-then-unlink. It invalidates only its still-pinned
inode to a fixed non-JSON failure tombstone and leaves any foreign replacement
untouched. Such a residue is not a valid qualification report and may require
operator cleanup before retrying the exclusive output path. Failures before
the atomic link close the unnamed inode and leave no output name.
