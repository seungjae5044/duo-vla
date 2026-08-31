# Reproducible CALVIN evaluator bootstrap

CALVIN is isolated from the model runtime because its official evaluator targets Python 3.8 and PyTorch 1.13.1. The
source-only preflight does not download the 517 GB ABC_D archive.

```bash
scripts/calvin/checkout.sh
scripts/calvin/bootstrap_env.sh
scripts/calvin/run_official_evaluator.sh preflight --egl
```

The checkout is pinned to parent commit `fa03f01f19c65920e18cf37398a9ce859274af76`, `calvin_env` commit
`1431a46bd36bde5903fb6345e68b5ccc30def666`, and `tacto` commit
`dd53360d9a8c186f0d6439372ec0be0fa5e21731`. The preflight also regenerates all 1,000 official seed-0 long-horizon
sequences and requires canonical SHA-256
`90191d9ac76baecb4f292ab766bbbf3ae65dbf43a83bbe5c376d99db10fd6446`.

Preflight is also the canonical runtime attestation producer. It rejects a dirty parent, `calvin_env`, or TACTO
checkout; verifies the origin of the imported `calvin_agent` and `calvin_env` packages; and requires every explicit
`name==version` entry in `constraints-py38.txt`. The attestation records the exact Python/platform/runtime identity,
the raw SHA-256 of both official evaluation YAML files, and the raw SHA-256 of `evaluate_calvin.py`,
`calvin_bridge.py`, and `preflight.py`. Preflight snapshots those three files at import time and rejects drift before
emitting its report, so the attested bytes cannot silently differ from the source boundary under which it ran. The
pinned raw hashes are:

- `new_playtable_validation.yaml`:
  `d14b0bf960f65158c5815b10ff11d2464073f26ac9b62c9f54e5c90ca352ccfa`
- `new_playtable_tasks.yaml`:
  `6e905de3ca05118efdd8a51f8a7756ec6e61ffdb2b9b6a2843f0b7e0e9e51dcf`

The optional EGL gate instantiates the official `calvin_scene_D_eval` config, renders the 200×200 static and 84×84
gripper cameras, validates the 15D/24D robot/scene observations and 30 Hz control rate, and takes one nonzero step. It
also confirms an important upstream behavior: `env.step` scales and mutates the first six action values in place, so
Duo-VLA must always give the simulator a fresh owned action copy. The preflight reserves stdout for one strict JSON
report and redirects native PyBullet/EGL diagnostics to stderr, including diagnostics flushed during process teardown.

Prepare the production archive-direct layout without extracting 1.89 million episode members:

```bash
scripts/calvin/download_dataset.sh archive-direct
# For an archive that was downloaded and verified separately:
scripts/calvin/download_dataset.sh prepare-archive-direct
# Idempotent strict v4 verification only:
scripts/calvin/download_dataset.sh verify-archive-direct
```

`archive-direct` reserves the remaining download bytes plus 16 GiB for the v2 SQLite index and the six projected
metadata files. It delegates archive hashing and publication to `prepare_archive_direct.py`, avoiding a duplicate
555 GB SHA-256 pass in the shell wrapper. That transaction accepts only a complete v4 generation and never blesses an
extracted v3 tree, a mixed generation, or a partially published root. The legacy
`all` and `extract` modes retain their original v3 extracted-tree meaning for parity work.

To authenticate the resulting projected dataset without downloading anything:

```bash
scripts/calvin/run_official_evaluator.sh preflight \
  --dataset-root /external/path/task_ABC_D \
  --require-dataset
```

With `CALVIN_DATASET_ROOT` set, preflight emits a canonical
`duovla-calvin-official-runtime-data-attestation-v2`. Production attestation accepts only the exact v4 manifest root
inventory with `storage.mode=archive-direct`, reader v1, and member index v2. The Python 3.8 stdlib-only verifier pins
all path components without following symlinks; hashes the 229,211,511-byte central directory and exact ZIP64 trailer;
the latter is the pinned 98-byte `c25191e6...fa359ffd` offset-only trailer (the classic central-size field remains
229,211,511 while the classic central-offset field is the ZIP64 sentinel). It hashes the complete small SQLite index;
checks its exact schema, metadata, every canonical member path/role and physical range; and authenticates all six
projected metadata files against their indexed raw-DEFLATE bytes. Official D reads no episode NPZ. These checks run
before sequence generation, OmegaConf/Hydra parsing, environment construction, or oracle construction. Preserve the
emitted `attestation_sha256` outside the evaluation host before freezing an official-score pre-registration.

The dataset attestation also emits the permanent training/run storage names: `archive_bytes`, `storage_mode`,
`storage_identity_sha256`, both dataset-manifest hashes and schema, `central_directory_sha256`, `reader_schema`, and
the member-index bytes/path/schema/SHA-256. Its nested `calvin_identity` uses the same `member_index` object and
`metadata_files` list as normalization v4, so checkpoints, serving health, pre-registration, and run artifacts bind one
canonical storage identity rather than translating field names.

An extracted v3 tree may be checked only as an explicitly non-promotable parity diagnostic:

```bash
scripts/calvin/run_official_evaluator.sh preflight \
  --dataset-root /external/path/task_ABC_D \
  --legacy-v3-parity
```

That report has `attestation: null`; the pre-registration creator and official evaluator cannot promote it.

Official scoring accepts only a `duovla-calvin-official-preregistration-v8` manifest. It freezes all 1,000 canonical
five-subtask sequences, evaluation seed 0 and its inference-seed domain, the runtime/data `attestation_sha256`, the
raw SHA-256 of `aggregate_calvin_official.py`, the exact Python 3.8.20 aggregation runtime, and
the exact 24-cell comparison matrix: training seeds `{0,1,2}` × flow NFE `{1,5,10}` or direct NFE `1` × execution
horizon `{1,4}`. Arbitrary, missing, duplicate, or off-matrix cells are rejected. Each seed/objective pair contributes
one distinct final checkpoint, reused across its NFE/K cells. Every cell also freezes the selected full policy-contract
digest, one common evaluation-only serving-runtime v4 digest across the complete matrix, and one common
grouped-MM/sample-isolated physical-B8 prefix geometry. The manifest
also freezes the discarded synthetic policy warm-up count (default two, minimum one) across every comparison cell.
Two dedicated, existing, empty real directories are frozen as the canonical run and external-claim roots, including
their absolute paths and device/inode identities. The creator derives each cell's sole
`<run-root>/<cell-id>` directory and `<claim-root>/<cell-id>.json` claim; neither path is accepted from the cell input.

Prepare a strict `{"cells": [...]}` input document with all 24 cells. A cell's `policy` object contains
`train_seed`, `objective`, `nfe`, `sampler`, and `inference_seed_behavior`; the creator derives the selected policy's
canonical `identity_sha256` rather than trusting one supplied by the caller. Then seal the matrix using the full JSON
report emitted by the dataset-backed unified evaluator launcher:

```bash
scripts/calvin/run_create_preregistration.sh \
  --cells /read-only/freeze/calvin-cells.json \
  --runtime-attestation /read-only/freeze/calvin-runtime-data-preflight.json \
  --evaluation-seed 0 \
  --policy-warmup-calls 2 \
  --final-freeze-token '<externally-held-token>' \
  --official-output-root /dedicated/empty/calvin-runs \
  --official-claim-root /dedicated/empty/calvin-claims \
  --output /read-only/freeze/calvin-preregistration.json
```

The creator must run under Python 3.8.20. Before importing evaluator code it snapshots itself, the evaluator, IPC
bridge, and preflight sources; after import it binds each imported module path and raw identity to that snapshot,
requires the evaluator snapshot to equal the authenticated preflight report, and verifies every live source through
the publication commit. It similarly holds one stable aggregator source identity. It durably publishes the `.sha256`
companion first and the manifest as the exclusive commit marker. Before and after every commit guard it opens each
target without following symlinks and requires the created inode, exact content, and link count; neither existing file
is overwritten, and a failed publication removes only links still owned by that attempt. Preserve
the manifest's raw digest outside the evaluation host; computing the expected digest from the candidate file at score
time would defeat the freeze gate.

```bash
scripts/calvin/run_official_evaluator.sh official-score \
  --dataset-root "$CALVIN_DATASET_ROOT" \
  --execution-horizon 4 \
  --policy-warmup-calls 2 \
  --socket /absolute/path/calvin-policy.sock \
  --output-dir /dedicated/empty/calvin-runs/<frozen-cell-id> \
  --preregistration-manifest /read-only/frozen-preregistration.json \
  --preregistration-sha256 '<externally-recorded-64hex-digest>' \
  --cell-id '<frozen-cell-id>' \
  --final-freeze-token '<externally-held-token>'
```

The same launcher is the only supported entry point for source-only preflight, dataset-backed infrastructure checks,
and official scoring. This is a fail-closed runtime contract, not an OS-level claim that a caller cannot manually
reconstruct the identical `env -i` invocation; an ordinary direct Python invocation is rejected. The launcher starts
Python 3.8.20, disables user-site and bytecode writes, fixes hash seed,
locale, timezone, and thread settings, and rejects every unlisted environment variable. The measured renderer contract
is `CUDA_VISIBLE_DEVICES=0`, `EGL_VISIBLE_DEVICES=0`, `EGL_PLATFORM=surfaceless`, and `PYOPENGL_PLATFORM=egl`, with
`DISPLAY`, `LD_LIBRARY_PATH`, `LD_PRELOAD`, `NVIDIA_VISIBLE_DEVICES`, and `PYTHONPATH` absent. This exact combination
passed the source-only EGL smoke on the evaluation host; there is no fallback device selection. Set
`CALVIN_SOURCE_ROOT`, `DUO_VLA_CACHE_ROOT`, or `CALVIN_VENV_ROOT` before invoking the launcher when their default paths
do not apply; they are canonicalized before the closed child environment is created.

The evaluator snapshots its own file, the IPC bridge, and preflight before importing either local dependency, then
requires each imported module's `__file__`, import-spec origin, and current raw identity to equal that snapshot. It
requires the recomputed full attestation to describe those exact bytes and rechecks them at infrastructure return,
official journal start/completion, and CLI output boundaries. Completion guards verify the created episode, summary,
and run-file inode, exact content, and link count before and after each source guard; a source or target mismatch after
the durable `run.json` commit rewrites the journal as failed. It performs these checks before loading any
YAML or constructing the oracle/environment. Authenticated YAML and the validation merged config are parsed from the
same stable, non-symlink descriptor bytes that were hashed; the environment is instantiated directly from that config
object, so the upstream path-based loader cannot reopen changed bytes. It authenticates the pre-registration's raw bytes before parsing JSON,
requires the manifest's attestation hash to equal the live hash, and requires live policy health—including
`serving_runtime_sha256`, `execution_geometry`, and the IPC-v4 nested `calvin_identity`—to equal both the selected cell
and the runtime-attested archive-direct dataset identity before the first prediction. The evaluator also requires the
legacy flat `normalization_metadata_sha256` health field to equal `calvin_identity.metadata_sha256`.
Infrastructure mode accepts none of the scoring/freeze arguments and cannot connect to a policy or inspect oracle
outcomes.

Official scoring performs the frozen warm-up calls on deterministic synthetic RGB/state inputs before constructing
the validation-D environment or task oracle. Warm-up requests use reserved replan indices starting at 360, outside the
official per-subtask action budget, so they cannot collide with a scored request identity. Before constructing a policy
client, the evaluator requires the exact pre-registered output path, rechecks both root inodes, exclusively creates the
canonical run directory, and exclusively publishes the external claim plus `.sha256` sidecar. An existing run path or
claim is a permanent collision; deleting a failed run directory does not make the surviving external claim reusable.
The evaluator durably records health plus partial warm-up progress,
including each request intent before dispatch and its completed report afterward. Their outputs are discarded and their request identity, input/output
digests, and request/server latency are durably recorded under `policy_warmup` in `run.json`; they are excluded from
all episode latency and throughput fields. `summary.json` reports authenticated episode-only request/server p50 and
p95 latency, total latency, call throughput, rollout elapsed time, and environment-action throughput, all recomputed
from the raw episode JSONL during aggregation.

Successful completion exclusively publishes `completion.json` and `completion.json.sha256` as the final commit marker.
The marker mutually binds the cell, pre-registration, external claim, episode JSONL, summary, final complete run JSON,
and the exact 1,000-record count. The journal schema is `duovla-calvin-official-run-v5`; a running or failed journal,
or a run without this marker, is not complete.

After all 24 cells finish, prepare and externally hash a strict run inventory. Paths are deliberately absent because
the aggregator derives them only from the frozen pre-registration:

```json
{
  "schema": "duovla-calvin-official-run-inventory-v3",
  "preregistration_sha256": "<frozen manifest raw SHA-256>",
  "runs": [{
    "cell_id": "seed-0-flow-nfe-1-k-1",
    "claim_json_sha256": "<64hex>",
    "completion_json_sha256": "<64hex>",
    "run_json_sha256": "<64hex>",
    "episodes_jsonl_sha256": "<64hex>",
    "summary_json_sha256": "<64hex>"
  }]
}
```

The list must contain exactly one entry for every canonical cell. Aggregate only with the independently recorded raw
inventory digest:

```bash
scripts/calvin/run_aggregate_official.sh \
  --preregistration-manifest /read-only/freeze/calvin-preregistration.json \
  --preregistration-sha256 '<externally-recorded-manifest-64hex>' \
  --run-inventory /read-only/freeze/calvin-run-inventory.json \
  --run-inventory-sha256 '<externally-recorded-inventory-64hex>' \
  --output /read-only/results/calvin-official-matrix-summary.json
```

Aggregation must also run under Python 3.8.20. Before importing evaluator code it snapshots the raw identities of
`aggregate_calvin_official.py`, `evaluate_calvin.py`, `calvin_bridge.py`, and `preflight.py`; it binds the aggregator
to preregistration and the other three sources to every run attestation, then verifies the snapshot in both the pre- and
post-commit publication guards. Those guards also bind the exclusive output and SHA-256 sidecar to the created inode,
content, and link count, rejecting target-entry substitution. It authenticates the validation-language YAML and requires every persisted instruction to be exactly
that task's first fixed phrase. It requires the run root to contain exactly the 24 canonical directories and the claim
root to contain exactly 24 claims plus their 24 sidecars, rejecting alternate, reused, missing, extra, symlinked, or
hard-linked paths. It authenticates every raw claim/completion/run/episode/summary artifact, attestation, cell,
live-policy identity, and all 1,000 ordered episode records per cell; recomputes each `summary.json`; and rejects
incomplete, failed, running, duplicate, or off-matrix results. The
`duovla-calvin-official-matrix-summary-v5` output is also an exclusive, durable payload/sidecar
pair. Its eight comparison rows report AvgLen, SR1–SR5, episode-only request/server latency and throughput, and rollout
action throughput as three-seed means and sample standard deviations using the `n-1` denominator while retaining every
per-seed value and artifact digest. The matrix records the frozen warm-up count and requires paired K=1/K=4 warm-up
action hashes to be identical for each seed/objective/NFE.

Passing these protocol and mutation tests is not an official benchmark result. A result claim requires the complete
real 24-cell run inventory and an independently authenticated v5 matrix summary; partial, synthetic, or dry-run
artifacts are never reportable as official CALVIN performance.

Production keeps the verified 555,309,812,705-byte ZIP, a compact SQLite index, and six projected metadata files.
The old extracted v3 parity path still needs approximately 517 GiB after extraction and roughly twice that while
retaining the download; it is not required for official v4 evaluation or archive-direct training.
