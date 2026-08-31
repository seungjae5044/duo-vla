# Training and policy runtime integrity

Final LIBERO and CALVIN runs must use the repository launchers:

- `scripts/run_libero_train.sh`
- `scripts/run_calvin_train.sh`
- `scripts/run_libero_policy_server.sh`
- `scripts/calvin/run_policy_server.sh`

Each launcher replaces the caller environment with an `env -i` allowlist. The
allowlist fixes visible-device order, deterministic math controls, thread
counts, offline model access, UTF-8 locale, UTC timezone, Python safe-path and
no-bytecode behavior, project import roots, and cache roots. Python starts with
`-P -B -X pycache_prefix=/dev/null`, user-site imports disabled, and no
`PYTHONPATH`; runtime validation checks the exact flags and import-path order.
Training seeds are
accepted only through an explicit `--seed 0`, `--seed 1`, or `--seed 2`; an
inherited `PYTHONHASHSEED` is not a training input. Torchrun creates rank state
after this reset, and every worker validates the complete standalone TP=2 rank
contract, including rank/local-rank equality.

Before training records a resolved configuration, it hashes every regular file
and symlink in the live train virtual environment using a stable no-follow
walk. The report binds the exact path inventory, file bytes, symlink targets,
file/symlink counts, total bytes, and Python `.pth` startup-hook inventory.
`sitecustomize` and `usercustomize` modules are forbidden. The report also
binds the resolved external base-Python executable and the complete
base-prefix/standard-library inventory, including the venv-to-base symlink
chain. This is `duo-vla-train-venv-identity-v2`. The complete report and
canonical static-environment hash are part of `execution_environment`;
therefore a resume with changed package code, compiled libraries, symlinks, or
startup hooks has a different run contract.

Real policy serving recomputes the train-venv identity and requires it to equal
the checkpoint identity before model construction. Serving runtime hashes also
include the live venv root digest. The LIBERO latency runtime additionally
binds the driver query, Python and framework binary hashes, logical-to-physical
device mapping, runtime-library versions, platform identity, and deterministic
backend selection. Final preregistration must therefore be generated from the
same immutable serving environment used for scoring.

The LIBERO simulator attestation v3 applies the same content-addressing to the
evaluator venv and its resolved base Python. Expert-replay qualification also
publishes a validator-runtime identity covering its closed process,
distribution records, module origins, site-packages inventory, and project
source hashes, and recomputes the identity immediately before exclusive report
publication.

LIBERO training authenticates the complete pinned Hugging Face dataset snapshot
before constructing the parquet reader. The resolved configuration and every
checkpoint bind these fields:

- `dataset_tree_sha256`
- `dataset_content_inventory_sha256`
- `dataset_files_verified`
- `dataset_total_bytes`

The policy health checkpoint object exposes those fields and the corresponding
model snapshot tree/content/count/byte identities. A changed parquet shard,
extra file, missing file, or changed tree metadata makes the checkpoint
ineligible for real serving.
