# CALVIN archive-direct backend: Phase A

Phase A defines the standalone authenticated storage layer for the pinned
CALVIN `task_ABC_D.zip`. It does not change the trainer, policy server,
qualification tools, legacy downloader, or the v3 extracted backend. Those
consumers must not use this implementation until the integration phase binds
their run and resume contracts to the v4 identities.

## Publication layout

The preparer keeps the existing logical dataset root while avoiding episode
extraction:

```text
<data-root>/
  task_ABC_D.zip
  task_ABC_D/
    training/ep_start_end_ids.npy
    training/lang_annotations/auto_lang_ann.npy
    training/scene_info.npy
    training/.hydra/merged_config.yaml
    validation/ep_start_end_ids.npy
    validation/.hydra/merged_config.yaml
  task_ABC_D.members-v2.sqlite3
  task_ABC_D.manifest.json
```

No `episode_*.npz` is materialized. The six files are an authenticated
metadata projection for the existing training-root and official-D filesystem
contracts. Official rollout construction functionally reads only
`validation/.hydra/merged_config.yaml`; `validation/ep_start_end_ids.npy`
remains projected because the official data attestation authenticates it. The
pinned official archive contains no `validation/scene_info.npy`; requiring or
synthesizing that path would violate the archive identity.

The manifest is the final commit marker. Preparation refuses every existing
root, v2 index, manifest, or private-stage residue and never relabels a v3
publication. An interrupted root/index publication without a manifest is
intentionally fail-closed and requires an explicit recovery or migration
operation in a later phase. Phase A never deletes an ambiguous stage, source,
destination, or lock residue.

The data-root path is opened from `/` one component at a time with
`O_DIRECTORY|O_NOFOLLOW`. Every ancestor descriptor and device/inode/mode
binding stays pinned through commit. The prepare lock, archive parent, archive,
private stage, projection, index source, and manifest source are likewise
pinned and rebound at publication checkpoints. Enumeration, publication, and
directory fsyncs are dirfd-relative.

Projection publication uses `renameat2(RENAME_NOREPLACE)`. A manifest is
written to `O_TMPFILE` and linked from its descriptor when the target
filesystem supports it. On filesystems without `O_TMPFILE` (including the
current CALVIN cache filesystem), the fully written, fsynced, unpredictable
named source is atomically moved with `renameat2(RENAME_NOREPLACE)`. The v2
SQLite index uses the same pinned named-source atomic move. Final destinations
must reopen no-follow as the exact expected inode, size, single-link file, and
SHA-256. A manifest becomes committed only after that validation. Failure of a
cleanup or redundant fsync after this point is returned as an explicit
`publication_warnings` outcome, rather than a false preparation failure.

## Authentication sequence

`prepare_archive_direct.py prepare` requires the `.aria2` control file to be
absent. It opens the archive with `O_NOFOLLOW`, requires a regular file with one
hard link, pins its device/inode/stat identity, and computes the pinned full
SHA-256 through that descriptor. Parsing starts only after the exact official
byte length and SHA-256 pass.

The parser then:

1. validates the terminal EOCD or ZIP64 EOCD/locator and rejects multi-disk
   archives, ambiguous trailers, and unaccounted bytes between the central
   directory and trailer;
2. streams the central directory into SQLite without retaining `ZipInfo`
   objects;
3. accepts only ASCII canonical paths below `task_ABC_D/`, regular files,
   unique namespaces, file method DEFLATE, directory method STORED, and
   general-purpose flags exactly zero; every non-root parent directory must be
   declared somewhere in the archive, but declaration order is not semantic
   and the virtual `task_ABC_D/` root entry itself is optional, matching the
   pinned official archive;
4. resolves ZIP64 values in spec order, cross-checks every non-sentinel classic
   EOCD value, requires ZIP64 member *size* fields to declare version-needed 45
   or newer, permits the official archive's central-only large-offset extension
   to retain DEFLATE extraction version 20, and validates the allowed ZIP64,
   extended-timestamp, and Unix UID/GID extra fields;
5. re-reads every local header, requires exact central/local version-needed,
   resolves local ZIP64 sizes, derives
   `data_offset = local_offset + 30 + name_bytes + extra_bytes`, and rejects
   local/central mismatches, overflows, overlaps, or payload ranges entering
   the central directory;
6. reads members in physical offset order using `pread`, demands strict raw
   DEFLATE EOF/input consumption/output length, verifies CRC32, and records a
   SHA-256 of the exact logical member bytes.

The finalized private v2 schema stores central `version_needed`, uses exact
`CREATE TABLE` and index SQL (including the partial-index predicate), and uses
a nullable `state_action_row` with explicit absent-sidecar metadata. Phase A
does not create or trust a compact state/action sidecar. Since no production
v2 index has been published and full-archive prepare remains prohibited during
Phase A, pre-hardening fixture indexes are rejected rather than migrated.

## Runtime session

`CalvinArchiveReader` pins both archive and index using `O_NOFOLLOW`, regular
file, single-link, and full stat identities. It authenticates both whole-file
hashes before connecting to SQLite through
`file:/proc/self/fd/<fd>?mode=ro&immutable=1`, enables query-only mode, and
checks exact SQLite `sqlite_schema` SQL, columns, partial predicates,
application/user versions, and integrity. Metadata integers must be canonical
ASCII decimal, and row semantics derive `(split, global_index)` from the
canonical episode path rather than trusting database columns. Counts and
compressed/uncompressed aggregates are recomputed from all semantic rows and
cross-bound to the full-archive central identity and manifest.

Every member read performs a strict index lookup, revalidates the local header
and derived offset, uses bounded `pread`, and repeats raw-DEFLATE EOF, length,
CRC32, and logical SHA-256 verification before returning bytes. Callers that
load pickle-bearing NumPy metadata must load from those returned authenticated
bytes, never verify a path and reopen it.

`from_manifest` pins the root, archive, and index for the reader lifetime. It
reads all six projected files once through pinned component descriptors,
checks their inventory before and after hashing, and compares the exact bytes
to member bytes decoded from the authenticated archive. Consumers use
`read_authenticated_metadata_bytes()` (or the immutable
`authenticated_metadata_bytes` mapping), so pickle-bearing NumPy loaders never
reopen a verified pathname.

Distributed consumers perform the full 555 GB archive hash exactly once on
rank 0 with `from_manifest`. That authenticated reader may expose its exact
live archive identity (`device`, `inode`, `mode`, single-link count, size,
mtime, and ctime) as a job-local capability. Other ranks may then call
`from_manifest_fast` with that freshly broadcast identity. The fast path skips
only the redundant full-archive hash: it still no-follow opens and pins the
entire path chain, requires the exact same live inode/stat identity, hashes the
authenticated central-directory range and complete SQLite index, validates all
index rows and aggregates, and compares every projected metadata file to its
archive member. This capability is included only in the transient distributed
generation contract; it is not a persistent authenticity credential and must
never be loaded from a checkpoint, normalization artifact, or user-supplied
file.

The reader exposes physical-order frame records for the later normalization
integration and revalidates each yielded row's local header/data offset. The
integration must place values into canonical global-index rows before quantiles
or ordered reductions.

## Commands

The Phase A CLI is deliberately separate from the running legacy downloader:

```bash
python scripts/calvin/prepare_archive_direct.py print-contract
python scripts/calvin/prepare_archive_direct.py prepare \
  --data-root /root/.cache/duo-vla/data/calvin
python scripts/calvin/prepare_archive_direct.py verify \
  --data-root /root/.cache/duo-vla/data/calvin
```

The CLI deliberately has no `--archive` override: both commands use the exact
`<data-root>/task_ABC_D.zip` binding recorded by the manifest, and `verify`
reports the path held by the verified archive descriptor.

Do not run `prepare` until the archive download has completed, the `.aria2`
file is gone, and the integration owner has approved publication. Phase A does
not support partial-index resume: a private staging failure is retained
fail-closed for explicit inspection/recovery, and a fresh authenticated scan
requires removal by a separate, identity-aware recovery procedure. Download
resume remains the downloader's responsibility.

## Deferred integration

The next phase must explicitly update dataset loading, normalization,
dev-state generation, qualification, training/resume contracts, serving
preflight, reproducibility comparison, official attestation, and their source
identity inventories. A v3 checkpoint or normalization artifact must not be
silently resumed under v4. The existing positional
`<data-root>/task_ABC_D/training` interface can remain unchanged because the
v4 manifest, rather than frame-path existence, selects the backend.
