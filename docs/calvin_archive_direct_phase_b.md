# CALVIN archive-direct backend: Phase B data contracts

Production data loading is selected only by the committed
`task_ABC_D.manifest.json` schema. The v4 schema selects `archive-direct`; the
legacy v3 schema remains available only through explicit fixture/parity flags.
The presence or absence of `episode_*.npz` never selects a backend. The Phase-A
reader authenticates the exact v4 metadata-projection inventory, so an extra
file, directory, symlink, or materialized frame fails closed.

`CalvinNpzDataset` owns one `CalvinArchiveReader` for its complete lifetime.
Rank zero first authenticates the full archive and broadcasts a job-local
archive file identity. Each rank then uses the fast reader path, which skips
only the duplicate 555 GB SHA-256 pass; it retains the no-follow root and file
bindings, exact live inode identity, central-directory digest, full index
digest and semantic validation, and projected-metadata/archive parity. The
live inode identity is part of the ephemeral generation capability, not a
persistent run, checkpoint, normalization, prefix, or G3 identity.

Pickle-bearing episode, scene, and language metadata is loaded through
`BytesIO` from reader-authenticated bytes. V4 construction and sampling never
probe filesystem frame paths. A batch validates every requested anchor before
any member read, decodes both RGB views and state only for distinct anchor
frames, reads only `rel_actions` from future frames, and restores the original
request order. Reader identity checks also run on cache hits, so closing the
dataset or mutating/rebinding its pinned archive or index fails closed.

## Persistent storage identity

The canonical storage payload has schema
`duo-vla-calvin-storage-identity-v1` and contains:

- archive basename, byte length, SHA-256, and URL;
- checksum URL;
- manifest schema, semantic content SHA-256, and raw file SHA-256;
- v2 member-index basename, schema, byte length, and SHA-256;
- the complete member counts/byte totals and member-inventory SHA-256;
- central-directory offset, byte length, entry count, ZIP64 flag, and SHA-256;
- storage mode and reader schema.

`storage_identity_sha256` is the canonical JSON SHA-256 of that payload without
its self-hash. It intentionally excludes device, inode, timestamps, and every
other placement-specific live identity.

## Normalization v4

Production normalization uses
`duo-vla-calvin-abc-to-d-normalization-v4` and requires an authenticated v4
generation. It scans archive records in physical `data_offset` order for
sequential I/O, maps each selected frame into its canonical episode/global
ordinal, and maintains an exact-once bitmap before computing NumPy linear q01
and q99 state percentiles. A synthetic v3/v4 parity test requires identical
split, count, state, and action results despite shuffled physical ZIP order.

The artifact's `dataset` object contains `archive_bytes`, `archive_sha256`,
`storage_mode`, `storage_identity_sha256`, both manifest hashes and its schema,
the member-inventory hash, the nested member-index identity, reader schema,
central-directory hash, and the canonical training-metadata identity. CALVIN
actions are never percentile-normalized: the transform remains
`identity_official_scaled_rel_actions`, including the official binary gripper.
Publication into a pre-existing durable parent directory is no-clobber and
durable: the writer fsyncs a private same-directory file, links it into an
absent destination, fsyncs the directory, and the CLI
strictly reloads the committed artifact against the live authenticated v4
generation before reporting success. The recorded `constant_dimensions` list
must exactly equal `abs(q99 - q01) < 1e-6` in serialized float64 and retain the
same classification after conversion to the runtime float32 tensors.

The generic prefix-geometry artifact is content-addressed and binds only its
measured geometry, exact instruction inventory, ordered cameras, and pinned
model/processor snapshot identity. Persistent v4 storage identity is bound by
the normalization artifact and the G3 fixed-anchor report, not by the prefix
artifact itself. The prefix generator's stdout dataset report is an
informational audit record rather than a durable artifact or an independently
pinned experiment input. None of these persistent artifacts records the
ephemeral archive inode capability.
