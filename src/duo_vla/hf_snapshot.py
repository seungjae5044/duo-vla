"""Content authentication for a pinned Hugging Face snapshot tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _file_digest(path: Path, algorithm: str, *, git_blob_size: int | None = None) -> str:
    digest = hashlib.new(algorithm)
    if git_blob_size is not None:
        digest.update(f"blob {git_blob_size}\0".encode())
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_huggingface_snapshot(
    snapshot_root: str | Path,
    *,
    expected_revision: str,
    optional_tree_files: frozenset[str] = frozenset({".gitattributes"}),
) -> dict[str, Any]:
    """Hash every snapshot file against its pinned Git/LFS tree identity."""

    snapshot = Path(snapshot_root).resolve()
    if snapshot.name != expected_revision or not snapshot.is_dir():
        raise ValueError(f"snapshot must be the pinned revision directory {expected_revision}")
    tree_path = snapshot.parents[1] / "trees" / f"{expected_revision}.json"
    if not tree_path.is_file():
        raise FileNotFoundError(f"snapshot tree metadata is missing: {tree_path}")
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    files = tree.get("files") if isinstance(tree, dict) and tree.get("format_version") == 1 else None
    if not isinstance(files, dict) or not files:
        raise ValueError("snapshot tree metadata has an invalid file inventory")

    observed_all = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()}
    declared = set(files)
    ignored_optional = declared & set(optional_tree_files)
    observed = observed_all - ignored_optional
    expected = declared - ignored_optional
    missing = expected - observed_all
    extra = observed_all - declared
    if missing or extra:
        raise ValueError(
            f"snapshot file inventory differs from the pinned tree: missing={sorted(missing)}, extra={sorted(extra)}"
        )

    records: list[dict[str, Any]] = []
    total_bytes = 0
    for name in sorted(observed):
        entry = files[name]
        if not isinstance(entry, dict):
            raise ValueError(f"snapshot tree entry is invalid: {name}")
        path = snapshot / name
        size = path.stat().st_size
        if "lfs_sha256" in entry:
            expected_size = entry.get("lfs_size")
            expected_digest = entry.get("lfs_sha256")
            algorithm = "sha256"
            observed_digest = _file_digest(path, algorithm)
        else:
            expected_size = entry.get("size")
            expected_digest = entry.get("blob_id")
            algorithm = "git-sha1"
            observed_digest = _file_digest(path, "sha1", git_blob_size=size)
        if type(expected_size) is not int or expected_size < 0 or size != expected_size:
            raise ValueError(f"snapshot file size mismatch: {name}")
        if not isinstance(expected_digest, str) or observed_digest != expected_digest:
            raise ValueError(f"snapshot file content hash mismatch: {name}")
        records.append({"algorithm": algorithm, "bytes": size, "digest": observed_digest, "path": name})
        total_bytes += size

    canonical = json.dumps(records, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return {
        "content_inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "files_verified": len(records),
        "revision": expected_revision,
        "snapshot": str(snapshot),
        "total_bytes": total_bytes,
        "tree_metadata_sha256": _file_digest(tree_path, "sha256"),
    }
