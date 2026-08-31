from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from duo_vla.hf_snapshot import verify_huggingface_snapshot


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def _snapshot(tmp_path: Path) -> Path:
    revision = "a" * 40
    model_root = tmp_path / "models--example--model"
    snapshot = model_root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    config = b'{"model":"tiny"}\n'
    weights = b"safe-tensor-bytes"
    (snapshot / "config.json").write_bytes(config)
    (snapshot / "model.safetensors").write_bytes(weights)
    tree = {
        "format_version": 1,
        "files": {
            ".gitattributes": {"blob_id": "0" * 40, "size": 0},
            "config.json": {"blob_id": _git_blob_sha1(config), "size": len(config)},
            "model.safetensors": {
                "blob_id": "1" * 40,
                "lfs_sha256": hashlib.sha256(weights).hexdigest(),
                "lfs_size": len(weights),
                "size": len(weights),
            },
        },
    }
    tree_path = model_root / "trees" / f"{revision}.json"
    tree_path.parent.mkdir()
    tree_path.write_text(json.dumps(tree), encoding="utf-8")
    return snapshot


def test_huggingface_snapshot_hashes_git_and_lfs_content(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)

    report = verify_huggingface_snapshot(snapshot, expected_revision="a" * 40)

    assert report["files_verified"] == 2
    assert report["total_bytes"] == len(b'{"model":"tiny"}\n') + len(b"safe-tensor-bytes")


def test_huggingface_snapshot_allows_declared_optional_tree_file_to_be_present(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    (snapshot / ".gitattributes").write_bytes(b"")

    report = verify_huggingface_snapshot(snapshot, expected_revision="a" * 40)

    assert report["files_verified"] == 2
    assert report["total_bytes"] == len(b'{"model":"tiny"}\n') + len(b"safe-tensor-bytes")


def test_huggingface_snapshot_rejects_same_size_content_drift(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    (snapshot / "model.safetensors").write_bytes(b"unsafe-tensor-byt")

    with pytest.raises(ValueError, match="content hash mismatch"):
        verify_huggingface_snapshot(snapshot, expected_revision="a" * 40)


def test_huggingface_snapshot_rejects_untracked_files(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    (snapshot / "extra.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="file inventory differs"):
        verify_huggingface_snapshot(snapshot, expected_revision="a" * 40)
