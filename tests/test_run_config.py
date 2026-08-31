from __future__ import annotations

import json
from pathlib import Path

import pytest

from duo_vla.run_config import (
    canonical_config_sha256,
    load_resolved_toml,
    load_verified_resolved_config,
    save_resolved_config,
)


def test_recursive_config_merge_and_stable_hash(tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    child = tmp_path / "child.toml"
    base.write_text('[model]\nid="base"\n[train]\nbatch=64\nlr=0.001\n', encoding="utf-8")
    child.write_text('extends="base.toml"\n[model]\nid="child"\n[train]\nbatch=32\n', encoding="utf-8")

    resolved = load_resolved_toml(child)

    assert resolved == {"model": {"id": "child"}, "train": {"batch": 32, "lr": 0.001}}
    assert canonical_config_sha256(resolved) == canonical_config_sha256(
        {"train": {"lr": 0.001, "batch": 32}, "model": {"id": "child"}}
    )
    output = tmp_path / "resolved.json"
    digest = save_resolved_config(output, resolved)
    assert json.loads(output.read_text(encoding="utf-8"))["config_sha256"] == digest
    assert load_verified_resolved_config(output, expected_sha256=digest) == (resolved, digest)


def test_config_cycle_and_missing_parent_are_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first.write_text('extends="second.toml"\n', encoding="utf-8")
    second.write_text('extends="first.toml"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        load_resolved_toml(first)

    missing = tmp_path / "missing.toml"
    missing.write_text('extends="absent.toml"\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing"):
        load_resolved_toml(missing)


def test_saved_resolved_config_is_loaded_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "resolved.json"
    digest = save_resolved_config(path, {"policy": {"objective": "rectified_flow"}})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["policy"]["objective"] = "direct_regression"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic SHA-256 mismatch"):
        load_verified_resolved_config(path)

    path.write_text(
        f'{{"config":{{}},"config":{{}},"config_sha256":"{digest}"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict JSON"):
        load_verified_resolved_config(path)
