"""Recursive TOML inheritance and canonical run-configuration hashing."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        inherited = merged.get(key)
        if isinstance(inherited, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(inherited, value)
        else:
            merged[key] = value
    return merged


def load_resolved_toml(path: str | Path) -> dict[str, Any]:
    """Load one config, resolving a relative single-parent ``extends`` chain."""

    return _load_resolved_toml(Path(path).resolve(), stack=())


def _load_resolved_toml(path: Path, *, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"configuration inheritance cycle: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(f"configuration file is missing: {path}")
    with path.open("rb") as handle:
        current = tomllib.load(handle)
    parent_value = current.pop("extends", None)
    if parent_value is None:
        return current
    if not isinstance(parent_value, str) or not parent_value:
        raise ValueError(f"configuration extends must be a nonempty path string: {path}")
    parent = (path.parent / parent_value).resolve()
    inherited = _load_resolved_toml(parent, stack=(*stack, path))
    return _deep_merge(inherited, current)


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    serialized = json.dumps(config, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()


def save_resolved_config(path: str | Path, config: Mapping[str, Any]) -> str:
    """Atomically save canonical resolved JSON and return its semantic content hash."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = canonical_config_sha256(config)
    payload = {"config": dict(config), "config_sha256": digest}
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        descriptor = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def load_verified_resolved_config(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Load a saved resolved config and verify both its envelope and semantic hash."""

    input_path = Path(path)
    try:
        payload = json.loads(
            input_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ValueError(f"resolved configuration is not strict JSON: {input_path}") from exc
    if not isinstance(payload, dict) or set(payload) != {"config", "config_sha256"}:
        raise ValueError("resolved configuration envelope does not exactly match the schema")
    config = payload["config"]
    recorded_sha256 = payload["config_sha256"]
    if not isinstance(config, dict) or not isinstance(recorded_sha256, str) or len(recorded_sha256) != 64:
        raise ValueError("resolved configuration envelope has invalid field types")
    observed_sha256 = canonical_config_sha256(config)
    if recorded_sha256 != observed_sha256:
        raise ValueError("resolved configuration semantic SHA-256 mismatch")
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError("resolved configuration differs from the expected run configuration")
    return config, observed_sha256


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")
