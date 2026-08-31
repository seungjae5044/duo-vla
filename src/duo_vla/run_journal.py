"""Crash-safe run ownership, checkpoint, and metric-history journal.

Only rank zero should mutate a run journal.  Checkpoint directories are immutable:
the journal authenticates the latest one by the SHA-256 of its ``manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

RUN_JOURNAL_FILENAME = "run_journal.json"
RUN_JOURNAL_SCHEMA = "duo-vla-run-journal-v1"
RECOVERY_QUARANTINE_DIRNAME = "recovery_quarantine"
RECOVERY_QUARANTINE_SCHEMA = "duo-vla-recovery-quarantine-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_DIRECTORY = re.compile(r"update-([0-9]{6,})")
_RANK_STATE_DIRECTORY = re.compile(r"\.rank-state-update-([0-9]{6,})")
_RESOLVED_CONFIG_TEMP = re.compile(r"\.resolved_config\.json\.tmp-[0-9]+")
_RUN_JOURNAL_TEMP = re.compile(r"\.run_journal\.json\.tmp-[0-9a-f]{32}")
_JOURNAL_KEYS = {"schema", "run_uuid", "config_sha256", "latest_checkpoint"}
_CHECKPOINT_KEYS = {
    "relative_path",
    "update",
    "manifest_sha256",
    "parent_manifest_sha256",
    "last_metrics",
}


@dataclass(frozen=True)
class CheckpointRecord:
    """Authenticated pointer to one committed, immutable checkpoint."""

    relative_path: str
    update: int
    manifest_sha256: str
    parent_manifest_sha256: str | None
    last_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "update": self.update,
            "manifest_sha256": self.manifest_sha256,
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "last_metrics": _copy_json_object(self.last_metrics, name="last_metrics"),
        }


@dataclass(frozen=True)
class RunJournal:
    """The durable identity and latest committed state of one training run."""

    run_uuid: str
    config_sha256: str
    latest_checkpoint: CheckpointRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUN_JOURNAL_SCHEMA,
            "run_uuid": self.run_uuid,
            "config_sha256": self.config_sha256,
            "latest_checkpoint": (None if self.latest_checkpoint is None else self.latest_checkpoint.to_dict()),
        }


@dataclass(frozen=True)
class MetricsReconciliation:
    """Summary of bringing ``metrics.jsonl`` back to a committed checkpoint."""

    committed_update: int
    retained_entries: int
    truncated_entries: int
    recovered_last_metrics: bool
    discarded_partial_tail: bool = False

    @property
    def changed(self) -> bool:
        return self.truncated_entries > 0 or self.recovered_last_metrics or self.discarded_partial_tail


@dataclass(frozen=True)
class RecoveryQuarantine:
    """Recoverable relocation of one interrupted checkpoint transaction."""

    directory: str | None
    moved_paths: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return bool(self.moved_paths)


def _require_sha256(value: object, *, name: str, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
    return value


def _require_update(value: object, *, name: str = "update") -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("run_uuid must be a canonical UUID string")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError("run_uuid must be a canonical UUID string") from exc
    if str(parsed) != value:
        raise ValueError("run_uuid must be a canonical UUID string")
    return value


def _require_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("checkpoint relative_path must be a canonical POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or value == "." or ".." in path.parts:
        raise ValueError("checkpoint relative_path must remain below the run directory")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _parse_json(text: str, *, source: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ValueError(f"malformed JSON in {source}: {exc}") from exc


def _canonical_json_bytes(value: Any, *, trailing_newline: bool = True) -> bytes:
    try:
        text = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not strict JSON: {exc}") from exc
    if trailing_newline:
        text += "\n"
    return text.encode("utf-8")


def _copy_json_object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    copied = _parse_json(
        _canonical_json_bytes(dict(value), trailing_newline=False).decode("utf-8"),
        source=name,
    )
    if not isinstance(copied, dict):  # pragma: no cover - guarded by Mapping above
        raise ValueError(f"{name} must be a JSON object")
    return copied


def _checkpoint_record_from_dict(value: object) -> CheckpointRecord:
    if not isinstance(value, dict) or set(value) != _CHECKPOINT_KEYS:
        raise ValueError("latest_checkpoint has an unsupported schema")
    update = _require_update(value["update"])
    last_metrics = _copy_json_object(value["last_metrics"], name="last_metrics")
    if _require_update(last_metrics.get("update"), name="last_metrics.update") != update:
        raise ValueError("last_metrics.update does not match checkpoint update")
    manifest_sha256 = _require_sha256(value["manifest_sha256"], name="manifest_sha256")
    parent_sha256 = _require_sha256(
        value["parent_manifest_sha256"],
        name="parent_manifest_sha256",
        allow_none=True,
    )
    assert isinstance(manifest_sha256, str)
    assert parent_sha256 is None or isinstance(parent_sha256, str)
    return CheckpointRecord(
        relative_path=_require_relative_path(value["relative_path"]),
        update=update,
        manifest_sha256=manifest_sha256,
        parent_manifest_sha256=parent_sha256,
        last_metrics=last_metrics,
    )


def _journal_from_dict(value: object) -> RunJournal:
    if not isinstance(value, dict) or set(value) != _JOURNAL_KEYS:
        raise ValueError("run journal has an unsupported schema")
    if value["schema"] != RUN_JOURNAL_SCHEMA:
        raise ValueError("unsupported run journal schema version")
    config_sha256 = _require_sha256(value["config_sha256"], name="config_sha256")
    latest_value = value["latest_checkpoint"]
    latest = None if latest_value is None else _checkpoint_record_from_dict(latest_value)
    assert isinstance(config_sha256, str)
    return RunJournal(
        run_uuid=_require_uuid(value["run_uuid"]),
        config_sha256=config_sha256,
        latest_checkpoint=latest,
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes, *, exclusive: bool, mode: int = 0o600) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _journal_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / RUN_JOURNAL_FILENAME


def _file_mode(path: Path, *, default: int) -> int:
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        return default


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_manifest(checkpoint_dir: Path) -> tuple[str, dict[str, Any]]:
    manifest_path = checkpoint_dir / "manifest.json"
    try:
        data = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read checkpoint manifest: {manifest_path}: {exc}") from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"checkpoint manifest is not UTF-8: {manifest_path}") from exc
    value = _parse_json(text, source=str(manifest_path))
    if not isinstance(value, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    return _sha256_bytes(data), value


def _resolve_checkpoint(output_dir: Path, checkpoint: str | Path) -> tuple[Path, str]:
    output = output_dir.resolve()
    supplied = Path(checkpoint)
    if supplied.is_absolute():
        candidate = supplied.resolve()
    else:
        direct = supplied.resolve()
        try:
            direct.relative_to(output)
        except ValueError:
            candidate = (output / supplied).resolve()
        else:
            candidate = direct
    try:
        relative = candidate.relative_to(output)
    except ValueError as exc:
        raise ValueError("checkpoint path must remain below the run directory") from exc
    relative_path = _require_relative_path(relative.as_posix())
    return candidate, relative_path


def _manifest_last_metrics(manifest: Mapping[str, Any], *, update: int) -> dict[str, Any]:
    metrics = _copy_json_object(manifest.get("last_metrics"), name="checkpoint last_metrics")
    if _require_update(metrics.get("update"), name="checkpoint last_metrics.update") != update:
        raise ValueError("checkpoint last_metrics.update does not match checkpoint update")
    return metrics


def _validate_manifest_run_identity(manifest: Mapping[str, Any], journal: RunJournal) -> None:
    manifest_uuid = _require_uuid(manifest.get("run_uuid"))
    manifest_config = _require_sha256(manifest.get("config_sha256"), name="manifest config_sha256")
    if manifest_uuid != journal.run_uuid:
        raise ValueError("checkpoint manifest run_uuid does not match the run journal")
    if manifest_config != journal.config_sha256:
        raise ValueError("checkpoint manifest config_sha256 does not match the run journal")


def create_run_journal(
    output_dir: str | Path,
    *,
    config_sha256: str,
    run_uuid: str | None = None,
) -> RunJournal:
    """Create a new run journal without ever replacing an existing one."""

    output = Path(output_dir)
    if not output.is_dir():
        raise FileNotFoundError(f"run output directory does not exist: {output}")
    checked_config = _require_sha256(config_sha256, name="config_sha256")
    assert isinstance(checked_config, str)
    journal = RunJournal(
        run_uuid=_require_uuid(str(uuid.uuid4()) if run_uuid is None else run_uuid),
        config_sha256=checked_config,
    )
    _atomic_write(_journal_path(output), _canonical_json_bytes(journal.to_dict()), exclusive=True)
    return journal


def load_run_journal(
    output_dir: str | Path,
    *,
    expected_config_sha256: str | None = None,
) -> RunJournal:
    """Load a journal with exact field, type, UUID, hash, and path validation."""

    path = _journal_path(output_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"run journal is not UTF-8: {path}") from exc
    journal = _journal_from_dict(_parse_json(text, source=str(path)))
    if expected_config_sha256 is not None:
        expected = _require_sha256(expected_config_sha256, name="expected_config_sha256")
        if journal.config_sha256 != expected:
            raise ValueError("run journal configuration hash mismatch")
    return journal


def record_latest_checkpoint(
    output_dir: str | Path,
    *,
    checkpoint: str | Path,
    update: int,
    manifest_sha256: str,
    parent_manifest_sha256: str | None,
    last_metrics: Mapping[str, Any],
) -> RunJournal:
    """Atomically advance the journal after a checkpoint has fully committed.

    The supplied manifest hash and metrics are checked against the immutable
    checkpoint before the journal pointer moves.  Repeating an identical commit
    is idempotent.
    """

    output = Path(output_dir)
    journal = load_run_journal(output)
    checked_update = _require_update(update)
    checked_manifest_sha = _require_sha256(manifest_sha256, name="manifest_sha256")
    checked_parent_sha = _require_sha256(
        parent_manifest_sha256,
        name="parent_manifest_sha256",
        allow_none=True,
    )
    assert isinstance(checked_manifest_sha, str)
    assert checked_parent_sha is None or isinstance(checked_parent_sha, str)
    checked_metrics = _copy_json_object(last_metrics, name="last_metrics")
    if _require_update(checked_metrics.get("update"), name="last_metrics.update") != checked_update:
        raise ValueError("last_metrics.update does not match checkpoint update")

    checkpoint_dir, relative_path = _resolve_checkpoint(output, checkpoint)
    actual_manifest_sha, manifest = _read_manifest(checkpoint_dir)
    if actual_manifest_sha != checked_manifest_sha:
        raise ValueError("checkpoint manifest hash does not match the supplied SHA-256")
    _validate_manifest_run_identity(manifest, journal)
    manifest_metrics = _manifest_last_metrics(manifest, update=checked_update)
    if _canonical_json_bytes(manifest_metrics) != _canonical_json_bytes(checked_metrics):
        raise ValueError("journal last_metrics conflicts with checkpoint manifest")
    if "parent_manifest_sha256" in manifest and manifest["parent_manifest_sha256"] != checked_parent_sha:
        raise ValueError("checkpoint manifest parent SHA conflicts with the journal commit")
    trainer_state = manifest.get("trainer_state")
    if isinstance(trainer_state, Mapping) and trainer_state.get("next_update") != checked_update:
        raise ValueError("checkpoint trainer_state.next_update does not match journal update")

    record = CheckpointRecord(
        relative_path=relative_path,
        update=checked_update,
        manifest_sha256=checked_manifest_sha,
        parent_manifest_sha256=checked_parent_sha,
        last_metrics=checked_metrics,
    )
    previous = journal.latest_checkpoint
    if previous == record:
        return journal
    expected_parent = None if previous is None else previous.manifest_sha256
    if checked_parent_sha != expected_parent:
        raise ValueError("checkpoint parent SHA does not match the journal latest checkpoint")
    if previous is not None:
        if checked_update <= previous.update:
            raise ValueError("checkpoint update must strictly advance the run journal")
        if relative_path == previous.relative_path:
            raise ValueError("a checkpoint path cannot be reused for a later update")

    advanced = RunJournal(
        run_uuid=journal.run_uuid,
        config_sha256=journal.config_sha256,
        latest_checkpoint=record,
    )
    journal_path = _journal_path(output)
    _atomic_write(
        journal_path,
        _canonical_json_bytes(advanced.to_dict()),
        exclusive=False,
        mode=_file_mode(journal_path, default=0o600),
    )
    return advanced


def _validate_resume_checkpoint(
    output_dir: Path,
    selected_checkpoint: str | Path,
) -> tuple[CheckpointRecord, dict[str, Any]]:
    journal = load_run_journal(output_dir)
    record = journal.latest_checkpoint
    if record is None:
        raise ValueError("run journal has no committed checkpoint to resume")
    checkpoint_dir, relative_path = _resolve_checkpoint(output_dir, selected_checkpoint)
    if relative_path != record.relative_path:
        raise ValueError("selected resume checkpoint is not the journal latest checkpoint")
    actual_sha, manifest = _read_manifest(checkpoint_dir)
    if actual_sha != record.manifest_sha256:
        raise ValueError("latest checkpoint manifest hash does not match the run journal")
    _validate_manifest_run_identity(manifest, journal)
    manifest_metrics = _manifest_last_metrics(manifest, update=record.update)
    if _canonical_json_bytes(manifest_metrics) != _canonical_json_bytes(record.last_metrics):
        raise ValueError("latest checkpoint last_metrics does not match the run journal")
    return record, manifest


def validate_resume_checkpoint(
    output_dir: str | Path,
    selected_checkpoint: str | Path,
) -> CheckpointRecord:
    """Authenticate an explicit resume selection against the journal latest."""

    record, _ = _validate_resume_checkpoint(Path(output_dir), selected_checkpoint)
    return record


def quarantine_uncommitted_training_artifacts(
    output_dir: str | Path,
) -> RecoveryQuarantine:
    """Move one interrupted post-journal checkpoint transaction out of replay paths.

    The journal tip is authoritative. A process crash can leave a rank-state staging
    directory and/or checkpoint directory for one later update. They are preserved
    below ``recovery_quarantine`` so replay can safely reuse their original names.
    Multiple later update numbers are rejected as ambiguous rather than moved.
    """

    output = Path(output_dir).resolve()
    journal = load_run_journal(output)
    latest = journal.latest_checkpoint
    committed_update = -1 if latest is None else latest.update
    committed_manifest_sha256 = None if latest is None else latest.manifest_sha256

    candidates: list[tuple[Path, str, int | None]] = []
    locations = (
        (output, _RANK_STATE_DIRECTORY, "root"),
        (output / "checkpoints", _CHECKPOINT_DIRECTORY, "checkpoints"),
    )
    for parent, pattern, destination_group in locations:
        if not parent.exists():
            continue
        if not parent.is_dir() or parent.is_symlink():
            raise ValueError(f"training artifact parent is not a real directory: {parent}")
        for child in parent.iterdir():
            match = pattern.fullmatch(child.name)
            if match is None:
                continue
            if child.is_symlink() or not child.is_dir():
                raise ValueError(f"training artifact is not a real directory: {child}")
            update = int(match.group(1))
            if update > committed_update:
                candidates.append((child, destination_group, update))

    metrics_path = output / "metrics.jsonl"
    if latest is None and metrics_path.exists():
        if metrics_path.is_symlink() or not metrics_path.is_file():
            raise ValueError("uncommitted metrics path is not a real file")
        candidates.append((metrics_path, "root", None))
    if latest is None:
        for child in output.iterdir():
            if not (_RESOLVED_CONFIG_TEMP.fullmatch(child.name) or _RUN_JOURNAL_TEMP.fullmatch(child.name)):
                continue
            if child.is_symlink() or not child.is_file():
                raise ValueError(f"bootstrap temporary artifact is not a real file: {child}")
            candidates.append((child, "root", None))

    updates = {update for _, _, update in candidates if update is not None}
    if len(updates) > 1:
        raise ValueError(f"multiple uncommitted checkpoint updates require manual recovery: {sorted(updates)}")
    if not candidates:
        return RecoveryQuarantine(directory=None, moved_paths=())

    quarantine_root = output / RECOVERY_QUARANTINE_DIRNAME
    quarantine_root.mkdir(mode=0o700, exist_ok=True)
    if quarantine_root.is_symlink() or not quarantine_root.is_dir():
        raise ValueError("recovery quarantine root is not a real directory")
    _fsync_directory(output)
    boundary = "bootstrap" if latest is None else f"{latest.update:06d}"
    quarantine = quarantine_root / f"after-{boundary}-{uuid.uuid4().hex}"
    quarantine.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(quarantine_root)

    destination_groups = sorted({destination_group for _, destination_group, _ in candidates})
    for destination_group in destination_groups:
        (quarantine / destination_group).mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(quarantine)

    planned_paths = [
        source.relative_to(output).as_posix()
        for source, _, _ in sorted(candidates, key=lambda item: item[0].as_posix())
    ]
    recovery_manifest = {
        "schema": RECOVERY_QUARANTINE_SCHEMA,
        "status": "moving",
        "run_uuid": journal.run_uuid,
        "config_sha256": journal.config_sha256,
        "committed_update": committed_update,
        "committed_manifest_sha256": committed_manifest_sha256,
        "moved_paths": planned_paths,
    }
    recovery_path = quarantine / "recovery.json"
    _atomic_write(
        recovery_path,
        _canonical_json_bytes(recovery_manifest),
        exclusive=True,
    )

    moved_paths: list[str] = []
    for source, destination_group, _ in sorted(candidates, key=lambda item: item[0].as_posix()):
        relative_source = source.relative_to(output).as_posix()
        destination_parent = quarantine / destination_group
        destination = destination_parent / source.name
        os.replace(source, destination)
        _fsync_directory(destination_parent)
        _fsync_directory(source.parent)
        moved_paths.append(relative_source)

    recovery_manifest["status"] = "complete"
    recovery_manifest["moved_paths"] = moved_paths
    _atomic_write(
        recovery_path,
        _canonical_json_bytes(recovery_manifest),
        exclusive=False,
        mode=_file_mode(recovery_path, default=0o600),
    )
    return RecoveryQuarantine(
        directory=quarantine.relative_to(output).as_posix(),
        moved_paths=tuple(moved_paths),
    )


def _load_metrics(path: Path) -> tuple[list[dict[str, Any]], bool]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], False
    except UnicodeDecodeError as exc:
        raise ValueError(f"metrics history is not UTF-8: {path}") from exc
    if not text:
        return [], False
    entries: list[dict[str, Any]] = []
    previous_update = -1
    lines = text.splitlines(keepends=True)
    discarded_partial_tail = False
    for offset, raw_line in enumerate(lines):
        line_number = offset + 1
        terminated = raw_line.endswith(("\n", "\r"))
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            raise ValueError(f"malformed metrics history at line {line_number}: blank line")
        try:
            value = _parse_json(line, source=f"{path}:{line_number}")
        except ValueError:
            if offset == len(lines) - 1 and not terminated:
                discarded_partial_tail = True
                break
            raise
        if not isinstance(value, dict):
            raise ValueError(f"malformed metrics history at line {line_number}: expected JSON object")
        update = _require_update(value.get("update"), name=f"metrics line {line_number} update")
        if update <= previous_update:
            raise ValueError("metrics updates must be strictly increasing (no duplicates or reordering)")
        entries.append(value)
        previous_update = update
    return entries, discarded_partial_tail


def reconcile_metrics_jsonl(output_dir: str | Path) -> MetricsReconciliation:
    """Reconcile ``metrics.jsonl`` to the latest authenticated checkpoint.

    Well-formed entries newer than the checkpoint are uncommitted and are
    atomically removed.  If the checkpoint's own metric is absent, its signed
    copy from ``manifest.json`` is restored.  Existing committed data is never
    silently corrected: malformed, duplicate, reordered, or conflicting data
    raises before the file is changed.
    """

    output = Path(output_dir)
    journal = load_run_journal(output)
    if journal.latest_checkpoint is None:
        raise ValueError("run journal has no committed checkpoint for metrics reconciliation")
    selected = output / journal.latest_checkpoint.relative_path
    record, manifest = _validate_resume_checkpoint(output, selected)
    checkpoint_metrics = _manifest_last_metrics(manifest, update=record.update)
    metrics_path = output / "metrics.jsonl"
    entries, discarded_partial_tail = _load_metrics(metrics_path)

    committed = [entry for entry in entries if entry["update"] <= record.update]
    uncommitted = [entry for entry in entries if entry["update"] > record.update]
    matches = [entry for entry in committed if entry["update"] == record.update]
    if matches:
        if _canonical_json_bytes(matches[0]) != _canonical_json_bytes(checkpoint_metrics):
            raise ValueError("committed metrics conflict with checkpoint last_metrics")
        recovered = False
    else:
        committed.append(checkpoint_metrics)
        recovered = True

    reconciliation = MetricsReconciliation(
        committed_update=record.update,
        retained_entries=len(committed),
        truncated_entries=len(uncommitted),
        recovered_last_metrics=recovered,
        discarded_partial_tail=discarded_partial_tail,
    )
    if reconciliation.changed:
        data = b"".join(_canonical_json_bytes(entry) for entry in committed)
        _atomic_write(
            metrics_path,
            data,
            exclusive=False,
            mode=_file_mode(metrics_path, default=0o644),
        )
    return reconciliation
