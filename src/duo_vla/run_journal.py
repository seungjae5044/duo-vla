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

from duo_vla.run_config import load_verified_resolved_config

RUN_JOURNAL_FILENAME = "run_journal.json"
RUN_JOURNAL_SCHEMA = "duo-vla-run-journal-v1"
RECOVERY_QUARANTINE_DIRNAME = "recovery_quarantine"
RECOVERY_QUARANTINE_SCHEMA = "duo-vla-recovery-quarantine-v1"
CHECKPOINT_RETENTION_DIRNAME = "checkpoint_retention"
CHECKPOINT_RETIREMENT_SCHEMA = "duo-vla-checkpoint-retirement-v1"
CHECKPOINT_RETENTION_CONTRACT_SCHEMA = "duo-vla-checkpoint-retention-contract-v1"
CHECKPOINT_RETIRED_MANIFEST_FILENAME = "checkpoint-manifest.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_DIRECTORY = re.compile(r"update-([0-9]{6,})")
_RETIREMENT_DIRECTORY = re.compile(r"retire-update-([0-9]{6,})-([0-9a-f]{32})")
_RANK_STATE_DIRECTORY = re.compile(r"\.rank-state-update-([0-9]{6,})")
_RESOLVED_CONFIG_TEMP = re.compile(r"\.resolved_config\.json\.tmp-[0-9]+")
_RUN_JOURNAL_TEMP = re.compile(r"\.run_journal\.json\.tmp-[0-9a-f]{32}")
_RETIREMENT_MANIFEST_TEMP = re.compile(r"\.retirement\.json\.tmp-[0-9a-f]{32}")
_RETIRED_CHECKPOINT_MANIFEST_TEMP = re.compile(r"\.checkpoint-manifest\.json\.tmp-[0-9a-f]{32}")
_JOURNAL_KEYS = {"schema", "run_uuid", "config_sha256", "latest_checkpoint"}
_CHECKPOINT_KEYS = {
    "relative_path",
    "update",
    "manifest_sha256",
    "parent_manifest_sha256",
    "last_metrics",
}
_RETIREMENT_KEYS = {
    "schema",
    "status",
    "run_uuid",
    "config_sha256",
    "checkpoint",
    "authorized_tip",
    "directory_device",
    "directory_inode",
    "permanent_checkpoint_interval",
}
_RETENTION_CONTRACT_KEYS = {"schema", "permanent_checkpoint_interval", "parent_checkpoint"}
_RETENTION_PARENT_KEYS = {"relative_path", "update"}
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | os.O_NONBLOCK
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


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


@dataclass(frozen=True)
class CheckpointRetention:
    """Result of recovering and enforcing bounded checkpoint retention."""

    retired_paths: tuple[str, ...]
    recovered_transactions: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return bool(self.retired_paths or self.recovered_transactions)


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
    if "checkpoint_retention" in manifest:
        _retention_contract_from_manifest(
            manifest,
            expected_parent=previous,
            expected_interval=_authenticated_retention_interval(output.resolve(), journal),
        )
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


def _require_positive_interval(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _checkpoint_relative_path(update: int) -> str:
    return f"checkpoints/update-{update:06d}"


def _retention_parent_from_dict(value: object) -> dict[str, str | int]:
    if not isinstance(value, dict) or set(value) != _RETENTION_PARENT_KEYS:
        raise ValueError("checkpoint retention parent has an unsupported schema")
    update = _require_update(value["update"], name="checkpoint retention parent update")
    relative_path = _require_relative_path(value["relative_path"])
    if relative_path != _checkpoint_relative_path(update):
        raise ValueError("checkpoint retention parent path is not canonical for its update")
    return {
        "relative_path": relative_path,
        "update": update,
    }


def make_checkpoint_retention_contract(
    *,
    permanent_checkpoint_interval: int,
    parent_checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the manifest contract that authenticates the checkpoint parent.

    The returned value is embedded in the immutable child checkpoint manifest.
    It preserves enough information to prove whether a missing parent was
    eligible for retirement after its directory has been removed.
    """

    interval = _require_positive_interval(
        permanent_checkpoint_interval,
        name="permanent_checkpoint_interval",
    )
    parent = None if parent_checkpoint is None else _retention_parent_from_dict(dict(parent_checkpoint))
    return {
        "schema": CHECKPOINT_RETENTION_CONTRACT_SCHEMA,
        "permanent_checkpoint_interval": interval,
        "parent_checkpoint": parent,
    }


def _retention_contract_from_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_parent: CheckpointRecord | None,
    expected_interval: int | None = None,
    validate_parent: bool = True,
) -> dict[str, Any]:
    value = manifest.get("checkpoint_retention")
    if not isinstance(value, dict) or set(value) != _RETENTION_CONTRACT_KEYS:
        raise ValueError("checkpoint manifest has no exact checkpoint retention contract")
    if value["schema"] != CHECKPOINT_RETENTION_CONTRACT_SCHEMA:
        raise ValueError("checkpoint manifest retention contract schema is unsupported")
    interval = _require_positive_interval(
        value["permanent_checkpoint_interval"],
        name="checkpoint manifest permanent_checkpoint_interval",
    )
    if expected_interval is not None and interval != expected_interval:
        raise ValueError("checkpoint manifest retention interval differs from the authenticated resolved config")
    parent_value = value["parent_checkpoint"]
    parent = None if parent_value is None else _retention_parent_from_dict(parent_value)
    expected = (
        None
        if expected_parent is None
        else {
            "relative_path": expected_parent.relative_path,
            "update": expected_parent.update,
        }
    )
    if validate_parent and parent != expected:
        raise ValueError("checkpoint manifest retention parent differs from the run-journal parent")
    return {
        "schema": CHECKPOINT_RETENTION_CONTRACT_SCHEMA,
        "permanent_checkpoint_interval": interval,
        "parent_checkpoint": parent,
    }


def _authenticated_retention_interval(output: Path, journal: RunJournal) -> int:
    config, _ = load_verified_resolved_config(
        output / "resolved_config.json",
        expected_sha256=journal.config_sha256,
    )
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("authenticated resolved config has no training table")
    checkpoint_interval = _require_positive_interval(
        training.get("checkpoint_interval"),
        name="training.checkpoint_interval",
    )
    permanent_interval = _require_positive_interval(
        training.get("permanent_checkpoint_interval"),
        name="training.permanent_checkpoint_interval",
    )
    if permanent_interval % checkpoint_interval:
        raise ValueError("authenticated permanent checkpoint interval must be a multiple of checkpoint_interval")
    return permanent_interval


def _retirement_manifest_from_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _RETIREMENT_KEYS:
        raise ValueError("checkpoint retirement manifest has an unsupported schema")
    if value["schema"] != CHECKPOINT_RETIREMENT_SCHEMA:
        raise ValueError("unsupported checkpoint retirement schema version")
    if value["status"] not in {"planned", "moved", "complete"}:
        raise ValueError("checkpoint retirement status is invalid")
    run_uuid = _require_uuid(value["run_uuid"])
    config_sha256 = _require_sha256(value["config_sha256"], name="retirement config_sha256")
    checkpoint = _checkpoint_record_from_dict(value["checkpoint"])
    authorized_tip = _checkpoint_record_from_dict(value["authorized_tip"])
    permanent_checkpoint_interval = _require_positive_interval(
        value["permanent_checkpoint_interval"],
        name="retirement permanent_checkpoint_interval",
    )
    device = value["directory_device"]
    inode = value["directory_inode"]
    if type(device) is not int or device < 0 or type(inode) is not int or inode <= 0:
        raise ValueError("checkpoint retirement directory identity is invalid")
    assert isinstance(config_sha256, str)
    return {
        "schema": CHECKPOINT_RETIREMENT_SCHEMA,
        "status": value["status"],
        "run_uuid": run_uuid,
        "config_sha256": config_sha256,
        "checkpoint": checkpoint,
        "authorized_tip": authorized_tip,
        "directory_device": device,
        "directory_inode": inode,
        "permanent_checkpoint_interval": permanent_checkpoint_interval,
    }


def _retirement_manifest_to_dict(value: Mapping[str, Any], *, status: str | None = None) -> dict[str, Any]:
    checkpoint = value["checkpoint"]
    authorized_tip = value["authorized_tip"]
    if not isinstance(checkpoint, CheckpointRecord) or not isinstance(authorized_tip, CheckpointRecord):
        raise TypeError("checkpoint retirement records are invalid")
    result = {
        "schema": CHECKPOINT_RETIREMENT_SCHEMA,
        "status": value["status"] if status is None else status,
        "run_uuid": value["run_uuid"],
        "config_sha256": value["config_sha256"],
        "checkpoint": checkpoint.to_dict(),
        "authorized_tip": authorized_tip.to_dict(),
        "directory_device": value["directory_device"],
        "directory_inode": value["directory_inode"],
        "permanent_checkpoint_interval": value["permanent_checkpoint_interval"],
    }
    return _retirement_manifest_from_dict(result) | {
        "checkpoint": checkpoint,
        "authorized_tip": authorized_tip,
    }


def _write_retirement_manifest(path: Path, value: Mapping[str, Any], *, exclusive: bool) -> None:
    serialized = _retirement_manifest_to_dict(value)
    payload = {
        **serialized,
        "checkpoint": serialized["checkpoint"].to_dict(),
        "authorized_tip": serialized["authorized_tip"].to_dict(),
    }
    _atomic_write(
        path,
        _canonical_json_bytes(payload),
        exclusive=exclusive,
        mode=_file_mode(path, default=0o600),
    )


def _read_retirement_manifest(path: Path) -> dict[str, Any]:
    try:
        data = _read_private_regular_file(path, context="checkpoint retirement manifest")
    except FileNotFoundError:
        raise
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"checkpoint retirement manifest is not UTF-8: {path}") from exc
    return _retirement_manifest_from_dict(_parse_json(text, source=str(path)))


def _validate_retirement_transaction_entries(transaction: Path) -> None:
    allowed_names = {"retirement.json", CHECKPOINT_RETIRED_MANIFEST_FILENAME, "checkpoint"}
    for child in transaction.iterdir():
        if child.name in allowed_names:
            continue
        if (
            _RETIREMENT_MANIFEST_TEMP.fullmatch(child.name) is None
            and _RETIRED_CHECKPOINT_MANIFEST_TEMP.fullmatch(child.name) is None
        ):
            raise ValueError(f"checkpoint retirement transaction contains an unknown entry: {child}")
        observed = os.lstat(child)
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"checkpoint retirement temporary is not a regular file: {child}")


def _real_directory_stat(path: Path, *, context: str) -> os.stat_result:
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"cannot inspect {context}: {path}: {exc}") from exc
    if not stat.S_ISDIR(observed.st_mode):
        raise ValueError(f"{context} is not a real directory: {path}")
    return observed


def _checkpoint_record_at(
    output: Path,
    checkpoint_dir: Path,
    *,
    journal: RunJournal,
    expected_manifest_sha256: str | None = None,
) -> tuple[CheckpointRecord, os.stat_result]:
    directory_stat = _real_directory_stat(checkpoint_dir, context="checkpoint retirement source")
    try:
        relative = checkpoint_dir.relative_to(output).as_posix()
    except ValueError as exc:  # pragma: no cover - caller constructs contained paths
        raise ValueError("checkpoint retirement source escapes the run directory") from exc
    match = _CHECKPOINT_DIRECTORY.fullmatch(checkpoint_dir.name)
    if checkpoint_dir.parent != output / "checkpoints" or match is None:
        raise ValueError("checkpoint retirement source is not a canonical checkpoint directory")
    manifest_path = checkpoint_dir / "manifest.json"
    try:
        manifest_stat = os.lstat(manifest_path)
    except OSError as exc:
        raise ValueError(f"cannot inspect checkpoint manifest: {manifest_path}: {exc}") from exc
    if not stat.S_ISREG(manifest_stat.st_mode) or manifest_stat.st_nlink != 1:
        raise ValueError("checkpoint retirement requires a private regular manifest")
    manifest_sha256, manifest = _read_manifest(checkpoint_dir)
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("checkpoint retirement source manifest hash mismatch")
    _validate_manifest_run_identity(manifest, journal)
    record = _checkpoint_record_from_manifest(
        relative_path=relative,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        journal=journal,
    )
    if record.update != int(match.group(1)):
        raise ValueError("checkpoint directory update differs from its canonical path")
    return record, directory_stat


def _checkpoint_record_from_manifest(
    *,
    relative_path: str,
    manifest_sha256: str,
    manifest: Mapping[str, Any],
    journal: RunJournal,
) -> CheckpointRecord:
    relative = _require_relative_path(relative_path)
    path = PurePosixPath(relative)
    match = _CHECKPOINT_DIRECTORY.fullmatch(path.name)
    if len(path.parts) != 2 or path.parts[0] != "checkpoints" or match is None:
        raise ValueError("checkpoint record path is not a canonical checkpoint directory")
    update = int(match.group(1))
    if relative != _checkpoint_relative_path(update):
        raise ValueError("checkpoint record path is not canonical for its update")
    _validate_manifest_run_identity(manifest, journal)
    metrics = _manifest_last_metrics(manifest, update=update)
    parent_sha256 = _require_sha256(
        manifest.get("parent_manifest_sha256"),
        name="checkpoint parent_manifest_sha256",
        allow_none=True,
    )
    trainer_state = manifest.get("trainer_state")
    if not isinstance(trainer_state, Mapping) or trainer_state.get("next_update") != update:
        raise ValueError("checkpoint trainer_state.next_update does not match its directory update")
    checked_manifest_sha256 = _require_sha256(manifest_sha256, name="checkpoint manifest_sha256")
    assert isinstance(checked_manifest_sha256, str)
    assert parent_sha256 is None or isinstance(parent_sha256, str)
    return CheckpointRecord(
        relative_path=relative,
        update=update,
        manifest_sha256=checked_manifest_sha256,
        parent_manifest_sha256=parent_sha256,
        last_metrics=metrics,
    )


def _same_directory_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(left.st_mode)
        and stat.S_ISDIR(right.st_mode)
        and (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)
    )


def _regular_file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _remove_directory_contents_fd(descriptor: int, *, context: str) -> None:
    """Remove entries through an already-open, no-follow directory capability."""

    for name in sorted(os.listdir(descriptor)):
        before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        child_context = f"{context}/{name}"
        if stat.S_ISDIR(before.st_mode):
            child_descriptor = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=descriptor)
            try:
                opened = os.fstat(child_descriptor)
                if not _same_directory_identity(before, opened):
                    raise ValueError(f"checkpoint retirement directory changed while opening: {child_context}")
                _remove_directory_contents_fd(child_descriptor, context=child_context)
                after_descriptor = os.fstat(child_descriptor)
                after_path = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not _same_directory_identity(opened, after_descriptor) or not _same_directory_identity(
                    opened, after_path
                ):
                    raise ValueError(f"checkpoint retirement directory identity changed: {child_context}")
            finally:
                os.close(child_descriptor)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)
    os.fsync(descriptor)


def _remove_retired_tree(path: Path, *, device: int, inode: int) -> None:
    """Delete only the inode authorized by the retirement transaction.

    The parent and target are opened with ``O_NOFOLLOW`` and all recursive work
    is descriptor-relative.  A replacement at the root name is detected before
    the final ``rmdir`` and is never traversed.
    """

    parent_before = _real_directory_stat(path.parent, context="checkpoint retirement staging parent")
    parent_descriptor = os.open(path.parent, _DIRECTORY_OPEN_FLAGS)
    target_descriptor: int | None = None
    try:
        parent_opened = os.fstat(parent_descriptor)
        if not _same_directory_identity(parent_before, parent_opened):
            raise ValueError("checkpoint retirement staging parent identity changed")
        target_before = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(target_before.st_mode) or (target_before.st_dev, target_before.st_ino) != (device, inode):
            raise ValueError(f"checkpoint retirement staging directory identity changed: {path}")
        target_descriptor = os.open(path.name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor)
        target_opened = os.fstat(target_descriptor)
        if not _same_directory_identity(target_before, target_opened):
            raise ValueError(f"checkpoint retirement staging directory changed while opening: {path}")
        _remove_directory_contents_fd(target_descriptor, context=str(path))
        target_after = os.fstat(target_descriptor)
        path_after = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not _same_directory_identity(target_opened, target_after) or not _same_directory_identity(
            target_opened, path_after
        ):
            raise ValueError(f"checkpoint retirement staging directory identity changed: {path}")
        os.rmdir(path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        if target_descriptor is not None:
            os.close(target_descriptor)
        os.close(parent_descriptor)


def _read_private_regular_file(path: Path, *, context: str) -> bytes:
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"cannot inspect {context}: {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{context} is not a private regular file: {path}")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino) or not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{context} changed while opening: {path}")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after_descriptor = os.fstat(descriptor)
        after_path = os.lstat(path)
        if _regular_file_identity(opened) != _regular_file_identity(after_descriptor) or _regular_file_identity(
            opened
        ) != _regular_file_identity(after_path):
            raise ValueError(f"{context} changed while reading: {path}")
        data = b"".join(blocks)
        if len(data) != opened.st_size:
            raise ValueError(f"{context} size changed while reading: {path}")
        return data
    finally:
        os.close(descriptor)


def _retired_checkpoint_record(
    transaction: Path,
    value: Mapping[str, Any],
    *,
    journal: RunJournal,
) -> CheckpointRecord:
    checkpoint = value["checkpoint"]
    if not isinstance(checkpoint, CheckpointRecord):
        raise TypeError("checkpoint retirement record is invalid")
    path = transaction / CHECKPOINT_RETIRED_MANIFEST_FILENAME
    data = _read_private_regular_file(path, context="preserved retired checkpoint manifest")
    manifest_sha256 = _sha256_bytes(data)
    if manifest_sha256 != checkpoint.manifest_sha256:
        raise ValueError("preserved retired checkpoint manifest SHA-256 mismatch")
    try:
        manifest = _parse_json(data.decode("utf-8"), source=str(path))
    except UnicodeDecodeError as exc:
        raise ValueError("preserved retired checkpoint manifest is not UTF-8") from exc
    if not isinstance(manifest, dict):
        raise ValueError("preserved retired checkpoint manifest must be a JSON object")
    try:
        authenticated = _checkpoint_record_from_manifest(
            relative_path=checkpoint.relative_path,
            manifest_sha256=manifest_sha256,
            manifest=manifest,
            journal=journal,
        )
    except ValueError as exc:
        raise ValueError(f"preserved retired checkpoint manifest is invalid: {exc}") from exc
    if authenticated != checkpoint:
        raise ValueError("preserved retired checkpoint manifest differs from its retirement record")
    return authenticated


def _preserve_retired_checkpoint_manifest(
    transaction: Path,
    staged: Path,
    *,
    checkpoint: CheckpointRecord,
) -> None:
    source = staged / "manifest.json"
    data = _read_private_regular_file(source, context="retired checkpoint manifest")
    if _sha256_bytes(data) != checkpoint.manifest_sha256:
        raise ValueError("retired checkpoint manifest changed before preservation")
    destination = transaction / CHECKPOINT_RETIRED_MANIFEST_FILENAME
    if os.path.lexists(destination):
        if _read_private_regular_file(destination, context="preserved retired checkpoint manifest") != data:
            raise ValueError("preserved retired checkpoint manifest bytes changed")
        return
    _atomic_write(destination, data, exclusive=True)


def _cleanup_retirement_temporaries(transaction: Path) -> None:
    changed = False
    for child in transaction.iterdir():
        if (
            _RETIREMENT_MANIFEST_TEMP.fullmatch(child.name) is None
            and _RETIRED_CHECKPOINT_MANIFEST_TEMP.fullmatch(child.name) is None
        ):
            continue
        observed = os.lstat(child)
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"checkpoint retirement temporary is not a regular file: {child}")
        child.unlink()
        changed = True
    if changed:
        _fsync_directory(transaction)


def _retention_root(output: Path, *, create: bool) -> Path | None:
    root = output / CHECKPOINT_RETENTION_DIRNAME
    if create:
        root.mkdir(mode=0o700, exist_ok=True)
        _fsync_directory(output)
    elif not os.path.lexists(root):
        return None
    _real_directory_stat(root, context="checkpoint retention root")
    return root


def _retirement_authorized(
    value: Mapping[str, Any],
    *,
    journal: RunJournal,
    permanent_checkpoint_interval: int,
) -> tuple[CheckpointRecord, CheckpointRecord]:
    if value["run_uuid"] != journal.run_uuid or value["config_sha256"] != journal.config_sha256:
        raise ValueError("checkpoint retirement transaction belongs to another run")
    if value["permanent_checkpoint_interval"] != permanent_checkpoint_interval:
        raise ValueError("checkpoint retirement interval differs from the authenticated resolved config")
    checkpoint = value["checkpoint"]
    authorized_tip = value["authorized_tip"]
    if not isinstance(checkpoint, CheckpointRecord) or not isinstance(authorized_tip, CheckpointRecord):
        raise TypeError("checkpoint retirement records are invalid")
    if checkpoint.update % permanent_checkpoint_interval == 0:
        raise ValueError("a permanent checkpoint cannot be retired")
    if authorized_tip.update <= checkpoint.update:
        raise ValueError("checkpoint retirement tip does not supersede its checkpoint")
    if authorized_tip.parent_manifest_sha256 != checkpoint.manifest_sha256:
        raise ValueError("checkpoint retirement is not authorized by the next journal tip")
    return checkpoint, authorized_tip


def _finish_retirement_transaction(
    output: Path,
    transaction: Path,
    value: dict[str, Any],
    *,
    journal: RunJournal,
    permanent_checkpoint_interval: int,
) -> bool:
    checkpoint, authorized_tip = _retirement_authorized(
        value,
        journal=journal,
        permanent_checkpoint_interval=permanent_checkpoint_interval,
    )
    source = output / checkpoint.relative_path
    staged = transaction / "checkpoint"
    source_exists = os.path.lexists(source)
    staged_exists = os.path.lexists(staged)
    if value["status"] == "complete":
        if source_exists or staged_exists:
            raise ValueError("completed checkpoint retirement still has a live checkpoint path")
        _retired_checkpoint_record(transaction, value, journal=journal)
        _cleanup_retirement_temporaries(transaction)
        return False
    if journal.latest_checkpoint != authorized_tip:
        raise ValueError("pending checkpoint retirement is not authorized by the current journal tip")
    if source_exists and staged_exists:
        raise ValueError("checkpoint retirement has both source and staging paths")
    if value["status"] == "planned" and not source_exists and not staged_exists:
        raise ValueError("planned checkpoint retirement lost both source and staging paths")
    if value["status"] == "moved" and source_exists:
        raise ValueError("moved checkpoint retirement unexpectedly retained its source path")
    device = int(value["directory_device"])
    inode = int(value["directory_inode"])
    if source_exists:
        authenticated, observed = _checkpoint_record_at(
            output,
            source,
            journal=journal,
            expected_manifest_sha256=checkpoint.manifest_sha256,
        )
        if authenticated != checkpoint or (observed.st_dev, observed.st_ino) != (device, inode):
            raise ValueError("checkpoint retirement source identity changed")
        os.replace(source, staged)
        _fsync_directory(transaction)
        _fsync_directory(source.parent)
        staged_exists = True
    if staged_exists:
        if os.path.lexists(staged / "manifest.json"):
            staged_record, staged_stat = _checkpoint_record_in_staging(
                staged,
                checkpoint=checkpoint,
                journal=journal,
                expected_device=device,
                expected_inode=inode,
            )
            _preserve_retired_checkpoint_manifest(transaction, staged, checkpoint=checkpoint)
        else:
            staged_stat = _real_directory_stat(staged, context="checkpoint retirement staging")
            if (staged_stat.st_dev, staged_stat.st_ino) != (device, inode):
                raise ValueError("checkpoint retirement staging directory identity changed")
            staged_record = _retired_checkpoint_record(transaction, value, journal=journal)
        if staged_record != checkpoint or (staged_stat.st_dev, staged_stat.st_ino) != (device, inode):
            raise ValueError("checkpoint retirement staging identity changed")
        value = {**value, "status": "moved"}
        _write_retirement_manifest(transaction / "retirement.json", value, exclusive=False)
        _remove_retired_tree(staged, device=device, inode=inode)
    else:
        _retired_checkpoint_record(transaction, value, journal=journal)
    _cleanup_retirement_temporaries(transaction)
    value = {**value, "status": "complete"}
    _write_retirement_manifest(transaction / "retirement.json", value, exclusive=False)
    return True


def _checkpoint_record_in_staging(
    staged: Path,
    *,
    checkpoint: CheckpointRecord,
    journal: RunJournal,
    expected_device: int,
    expected_inode: int,
) -> tuple[CheckpointRecord, os.stat_result]:
    observed = _real_directory_stat(staged, context="checkpoint retirement staging")
    if (observed.st_dev, observed.st_ino) != (expected_device, expected_inode):
        raise ValueError("checkpoint retirement staging directory identity changed")
    data = _read_private_regular_file(staged / "manifest.json", context="checkpoint retirement staging manifest")
    manifest_sha256 = _sha256_bytes(data)
    try:
        manifest = _parse_json(data.decode("utf-8"), source=str(staged / "manifest.json"))
    except UnicodeDecodeError as exc:
        raise ValueError("checkpoint retirement staging manifest is not UTF-8") from exc
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint retirement staging manifest must be a JSON object")
    record = _checkpoint_record_from_manifest(
        relative_path=checkpoint.relative_path,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        journal=journal,
    )
    return record, observed


def _recover_checkpoint_retirements(
    output: Path,
    *,
    journal: RunJournal,
    permanent_checkpoint_interval: int,
) -> tuple[str, ...]:
    root = _retention_root(output, create=False)
    if root is None:
        return ()
    recovered: list[str] = []
    for transaction in sorted(root.iterdir(), key=lambda path: path.name):
        match = _RETIREMENT_DIRECTORY.fullmatch(transaction.name)
        if match is None or transaction.is_symlink() or not transaction.is_dir():
            raise ValueError(f"checkpoint retention root contains an unknown entry: {transaction}")
        manifest_path = transaction / "retirement.json"
        try:
            value = _read_retirement_manifest(manifest_path)
        except FileNotFoundError:
            children = list(transaction.iterdir())
            if any(child.name in {"checkpoint", CHECKPOINT_RETIRED_MANIFEST_FILENAME} for child in children) or any(
                _RETIREMENT_MANIFEST_TEMP.fullmatch(child.name) is None
                and _RETIRED_CHECKPOINT_MANIFEST_TEMP.fullmatch(child.name) is None
                for child in children
            ):
                raise ValueError(f"incomplete checkpoint retirement requires manual recovery: {transaction}") from None
            observed = _real_directory_stat(transaction, context="empty checkpoint retirement transaction")
            _remove_retired_tree(transaction, device=observed.st_dev, inode=observed.st_ino)
            recovered.append(transaction.relative_to(output).as_posix())
            continue
        _validate_retirement_transaction_entries(transaction)
        checkpoint = value["checkpoint"]
        if not isinstance(checkpoint, CheckpointRecord) or int(match.group(1)) != checkpoint.update:
            raise ValueError("checkpoint retirement directory update differs from its manifest")
        if _finish_retirement_transaction(
            output,
            transaction,
            value,
            journal=journal,
            permanent_checkpoint_interval=permanent_checkpoint_interval,
        ):
            recovered.append(transaction.relative_to(output).as_posix())
    return tuple(recovered)


def _completed_retirement_for_parent(
    output: Path,
    *,
    journal: RunJournal,
    tip: CheckpointRecord,
    parent_anchor: Mapping[str, Any],
    permanent_checkpoint_interval: int,
) -> CheckpointRecord | None:
    root = _retention_root(output, create=False)
    if root is None:
        return None
    matches: list[CheckpointRecord] = []
    for transaction in root.iterdir():
        if _RETIREMENT_DIRECTORY.fullmatch(transaction.name) is None or not transaction.is_dir():
            raise ValueError(f"checkpoint retention root contains an unknown entry: {transaction}")
        try:
            value = _read_retirement_manifest(transaction / "retirement.json")
        except FileNotFoundError as exc:
            raise ValueError(f"checkpoint retirement manifest is missing: {transaction}") from exc
        checkpoint = value["checkpoint"]
        if value["status"] != "complete" or not isinstance(checkpoint, CheckpointRecord):
            continue
        _retirement_authorized(
            value,
            journal=journal,
            permanent_checkpoint_interval=permanent_checkpoint_interval,
        )
        authenticated = _retired_checkpoint_record(transaction, value, journal=journal)
        if (
            value["authorized_tip"] == tip
            and authenticated.relative_path == parent_anchor["relative_path"]
            and authenticated.update == parent_anchor["update"]
            and authenticated.manifest_sha256 == tip.parent_manifest_sha256
        ):
            matches.append(authenticated)
    if len(matches) > 1:
        raise ValueError("multiple completed retirements claim the current journal parent")
    return None if not matches else matches[0]


def _active_checkpoint_records(
    output: Path,
    *,
    journal: RunJournal,
) -> tuple[CheckpointRecord, ...]:
    checkpoints = output / "checkpoints"
    if not checkpoints.exists():
        return ()
    _real_directory_stat(checkpoints, context="checkpoint parent")
    records: list[CheckpointRecord] = []
    for child in checkpoints.iterdir():
        if _CHECKPOINT_DIRECTORY.fullmatch(child.name) is None:
            continue
        record, _ = _checkpoint_record_at(output, child, journal=journal)
        records.append(record)
    hashes = [record.manifest_sha256 for record in records]
    updates = [record.update for record in records]
    if len(set(hashes)) != len(hashes) or len(set(updates)) != len(updates):
        raise ValueError("active checkpoint inventory contains duplicate hashes or updates")
    return tuple(records)


def _validate_active_checkpoint_bound(
    records: tuple[CheckpointRecord, ...],
    *,
    tip: CheckpointRecord,
    parent_anchor: Mapping[str, Any] | None,
    permanent_checkpoint_interval: int,
    allow_retiring_parent: bool,
) -> CheckpointRecord | None:
    tip_matches = [record for record in records if record.manifest_sha256 == tip.manifest_sha256]
    if tip_matches != [tip]:
        raise ValueError("active checkpoint inventory does not contain the exact journal tip")
    parent: CheckpointRecord | None = None
    for record in records:
        if record == tip:
            continue
        if record.update >= tip.update:
            raise ValueError("active checkpoint inventory contains an uncommitted future or peer checkpoint")
        is_parent = (
            parent_anchor is not None
            and record.relative_path == parent_anchor["relative_path"]
            and record.update == parent_anchor["update"]
            and record.manifest_sha256 == tip.parent_manifest_sha256
        )
        if is_parent:
            parent = record
        if record.update % permanent_checkpoint_interval and not (allow_retiring_parent and is_parent):
            raise ValueError("active checkpoint inventory contains a non-permanent backlog checkpoint")
    return parent


def _begin_checkpoint_retirement(
    output: Path,
    *,
    journal: RunJournal,
    checkpoint: CheckpointRecord,
    permanent_checkpoint_interval: int,
) -> str:
    tip = journal.latest_checkpoint
    if tip is None:
        raise ValueError("checkpoint retention requires a committed journal tip")
    value = {
        "schema": CHECKPOINT_RETIREMENT_SCHEMA,
        "status": "planned",
        "run_uuid": journal.run_uuid,
        "config_sha256": journal.config_sha256,
        "checkpoint": checkpoint,
        "authorized_tip": tip,
        "directory_device": -1,
        "directory_inode": -1,
        "permanent_checkpoint_interval": permanent_checkpoint_interval,
    }
    _retirement_authorized(
        value,
        journal=journal,
        permanent_checkpoint_interval=permanent_checkpoint_interval,
    )
    source = output / checkpoint.relative_path
    authenticated, observed = _checkpoint_record_at(
        output,
        source,
        journal=journal,
        expected_manifest_sha256=checkpoint.manifest_sha256,
    )
    if authenticated != checkpoint:
        raise ValueError("checkpoint retirement source record changed")
    value["directory_device"] = observed.st_dev
    value["directory_inode"] = observed.st_ino
    root = _retention_root(output, create=True)
    assert root is not None
    transaction = root / f"retire-update-{checkpoint.update:06d}-{uuid.uuid4().hex}"
    transaction.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(root)
    _write_retirement_manifest(transaction / "retirement.json", value, exclusive=True)
    _finish_retirement_transaction(
        output,
        transaction,
        value,
        journal=journal,
        permanent_checkpoint_interval=permanent_checkpoint_interval,
    )
    return transaction.relative_to(output).as_posix()


def apply_checkpoint_retention(
    output_dir: str | Path,
) -> CheckpointRetention:
    """Retain permanent multiples and the current journal tip.

    Only the checkpoint named by the current tip's authenticated parent hash can
    be retired.  Retirement starts after the new tip is durable, atomically
    moves the superseded directory into a transaction, then removes it with
    descriptor-relative symlink-safe traversal.  Durable transaction records
    let the next invocation finish a crash between any of those steps.
    """

    output = Path(output_dir).resolve()
    journal = load_run_journal(output)
    interval = _authenticated_retention_interval(output, journal)
    tip = journal.latest_checkpoint
    if tip is None:
        return CheckpointRetention(retired_paths=(), recovered_transactions=())
    authenticated_tip, tip_manifest = _validate_resume_checkpoint(output, output / tip.relative_path)
    if authenticated_tip != tip:
        raise ValueError("authenticated checkpoint tip differs from the run journal")
    retention_contract = _retention_contract_from_manifest(
        tip_manifest,
        expected_parent=None,
        expected_interval=interval,
        validate_parent=False,
    )
    parent_anchor = retention_contract["parent_checkpoint"]
    if tip.parent_manifest_sha256 is None:
        if parent_anchor is not None:
            raise ValueError("first checkpoint retention contract unexpectedly names a parent")
    elif parent_anchor is None:
        raise ValueError("checkpoint retention contract omits the journal parent anchor")
    recovered = _recover_checkpoint_retirements(
        output,
        journal=journal,
        permanent_checkpoint_interval=interval,
    )
    active = _active_checkpoint_records(
        output,
        journal=journal,
    )
    parent = _validate_active_checkpoint_bound(
        active,
        tip=tip,
        parent_anchor=parent_anchor,
        permanent_checkpoint_interval=interval,
        allow_retiring_parent=True,
    )
    if parent_anchor is None:
        _validate_active_checkpoint_bound(
            active,
            tip=tip,
            parent_anchor=None,
            permanent_checkpoint_interval=interval,
            allow_retiring_parent=False,
        )
        return CheckpointRetention(retired_paths=(), recovered_transactions=recovered)
    if int(parent_anchor["update"]) % interval == 0:
        if parent is None:
            raise ValueError("permanent journal parent checkpoint is missing")
        return CheckpointRetention(retired_paths=(), recovered_transactions=recovered)
    if parent is None:
        completed = _completed_retirement_for_parent(
            output,
            journal=journal,
            tip=tip,
            parent_anchor=parent_anchor,
            permanent_checkpoint_interval=interval,
        )
        if completed is None:
            raise ValueError("journal parent checkpoint is absent without an authenticated completed retirement")
        return CheckpointRetention(retired_paths=(), recovered_transactions=recovered)
    _begin_checkpoint_retirement(
        output,
        journal=journal,
        checkpoint=parent,
        permanent_checkpoint_interval=interval,
    )
    remaining = _active_checkpoint_records(output, journal=journal)
    _validate_active_checkpoint_bound(
        remaining,
        tip=tip,
        parent_anchor=parent_anchor,
        permanent_checkpoint_interval=interval,
        allow_retiring_parent=False,
    )
    return CheckpointRetention(
        retired_paths=(parent.relative_path,),
        recovered_transactions=recovered,
    )


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
