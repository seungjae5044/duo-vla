from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import duo_vla.run_journal as run_journal_module
from duo_vla.run_config import canonical_config_sha256, save_resolved_config
from duo_vla.run_journal import (
    RUN_JOURNAL_FILENAME,
    CheckpointRecord,
    apply_checkpoint_retention,
    create_run_journal,
    load_run_journal,
    make_checkpoint_retention_contract,
    quarantine_uncommitted_training_artifacts,
    reconcile_metrics_jsonl,
    record_latest_checkpoint,
    validate_resume_checkpoint,
)

RETENTION_CONFIG = {
    "training": {
        "checkpoint_interval": 1000,
        "permanent_checkpoint_interval": 5000,
    }
}
CONFIG_SHA = canonical_config_sha256(RETENTION_CONFIG)
RUN_UUID = "12345678-1234-4234-8234-123456789abc"
_NO_RETENTION_CONTRACT = object()


def _write_checkpoint(
    output: Path,
    update: int,
    metrics: dict | None = None,
    *,
    parent_sha: str | None = None,
    retention_parent: CheckpointRecord | object | None = _NO_RETENTION_CONTRACT,
) -> tuple[Path, str, dict]:
    checkpoint = output / "checkpoints" / f"update-{update:06d}"
    checkpoint.mkdir(parents=True)
    last_metrics = {"train_loss": 1.0 / update, "update": update} if metrics is None else metrics
    manifest = {
        "config_sha256": CONFIG_SHA,
        "last_metrics": last_metrics,
        "parent_manifest_sha256": parent_sha,
        "run_uuid": RUN_UUID,
        "schema": "test-checkpoint-v1",
        "trainer_state": {"next_update": update},
    }
    if retention_parent is not _NO_RETENTION_CONTRACT:
        parent = retention_parent
        assert parent is None or isinstance(parent, CheckpointRecord)
        manifest["checkpoint_retention"] = make_checkpoint_retention_contract(
            permanent_checkpoint_interval=5000,
            parent_checkpoint=(
                None
                if parent is None
                else {
                    "relative_path": parent.relative_path,
                    "update": parent.update,
                }
            ),
        )
    data = (json.dumps(manifest, allow_nan=False, sort_keys=True) + "\n").encode()
    (checkpoint / "manifest.json").write_bytes(data)
    return checkpoint, hashlib.sha256(data).hexdigest(), last_metrics


def _create_committed_checkpoint(
    output: Path,
    update: int = 2,
    metrics: dict | None = None,
) -> tuple[Path, CheckpointRecord]:
    create_run_journal(output, config_sha256=CONFIG_SHA, run_uuid=RUN_UUID)
    checkpoint, digest, last_metrics = _write_checkpoint(output, update, metrics)
    journal = record_latest_checkpoint(
        output,
        checkpoint=checkpoint,
        update=update,
        manifest_sha256=digest,
        parent_manifest_sha256=None,
        last_metrics=last_metrics,
    )
    assert journal.latest_checkpoint is not None
    return checkpoint, journal.latest_checkpoint


def _write_metrics(path: Path, entries: list[object]) -> None:
    path.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")


def _advance_checkpoint(
    output: Path,
    update: int,
    *,
    parent: CheckpointRecord | None,
) -> CheckpointRecord:
    parent_sha256 = None if parent is None else parent.manifest_sha256
    checkpoint, digest, metrics = _write_checkpoint(
        output,
        update,
        parent_sha=parent_sha256,
        retention_parent=parent,
    )
    journal = record_latest_checkpoint(
        output,
        checkpoint=checkpoint,
        update=update,
        manifest_sha256=digest,
        parent_manifest_sha256=parent_sha256,
        last_metrics=metrics,
    )
    assert journal.latest_checkpoint is not None
    return journal.latest_checkpoint


def _create_retention_run(output: Path) -> None:
    save_resolved_config(output / "resolved_config.json", RETENTION_CONFIG)
    create_run_journal(output, config_sha256=CONFIG_SHA, run_uuid=RUN_UUID)


def test_create_and_strictly_load_run_journal(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()

    created = create_run_journal(output, config_sha256=CONFIG_SHA, run_uuid=RUN_UUID)
    loaded = load_run_journal(output, expected_config_sha256=CONFIG_SHA)

    assert loaded == created
    assert loaded.latest_checkpoint is None
    assert not list(output.glob(f".{RUN_JOURNAL_FILENAME}.tmp-*"))
    with pytest.raises(FileExistsError):
        create_run_journal(output, config_sha256=CONFIG_SHA)
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        load_run_journal(output, expected_config_sha256="b" * 64)


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda value: value.update(schema="future-v2"), "schema version"),
        (lambda value: value.update(extra=True), "unsupported schema"),
        (lambda value: value.update(run_uuid="not-a-uuid"), "canonical UUID"),
        (lambda value: value.update(config_sha256="ABC"), "SHA-256"),
    ],
)
def test_load_rejects_invalid_journal_schema(tmp_path: Path, mutation, match: str) -> None:
    output = tmp_path / "run"
    output.mkdir()
    create_run_journal(output, config_sha256=CONFIG_SHA, run_uuid=RUN_UUID)
    path = output / RUN_JOURNAL_FILENAME
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_run_journal(output)


def test_load_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    path = output / RUN_JOURNAL_FILENAME
    path.write_text(
        '{"schema":"duo-vla-run-journal-v1","schema":"duo-vla-run-journal-v1",'
        f'"run_uuid":"{RUN_UUID}","config_sha256":"{CONFIG_SHA}","latest_checkpoint":null}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_run_journal(output)


def test_record_checkpoint_chain_is_atomic_strict_and_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    create_run_journal(output, config_sha256=CONFIG_SHA, run_uuid=RUN_UUID)
    first, first_sha, first_metrics = _write_checkpoint(output, 2)

    first_journal = record_latest_checkpoint(
        output,
        checkpoint=first,
        update=2,
        manifest_sha256=first_sha,
        parent_manifest_sha256=None,
        last_metrics=first_metrics,
    )
    assert first_journal.latest_checkpoint == CheckpointRecord(
        relative_path="checkpoints/update-000002",
        update=2,
        manifest_sha256=first_sha,
        parent_manifest_sha256=None,
        last_metrics=first_metrics,
    )
    assert (
        record_latest_checkpoint(
            output,
            checkpoint="checkpoints/update-000002",
            update=2,
            manifest_sha256=first_sha,
            parent_manifest_sha256=None,
            last_metrics=first_metrics,
        )
        == first_journal
    )

    second, second_sha, second_metrics = _write_checkpoint(output, 4, parent_sha=first_sha)
    with pytest.raises(ValueError, match="parent SHA"):
        record_latest_checkpoint(
            output,
            checkpoint=second,
            update=4,
            manifest_sha256=second_sha,
            parent_manifest_sha256=None,
            last_metrics=second_metrics,
        )
    advanced = record_latest_checkpoint(
        output,
        checkpoint=second,
        update=4,
        manifest_sha256=second_sha,
        parent_manifest_sha256=first_sha,
        last_metrics=second_metrics,
    )
    assert advanced.latest_checkpoint is not None
    assert advanced.latest_checkpoint.parent_manifest_sha256 == first_sha
    assert advanced.latest_checkpoint.manifest_sha256 == second_sha
    assert not list(output.glob(f".{RUN_JOURNAL_FILENAME}.tmp-*"))


def test_record_rejects_manifest_or_metadata_mismatch_and_escape(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    create_run_journal(output, config_sha256=CONFIG_SHA, run_uuid=RUN_UUID)
    checkpoint, digest, metrics = _write_checkpoint(output, 2)

    with pytest.raises(ValueError, match="supplied SHA-256"):
        record_latest_checkpoint(
            output,
            checkpoint=checkpoint,
            update=2,
            manifest_sha256="f" * 64,
            parent_manifest_sha256=None,
            last_metrics=metrics,
        )
    with pytest.raises(ValueError, match="last_metrics conflicts"):
        record_latest_checkpoint(
            output,
            checkpoint=checkpoint,
            update=2,
            manifest_sha256=digest,
            parent_manifest_sha256=None,
            last_metrics={"train_loss": 999.0, "update": 2},
        )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="below the run directory"):
        record_latest_checkpoint(
            output,
            checkpoint=outside,
            update=2,
            manifest_sha256="f" * 64,
            parent_manifest_sha256=None,
            last_metrics=metrics,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("run_uuid", "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "run_uuid does not match"),
        ("config_sha256", "b" * 64, "config_sha256 does not match"),
    ],
)
def test_record_rejects_checkpoint_from_another_run(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    create_run_journal(output, config_sha256=CONFIG_SHA, run_uuid=RUN_UUID)
    checkpoint, _, metrics = _write_checkpoint(output, 2)
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    data = (json.dumps(manifest, allow_nan=False, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(data)

    with pytest.raises(ValueError, match=match):
        record_latest_checkpoint(
            output,
            checkpoint=checkpoint,
            update=2,
            manifest_sha256=hashlib.sha256(data).hexdigest(),
            parent_manifest_sha256=None,
            last_metrics=metrics,
        )


def test_validate_resume_requires_exact_latest_and_authentic_manifest(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    checkpoint, record = _create_committed_checkpoint(output)
    other, _, _ = _write_checkpoint(output, 3, parent_sha=record.manifest_sha256)

    assert validate_resume_checkpoint(output, checkpoint) == record
    with pytest.raises(ValueError, match="not the journal latest"):
        validate_resume_checkpoint(output, other)

    (checkpoint / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        validate_resume_checkpoint(output, checkpoint)


def test_validate_resume_rechecks_manifest_run_identity(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    checkpoint, _ = _create_committed_checkpoint(output)
    journal_path = output / RUN_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text())
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["run_uuid"] = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    data = (json.dumps(manifest, allow_nan=False, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(data)
    journal["latest_checkpoint"]["manifest_sha256"] = hashlib.sha256(data).hexdigest()
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(ValueError, match="run_uuid does not match"):
        validate_resume_checkpoint(output, checkpoint)


def test_checkpoint_retention_keeps_permanent_multiples_and_current_tip(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_retention_run(output)
    parent = None

    for update in range(1000, 7000, 1000):
        record = _advance_checkpoint(output, update, parent=parent)
        apply_checkpoint_retention(output)
        if update == 1000:
            assert (output / "checkpoints/update-001000").is_dir()
            assert not (output / "checkpoint_retention").exists()
        parent = record

    assert sorted(path.name for path in (output / "checkpoints").iterdir()) == [
        "update-005000",
        "update-006000",
    ]
    assert validate_resume_checkpoint(output, "checkpoints/update-006000").update == 6000
    transactions = sorted((output / "checkpoint_retention").iterdir())
    assert len(transactions) == 4
    assert all(json.loads((path / "retirement.json").read_text())["status"] == "complete" for path in transactions)
    assert all((path / "checkpoint-manifest.json").is_file() for path in transactions)


def test_checkpoint_retention_recovers_crash_after_atomic_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_retention_run(output)
    first = _advance_checkpoint(output, 1000, parent=None)
    _advance_checkpoint(output, 2000, parent=first)

    real_remove = run_journal_module._remove_retired_tree

    def fail_after_move(*_args, **_kwargs) -> None:
        raise OSError("simulated crash after retirement move")

    monkeypatch.setattr(run_journal_module, "_remove_retired_tree", fail_after_move)
    with pytest.raises(OSError, match="simulated crash"):
        apply_checkpoint_retention(output)
    assert not (output / "checkpoints/update-001000").exists()
    transaction = next((output / "checkpoint_retention").iterdir())
    assert (transaction / "checkpoint").is_dir()

    monkeypatch.setattr(run_journal_module, "_remove_retired_tree", real_remove)
    recovered = apply_checkpoint_retention(output)

    assert recovered.retired_paths == ()
    assert recovered.recovered_transactions == (transaction.relative_to(output).as_posix(),)
    assert not (transaction / "checkpoint").exists()
    assert json.loads((transaction / "retirement.json").read_text())["status"] == "complete"
    assert (output / "checkpoints/update-002000").is_dir()


def test_checkpoint_retention_recovers_crash_after_delete_before_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_retention_run(output)
    first = _advance_checkpoint(output, 1000, parent=None)
    _advance_checkpoint(output, 2000, parent=first)
    real_write = run_journal_module._write_retirement_manifest

    def fail_complete(path: Path, value: dict, *, exclusive: bool) -> None:
        if value["status"] == "complete":
            raise OSError("simulated crash before retirement completion record")
        real_write(path, value, exclusive=exclusive)

    monkeypatch.setattr(run_journal_module, "_write_retirement_manifest", fail_complete)
    with pytest.raises(OSError, match="simulated crash"):
        apply_checkpoint_retention(output)
    transaction = next((output / "checkpoint_retention").iterdir())
    assert not (output / "checkpoints/update-001000").exists()
    assert not (transaction / "checkpoint").exists()
    assert json.loads((transaction / "retirement.json").read_text())["status"] == "moved"

    monkeypatch.setattr(run_journal_module, "_write_retirement_manifest", real_write)
    recovered = apply_checkpoint_retention(output)

    assert recovered.retired_paths == ()
    assert recovered.recovered_transactions == (transaction.relative_to(output).as_posix(),)
    assert json.loads((transaction / "retirement.json").read_text())["status"] == "complete"
    assert (output / "checkpoints/update-002000").is_dir()


def test_checkpoint_retention_never_deletes_unauthenticated_parent(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_retention_run(output)
    first = _advance_checkpoint(output, 1000, parent=None)
    _advance_checkpoint(output, 2000, parent=first)
    manifest = output / "checkpoints/update-001000/manifest.json"
    manifest.write_text(manifest.read_text() + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="non-permanent backlog"):
        apply_checkpoint_retention(output)

    assert (output / "checkpoints/update-001000").is_dir()
    assert (output / "checkpoints/update-002000").is_dir()
    assert not (output / "checkpoint_retention").exists()


def test_checkpoint_retention_rejects_replaced_staging_inode_without_deleting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_retention_run(output)
    first = _advance_checkpoint(output, 1000, parent=None)
    _advance_checkpoint(output, 2000, parent=first)

    def fail_after_move(*_args, **_kwargs) -> None:
        raise OSError("simulated crash after retirement move")

    monkeypatch.setattr(run_journal_module, "_remove_retired_tree", fail_after_move)
    with pytest.raises(OSError):
        apply_checkpoint_retention(output)
    transaction = next((output / "checkpoint_retention").iterdir())
    staged = transaction / "checkpoint"
    original = output / "original-checkpoint"
    staged.rename(original)
    staged.mkdir()
    marker = staged / "foreign"
    marker.write_text("do not delete", encoding="utf-8")
    monkeypatch.undo()

    with pytest.raises(ValueError, match="checkpoint retirement staging"):
        apply_checkpoint_retention(output)

    assert marker.read_text() == "do not delete"
    assert original.is_dir()


def test_checkpoint_retention_rejects_nonpermanent_backlog_before_deleting(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_retention_run(output)
    first = _advance_checkpoint(output, 1000, parent=None)
    second = _advance_checkpoint(output, 2000, parent=first)
    _advance_checkpoint(output, 3000, parent=second)

    with pytest.raises(ValueError, match="non-permanent backlog"):
        apply_checkpoint_retention(output)

    assert sorted(path.name for path in (output / "checkpoints").iterdir()) == [
        "update-001000",
        "update-002000",
        "update-003000",
    ]
    assert not (output / "checkpoint_retention").exists()


def test_checkpoint_retention_interval_comes_from_authenticated_resolved_config(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_retention_run(output)
    first = _advance_checkpoint(output, 1000, parent=None)
    _advance_checkpoint(output, 2000, parent=first)
    envelope_path = output / "resolved_config.json"
    envelope = json.loads(envelope_path.read_text())
    envelope["config"]["training"]["permanent_checkpoint_interval"] = 10000
    envelope["config_sha256"] = canonical_config_sha256(envelope["config"])
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="expected run configuration"):
        apply_checkpoint_retention(output)

    assert (output / "checkpoints/update-001000").is_dir()
    assert (output / "checkpoints/update-002000").is_dir()


def test_checkpoint_retention_rejects_transaction_interval_drift(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_retention_run(output)
    first = _advance_checkpoint(output, 1000, parent=None)
    _advance_checkpoint(output, 2000, parent=first)
    apply_checkpoint_retention(output)
    transaction = next((output / "checkpoint_retention").iterdir())
    manifest_path = transaction / "retirement.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["permanent_checkpoint_interval"] = 10000
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="interval differs from the authenticated"):
        apply_checkpoint_retention(output)


def test_completed_retirement_cannot_mask_missing_permanent_parent(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_retention_run(output)
    permanent = _advance_checkpoint(output, 5000, parent=None)
    tip = _advance_checkpoint(output, 6000, parent=permanent)
    permanent_path = output / permanent.relative_path
    permanent_manifest = (permanent_path / "manifest.json").read_bytes()
    permanent_path.rename(output / "displaced-permanent-checkpoint")

    root = output / "checkpoint_retention"
    root.mkdir()
    transaction = root / "retire-update-001000-0123456789abcdef0123456789abcdef"
    transaction.mkdir()
    (transaction / "checkpoint-manifest.json").write_bytes(permanent_manifest)
    fake = CheckpointRecord(
        relative_path="checkpoints/update-001000",
        update=1000,
        manifest_sha256=permanent.manifest_sha256,
        parent_manifest_sha256=None,
        last_metrics={"train_loss": 0.001, "update": 1000},
    )
    run_journal_module._write_retirement_manifest(
        transaction / "retirement.json",
        {
            "schema": run_journal_module.CHECKPOINT_RETIREMENT_SCHEMA,
            "status": "complete",
            "run_uuid": RUN_UUID,
            "config_sha256": CONFIG_SHA,
            "checkpoint": fake,
            "authorized_tip": tip,
            "directory_device": 1,
            "directory_inode": 1,
            "permanent_checkpoint_interval": 5000,
        },
        exclusive=True,
    )

    with pytest.raises(ValueError, match="preserved retired checkpoint manifest"):
        apply_checkpoint_retention(output)

    assert (output / "checkpoints/update-006000").is_dir()


def test_descriptor_relative_retirement_does_not_delete_replacement_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoint"
    target.mkdir()
    (target / "original").write_text("retire me", encoding="utf-8")
    observed = target.stat()
    displaced = tmp_path / "authorized-original"
    real_remove_contents = run_journal_module._remove_directory_contents_fd
    swapped = False

    def swap_root_after_open(descriptor: int, *, context: str) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            target.rename(displaced)
            target.mkdir()
            (target / "foreign").write_text("must survive", encoding="utf-8")
        real_remove_contents(descriptor, context=context)

    monkeypatch.setattr(run_journal_module, "_remove_directory_contents_fd", swap_root_after_open)

    with pytest.raises(ValueError, match="directory identity changed"):
        run_journal_module._remove_retired_tree(
            target,
            device=observed.st_dev,
            inode=observed.st_ino,
        )

    assert swapped
    assert (target / "foreign").read_text() == "must survive"
    assert displaced.is_dir()


def test_descriptor_relative_retirement_unlinks_symlink_without_following_it(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "must-survive"
    marker.write_text("outside the retired tree", encoding="utf-8")
    target = tmp_path / "checkpoint"
    target.mkdir()
    (target / "external-link").symlink_to(external, target_is_directory=True)
    observed = target.stat()

    run_journal_module._remove_retired_tree(
        target,
        device=observed.st_dev,
        inode=observed.st_ino,
    )

    assert not target.exists()
    assert marker.read_text(encoding="utf-8") == "outside the retired tree"


def test_checkpoint_retention_recovers_crash_after_move_before_manifest_preservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_retention_run(output)
    first = _advance_checkpoint(output, 1000, parent=None)
    _advance_checkpoint(output, 2000, parent=first)
    real_preserve = run_journal_module._preserve_retired_checkpoint_manifest

    def fail_before_preservation(*_args, **_kwargs) -> None:
        raise OSError("simulated crash before manifest preservation")

    monkeypatch.setattr(run_journal_module, "_preserve_retired_checkpoint_manifest", fail_before_preservation)
    with pytest.raises(OSError, match="before manifest preservation"):
        apply_checkpoint_retention(output)
    transaction = next((output / "checkpoint_retention").iterdir())
    assert (transaction / "checkpoint").is_dir()
    assert not (transaction / "checkpoint-manifest.json").exists()
    assert json.loads((transaction / "retirement.json").read_text())["status"] == "planned"

    monkeypatch.setattr(run_journal_module, "_preserve_retired_checkpoint_manifest", real_preserve)
    recovered = apply_checkpoint_retention(output)

    assert recovered.recovered_transactions == (transaction.relative_to(output).as_posix(),)
    assert (transaction / "checkpoint-manifest.json").is_file()
    assert json.loads((transaction / "retirement.json").read_text())["status"] == "complete"


def test_checkpoint_retention_recovers_partial_payload_delete_after_manifest_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_retention_run(output)
    first = _advance_checkpoint(output, 1000, parent=None)
    _advance_checkpoint(output, 2000, parent=first)
    real_remove = run_journal_module._remove_retired_tree

    def remove_manifest_then_fail(path: Path, *, device: int, inode: int) -> None:
        del device, inode
        (path / "manifest.json").unlink()
        raise OSError("simulated crash during payload deletion")

    monkeypatch.setattr(run_journal_module, "_remove_retired_tree", remove_manifest_then_fail)
    with pytest.raises(OSError, match="during payload deletion"):
        apply_checkpoint_retention(output)
    transaction = next((output / "checkpoint_retention").iterdir())
    assert (transaction / "checkpoint").is_dir()
    assert not (transaction / "checkpoint/manifest.json").exists()
    assert (transaction / "checkpoint-manifest.json").is_file()

    monkeypatch.setattr(run_journal_module, "_remove_retired_tree", real_remove)
    recovered = apply_checkpoint_retention(output)

    assert recovered.recovered_transactions == (transaction.relative_to(output).as_posix(),)
    assert not (transaction / "checkpoint").exists()
    assert json.loads((transaction / "retirement.json").read_text())["status"] == "complete"


def test_reconcile_truncates_valid_uncommitted_metrics_atomically(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _, record = _create_committed_checkpoint(output)
    metrics_path = output / "metrics.jsonl"
    entries = [
        {"train_loss": 1.0, "update": 1},
        record.last_metrics,
        {"train_loss": 0.1, "update": 3},
        {"train_loss": 0.05, "update": 4},
    ]
    _write_metrics(metrics_path, entries)

    result = reconcile_metrics_jsonl(output)

    assert result.committed_update == 2
    assert result.retained_entries == 2
    assert result.truncated_entries == 2
    assert result.recovered_last_metrics is False
    assert result.changed is True
    assert [json.loads(line) for line in metrics_path.read_text().splitlines()] == entries[:2]
    assert not list(output.glob(".metrics.jsonl.tmp-*"))


def test_reconcile_recovers_missing_checkpoint_metric(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _, record = _create_committed_checkpoint(output)
    metrics_path = output / "metrics.jsonl"
    _write_metrics(
        metrics_path,
        [
            {"train_loss": 1.0, "update": 1},
            {"train_loss": 0.1, "update": 3},
        ],
    )

    result = reconcile_metrics_jsonl(output)

    assert result.retained_entries == 2
    assert result.truncated_entries == 1
    assert result.recovered_last_metrics is True
    assert [json.loads(line) for line in metrics_path.read_text().splitlines()] == [
        {"train_loss": 1.0, "update": 1},
        record.last_metrics,
    ]


def test_reconcile_creates_missing_metrics_file_from_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _, record = _create_committed_checkpoint(output)

    result = reconcile_metrics_jsonl(output)

    assert result.recovered_last_metrics is True
    assert json.loads((output / "metrics.jsonl").read_text()) == record.last_metrics


def test_reconcile_discards_only_an_unterminated_malformed_tail(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _, record = _create_committed_checkpoint(output)
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text('{"update":1}\n{"update":2', encoding="utf-8")

    result = reconcile_metrics_jsonl(output)

    assert result.discarded_partial_tail is True
    assert result.recovered_last_metrics is True
    assert [json.loads(line) for line in metrics_path.read_text().splitlines()] == [
        {"update": 1},
        record.last_metrics,
    ]


def test_quarantine_preserves_one_interrupted_checkpoint_transaction(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _, record = _create_committed_checkpoint(output)
    stage = output / ".rank-state-update-000003"
    stage.mkdir()
    (stage / "rank-000.pt").write_bytes(b"partial")
    orphan, _, _ = _write_checkpoint(
        output,
        3,
        parent_sha=record.manifest_sha256,
    )

    result = quarantine_uncommitted_training_artifacts(output)

    assert result.changed is True
    assert set(result.moved_paths) == {
        ".rank-state-update-000003",
        "checkpoints/update-000003",
    }
    assert not stage.exists()
    assert not orphan.exists()
    assert result.directory is not None
    quarantine = output / result.directory
    assert (quarantine / "root" / stage.name / "rank-000.pt").read_bytes() == b"partial"
    assert (quarantine / "checkpoints" / orphan.name / "manifest.json").is_file()
    recovery = json.loads((quarantine / "recovery.json").read_text())
    assert recovery["status"] == "complete"
    assert recovery["committed_update"] == 2
    assert recovery["committed_manifest_sha256"] == record.manifest_sha256


def test_quarantine_rejects_ambiguous_future_updates_without_mutation(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_committed_checkpoint(output)
    first = output / ".rank-state-update-000003"
    second = output / ".rank-state-update-000004"
    first.mkdir()
    second.mkdir()

    with pytest.raises(ValueError, match="multiple uncommitted"):
        quarantine_uncommitted_training_artifacts(output)

    assert first.is_dir()
    assert second.is_dir()
    assert not (output / "recovery_quarantine").exists()


def test_quarantine_recovers_bootstrap_artifacts_without_a_journal_tip(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    create_run_journal(output, config_sha256=CONFIG_SHA, run_uuid=RUN_UUID)
    stage = output / ".rank-state-update-000001"
    stage.mkdir()
    checkpoint, _, _ = _write_checkpoint(output, 1)
    metrics = output / "metrics.jsonl"
    metrics.write_text('{"update":1}\n', encoding="utf-8")
    config_temp = output / ".resolved_config.json.tmp-123"
    config_temp.write_text("partial", encoding="utf-8")
    journal_temp = output / ".run_journal.json.tmp-0123456789abcdef0123456789abcdef"
    journal_temp.write_text("partial", encoding="utf-8")

    result = quarantine_uncommitted_training_artifacts(output)

    assert set(result.moved_paths) == {
        ".rank-state-update-000001",
        ".resolved_config.json.tmp-123",
        ".run_journal.json.tmp-0123456789abcdef0123456789abcdef",
        "checkpoints/update-000001",
        "metrics.jsonl",
    }
    assert not stage.exists()
    assert not checkpoint.exists()
    assert not metrics.exists()
    assert result.directory is not None
    quarantine = output / result.directory
    recovery = json.loads((quarantine / "recovery.json").read_text())
    assert recovery["status"] == "complete"
    assert recovery["committed_update"] == -1
    assert recovery["committed_manifest_sha256"] is None


@pytest.mark.parametrize(
    "content, match",
    [
        ('{"update":1}\n{"update":1}\n', "strictly increasing"),
        ('{"update":2}\n{"update":1}\n', "strictly increasing"),
        ('{"update":1}\nnot-json\n', "malformed JSON"),
        ('{"update":true}\n', "nonnegative integer"),
        ('{"loss":1}\n', "nonnegative integer"),
        ('{"update":1,"update":2}\n', "duplicate JSON object key"),
    ],
)
def test_reconcile_rejects_duplicate_reordered_or_malformed_history(
    tmp_path: Path,
    content: str,
    match: str,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_committed_checkpoint(output)
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text(content, encoding="utf-8")
    before = metrics_path.read_bytes()

    with pytest.raises(ValueError, match=match):
        reconcile_metrics_jsonl(output)

    assert metrics_path.read_bytes() == before


def test_reconcile_rejects_conflicting_committed_metric_without_mutation(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _create_committed_checkpoint(output)
    metrics_path = output / "metrics.jsonl"
    _write_metrics(metrics_path, [{"update": 1}, {"train_loss": 999.0, "update": 2}, {"update": 3}])
    before = metrics_path.read_bytes()

    with pytest.raises(ValueError, match="conflict"):
        reconcile_metrics_jsonl(output)

    assert metrics_path.read_bytes() == before


def test_failed_atomic_metrics_replace_preserves_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _, record = _create_committed_checkpoint(output)
    metrics_path = output / "metrics.jsonl"
    entries = [{"update": 1}, record.last_metrics, {"update": 3}]
    _write_metrics(metrics_path, entries)
    before = metrics_path.read_bytes()

    def fail_replace(source, destination) -> None:
        raise OSError("simulated crash before replace")

    monkeypatch.setattr(run_journal_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated crash"):
        reconcile_metrics_jsonl(output)

    assert metrics_path.read_bytes() == before
    assert not list(output.glob(".metrics.jsonl.tmp-*"))
