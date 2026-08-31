from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/create_calvin_prefix_geometry.py"
SPEC = importlib.util.spec_from_file_location("create_calvin_prefix_geometry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def test_validation_yaml_rejects_duplicate_keys() -> None:
    raw = b"task_a: [first phrase]\ntask_a: [replacement phrase]\n"

    with pytest.raises(RuntimeError, match="duplicate YAML key 'task_a'"):
        GENERATOR.parse_validation_annotations_yaml(raw)


def test_validation_yaml_selects_first_phrase_and_validates_every_phrase() -> None:
    selected = GENERATOR.parse_validation_annotations_yaml(
        "task_b: [first exact phrase, ignored alternative]\ntask_a: [Only phrase]\n"
    )

    assert selected == {"task_a": "Only phrase", "task_b": "first exact phrase"}
    with pytest.raises(RuntimeError, match="nonempty list"):
        GENERATOR.parse_validation_annotations_yaml("task: scalar phrase\n")
    with pytest.raises(RuntimeError, match="nonempty strings"):
        GENERATOR.parse_validation_annotations_yaml("task: [valid first phrase, 7]\n")


def test_training_and_evaluation_inventory_union_preserves_exact_text() -> None:
    dataset = SimpleNamespace(
        annotations=(
            SimpleNamespace(instruction="train wording", task="task_a"),
            SimpleNamespace(instruction="Alternate Wording", task="task_a"),
            SimpleNamespace(instruction="train wording", task="task_a"),
            SimpleNamespace(instruction="shared wording", task="task_b"),
        ),
        tasks=("task_a", "task_b"),
    )
    training_instructions, training_tasks = GENERATOR.training_calvin_inventory(dataset)

    union = GENERATOR.union_calvin_instruction_inventories(
        training_instructions=training_instructions,
        training_tasks=training_tasks,
        evaluation_by_task={"task_a": "official wording", "task_b": "shared wording"},
    )

    assert training_instructions == ("Alternate Wording", "shared wording", "train wording")
    assert training_tasks == ("task_a", "task_b")
    assert union == ("Alternate Wording", "official wording", "shared wording", "train wording")


def test_training_and_evaluation_task_sets_must_be_exactly_equal() -> None:
    with pytest.raises(RuntimeError, match="task sets differ"):
        GENERATOR.union_calvin_instruction_inventories(
            training_instructions=("training",),
            training_tasks=("task_a", "task_b"),
            evaluation_by_task={"task_a": "evaluation", "task_c": "unexpected"},
        )


def test_auto_width_is_unbounded_maximum_plus_sentinel_then_requires_exact_fixed_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int | None]] = []
    unbounded = {"short": 5, "long": 11}

    def measure_unbounded(*_args: object, **_kwargs: object) -> dict[str, int]:
        calls.append(("unbounded", None))
        return dict(unbounded)

    def measure_fixed(*_args: object, **kwargs: object) -> dict[str, int]:
        calls.append(("fixed", kwargs["fixed_physical_prefix_width"]))
        return dict(unbounded)

    monkeypatch.setattr(GENERATOR, "measure_unbounded_prefix_valid_lengths", measure_unbounded)
    monkeypatch.setattr(GENERATOR, "measure_prefix_valid_lengths", measure_fixed)

    width, measured = GENERATOR.measure_sentinel_padded_prefix_geometry(
        object(),
        instructions=("short", "long"),
        ordered_cameras=GENERATOR.CALVIN_CAMERAS,
        padding_side="left",
        expected_fixed_prefix_width=12,
    )

    assert width == 12
    assert measured == unbounded
    assert calls == [("unbounded", None), ("fixed", 12)]

    with pytest.raises(RuntimeError, match="differs from --expected-fixed-prefix-width"):
        GENERATOR.measure_sentinel_padded_prefix_geometry(
            object(),
            instructions=("short", "long"),
            ordered_cameras=GENERATOR.CALVIN_CAMERAS,
            padding_side="left",
            expected_fixed_prefix_width=11,
        )


def test_auto_width_rejects_any_fixed_measurement_difference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        GENERATOR,
        "measure_unbounded_prefix_valid_lengths",
        lambda *_args, **_kwargs: {"instruction": 9},
    )
    monkeypatch.setattr(
        GENERATOR,
        "measure_prefix_valid_lengths",
        lambda *_args, **_kwargs: {"instruction": 8},
    )

    with pytest.raises(RuntimeError, match="differ from the authenticated unbounded"):
        GENERATOR.measure_sentinel_padded_prefix_geometry(
            object(),
            instructions=("instruction",),
            ordered_cameras=GENERATOR.CALVIN_CAMERAS,
            padding_side="left",
        )


def test_dataset_report_binds_complete_archive_direct_storage_identity() -> None:
    storage_payload = {"mode": "archive-direct", "reader_schema": "reader-v1"}
    storage = SimpleNamespace(
        mode=GENERATOR.CALVIN_STORAGE_MODE_ARCHIVE_DIRECT,
        manifest_schema="manifest-v4",
        manifest_content_sha256="a" * 64,
        content_sha256="b" * 64,
        to_dict=lambda: dict(storage_payload),
    )
    generation = SimpleNamespace(
        storage=storage,
        critical_files=((GENERATOR.CALVIN_TRAINING_ANNOTATION_PATH, "c" * 64),),
        dataset_manifest_file_sha256="d" * 64,
        to_dict=lambda: {"schema": "generation-v2"},
    )

    report = GENERATOR.authenticated_dataset_report(  # type: ignore[arg-type]
        {"schema": "manifest-v4", "content_sha256": "a" * 64},
        generation,
    )

    assert report["storage"] == storage_payload
    assert report["storage_identity_sha256"] == "b" * 64
    assert report["critical_annotation_sha256"] == "c" * 64

    storage.mode = "verified-extraction"
    with pytest.raises(RuntimeError, match="archive-direct v4"):
        GENERATOR.authenticated_dataset_report(  # type: ignore[arg-type]
            {"schema": "manifest-v4", "content_sha256": "a" * 64},
            generation,
        )


def test_main_uses_verified_snapshot_path_and_strictly_reloads_published_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_snapshot = tmp_path / "model" / GENERATOR.DEFAULT_DIFFUSION_GEMMA_SPEC.revision
    output = tmp_path / "output" / "prefix.json"
    output.parent.mkdir()
    source_root = tmp_path / "calvin"
    training_root = tmp_path / "task_ABC_D" / "training"
    args = SimpleNamespace(
        training_root=training_root,
        model_snapshot=model_snapshot,
        calvin_source=source_root,
        output=output,
        expected_fixed_prefix_width=None,
    )
    dataset = SimpleNamespace(
        annotations=(SimpleNamespace(instruction="training wording", task="task_a"),),
        tasks=("task_a",),
    )

    class DatasetContext:
        def __enter__(self) -> Any:
            return dataset

        def __exit__(self, *_args: object) -> None:
            return None

    processor = object()
    processor_calls: list[tuple[object, dict[str, object]]] = []
    published: dict[str, Any] = {}
    reload_calls: list[tuple[Path, dict[str, Any]]] = []

    monkeypatch.setattr(GENERATOR, "parse_args", lambda: args)
    monkeypatch.setattr(GENERATOR, "authenticate_calvin_dataset_generation", lambda _root: ({}, object()))
    monkeypatch.setattr(
        GENERATOR,
        "authenticated_dataset_report",
        lambda _manifest, _generation: {"storage_identity_sha256": "1" * 64},
    )
    monkeypatch.setattr(GENERATOR, "CalvinNpzDataset", lambda *_args, **_kwargs: DatasetContext())
    monkeypatch.setattr(
        GENERATOR,
        "authenticate_calvin_source",
        lambda _root: {"revision": GENERATOR.CALVIN_SOURCE_REVISION},
    )
    monkeypatch.setattr(
        GENERATOR,
        "official_calvin_evaluation_inventory",
        lambda _root: ({"task_a": "official wording"}, {"path": "validation.yaml", "sha256": "2" * 64}),
    )
    monkeypatch.setattr(
        GENERATOR,
        "verify_huggingface_snapshot",
        lambda path, *, expected_revision: {
            "content_inventory_sha256": "3" * 64,
            "files_verified": 7,
            "revision": expected_revision,
            "snapshot": str(path),
            "total_bytes": 1234,
            "tree_metadata_sha256": "4" * 64,
        },
    )

    def load_processor(path: object, **kwargs: object) -> object:
        processor_calls.append((path, dict(kwargs)))
        return processor

    monkeypatch.setattr(GENERATOR.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(
        GENERATOR,
        "measure_sentinel_padded_prefix_geometry",
        lambda processor_arg, **_kwargs: (
            (
                12,
                {"official wording": 11, "training wording": 10},
            )
            if processor_arg is processor
            else pytest.fail("main measured prefix geometry with the wrong processor")
        ),
    )

    def save(path: Path, contract: dict[str, Any]) -> str:
        published["path"] = path
        published["contract"] = contract
        return contract["content_sha256"]

    def reload(path: Path, **kwargs: Any) -> dict[str, Any]:
        reload_calls.append((path, kwargs))
        return published["contract"]

    monkeypatch.setattr(GENERATOR, "save_prefix_geometry_contract", save)
    monkeypatch.setattr(GENERATOR, "load_prefix_geometry_contract", reload)

    GENERATOR.main()

    result = json.loads(capsys.readouterr().out)
    assert processor_calls == [(model_snapshot.resolve(), {"local_files_only": True})]
    assert published["path"] == output.resolve()
    assert len(reload_calls) == 1
    reload_path, reload_kwargs = reload_calls[0]
    assert reload_path == output.resolve()
    assert reload_kwargs["expected_content_sha256"] == published["contract"]["content_sha256"]
    assert reload_kwargs["expected_ordered_cameras"] == GENERATOR.CALVIN_CAMERAS
    assert reload_kwargs["expected_instructions"] == ("official wording", "training wording")
    assert reload_kwargs["expected_fixed_physical_prefix_width"] == 12
    assert result["status"] == "ok"

    monkeypatch.setattr(GENERATOR, "load_prefix_geometry_contract", lambda *_args, **_kwargs: {})
    with pytest.raises(RuntimeError, match="differs after strict reload"):
        GENERATOR.main()
