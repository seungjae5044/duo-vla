from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/calvin/generate_calvin_dev_states.py"
SPEC = importlib.util.spec_from_file_location("calvin_dev_state_generator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


def test_generator_boundary_has_no_archive_reader_or_episode_path_access() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT), feature_version=(3, 8))
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names
    }
    assert "duo_vla.data.calvin_archive" not in imported
    assert "zipfile" not in imported
    assert ".npz" not in source
    assert "read_member_bytes" not in source
    assert "load_replay_bundle" in source


def test_generator_cli_requires_the_replay_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--normalization",
            str(tmp_path / "normalization.json"),
            "--output-dir",
            str(tmp_path / "bank"),
        ],
    )
    with pytest.raises(SystemExit):
        GENERATOR.parse_args()


def test_generator_main_passes_only_authenticated_bytes_and_bundled_replays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    training = tmp_path / "task_ABC_D/training"
    normalization = tmp_path / "normalization.json"
    source = tmp_path / "source"
    revisions = tmp_path / "revisions.env"
    replay_bundle = tmp_path / "replay-bundle"
    output = tmp_path / "bank"
    replay_manifest = {"root_sha256": "a" * 64}
    replays = (object(),)
    inputs = SimpleNamespace(
        identity={"bound": True},
        metadata={"training/.hydra/merged_config.yaml": b"authenticated merged config"},
        source_files={
            "scene/calvin_scene_A.yaml": b"authenticated scene config",
            "task_oracle.yaml": b"authenticated task oracle",
        },
        split={"bound": True},
    )
    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--training-root",
            str(training),
            "--normalization",
            str(normalization),
            "--source-root",
            str(source),
            "--revision-file",
            str(revisions),
            "--replay-bundle",
            str(replay_bundle),
            "--output-dir",
            str(output),
            "--smoke-tasks-per-scene",
            "1",
        ],
    )
    monkeypatch.setattr(GENERATOR, "authenticate_dev_inputs", lambda *args: inputs)

    def load_bundle(path: Path, observed_inputs: object) -> tuple[dict[str, Any], tuple[object, ...]]:
        assert path == replay_bundle and observed_inputs is inputs
        return replay_manifest, replays

    monkeypatch.setattr(GENERATOR, "load_replay_bundle", load_bundle)

    def instantiate_oracle(path: Path, *, task_oracle_bytes: bytes) -> object:
        assert path == source and task_oracle_bytes == b"authenticated task oracle"
        return object()

    monkeypatch.setattr(GENERATOR, "instantiate_task_oracle", instantiate_oracle)

    def instantiate_environment(
        observed_training: Path,
        observed_source: Path,
        scene: str,
        *,
        merged_config_bytes: bytes,
        scene_config_bytes: bytes,
    ) -> object:
        assert observed_training == training and observed_source == source and scene == "calvin_scene_A"
        assert merged_config_bytes == b"authenticated merged config"
        assert scene_config_bytes == b"authenticated scene config"
        return object()

    monkeypatch.setattr(GENERATOR, "instantiate_abc_environment", instantiate_environment)

    def materialize(
        values: object,
        oracle: object,
        factory: object,
        base_seed: int,
    ) -> tuple[tuple[object, ...], tuple[object, ...]]:
        assert values is replays and callable(factory) and base_seed == GENERATOR.DEFAULT_BASE_SEED
        factory("calvin_scene_A")
        calls["oracle"] = oracle
        return (object(),), ()

    monkeypatch.setattr(GENERATOR, "materialize_replay_valid_resets", materialize)

    def build_bank(*args: object, **kwargs: object) -> tuple[dict[str, Any], dict[str, bytes]]:
        assert args[1] == inputs.identity and args[2] == inputs.split and args[3] is replay_manifest
        assert kwargs["smoke_tasks_per_scene"] == 1
        return {
            "records": [{}],
            "rejections": [],
            "root_sha256": "b" * 64,
            "selection": {"smoke_reset_indices": [0]},
        }, {"robot_obs.npy": b"fixture"}

    monkeypatch.setattr(GENERATOR, "build_bank", build_bank)

    def write_bank(path: Path, manifest: object, artifacts: object) -> None:
        calls["write"] = (path, manifest, artifacts)

    monkeypatch.setattr(GENERATOR, "write_bank_exclusive", write_bank)
    GENERATOR.main()
    assert calls["write"][0] == output
