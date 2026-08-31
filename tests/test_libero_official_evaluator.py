"""Fail-closed tests for the sealed official LIBERO evaluation contract."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_libero as EVALUATOR  # noqa: E402
import preflight_libero_env as PREFLIGHT  # noqa: E402

from duo_vla.run_journal import create_run_journal, record_latest_checkpoint  # noqa: E402


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cells(episode_sha256: str) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for seed in EVALUATOR.OFFICIAL_TRAIN_SEEDS:
        for objective, nfes in (
            ("rectified_flow", EVALUATOR.OFFICIAL_FLOW_NFES),
            ("direct_regression", (1,)),
        ):
            checkpoint = {
                "manifest_sha256": _digest(f"checkpoint:{seed}:{objective}"),
                "source_tree_sha256": "f" * 64,
                "update": EVALUATOR.FINAL_CHECKPOINT_UPDATE,
            }
            policy_contract_sha256 = _digest(f"contract:{objective}")
            runtime_sha256 = _digest(f"runtime:{seed}:{objective}")
            for nfe in nfes:
                for execution_horizon in EVALUATOR.OFFICIAL_EXECUTION_HORIZONS:
                    selected_policy = {
                        "inference_seed_behavior": (
                            "episode_identity_gaussian_noise"
                            if objective == "rectified_flow"
                            else "episode_identity_echo_only"
                        ),
                        "nfe": nfe,
                        "objective": objective,
                        "sampler": "euler_uniform" if objective == "rectified_flow" else "single_forward",
                    }
                    cells.append(
                        {
                            "cell_id": EVALUATOR.official_cell_id(seed, objective, nfe, execution_horizon),
                            "checkpoint": checkpoint,
                            "episode_matrix_sha256": episode_sha256,
                            "execution_geometry": copy.deepcopy(EVALUATOR.LIBERO_EXECUTION_GEOMETRY),
                            "execution_horizon": execution_horizon,
                            **selected_policy,
                            "policy_contract_sha256": policy_contract_sha256,
                            "policy_warmup_calls": EVALUATOR.OFFICIAL_POLICY_WARMUP_CALLS,
                            "serving_policy_sha256": EVALUATOR.canonical_sha256(selected_policy),
                            "serving_runtime_sha256": runtime_sha256,
                            "train_seed": seed,
                        }
                    )
    return cells


def _manifest(*, token: str = "sealed", attestation_sha256: str = "a" * 64) -> dict[str, Any]:
    contamination = EVALUATOR.load_contamination_contract(ROOT)
    episodes = EVALUATOR.official_episode_matrix(contamination)
    episode_sha256 = EVALUATOR.canonical_sha256(episodes)
    return {
        "benchmark_protocol": EVALUATOR.PROTOCOL,
        "cells": _cells(episode_sha256),
        "contamination": contamination,
        "episode_count": EVALUATOR.OFFICIAL_PRIMARY_EPISODES,
        "episode_matrix_sha256": episode_sha256,
        "episodes": episodes,
        "evaluation_seed": 123,
        "execution_horizons": list(EVALUATOR.OFFICIAL_EXECUTION_HORIZONS),
        "final_checkpoint_update": EVALUATOR.FINAL_CHECKPOINT_UPDATE,
        "final_freeze_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "official_resets_per_task": EVALUATOR.OFFICIAL_RESETS_PER_TASK,
        "schema": EVALUATOR.PREREGISTRATION_SCHEMA,
        "simulator_attestation_sha256": attestation_sha256,
        "suites": list(EVALUATOR.SUITES),
        "task_ids": list(range(10)),
        "training_seeds": list(EVALUATOR.OFFICIAL_TRAIN_SEEDS),
    }


def test_contamination_contract_forces_exact_1999_episode_primary_matrix() -> None:
    contamination = EVALUATOR.load_contamination_contract(ROOT)
    episodes = EVALUATOR.official_episode_matrix(contamination)

    assert len(episodes) == 1999
    assert EVALUATOR.OFFICIAL_EXCLUDED_EPISODE not in episodes
    assert sum(episode["suite"] == "libero_goal" and episode["task_id"] == 7 for episode in episodes) == 49
    assert episodes[0] == {"reset_id": 0, "suite": "libero_spatial", "task_id": 0}
    assert episodes[-1] == {"reset_id": 49, "suite": "libero_10", "task_id": 9}


def test_preregistration_binds_raw_sha_token_attestation_and_exact_24_cells(tmp_path: Path) -> None:
    token = "external final freeze token"
    manifest = _manifest(token=token)
    path = tmp_path / "preregistered.json"
    path.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
    raw_sha256 = EVALUATOR.sha256_file(path)

    loaded, cell, observed_sha256 = EVALUATOR.load_preregistration(
        path,
        cell_id="seed-1-flow-nfe-5-k-4",
        execution_horizon=4,
        evaluation_seed=123,
        final_freeze_token=token,
        preregistration_sha256=raw_sha256,
        simulator_attestation_sha256="a" * 64,
        contamination=manifest["contamination"],
    )

    assert loaded == manifest
    assert cell["train_seed"] == 1
    assert cell["objective"] == "rectified_flow"
    assert cell["nfe"] == 5
    assert observed_sha256 == raw_sha256

    with pytest.raises(RuntimeError, match="freeze token"):
        EVALUATOR.load_preregistration(
            path,
            cell_id=cell["cell_id"],
            execution_horizon=4,
            evaluation_seed=123,
            final_freeze_token="post-hoc",
            preregistration_sha256=raw_sha256,
            simulator_attestation_sha256="a" * 64,
            contamination=manifest["contamination"],
        )


def test_preregistration_creator_derives_episode_and_serving_policy_hashes(tmp_path: Path) -> None:
    manifest = _manifest()
    cells = []
    for cell in manifest["cells"]:
        cells.append(
            {
                name: value
                for name, value in cell.items()
                if name not in {"episode_matrix_sha256", "serving_policy_sha256"}
            }
        )
    cells_path = tmp_path / "cells.json"
    cells_path.write_text(json.dumps({"cells": cells}, sort_keys=True))
    attestation = {
        "environment_constructed": True,
        "schema": "duo-vla-libero-simulator-attestation-v2",
        "status": "ok",
    }
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(json.dumps(attestation, sort_keys=True))
    output = tmp_path / "preregistered.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/create_libero_preregistration.py"),
            "--cells",
            str(cells_path),
            "--simulator-attestation",
            str(attestation_path),
            "--evaluation-seed",
            "123",
            "--final-freeze-token",
            "sealed",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    created = json.loads(output.read_text())
    assert created["episode_matrix_sha256"] == manifest["episode_matrix_sha256"]
    assert all(cell["serving_policy_sha256"] for cell in created["cells"])
    assert output.with_suffix(".json.sha256").is_file()


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda value: value["cells"].pop(), "exact 24-cell"),
        (lambda value: value["episodes"].append(dict(EVALUATOR.OFFICIAL_EXCLUDED_EPISODE)), "episode matrix changed"),
        (lambda value: value["cells"][0].update(nfe=7), "flow cell NFE"),
        (lambda value: value["cells"][0].update(policy_warmup_calls=1), "warm-up count"),
        (
            lambda value: value["cells"][0]["checkpoint"].update(update=29_999),
            "checkpoint update",
        ),
    ),
)
def test_preregistration_rejects_incomplete_or_mutated_factor_matrix(mutate: Any, message: str) -> None:
    manifest = _manifest()
    mutate(manifest)
    with pytest.raises(RuntimeError, match=message):
        EVALUATOR.validate_preregistration_manifest(
            manifest,
            contamination=manifest["contamination"],
            simulator_attestation_sha256="a" * 64,
        )


def _committed_checkpoint(tmp_path: Path, *, selected_cell: dict[str, Any]) -> tuple[Path, str]:
    run_root = tmp_path / "run"
    run_root.mkdir()
    journal = create_run_journal(run_root, config_sha256="c" * 64)
    checkpoint = run_root / "checkpoints/update-030000"
    checkpoint.mkdir(parents=True)
    manifest = {
        "config_sha256": "c" * 64,
        "kind": "resumable-libero-training",
        "last_metrics": {"update": EVALUATOR.FINAL_CHECKPOINT_UPDATE},
        "parent_manifest_sha256": None,
        "policy_contract": {
            "inference_seed_behavior": selected_cell["inference_seed_behavior"],
            "nfe": 10,
            "objective": selected_cell["objective"],
            "sampler": selected_cell["sampler"],
        },
        "policy_contract_sha256": selected_cell["policy_contract_sha256"],
        "run_seed": selected_cell["train_seed"],
        "run_uuid": journal.run_uuid,
        "schema": "duo-vla-checkpoint-v1",
        "source_tree_sha256": selected_cell["checkpoint"]["source_tree_sha256"],
        "trainer_state": {"next_update": EVALUATOR.FINAL_CHECKPOINT_UPDATE},
    }
    path = checkpoint / "manifest.json"
    path.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
    manifest_sha256 = EVALUATOR.sha256_file(path)
    record_latest_checkpoint(
        run_root,
        checkpoint=checkpoint,
        update=EVALUATOR.FINAL_CHECKPOINT_UPDATE,
        manifest_sha256=manifest_sha256,
        parent_manifest_sha256=None,
        last_metrics=manifest["last_metrics"],
    )
    return checkpoint, manifest_sha256


def test_official_health_binds_selected_nfe_and_final_journal_tip(tmp_path: Path) -> None:
    manifest = _manifest()
    selected = next(cell for cell in manifest["cells"] if cell["cell_id"] == "seed-1-flow-nfe-5-k-4")
    checkpoint_path, manifest_sha256 = _committed_checkpoint(tmp_path, selected_cell=selected)
    selected["checkpoint"]["manifest_sha256"] = manifest_sha256
    contract = {
        "inference_seed_behavior": selected["inference_seed_behavior"],
        "nfe": selected["nfe"],
        "objective": selected["objective"],
        "sampler": selected["sampler"],
    }
    health = {
        **contract,
        "checkpoint": {
            "execution_geometry": copy.deepcopy(EVALUATOR.LIBERO_EXECUTION_GEOMETRY),
            "kind": "resumable-libero-training",
            "manifest_sha256": manifest_sha256,
            "path": str(checkpoint_path),
            "policy_contract": {**contract, "nfe": 10},
            "policy_contract_sha256": selected["policy_contract_sha256"],
            "source_tree_sha256": selected["checkpoint"]["source_tree_sha256"],
            "train_seed": selected["train_seed"],
        },
        "dataset_revision": EVALUATOR.DATASET_REVISION,
        "execution_geometry": copy.deepcopy(EVALUATOR.LIBERO_EXECUTION_GEOMETRY),
        "mode": "real",
        "model_revision": EVALUATOR.MODEL_REVISION,
        "normalization_content_sha256": EVALUATOR.NORMALIZATION_SHA256,
        "prefix_cache_scope": "request",
        "serving_runtime_sha256": selected["serving_runtime_sha256"],
        "train_seed": selected["train_seed"],
    }

    report = EVALUATOR.validate_official_policy_health(health, selected_cell=selected, execution_horizon=4)

    assert report["update"] == EVALUATOR.FINAL_CHECKPOINT_UPDATE
    assert report["run_journal_latest"] is True
    changed = copy.deepcopy(health)
    changed["nfe"] = 10
    with pytest.raises(RuntimeError, match="serving policy"):
        EVALUATOR.validate_official_policy_health(changed, selected_cell=selected, execution_horizon=4)


def test_official_mode_requires_full_primary_selection_and_seal() -> None:
    args = EVALUATOR.parse_args(
        [
            "--mode",
            "official-score",
            "--suite",
            "all",
            "--task-ids",
            "all",
            "--init-state-ids",
            "all",
            "--evaluation-seed",
            "123",
            "--execution-horizon",
            "4",
            "--output-dir",
            "/tmp/output",
            "--preregistration-manifest",
            "/tmp/preregistered.json",
            "--preregistration-sha256",
            "a" * 64,
            "--cell-id",
            "seed-0-flow-nfe-10-k-4",
            "--final-freeze-token",
            "sealed",
        ]
    )
    EVALUATOR.validate_mode_arguments(args)

    args.task_ids = "0"
    with pytest.raises(RuntimeError, match="task-ids all"):
        EVALUATOR.validate_mode_arguments(args)


def test_official_summary_states_clean_denominator_and_exclusion() -> None:
    contamination = EVALUATOR.load_contamination_contract(ROOT)
    task_metrics = [
        {
            "episodes": 49 if (suite, task_id) == ("libero_goal", 7) else 50,
            "suite": suite,
            "task_id": task_id,
        }
        for suite in EVALUATOR.SUITES
        for task_id in range(10)
    ]
    summary = {
        "complete_40_task_macro": True,
        "episodes": 1999,
        "task_metrics": task_metrics,
    }

    result = EVALUATOR.bind_official_summary(
        summary,
        contamination=contamination,
        episode_matrix_sha256="e" * 64,
    )

    assert result["reporting"]["denominator"] == 1999
    assert result["reporting"]["full_official_episode_count"] == 2000
    assert result["reporting"]["excluded_episodes"] == [EVALUATOR.OFFICIAL_EXCLUDED_EPISODE]
    assert result["reporting"]["non_blind_full_set_reported"] is False


def test_eval_launchers_scrub_injection_and_do_not_append_pythonpath() -> None:
    for name in (
        "run_create_libero_preregistration.sh",
        "run_libero_eval.sh",
        "run_libero_preflight.sh",
    ):
        source = (ROOT / "scripts" / name).read_text()
        assert source.startswith('#!/bin/bash\nset -euo pipefail\nexport PATH="/usr/bin:/bin"')
        assert "compgen -v" in source
        assert "exec /usr/bin/env -i" in source
        assert '"PYTHONPATH=${project_dir}/src:${project_dir}/scripts"' in source
        assert "${PYTHONPATH:+" not in source
        assert '"MUJOCO_EGL_DEVICE_ID=0"' in source
        assert '"HF_HOME=/root/.cache/huggingface"' in source
        assert '"LANG=C.UTF-8"' in source
        assert '"LC_ALL=C.UTF-8"' in source


def test_evaluator_rejects_direct_python_environment_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONWARNINGS", "error")
    with pytest.raises(RuntimeError, match="injection_overrides"):
        EVALUATOR.validate_evaluator_process_environment(ROOT)


def test_evaluator_rejects_unlisted_render_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    cache_root = Path("/root/.cache/duo-vla")
    expected = {
        **EVALUATOR._REQUIRED_EVALUATOR_ENVIRONMENT,
        "DUO_VLA_CACHE_ROOT": str(cache_root),
        "HF_HOME": "/root/.cache/huggingface",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LIBERO_CONFIG_PATH": str(cache_root / "simulators/libero/config"),
        "PATH": f"{cache_root}/venvs/libero-eval/bin:/usr/bin:/bin",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}",
    }
    monkeypatch.setattr(EVALUATOR.os, "environ", {**expected, "LIBGL_ALWAYS_SOFTWARE": "1"})
    monkeypatch.setattr(EVALUATOR.sys, "prefix", str(cache_root / "venvs/libero-eval"))
    with pytest.raises(RuntimeError, match="exact closed launcher allowlist"):
        EVALUATOR.validate_evaluator_process_environment(ROOT)


def test_simulator_manifest_parser_rejects_duplicate_and_extra_fields(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="strict finite UTF-8 JSON"):
        PREFLIGHT.load_strict_manifest(duplicate)

    extra = tmp_path / "extra.json"
    extra.write_text(
        json.dumps(
            {
                "assets": {},
                "environment": {},
                "extra": True,
                "paths": {},
                "schema": "duo-vla-libero-simulator-v1",
                "source": {},
                "training_data_downloaded": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="top-level fields changed"):
        PREFLIGHT.load_strict_manifest(extra)
