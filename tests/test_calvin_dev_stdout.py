"""Descriptor-level stdout isolation regressions for CALVIN development CLIs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CALVIN_SCRIPTS = ROOT / "scripts" / "calvin"


def _run_main_harness(module_name: str, outcome: str) -> subprocess.CompletedProcess[str]:
    harness = f"""
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, {str(CALVIN_SCRIPTS)!r})
import {module_name} as target

args = types.SimpleNamespace(
    all_resets=False,
    allow_fake_policy=True,
    base_seed=0,
    evaluation_seed=0,
    execution_horizon=1,
    normalization=Path("/normalization.json"),
    output_dir=Path("/output"),
    replay_bundle=Path("/replay"),
    reset_bank=Path("/bank"),
    revision_file=Path("/revisions.env"),
    smoke_tasks_per_scene=1,
    socket=Path("/policy.sock"),
    source_root=Path("/source"),
    training_root=Path("/training"),
)
inputs = types.SimpleNamespace(
    identity={{}},
    metadata={{"training/.hydra/merged_config.yaml": b"config"}},
    source_files={{"task_oracle.yaml": b"oracle"}},
    split={{}},
)
target.parse_args = lambda: args
target.authenticate_dev_inputs = lambda *_args: inputs
target.instantiate_task_oracle = lambda *_args, **_kwargs: object()

def native_boundary(*_args, **_kwargs):
    os.write(1, b"native-before-report\\n")
    if {outcome!r} == "failure":
        raise RuntimeError("simulated native failure")

if {module_name!r} == "generate_calvin_dev_states":
    manifest = {{
        "records": [],
        "rejections": [],
        "root_sha256": "a" * 64,
        "selection": {{"smoke_reset_indices": []}},
    }}
    target.load_replay_bundle = lambda *_args: ({{}}, ())
    target.materialize_replay_valid_resets = lambda *_args: ((), ())
    target.build_bank = lambda *_args, **_kwargs: (manifest, {{}})
    target.write_bank_exclusive = native_boundary
else:
    manifest = {{
        "records": [],
        "replay_bundle": {{"root_sha256": "b" * 64}},
        "root_sha256": "c" * 64,
        "selection": {{"smoke_reset_indices": []}},
    }}
    target.load_bank = lambda *_args: (manifest, None, None)
    target.assert_bank_matches_inputs = lambda *_args: None
    target.validate_live_policy = lambda *_args, **_kwargs: 0
    target.time.perf_counter = lambda: 1.0

    class Client:
        def __init__(self, *_args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def health(self):
            return {{"mode": "fake"}}

    class Journal:
        def __init__(self, *_args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def start_run(self, *_args):
            pass
        def append(self, *_args):
            pass
        def write_json(self, *_args):
            pass
        def complete(self):
            pass

    def evaluate(*_args, **_kwargs):
        native_boundary()
        return ()

    target.DevPolicyClient = Client
    target.DevelopmentJournal = Journal
    target.evaluate_reset_indices = evaluate
    target.summarize_development = lambda *_args, **_kwargs: {{"status": "ok"}}

try:
    target.main()
except BaseException:
    if {outcome!r} != "failure":
        raise
    os.write(1, b"native-after-main\\n")
else:
    if {outcome!r} != "success":
        raise RuntimeError("expected development CLI failure")
    os.write(1, b"native-after-main\\n")
"""
    return subprocess.run(
        [sys.executable, "-c", harness],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("module_name", ["generate_calvin_dev_states", "evaluate_calvin_dev"])
def test_development_cli_reserves_stdout_for_one_report_through_native_teardown(module_name: str) -> None:
    completed = _run_main_harness(module_name, "success")

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "ok"
    assert completed.stdout.count("\n") == 1
    assert "native-before-report\n" in completed.stderr
    assert "native-after-main\n" in completed.stderr


@pytest.mark.parametrize("module_name", ["generate_calvin_dev_states", "evaluate_calvin_dev"])
def test_development_cli_failure_emits_no_report_and_keeps_native_teardown_on_stderr(module_name: str) -> None:
    completed = _run_main_harness(module_name, "failure")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert "native-before-report\n" in completed.stderr
    assert "native-after-main\n" in completed.stderr


@pytest.mark.parametrize("script_name", ["generate_calvin_dev_states.py", "evaluate_calvin_dev.py"])
def test_development_cli_help_remains_on_stdout(script_name: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(CALVIN_SCRIPTS / script_name), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("usage:")
    assert "usage:" not in completed.stderr
