from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path

import pytest

from duo_vla.runtime_integrity import (
    canonical_sha256,
    canonical_visible_cuda_world_size,
    content_address_train_venv,
    require_matching_train_venv,
    static_environment_identity,
    validate_torchrun_rank_environment,
)


def _fake_venv(root: Path) -> Path:
    base = root / "base-python"
    (base / "bin").mkdir(parents=True)
    (base / "lib/python3.11/site-packages").mkdir(parents=True)
    executable = base / "bin/python3.11"
    executable.write_bytes(b"fake-cpython-runtime\n")
    executable.chmod(0o755)
    venv = root / "venv"
    site_packages = venv / "lib/python3.11/site-packages"
    package = site_packages / "package"
    package.mkdir(parents=True)
    (venv / "bin").mkdir()
    (venv / "pyvenv.cfg").write_text(
        f"home = {base / 'bin'}\nversion_info = 3.11.15\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (site_packages / "known-hook.pth").write_text("import known_hook\n", encoding="utf-8")
    (venv / "bin/python").symlink_to(executable)
    return venv


def test_train_venv_identity_covers_every_regular_file_symlink_and_startup_hook(tmp_path: Path) -> None:
    venv = _fake_venv(tmp_path)

    first = content_address_train_venv(venv)
    second = content_address_train_venv(venv)

    assert first == second
    assert first["files_verified"] == 3
    assert first["symlinks_verified"] == 1
    assert first["startup_hooks"] == ["lib/python3.11/site-packages/known-hook.pth"]
    assert first["root_sha256"] == canonical_sha256(
        {
            "content_inventory_sha256": first["content_inventory_sha256"],
            "base_python_runtime_root_sha256": first["base_python_runtime"]["root_sha256"],
            "files_verified": 3,
            "schema": first["schema"],
            "startup_hooks_sha256": first["startup_hooks_sha256"],
            "symlinks_verified": 1,
            "total_bytes": first["total_bytes"],
            "tree_metadata_sha256": first["tree_metadata_sha256"],
        }
    )

    (venv / "lib/python3.11/site-packages/package/__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed_file = content_address_train_venv(venv)
    assert changed_file["content_inventory_sha256"] != first["content_inventory_sha256"]
    assert changed_file["root_sha256"] != first["root_sha256"]
    with pytest.raises(RuntimeError, match="differs from the checkpoint"):
        require_matching_train_venv(first, changed_file)

    base_executable = Path(changed_file["base_python_runtime"]["resolved_executable"])
    base_executable.write_bytes(b"patched-cpython-runtime\n")
    changed_base = content_address_train_venv(venv)
    assert changed_base["content_inventory_sha256"] == changed_file["content_inventory_sha256"]
    assert changed_base["base_python_runtime"]["root_sha256"] != changed_file["base_python_runtime"]["root_sha256"]
    assert changed_base["root_sha256"] != changed_file["root_sha256"]

    (venv / "empty-runtime-directory").mkdir()
    changed_directory_inventory = content_address_train_venv(venv)
    assert changed_directory_inventory["tree_metadata_sha256"] != changed_base["tree_metadata_sha256"]
    assert changed_directory_inventory["root_sha256"] != changed_base["root_sha256"]


def test_train_venv_identity_tracks_pth_bytes_and_rejects_customization_hooks(tmp_path: Path) -> None:
    venv = _fake_venv(tmp_path)
    pth = venv / "lib/python3.11/site-packages/known-hook.pth"
    first = content_address_train_venv(venv)

    pth.write_text("import changed_hook\n", encoding="utf-8")
    changed = content_address_train_venv(venv)
    assert changed["startup_hooks_sha256"] != first["startup_hooks_sha256"]
    assert changed["root_sha256"] != first["root_sha256"]

    (pth.parent / "sitecustomize.py").write_text("raise SystemExit\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden Python startup hook"):
        content_address_train_venv(venv)


def _torchrun_environment() -> dict[str, str]:
    return {
        "GROUP_RANK": "0",
        "GROUP_WORLD_SIZE": "1",
        "LOCAL_RANK": "1",
        "LOCAL_WORLD_SIZE": "2",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29400",
        "RANK": "1",
        "ROLE_NAME": "default",
        "ROLE_RANK": "1",
        "ROLE_WORLD_SIZE": "2",
        "TORCHELASTIC_ERROR_FILE": "/tmp/torchelastic/error.json",
        "TORCHELASTIC_MAX_RESTARTS": "0",
        "TORCHELASTIC_RESTART_COUNT": "0",
        "TORCHELASTIC_RUN_ID": "run-id",
        "TORCHELASTIC_SIGNALS_TO_HANDLE": "SIGTERM,SIGINT,SIGHUP,SIGQUIT",
        "TORCHELASTIC_USE_AGENT_STORE": "True",
        "WORLD_SIZE": "2",
    }


def test_torchrun_rank_environment_is_explicit_and_tp_ranks_must_match() -> None:
    environment = _torchrun_environment()
    assert validate_torchrun_rank_environment(environment, required=True) == {
        "group_world_size": 1,
        "local_rank_equals_rank": True,
        "local_world_size": 2,
        "role_world_size": 2,
        "world_size": 2,
    }
    assert validate_torchrun_rank_environment({}, required=False) == validate_torchrun_rank_environment(
        environment, required=True
    )

    changed = copy.deepcopy(environment)
    changed["LOCAL_RANK"] = "0"
    with pytest.raises(RuntimeError, match="RANK=LOCAL_RANK"):
        validate_torchrun_rank_environment(changed, required=True)

    missing = copy.deepcopy(environment)
    del missing["ROLE_WORLD_SIZE"]
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_torchrun_rank_environment(missing, required=True)


def test_single_gpu_torchrun_rank_environment_and_visible_device_are_exact() -> None:
    environment = _torchrun_environment()
    environment.update(
        {
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "RANK": "0",
            "ROLE_RANK": "0",
            "ROLE_WORLD_SIZE": "1",
            "WORLD_SIZE": "1",
        }
    )
    assert validate_torchrun_rank_environment(environment, required=True, expected_world_size=1) == {
        "group_world_size": 1,
        "local_rank_equals_rank": True,
        "local_world_size": 1,
        "role_world_size": 1,
        "world_size": 1,
    }
    assert canonical_visible_cuda_world_size({"CUDA_VISIBLE_DEVICES": "0"}) == 1
    assert canonical_visible_cuda_world_size({"CUDA_VISIBLE_DEVICES": "1"}) == 1
    assert canonical_visible_cuda_world_size({"CUDA_VISIBLE_DEVICES": "0,1"}) == 2
    with pytest.raises(RuntimeError, match="canonical launcher"):
        canonical_visible_cuda_world_size({"CUDA_VISIBLE_DEVICES": "1,0"})


def test_static_environment_identity_is_order_independent_and_tracks_overrides() -> None:
    first = static_environment_identity({"OMP_NUM_THREADS": "1", "LANG": "C.UTF-8"})
    reordered = static_environment_identity({"LANG": "C.UTF-8", "OMP_NUM_THREADS": "1"})
    changed = static_environment_identity({"LANG": "C.UTF-8", "OMP_NUM_THREADS": "8"})

    assert first == reordered
    assert changed["sha256"] != first["sha256"]


def test_train_venv_rejects_nonregular_entries(tmp_path: Path) -> None:
    venv = _fake_venv(tmp_path)
    fifo = venv / "runtime.fifo"
    os.mkfifo(fifo)
    with pytest.raises(RuntimeError, match="regular file, symlink, or real directory"):
        content_address_train_venv(venv)


@pytest.mark.parametrize(
    "launcher",
    (
        "scripts/run_libero_train.sh",
        "scripts/run_calvin_train.sh",
        "scripts/run_libero_policy_server.sh",
        "scripts/calvin/run_policy_server.sh",
    ),
)
def test_real_launchers_remove_caller_runtime_and_injection_overrides(tmp_path: Path, launcher: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    cache_root = tmp_path / "cache"
    bin_dir = cache_root / "venvs/train/bin"
    bin_dir.mkdir(parents=True)
    environment_probe = "#!/bin/bash\n/usr/bin/env\n"
    for name in ("python", "torchrun"):
        executable = bin_dir / name
        executable.write_text(environment_probe, encoding="utf-8")
        executable.chmod(0o755)
    inherited = dict(os.environ)
    inherited.update(
        {
            "BASH_ENV": "/definitely/not/a/startup/file",
            "DUO_VLA_CACHE_ROOT": str(cache_root),
            "HF_HOME": str(tmp_path / "hf"),
            "LANG": "injected-locale",
            "LC_ALL": "injected-locale",
            "MALLOC_CONF": "background_thread:true",
            "MKL_NUM_THREADS": "99",
            "NCCL_ALGO": "injected",
            "OMP_NUM_THREADS": "99",
            "PYTHONPATH": "/tmp/injected",
        }
    )

    completed = subprocess.run(
        [str(project_root / launcher), "--help"],
        check=True,
        capture_output=True,
        env=inherited,
        text=True,
    )
    observed = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)

    assert observed["LANG"] == "C.UTF-8"
    assert observed["LC_ALL"] == "C.UTF-8"
    assert observed["MKL_NUM_THREADS"] == "1"
    assert observed["OMP_NUM_THREADS"] == "1"
    assert observed["PYTHONSAFEPATH"] == "1"
    assert observed["PYTHONDONTWRITEBYTECODE"] == "1"
    for name in ("BASH_ENV", "MALLOC_CONF", "NCCL_ALGO"):
        assert name not in observed
        assert "/tmp/injected" not in observed.get("PYTHONPATH", "")


@pytest.mark.parametrize(
    "launcher",
    (
        "scripts/run_libero_train_single_gpu.sh",
        "scripts/run_calvin_train_single_gpu.sh",
        "scripts/run_libero_policy_server_single_gpu.sh",
        "scripts/calvin/run_policy_server_single_gpu.sh",
    ),
)
def test_single_gpu_launchers_seal_gpu_zero_and_tp1(tmp_path: Path, launcher: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    cache_root = tmp_path / "cache"
    bin_dir = cache_root / "venvs/train-single-gpu/bin"
    bin_dir.mkdir(parents=True)
    environment_probe = "#!/bin/bash\n/usr/bin/env\n"
    for name in ("python", "torchrun"):
        executable = bin_dir / name
        executable.write_text(environment_probe, encoding="utf-8")
        executable.chmod(0o755)
    inherited = dict(os.environ)
    inherited.update(
        {
            "DUO_VLA_CACHE_ROOT": str(cache_root),
            "HF_HOME": str(tmp_path / "hf"),
            "NCCL_ALGO": "injected",
            "PYTHONPATH": "/tmp/injected",
        }
    )

    completed = subprocess.run(
        [str(project_root / launcher), "--help"],
        check=True,
        capture_output=True,
        env=inherited,
        text=True,
    )
    observed = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)

    assert observed["CUDA_VISIBLE_DEVICES"] == "0"
    assert observed["DUO_VLA_TRAIN_VENV"] == str(cache_root / "venvs/train-single-gpu")
    assert "NCCL_ALGO" not in observed
    assert "/tmp/injected" not in observed.get("PYTHONPATH", "")


def test_libero_train_single_gpu_launcher_can_seal_physical_gpu_one(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    cache_root = tmp_path / "cache"
    bin_dir = cache_root / "venvs/train-single-gpu/bin"
    bin_dir.mkdir(parents=True)
    for executable_name in ("python", "torchrun"):
        probe = bin_dir / executable_name
        probe.write_text("#!/bin/bash\n/usr/bin/env\n", encoding="utf-8")
        probe.chmod(0o755)
    inherited = dict(os.environ)
    inherited.update(
        {
            "DUO_VLA_CACHE_ROOT": str(cache_root),
            "DUO_VLA_PHYSICAL_GPU": "1",
            "HF_HOME": str(tmp_path / "hf"),
        }
    )

    completed = subprocess.run(
        [str(project_root / "scripts/run_libero_train_single_gpu.sh"), "--help"],
        check=True,
        capture_output=True,
        env=inherited,
        text=True,
    )
    observed = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)

    assert observed["CUDA_VISIBLE_DEVICES"] == "1"
    assert "DUO_VLA_PHYSICAL_GPU" not in observed
