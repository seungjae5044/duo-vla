#!/usr/bin/env bash
set -euo pipefail

readonly project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly runtime_root="${cache_root}/simulators/libero"
readonly environment_path="${cache_root}/venvs/libero-eval"
readonly source_path="${runtime_root}/source"
readonly assets_path="${runtime_root}/assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
readonly config_path="${runtime_root}/config"
readonly dataset_path="${cache_root}/data/libero/simulator-datasets"
readonly hf_home="${HF_HOME:-/root/.cache/huggingface}"
readonly uv_cache="${cache_root}/uv-cache"
readonly uv_python_root="${UV_PYTHON_INSTALL_DIR:-${cache_root}/uv-python}"
readonly python_version="3.12.13"
readonly source_url="https://github.com/huggingface/LIBERO.git"
readonly source_revision="8561c60eea2fb93096146f240194649df73d8b1e"
readonly assets_repo="lerobot/libero-assets"
readonly assets_revision="0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
readonly http_timeout="${UV_HTTP_TIMEOUT:-600}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "LIBERO evaluation is supported only on Linux" >&2
  exit 1
fi
for command_name in git uv; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "required command not found: ${command_name}" >&2
    exit 1
  fi
done

mkdir -p \
  "${runtime_root}/assets" \
  "${config_path}" \
  "${dataset_path}" \
  "${cache_root}/venvs" \
  "${hf_home}" \
  "${uv_cache}" \
  "${uv_python_root}"

if [[ -e "${source_path}" && ! -d "${source_path}/.git" ]]; then
  echo "refusing to replace non-git path: ${source_path}" >&2
  exit 1
fi
if [[ ! -d "${source_path}/.git" ]]; then
  git clone --filter=blob:none "${source_url}" "${source_path}"
fi
if [[ "$(git -C "${source_path}" remote get-url origin)" != "${source_url}" ]]; then
  echo "unexpected LIBERO origin at ${source_path}" >&2
  exit 1
fi
if [[ -n "$(git -C "${source_path}" status --porcelain)" ]]; then
  echo "refusing to alter a modified LIBERO source checkout: ${source_path}" >&2
  exit 1
fi
git -C "${source_path}" fetch --depth 1 origin "${source_revision}"
git -C "${source_path}" checkout --quiet --detach "${source_revision}"
if [[ "$(git -C "${source_path}" rev-parse HEAD)" != "${source_revision}" ]]; then
  echo "LIBERO source revision verification failed" >&2
  exit 1
fi

UV_CACHE_DIR="${uv_cache}" UV_PYTHON_INSTALL_DIR="${uv_python_root}" \
  uv python install "${python_version}"

UV_CACHE_DIR="${uv_cache}" \
UV_PYTHON_INSTALL_DIR="${uv_python_root}" \
UV_HTTP_TIMEOUT="${http_timeout}" \
UV_PROJECT_ENVIRONMENT="${environment_path}" \
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  uv sync --project "${project_dir}/envs/libero-eval" --python "${python_version}" --frozen

HF_HOME="${hf_home}" "${environment_path}/bin/python" - "${assets_path}" "${assets_repo}" "${assets_revision}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

destination = Path(sys.argv[1])
repo_id = sys.argv[2]
revision = sys.argv[3]
local_metadata = destination / ".cache"
external_metadata = destination.parents[1] / "download-metadata" / f"assets-{revision}"
if local_metadata.exists() and external_metadata.exists():
    raise RuntimeError("LIBERO asset download metadata exists in both the content and external roots")
if external_metadata.exists():
    local_metadata.parent.mkdir(parents=True, exist_ok=True)
    external_metadata.rename(local_metadata)
observed = HfApi().dataset_info(repo_id, revision=revision).sha
if observed != revision:
    raise RuntimeError(f"asset revision mismatch: expected {revision}, observed {observed}")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    revision=revision,
    local_dir=destination,
)
if not local_metadata.is_dir():
    raise RuntimeError("LIBERO asset download did not create its expected local metadata cache")
external_metadata.parent.mkdir(parents=True, exist_ok=True)
local_metadata.rename(external_metadata)
required = (
    "articulated_objects",
    "scenes",
    "stable_hope_objects",
    "stable_scanned_objects",
    "turbosquid_objects",
)
missing = [name for name in required if not (destination / name).is_dir()]
if missing:
    raise RuntimeError(f"incomplete LIBERO assets at {destination}: {missing}")
PY

LIBERO_RUNTIME_ROOT="${runtime_root}" \
LIBERO_ASSETS_PATH="${assets_path}" \
LIBERO_CONFIG_PATH="${config_path}" \
LIBERO_DATASET_PATH="${dataset_path}" \
LIBERO_SOURCE_PATH="${source_path}" \
LIBERO_SOURCE_URL="${source_url}" \
LIBERO_SOURCE_REVISION="${source_revision}" \
LIBERO_ASSETS_REPO="${assets_repo}" \
LIBERO_ASSETS_REVISION="${assets_revision}" \
LIBERO_LOCK_PATH="${project_dir}/envs/libero-eval/uv.lock" \
  "${environment_path}/bin/python" <<'PY'
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

import yaml


def write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


runtime_root = Path(os.environ["LIBERO_RUNTIME_ROOT"])
assets_path = Path(os.environ["LIBERO_ASSETS_PATH"]).resolve()
config_path = Path(os.environ["LIBERO_CONFIG_PATH"])
dataset_path = Path(os.environ["LIBERO_DATASET_PATH"])
source_path = Path(os.environ["LIBERO_SOURCE_PATH"])
lock_path = Path(os.environ["LIBERO_LOCK_PATH"])

# find_spec avoids executing libero/libero/__init__.py before its noninteractive
# configuration exists.
package_spec = importlib.util.find_spec("libero")
if package_spec is None or not package_spec.submodule_search_locations:
    raise RuntimeError("hf-libero is installed but the libero package cannot be located")
package_root = Path(next(iter(package_spec.submodule_search_locations))) / "libero"
asset_link = package_root / "assets"
if asset_link.is_symlink():
    if asset_link.resolve() != assets_path:
        raise RuntimeError(f"refusing to replace unexpected asset link: {asset_link}")
elif asset_link.exists():
    raise RuntimeError(f"refusing to replace unexpected asset path: {asset_link}")
else:
    asset_link.symlink_to(assets_path, target_is_directory=True)

config = {
    "assets": str(assets_path),
    "bddl_files": str(package_root / "bddl_files"),
    "benchmark_root": str(package_root),
    "datasets": str(dataset_path),
    "init_states": str(package_root / "init_files"),
}
write_if_changed(config_path / "config.yaml", yaml.safe_dump(config, sort_keys=True))

manifest = {
    "schema": "duo-vla-libero-simulator-v1",
    "source": {
        "url": os.environ["LIBERO_SOURCE_URL"],
        "revision": os.environ["LIBERO_SOURCE_REVISION"],
        "path": str(source_path),
    },
    "assets": {
        "repo_type": "dataset",
        "repo_id": os.environ["LIBERO_ASSETS_REPO"],
        "revision": os.environ["LIBERO_ASSETS_REVISION"],
        "path": str(assets_path),
    },
    "environment": {
        "python": os.sys.version.split()[0],
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "cmake",
                "hf-egl-probe",
                "hf-libero",
                "huggingface-hub",
                "mujoco",
                "numpy",
                "robosuite",
                "torch",
                "torchvision",
            )
        },
        "uv_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    },
    "paths": {
        "config": str(config_path / "config.yaml"),
        "datasets": str(dataset_path),
    },
    "training_data_downloaded": False,
}
write_if_changed(runtime_root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

DUO_VLA_CACHE_ROOT="${cache_root}" \
  "${project_dir}/scripts/run_libero_preflight.sh"
