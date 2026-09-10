#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cache_root=${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}
source_root=${CALVIN_SOURCE_ROOT:-$cache_root/simulators/calvin}
venv_root=${CALVIN_VENV_ROOT:-$cache_root/venvs/calvin-eval}
uv_python_root=${UV_PYTHON_INSTALL_DIR:-$cache_root/uv-python}
uv_cache_root=${UV_CACHE_DIR:-$cache_root/uv-cache}
pip_cache_root=${PIP_CACHE_DIR:-$cache_root/pip-cache}
pip_timeout=${PIP_DEFAULT_TIMEOUT:-600}
pip_retries=${PIP_RETRIES:-10}

export UV_PYTHON_INSTALL_DIR="$uv_python_root"
export UV_CACHE_DIR="$uv_cache_root"
export PIP_CACHE_DIR="$pip_cache_root"
export PIP_DEFAULT_TIMEOUT="$pip_timeout"
export PIP_RETRIES="$pip_retries"

if [[ ! -d "$source_root/.git" ]]; then
  echo "CALVIN source is absent; run $script_dir/checkout.sh first" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to provision exact Python 3.8.20" >&2
  exit 1
fi

if [[ ! -x "$venv_root/bin/python" ]]; then
  mkdir -p "$(dirname -- "$venv_root")" "$uv_python_root" "$uv_cache_root" "$pip_cache_root"
  uv python install 3.8.20
  uv venv --python 3.8.20 --seed "$venv_root"
fi

python_version=$($venv_root/bin/python -c 'import platform; print(platform.python_version())')
if [[ "$python_version" != 3.8.20 ]]; then
  echo "Expected Python 3.8.20, found $python_version in $venv_root" >&2
  exit 1
fi

pip_cmd=("$venv_root/bin/python" -m pip)
"${pip_cmd[@]}" install pip==23.3.2 setuptools==57.5.0 wheel==0.38.4
"${pip_cmd[@]}" install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -c "$script_dir/constraints-py38.txt" \
  torch==1.13.1+cpu torchvision==0.14.1+cpu
"${pip_cmd[@]}" install -c "$script_dir/constraints-py38.txt" \
  GitPython PyYAML cloudpickle gym hydra-colorlog hydra-core lightning-utilities matplotlib numba \
  numpy numpy-quaternion omegaconf opencv-python-headless pandas pybullet pytorch-lightning rich scipy \
  tensorboardX termcolor torchmetrics
"${pip_cmd[@]}" install --no-build-isolation -c "$script_dir/constraints-py38.txt" pyhash
"${pip_cmd[@]}" install --no-deps -e "$source_root/calvin_env" -e "$source_root/calvin_models"

DUO_VLA_CACHE_ROOT="$cache_root" \
CALVIN_SOURCE_ROOT="$source_root" \
CALVIN_VENV_ROOT="$venv_root" \
  "$script_dir/run_official_evaluator.sh" preflight
