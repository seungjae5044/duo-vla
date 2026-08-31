#!/bin/bash -p
set -euo pipefail

readonly requested_cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly requested_hf_home="${HF_HOME:-/root/.cache/huggingface}"
readonly script_directory="${BASH_SOURCE[0]%/*}"

# The development endpoint uses the same closed Python/CUDA environment as the
# official inference core, but a different executable, socket, and wire schema.
while IFS= read -r environment_name; do
  case "${environment_name}" in
    BLIS_*|CUBLAS_*|CUDA_*|CUDNN_*|DUO_VLA_*|GOMP_*|HF_*|KMP_*|LANG|LANGUAGE|LC_*|LD_*|MKL_*|NCCL_*|NVIDIA_*|NUMEXPR_*|OMP_*|OPENBLAS_*|PYTHON*|PYTORCH_*|RAYON_*|TOKENIZERS_*|TORCH_*|TRANSFORMERS_*|TZ|VECLIB_*)
      unset "${environment_name}"
      ;;
  esac
done < <(compgen -e)
unset BASH_ENV CDPATH ENV GLOBIGNORE

export PATH="/usr/bin:/bin"
readonly project_dir="$(cd "${script_directory}/../.." && pwd -P)"
readonly cache_root="$(realpath -m "${requested_cache_root}")"
readonly hf_home="$(realpath -m "${requested_hf_home}")"
readonly environment_path="${cache_root}/venvs/train"

if [[ ! -x "${environment_path}/bin/python" ]]; then
  echo "Train environment is missing; run ./scripts/bootstrap_train_env.sh" >&2
  exit 1
fi

export BLIS_NUM_THREADS="1"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1"
export DUO_VLA_CACHE_ROOT="${cache_root}"
export DUO_VLA_PROJECT_ROOT="${project_dir}"
export DUO_VLA_TRAIN_VENV="${environment_path}"
export HF_HOME="${hf_home}"
export HF_HUB_OFFLINE="1"
export LANG="C.UTF-8"
export LC_ALL="C.UTF-8"
export MKL_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export OMP_DYNAMIC="FALSE"
export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export PYTHONHASHSEED="0"
export PYTHONNOUSERSITE="1"
export PYTHONPATH="${project_dir}/src:${project_dir}/scripts/calvin"
export PYTHONSAFEPATH="1"
export PYTHONDONTWRITEBYTECODE="1"
export RAYON_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="false"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export TRANSFORMERS_OFFLINE="1"
export TZ="UTC"
export VECLIB_MAXIMUM_THREADS="1"

for argument in "$@"; do
  if [[ "${argument}" == "--preflight-only" || "${argument}" == "--fake-policy" || "${argument}" == "--help" || "${argument}" == "-h" ]]; then
    exec "${environment_path}/bin/python" "${project_dir}/scripts/calvin/serve_policy_dev.py" "$@"
  fi
done

exec "${environment_path}/bin/torchrun" \
  --standalone \
  --nproc-per-node=2 \
  "${project_dir}/scripts/calvin/serve_policy_dev.py" \
  "$@"
