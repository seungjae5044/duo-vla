#!/usr/bin/env bash
set -euo pipefail

readonly project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly environment_path="${cache_root}/venvs/train"

if [[ ! -x "${environment_path}/bin/python" ]]; then
  echo "Train environment is missing; run ./scripts/bootstrap_train_env.sh" >&2
  exit 1
fi

while IFS='=' read -r name _value; do
  case "${name}" in
    CUBLAS_*|CUDA_*|CUDNN_*|NCCL_*|PYTORCH_*|TORCH_*)
      unset "${name}"
      ;;
  esac
done < <(env)
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONINSPECT PYTHONSTARTUP

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1"
export DUO_VLA_CACHE_ROOT="${cache_root}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="1"
export MKL_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export OMP_DYNAMIC="FALSE"
export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export PYTHONHASHSEED="0"
export PYTHONNOUSERSITE="1"
export PYTHONPATH="${project_dir}/src:${project_dir}/scripts"
export TOKENIZERS_PARALLELISM="false"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export TRANSFORMERS_OFFLINE="1"

for argument in "$@"; do
  if [[ "${argument}" == "--preflight-only" || "${argument}" == "--fake-policy" || "${argument}" == "--help" || "${argument}" == "-h" ]]; then
    exec "${environment_path}/bin/python" "${project_dir}/scripts/serve_libero_policy.py" "$@"
  fi
done

exec "${environment_path}/bin/torchrun" \
  --standalone \
  --nproc-per-node=2 \
  "${project_dir}/scripts/serve_libero_policy.py" \
  "$@"
