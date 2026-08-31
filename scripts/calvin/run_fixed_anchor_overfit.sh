#!/usr/bin/env bash
set -euo pipefail

readonly project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly environment_path="${cache_root}/venvs/train"

if [[ ! -x "${environment_path}/bin/torchrun" ]]; then
  echo "Train environment is missing; run ./scripts/bootstrap_train_env.sh" >&2
  exit 1
fi

help_only="false"
preflight_only="false"
for argument in "$@"; do
  case "${argument}" in
    --help|-h)
      help_only="true"
      ;;
    --runtime-preflight-only)
      preflight_only="true"
      ;;
  esac
done

while IFS='=' read -r name _value; do
  if [[ "${name}" == NCCL_* ]]; then
    unset "${name}"
  fi
done < <(env)
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONINSPECT PYTHONSTARTUP

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1"
export DUO_VLA_CACHE_ROOT="${cache_root}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONPATH="${project_dir}/src"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1

if [[ "${help_only}" == "true" || "${preflight_only}" == "true" ]]; then
  exec "${environment_path}/bin/python" \
    "${project_dir}/scripts/qualify_calvin_fixed_anchor_overfit.py" \
    "$@"
fi

exec "${environment_path}/bin/torchrun" \
  --standalone \
  --nproc-per-node=2 \
  "${project_dir}/scripts/qualify_calvin_fixed_anchor_overfit.py" \
  "$@"
