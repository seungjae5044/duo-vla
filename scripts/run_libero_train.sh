#!/usr/bin/env bash
set -euo pipefail

readonly project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly environment_path="${cache_root}/venvs/train"

if [[ ! -x "${environment_path}/bin/torchrun" ]]; then
  echo "Train environment is missing; run ./scripts/bootstrap_train_env.sh" >&2
  exit 1
fi
train_seed="${PYTHONHASHSEED:-}"
help_only="false"
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  case "${arguments[index]}" in
    --help|-h)
      help_only="true"
      ;;
    --seed)
      ((index + 1 < ${#arguments[@]})) || {
        echo "--seed requires a value" >&2
        exit 2
      }
      train_seed="${arguments[index + 1]}"
      ((index += 1))
      ;;
    --seed=*)
      train_seed="${arguments[index]#--seed=}"
      ;;
  esac
done
if [[ "${help_only}" == "true" ]]; then
  train_seed="0"
elif [[ ! "${train_seed}" =~ ^(0|1|2)$ ]]; then
  echo "Canonical LIBERO training requires --seed or PYTHONHASHSEED in {0,1,2}" >&2
  exit 2
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
export HF_HUB_DISABLE_PROGRESS_BARS="1"
export HF_HUB_OFFLINE="1"
export MKL_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export OMP_DYNAMIC="FALSE"
export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export PYTHONHASHSEED="${train_seed}"
export PYTHONNOUSERSITE="1"
export PYTHONPATH="${project_dir}/src"
export TOKENIZERS_PARALLELISM="false"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export TRANSFORMERS_OFFLINE="1"

# The host image exposes cuDNN 9.8 through LD_LIBRARY_PATH, while the locked
# PyTorch wheel bundles the required cuDNN 9.10.2. Let the wheel's rpath win.

if [[ "${help_only}" == "true" ]]; then
  exec "${environment_path}/bin/python" "${project_dir}/scripts/train_libero.py" "$@"
fi

cd "${project_dir}"
exec "${environment_path}/bin/torchrun" \
  --standalone \
  --nproc-per-node=2 \
  "${project_dir}/scripts/train_libero.py" \
  "$@"
