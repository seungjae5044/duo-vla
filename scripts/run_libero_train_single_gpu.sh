#!/bin/bash -p
set -euo pipefail

readonly requested_cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly requested_hf_home="${HF_HOME:-/root/.cache/huggingface}"
readonly requested_physical_gpu="${DUO_VLA_PHYSICAL_GPU:-0}"
export PATH="/usr/bin:/bin"
unset BASH_ENV CDPATH ENV GLOBIGNORE

if [[ ! "${requested_physical_gpu}" =~ ^(0|1)$ ]]; then
  echo "DUO_VLA_PHYSICAL_GPU must select qualified physical GPU 0 or 1" >&2
  exit 2
fi

readonly project_dir="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd -P)"
readonly cache_root="$(realpath -m -- "${requested_cache_root}")"
readonly hf_home="$(realpath -m -- "${requested_hf_home}")"
readonly environment_path="${cache_root}/venvs/train-single-gpu"

if [[ ! -x "${environment_path}/bin/torchrun" ]]; then
  echo "Single-GPU train environment is missing; run ./scripts/bootstrap_train_single_gpu_env.sh" >&2
  exit 1
fi

train_seed=""
topology_fork_mode="false"
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
    --topology-fork-from|--topology-fork-from=*)
      topology_fork_mode="true"
      ;;
  esac
done
if [[ "${help_only}" == "true" ]]; then
  train_seed="0"
elif [[ "${topology_fork_mode}" == "true" ]]; then
  if [[ -n "${train_seed}" ]]; then
    echo "--seed is sealed by --topology-fork-from and must not be supplied" >&2
    exit 2
  fi
  train_seed="0"
elif [[ ! "${train_seed}" =~ ^(0|1|2)$ ]]; then
  echo "Single-GPU LIBERO training requires an explicit --seed in {0,1,2}" >&2
  exit 2
fi

canonical_environment=(
  "BLIS_NUM_THREADS=1"
  "CUBLAS_WORKSPACE_CONFIG=:4096:8"
  "CUDA_DEVICE_ORDER=PCI_BUS_ID"
  "CUDA_VISIBLE_DEVICES=${requested_physical_gpu}"
  "DUO_VLA_CACHE_ROOT=${cache_root}"
  "DUO_VLA_PROJECT_ROOT=${project_dir}"
  "DUO_VLA_TRAIN_VENV=${environment_path}"
  "HF_HOME=${hf_home}"
  "HF_HUB_DISABLE_PROGRESS_BARS=1"
  "HF_HUB_OFFLINE=1"
  "HOME=/root"
  "LANG=C.UTF-8"
  "LC_ALL=C.UTF-8"
  "MKL_NUM_THREADS=1"
  "NUMEXPR_NUM_THREADS=1"
  "OMP_DYNAMIC=FALSE"
  "OMP_NUM_THREADS=1"
  "OPENBLAS_NUM_THREADS=1"
  "PATH=/usr/bin:/bin"
  "PYTHONHASHSEED=${train_seed}"
  "PYTHONNOUSERSITE=1"
  "PYTHONPYCACHEPREFIX=/dev/null"
  "PYTHONSAFEPATH=1"
  "PYTHONDONTWRITEBYTECODE=1"
  "RAYON_NUM_THREADS=1"
  "TOKENIZERS_PARALLELISM=false"
  "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
  "TRANSFORMERS_OFFLINE=1"
  "TZ=UTC"
  "VECLIB_MAXIMUM_THREADS=1"
)

cd "${project_dir}"
if [[ "${help_only}" == "true" ]]; then
  exec /usr/bin/env -i "${canonical_environment[@]}" \
    "${environment_path}/bin/python" -P -B -X pycache_prefix=/dev/null \
    "${project_dir}/scripts/train_libero.py" "$@"
fi

exec /usr/bin/env -i "${canonical_environment[@]}" \
  "${environment_path}/bin/python" -P -B -X pycache_prefix=/dev/null \
  -m torch.distributed.run \
  --standalone \
  --nproc-per-node=1 \
  "${project_dir}/scripts/train_libero.py" \
  "$@"
