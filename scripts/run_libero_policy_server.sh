#!/bin/bash -p
set -euo pipefail

readonly requested_cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly requested_hf_home="${HF_HOME:-/root/.cache/huggingface}"
export PATH="/usr/bin:/bin"
unset BASH_ENV CDPATH ENV GLOBIGNORE

readonly project_dir="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd -P)"
readonly cache_root="$(realpath -m -- "${requested_cache_root}")"
readonly hf_home="$(realpath -m -- "${requested_hf_home}")"
readonly environment_path="${cache_root}/venvs/train"

if [[ ! -x "${environment_path}/bin/python" ]]; then
  echo "Train environment is missing; run ./scripts/bootstrap_train_env.sh" >&2
  exit 1
fi

canonical_environment=(
  "BLIS_NUM_THREADS=1"
  "CUBLAS_WORKSPACE_CONFIG=:4096:8"
  "CUDA_DEVICE_ORDER=PCI_BUS_ID"
  "CUDA_VISIBLE_DEVICES=0,1"
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
  "PYTHONHASHSEED=0"
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
for argument in "$@"; do
  if [[ "${argument}" == "--preflight-only" || "${argument}" == "--fake-policy" || "${argument}" == "--help" || "${argument}" == "-h" ]]; then
    exec /usr/bin/env -i "${canonical_environment[@]}" \
      "${environment_path}/bin/python" -P -B -X pycache_prefix=/dev/null \
      "${project_dir}/scripts/serve_libero_policy.py" "$@"
  fi
done

exec /usr/bin/env -i "${canonical_environment[@]}" \
  "${environment_path}/bin/python" -P -B -X pycache_prefix=/dev/null \
  -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  "${project_dir}/scripts/serve_libero_policy.py" \
  "$@"
