#!/bin/bash -p
set -euo pipefail

readonly requested_cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
export PATH="/usr/bin:/bin"
unset BASH_ENV CDPATH ENV GLOBIGNORE
while IFS= read -r name; do
  [[ "${name}" == LD_* ]] && unset "${name}"
done < <(compgen -v)

readonly project_dir="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd -P)"
readonly cache_root="$(realpath -m -- "${requested_cache_root}")"
readonly environment_path="${cache_root}/venvs/train-single-gpu"

if [[ ! -x "${environment_path}/bin/python" ]]; then
  echo "Single-GPU train environment is missing; run ./scripts/bootstrap_train_single_gpu_env.sh" >&2
  exit 1
fi

exec /usr/bin/env -i \
  "DUO_VLA_CACHE_ROOT=${cache_root}" \
  "DUO_VLA_PROJECT_ROOT=${project_dir}" \
  "DUO_VLA_TRAIN_VENV=${environment_path}" \
  "HF_HOME=/root/.cache/huggingface" \
  "HF_HUB_OFFLINE=1" \
  "HOME=/root" \
  "LANG=C.UTF-8" \
  "LC_ALL=C.UTF-8" \
  "MKL_NUM_THREADS=1" \
  "NUMEXPR_NUM_THREADS=1" \
  "OMP_DYNAMIC=FALSE" \
  "OMP_NUM_THREADS=1" \
  "OPENBLAS_NUM_THREADS=1" \
  "PATH=${environment_path}/bin:/usr/bin:/bin" \
  "PYTHONHASHSEED=0" \
  "PYTHONNOUSERSITE=1" \
  "PYTHONSAFEPATH=1" \
  "PYTHONDONTWRITEBYTECODE=1" \
  "RAYON_NUM_THREADS=1" \
  "TOKENIZERS_PARALLELISM=false" \
  "TRANSFORMERS_OFFLINE=1" \
  "TZ=UTC" \
  "VECLIB_MAXIMUM_THREADS=1" \
  "${environment_path}/bin/python" -P -B -X pycache_prefix=/dev/null \
  "${project_dir}/scripts/bind_libero_expert_replay.py" \
  "$@"
