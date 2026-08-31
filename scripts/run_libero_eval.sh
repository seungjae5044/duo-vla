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
readonly environment_path="${cache_root}/venvs/libero-eval"

if [[ ! -x "${environment_path}/bin/python" ]]; then
  echo "LIBERO environment is missing; run ./scripts/bootstrap_libero_env.sh" >&2
  exit 1
fi

exec /usr/bin/env -i \
  "DUO_VLA_CACHE_ROOT=${cache_root}" \
  "HF_HOME=/root/.cache/huggingface" \
  "LANG=C.UTF-8" \
  "LC_ALL=C.UTF-8" \
  "LIBERO_CONFIG_PATH=${cache_root}/simulators/libero/config" \
  "MKL_NUM_THREADS=1" \
  "MUJOCO_EGL_DEVICE_ID=0" \
  "MUJOCO_GL=egl" \
  "NUMEXPR_NUM_THREADS=1" \
  "OMP_DYNAMIC=FALSE" \
  "OMP_NUM_THREADS=1" \
  "OPENBLAS_NUM_THREADS=1" \
  "PATH=${environment_path}/bin:/usr/bin:/bin" \
  "PYOPENGL_PLATFORM=egl" \
  "PYTHONHASHSEED=0" \
  "PYTHONNOUSERSITE=1" \
  "PYTHONSAFEPATH=1" \
  "PYTHONDONTWRITEBYTECODE=1" \
  "${environment_path}/bin/python" -P -B -X pycache_prefix=/dev/null \
  "${project_dir}/scripts/evaluate_libero.py" \
  "$@"
