#!/bin/bash
set -euo pipefail
export PATH="/usr/bin:/bin"
readonly cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
unset BASH_ENV ENV
while IFS= read -r name; do
  [[ "${name}" == LD_* ]] && unset "${name}"
done < <(compgen -v)

readonly project_dir="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
  "PYTHONPATH=${project_dir}/src:${project_dir}/scripts" \
  "${environment_path}/bin/python" \
  "${project_dir}/scripts/create_libero_preregistration.py" \
  "$@"
