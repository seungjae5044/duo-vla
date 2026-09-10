#!/bin/bash -p
set -euo pipefail

readonly requested_cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly requested_physical_gpu="${DUO_VLA_PHYSICAL_GPU:-0}"
export PATH="/usr/bin:/bin"
unset BASH_ENV CDPATH ENV GLOBIGNORE
while IFS= read -r name; do
  [[ "${name}" == LD_* ]] && unset "${name}"
done < <(compgen -v)

if [[ ! "${requested_physical_gpu}" =~ ^(0|1)$ ]]; then
  echo "DUO_VLA_PHYSICAL_GPU must be exactly 0 or 1" >&2
  exit 2
fi

readonly project_dir="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd -P)"
readonly cache_root="$(realpath -m -- "${requested_cache_root}")"
readonly environment_path="${cache_root}/venvs/libero-eval"
readonly python_bootstrap='import runpy, sys; source_root, entrypoint, *arguments = sys.argv[1:]; sys.path.insert(0, source_root); sys.argv = [entrypoint, *arguments]; runpy.run_path(entrypoint, run_name="__main__")'

if [[ ! -x "${environment_path}/bin/python" ]]; then
  echo "LIBERO environment is missing; run ./scripts/bootstrap_libero_env.sh" >&2
  exit 1
fi

exec /usr/bin/env -i \
  "CUDA_DEVICE_ORDER=PCI_BUS_ID" \
  "CUDA_VISIBLE_DEVICES=${requested_physical_gpu}" \
  "DUO_VLA_CACHE_ROOT=${cache_root}" \
  "HF_HOME=/root/.cache/huggingface" \
  "LANG=C.UTF-8" \
  "LC_ALL=C.UTF-8" \
  "LIBERO_CONFIG_PATH=${cache_root}/simulators/libero/config" \
  "MKL_NUM_THREADS=1" \
  "MUJOCO_EGL_DEVICE_ID=${requested_physical_gpu}" \
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
  "TZ=UTC" \
  "${environment_path}/bin/python" -P -B -X pycache_prefix=/dev/null \
  -c "${python_bootstrap}" \
  "${project_dir}/src" \
  "${project_dir}/scripts/generate_libero_dev_states.py" \
  "$@"
