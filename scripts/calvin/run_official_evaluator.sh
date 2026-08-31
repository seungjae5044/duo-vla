#!/bin/bash -p
set -euo pipefail

readonly requested_cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly requested_venv_root="${CALVIN_VENV_ROOT:-${requested_cache_root}/venvs/calvin-eval}"
readonly requested_source_root="${CALVIN_SOURCE_ROOT:-${requested_cache_root}/simulators/calvin}"
export PATH="/usr/bin:/bin"
unset BASH_ENV CDPATH ENV GLOBIGNORE

readonly script_directory="$(cd -- "$(/usr/bin/dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly project_root="$(cd -- "${script_directory}/../.." && pwd -P)"
readonly cache_root="$(realpath -m -- "${requested_cache_root}")"
readonly venv_root="$(realpath -m -- "${requested_venv_root}")"
readonly source_root="$(realpath -m -- "${requested_source_root}")"

if [[ ! -x "${venv_root}/bin/python" ]]; then
  echo "CALVIN evaluation environment is missing; run scripts/calvin/bootstrap_env.sh" >&2
  exit 1
fi
if [[ ! -d "${source_root}" ]]; then
  echo "CALVIN source checkout is missing: ${source_root}" >&2
  exit 1
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 {preflight|infrastructure|official-score} [arguments...]" >&2
  exit 2
fi

readonly operation="$1"
shift
for argument in "$@"; do
  if [[ "${argument}" == "--source-root" || "${argument}" == --source-root=* ]]; then
    echo "--source-root is owned by the canonical launcher; set CALVIN_SOURCE_ROOT before invoking it" >&2
    exit 2
  fi
done

case "${operation}" in
  preflight)
    readonly entrypoint="${project_root}/scripts/calvin/preflight.py"
    readonly mode_arguments=(--source-root "${source_root}")
    ;;
  infrastructure)
    readonly entrypoint="${project_root}/scripts/calvin/evaluate_calvin.py"
    readonly mode_arguments=(--mode infrastructure --source-root "${source_root}")
    ;;
  official-score)
    readonly entrypoint="${project_root}/scripts/calvin/evaluate_calvin.py"
    readonly mode_arguments=(--mode official-score --source-root "${source_root}")
    ;;
  *)
    echo "unknown CALVIN evaluator operation: ${operation}" >&2
    exit 2
    ;;
esac

# This exact environment was qualified with the source-only EGL smoke on the
# evaluation host.  No fallback or caller-selected device mapping is allowed.
exec /usr/bin/env -i \
  CALVIN_SOURCE_ROOT="${source_root}" \
  CUDA_VISIBLE_DEVICES=0 \
  DUO_VLA_CACHE_ROOT="${cache_root}" \
  EGL_PLATFORM=surfaceless \
  EGL_VISIBLE_DEVICES=0 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  PATH="${venv_root}/bin:/usr/bin:/bin" \
  PYOPENGL_PLATFORM=egl \
  PYTHONHASHSEED=0 \
  PYTHONNOUSERSITE=1 \
  TZ=UTC \
  "${venv_root}/bin/python" -B \
  "${entrypoint}" \
  "${mode_arguments[@]}" \
  "$@"
