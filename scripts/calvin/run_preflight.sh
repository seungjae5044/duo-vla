#!/bin/bash -p
set -euo pipefail

export PATH="/usr/bin:/bin"
unset BASH_ENV CDPATH ENV GLOBIGNORE
script_dir=$(cd -- "$(/usr/bin/dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

args=()
if [[ -n "${CALVIN_DATASET_ROOT:-}" ]]; then
  args+=(--dataset-root "$CALVIN_DATASET_ROOT")
fi
if [[ "${CALVIN_REQUIRE_DATASET:-0}" == 1 ]]; then
  args+=(--require-dataset)
fi
if [[ "${CALVIN_PREFLIGHT_EGL:-0}" == 1 ]]; then
  args+=(--egl)
fi

exec "$script_dir/run_official_evaluator.sh" preflight "${args[@]}"
