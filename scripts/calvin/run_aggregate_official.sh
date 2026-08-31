#!/bin/bash -p
set -euo pipefail

export PATH="/usr/bin:/bin"
unset BASH_ENV CDPATH ENV GLOBIGNORE
script_dir=$(cd -- "$(/usr/bin/dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cache_root=${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}
venv_root=${CALVIN_VENV_ROOT:-$cache_root/venvs/calvin-eval}

if [[ ! -x "$venv_root/bin/python" ]]; then
  echo "CALVIN evaluation environment is missing; run scripts/calvin/bootstrap_env.sh" >&2
  exit 1
fi

exec /usr/bin/env -i \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PATH="$venv_root/bin:/usr/bin:/bin" \
  PYTHONHASHSEED=0 \
  PYTHONNOUSERSITE=1 \
  "$venv_root/bin/python" -B \
  "$script_dir/aggregate_calvin_official.py" \
  "$@"
