#!/usr/bin/env bash
set -euo pipefail

readonly repo_id="HuggingFaceVLA/libero"
readonly revision="86958911c0f959db2bbbdb107eb3e17c5f9c798e"
readonly cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly hf_home="${HF_HOME:-/root/.cache/huggingface}"
readonly environment_path="${DUO_VLA_TRAIN_VENV:-${cache_root}/venvs/train}"

if [[ ! -x "${environment_path}/bin/hf" ]]; then
  echo "training environment is missing; run the matching bootstrap_train*_env.sh first" >&2
  exit 1
fi

mkdir -p "${hf_home}"

HF_HOME="${hf_home}" \
HF_XET_HIGH_PERFORMANCE=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/dev/null \
  "${environment_path}/bin/hf" download "${repo_id}" \
  --repo-type dataset \
  --revision "${revision}"
