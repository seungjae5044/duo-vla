#!/usr/bin/env bash
set -euo pipefail

cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
environment_path="${cache_root}/venvs/train"
hf_home="${HF_HOME:-/root/.cache/huggingface}"

if [[ ! -x "${environment_path}/bin/hf" ]]; then
  echo "training environment is missing; run scripts/bootstrap_train_env.sh first" >&2
  exit 1
fi

HF_HOME="${hf_home}" HF_XET_HIGH_PERFORMANCE=1 \
  "${environment_path}/bin/hf" download google/diffusiongemma-26B-A4B-it \
  --revision f7f5b7f5fa82ffc52addd066915886d497f5517b \
  --include '*.safetensors' --include '*.json' --include '*.jinja'
