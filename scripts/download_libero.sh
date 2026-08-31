#!/usr/bin/env bash
set -euo pipefail

readonly repo_id="HuggingFaceVLA/libero"
readonly revision="86958911c0f959db2bbbdb107eb3e17c5f9c798e"
readonly hf_home="/root/.cache/huggingface"
readonly environment_path="/root/.cache/duo-vla/venvs/train"

if [[ ! -x "${environment_path}/bin/hf" ]]; then
  echo "training environment is missing; run scripts/bootstrap_train_env.sh first" >&2
  exit 1
fi

mkdir -p "${hf_home}"

HF_HOME="${hf_home}" \
HF_XET_HIGH_PERFORMANCE=1 \
  "${environment_path}/bin/hf" download "${repo_id}" \
  --repo-type dataset \
  --revision "${revision}"
