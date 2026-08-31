#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
environment_path="${cache_root}/venvs/train"
http_timeout="${UV_HTTP_TIMEOUT:-600}"

mkdir -p "${cache_root}/models" "${cache_root}/data" "${cache_root}/venvs"
cd "${project_dir}"

UV_HTTP_TIMEOUT="${http_timeout}" UV_PROJECT_ENVIRONMENT="${environment_path}" \
  uv sync --frozen --extra train --extra data --extra dev

env -u LD_LIBRARY_PATH "${environment_path}/bin/python" - <<'PY'
import torch
import transformers
import peft
import tokenizers

from transformers import DiffusionGemmaForBlockDiffusion
from transformers.integrations.tensor_parallel import add_tensor_parallel_hooks_to_module

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.is_bf16_supported(), "the selected GPU/runtime does not support BF16"
assert transformers.__version__ == "5.15.0", "unsupported Transformers version"
assert peft.__version__ == "0.20.0", "unsupported PEFT version"
assert tokenizers.__version__ == "0.22.2", "unsupported Tokenizers version"
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__} peft={peft.__version__} tokenizers={tokenizers.__version__}")
print(f"diffusion_gemma={DiffusionGemmaForBlockDiffusion.__name__}")
print(f"tp_peft_hook={add_tensor_parallel_hooks_to_module.__name__}")
print(f"gpus={torch.cuda.device_count()}")
PY
