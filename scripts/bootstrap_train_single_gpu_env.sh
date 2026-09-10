#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
environment_path="${cache_root}/venvs/train-single-gpu"
http_timeout="${UV_HTTP_TIMEOUT:-600}"

mkdir -p "${cache_root}/models" "${cache_root}/data" "${cache_root}/venvs"

UV_HTTP_TIMEOUT="${http_timeout}" UV_PROJECT_ENVIRONMENT="${environment_path}" \
  uv sync --project "${project_dir}/envs/train-single-gpu" --frozen
UV_HTTP_TIMEOUT="${http_timeout}" uv pip install \
  --python "${environment_path}/bin/python" \
  --no-deps \
  --editable "${project_dir}"

CUDA_VISIBLE_DEVICES=0 env -u LD_LIBRARY_PATH "${environment_path}/bin/python" - <<'PY'
import torch
import transformers
import peft
import tokenizers

from transformers import DiffusionGemmaForBlockDiffusion

assert torch.__version__ == "2.13.0+cu129", "unsupported single-GPU PyTorch build"
assert torch.version.cuda == "12.9", "single-GPU runtime must use CUDA 12.9"
assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() == 1, "single-GPU runtime must expose only GPU 0"
assert torch.cuda.is_bf16_supported(), "GPU 0 does not support BF16"
assert torch.cuda.get_device_capability(0) == (12, 0), "GPU 0 is not the qualified Blackwell device"
probe = torch.ones(1, device="cuda", dtype=torch.bfloat16)
assert probe.item() == 1.0, "the CUDA 12.9 build could not launch a BF16 kernel on GPU 0"
assert transformers.__version__ == "5.15.0", "unsupported Transformers version"
assert peft.__version__ == "0.20.0", "unsupported PEFT version"
assert tokenizers.__version__ == "0.22.2", "unsupported Tokenizers version"
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__} peft={peft.__version__} tokenizers={tokenizers.__version__}")
print(f"diffusion_gemma={DiffusionGemmaForBlockDiffusion.__name__}")
print(f"gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
PY
