# Environment and storage

## Observed host

The initial bring-up was performed on a dual-GPU host with 48 GiB per GPU and driver `570.133.20`. The root
filesystem had about 1.4 TiB free, while `/workspace` had only about 19 GiB free. Model weights, datasets, virtual
environments, and simulator assets therefore belong under `/root/.cache/duo-vla` (or another large path supplied via
`DUO_VLA_CACHE_ROOT`), not inside the repository.

`nvidia-smi topo -m` reports `NODE` between the two GPUs, not `NV#`: this host has no active NVLink path. Native TP=2
is still required for BF16 capacity, but inter-GPU collectives traverse PCIe/host bridges and may dominate latency.
Throughput measurements on this exact topology are therefore a required feasibility result, not an assumed property.
The bring-up collective gate measured a 64 MiB BF16 all-reduce at 3.68 ms averaged over ten iterations.

The pre-existing global Python environment is not usable for this project: its Transformers and Hugging Face Hub
versions conflict, and it predates DiffusionGemma support. Duo-VLA uses an isolated Python 3.11 environment.

## Reproduce the training environment

Install `uv`, then run from the repository root:

```bash
export DUO_VLA_CACHE_ROOT=/root/.cache/duo-vla
./scripts/bootstrap_train_env.sh
```

The script resolves the committed `uv.lock` into `${DUO_VLA_CACHE_ROOT}/venvs/train` and verifies CUDA, BF16, and the
DiffusionGemma class import. It does not change the global interpreter. The locked core versions are PyTorch 2.13.0
from the CUDA 12.6 index, Transformers 5.15.0, Tokenizers 0.22.2, PEFT 0.20.0, and Accelerate 1.14.0. The explicit CUDA index avoids
accidentally resolving a CUDA 13 wheel, which is not compatible with the installed R570 driver branch.

Use these cache locations for subsequent commands:

```bash
export HF_HOME=/root/.cache/huggingface
export DUO_VLA_CACHE_ROOT=/root/.cache/duo-vla
export MUJOCO_GL=egl
```

Launch training through `scripts/run_libero_train.sh`. The wrapper removes the host's `LD_LIBRARY_PATH`: this host
advertises system cuDNN 9.8 there, while the locked PyTorch wheel was compiled with and bundles cuDNN 9.10.2. Directly
inheriting the host path fails before model loading. Set `PYTHONHASHSEED` to the run seed; for example:

```bash
PYTHONHASHSEED=0 ./scripts/run_libero_train.sh \
  <libero-snapshot> <normalization.json> <output-dir> \
  --prefix-geometry-artifact \
  /root/.cache/duo-vla/contracts/prefix-geometry/libero-v2.json \
  [trainer options]
```

The artifact path is a required input, but its semantic SHA-256 and fixed width are not accepted from the command
line. They are independently pinned by `configs/libero.toml`; training authenticates the artifact against those pins,
the complete model/processor snapshot identity, both `256x256x3` cameras, and all 40 canonical instructions before
constructing DiffusionGemma. The current `libero-v2.json` artifact has semantic SHA-256
`cc907e22ccd5ae704767edba606233dede39989a119ac544764b47aaa4fbe634`, canonical file SHA-256
`e0c5084fcccf7fa223fd430ff55569b6dd74f12f75cb82bcd7d7aff5358c0bc8`, and fixed width `P=545`.

The Google checkpoint is pinned to repository revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`. It contains
11 BF16 safetensor shards totaling approximately 51.4 GB (decimal), so it must be loaded across both 48 GiB GPUs for the
initial full-precision experiment. Do not substitute the partially cached NVIDIA NVFP4 conversion: the baseline design
and trainability assumptions are for the official BF16 checkpoint.

The environment deliberately pins Transformers 5.15.0 rather than 5.16.1. PEFT 0.20.0 obtains tensor-parallel LoRA
metadata and gradient/save hooks through
`transformers.integrations.tensor_parallel.add_tensor_parallel_hooks_to_module`. The 5.16.1 compatibility module no
longer exports that API and the newer TP path does not annotate the target layers in the form PEFT expects. The model
can appear to train in that combination while replicated LoRA factors silently diverge between ranks and a checkpoint
omits part of the distributed adapter. Duo-VLA checks every TP LoRA target for compatible metadata and fails fast if
the hook contract is absent. A 100-update trained-checkpoint round trip, including a byte-identical prediction hash,
guards the pinned combination.

The recursively collected automatic DiffusionGemma TP plan can also install asymmetric hooks around tied encoder and
decoder weights. Duo-VLA passes an explicit symmetric plan covering both `model.encoder.language_model` and
`model.decoder`. The published Gemma4 vision TP plan remains incompatible with its clippable-linear wrappers, so the
large tied text/MoE stacks are sharded while the much smaller frozen vision tower is replicated on both ranks. These
choices are guarded by the real two-image memory and backward smoke gates.

## Simulator separation

LIBERO and especially CALVIN have dependency constraints that conflict with the modern model process. Use one isolated
environment/process per simulator and communicate with the policy process using lossless local IPC. The simulator sends
raw RGB arrays, state, and language; the policy returns a new copied `float32[7]` action. Timeouts invalidate a rollout
instead of silently injecting a no-op.

Recommended external layout:

```text
/root/.cache/duo-vla/
  data/libero/
  data/calvin/
  models/
  simulators/libero/
  simulators/calvin/
  venvs/train/
  venvs/libero-eval/
  venvs/calvin-eval/
```

Every downloaded model/dataset is addressed by an immutable revision and verified by the upstream checksum where one
is published. A run manifest records those revisions and the simulator repository/submodule commits.
