# CALVIN fixed-anchor overfit qualification

This gate implements G3 from the experiment plan. It is an engineering
qualification on authenticated `task_ABC_D/training` data from scenes A/B/C;
it is not the official ABC→D benchmark and it never reads policy observations
from environment D.

The v2 gate deterministically draws 32 distinct physical action anchors by
default (configurable only to a multiple of eight in the inclusive range
32--128). It materializes and content-hashes those camera/state/language/action
examples once. It also creates one fixed rectified-flow input/target pair per
physical B=8 microbatch. Every optimizer update traverses every microbatch once
in the same order. There is no sampler call or data seed inside the update
loop.

It reuses the production CALVIN trainer's fail-closed contracts for:

- authenticated archive/extraction generation, A/B/C episode split, and
  normalization;
- the exact canonical flow config, model snapshot, prefix geometry, camera
  order, and instruction inventory;
- BF16 native TP=2, physical B=8 sample-isolated grouped MoE execution,
  decoder-attention LoRA topology, FP32 trainables/optimizer state, and
  TP-aware gradient clipping;
- the pinned Python environment, package lock, deterministic CUDA settings,
  and source identities.

The report is published exclusively and includes a semantic qualification
contract hash plus a self-hash. A pass requires all of the following:

- 32--128 distinct physical action anchors and one identical inventory hash at
  every update;
- final loss on the same fixed flow tensors at least 20x lower than its initial
  value;
- every declared LoRA and action-interface tensor changed;
- the optimizer contained exactly those declared trainables;
- no frozen parameter, module buffer, registration, or prefix-cache tensor was
  modified;
- CALVIN D was not accessed.

The report also embeds the canonical archive-direct v4 storage identity and
its `storage_identity_sha256`. The job-local archive inode capability used to
avoid redundant full-archive hashing is deliberately absent from the
persistent report.

The script emits no checkpoint, and the report explicitly records
`qualification_only=true`, `checkpoint_emitted=false`, and
`official_benchmark_claim=false`. Thus it cannot be served or mistaken for an
official score.

After the authenticated dataset, normalization, and measured prefix geometry
exist, create an empty report directory and run:

```bash
mkdir -p /root/.cache/duo-vla/reports/calvin-fixed-anchor-g3-v2

/bin/bash scripts/calvin/run_fixed_anchor_overfit.sh \
  /root/.cache/duo-vla/data/calvin/task_ABC_D/training \
  /root/.cache/duo-vla/contracts/normalization/calvin-abc-to-d-v4.json \
  /root/.cache/duo-vla/contracts/prefix-geometry/calvin-abc-to-d-v1.json \
  /root/.cache/duo-vla/reports/calvin-fixed-anchor-g3-v2/report.json \
  --anchor-count 32 \
  --updates 200 \
  --warmup-updates 10
```

The report path must not already exist. A failed gate exits nonzero without
publishing a passing report. Runtime qualification still requires the same two
GPUs as production training; the source and unit tests themselves are CPU-only.
