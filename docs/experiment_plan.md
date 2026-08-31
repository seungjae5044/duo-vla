# Experiment and evaluation plan

## Reproducibility envelope

Every run records the git revision, dirty diff hash, Python/package lock, CUDA/driver versions, GPU model, model revision,
dataset revision, normalization artifact hash, seed, and complete resolved config. Heavy artifacts live under a cache
root outside `/workspace`; only small manifests, metrics, and checkpoints of trainable weights belong in this repository.
Training must be launched with `PYTHONHASHSEED` equal to the configured run seed. A run-owned journal authenticates the
latest immutable checkpoint, its parent-manifest hash, and the last committed metric; resume rejects a checkpoint that
is not the journal tip and reconciles any metrics written after that tip.
An OS file lock gives the output directory one writer. Interrupted staging/checkpoint paths are moved into a recoverable
`recovery_quarantine/` directory before replay; they are never silently deleted. If a run stops before its first
checkpoint, launching the same resolved configuration at the same output path safely restarts it from update zero.
An invocation-local `--stop-after-updates` boundary writes a resumable checkpoint but does not introduce an
out-of-schedule validation pass; validation remains tied to its declared interval and the configured final update.

Use three training seeds for reported comparisons. Development smoke tests may use one seed but are labeled as such.
The benchmark checkpoint is the predeclared final optimizer update (30,000 for the initial recipe); validation is a
diagnostic and does not trigger early stopping or best-checkpoint selection. A different selection rule requires a new
pre-registration and implementation before training. The evaluation seed set is independent of the training seed and
is shared across checkpoint/objective/NFE/K comparisons as common random numbers. Policy-scored development uses
materialized, hashed development resets from training environments; it never uses published LIBERO fixed states or
CALVIN-D. Benchmark test environments are scored once only after every checkpoint hash, factor cell, training seed,
evaluation seed set, and aggregation rule is frozen and pre-registered.

### Compute-budget gate

The benchmark protocol does not make a long run affordable merely by making it reproducible. On the current dual-GPU
host, the fixed-physical-B=8 LIBERO qualification measured 44.46 seconds per flow optimizer update and 44.16 seconds
per direct-regression update over ten updates. At that provisional rate, 30,000 updates take about 15.4 days for one
checkpoint; two objectives, three seeds, and two independently trained benchmarks take about 185 serial GPU-days
before rollout evaluation. These numbers are planning estimates, not benchmark timings, because they were measured on
an earlier source qualification before CALVIN's now-pinned `P=538` prefix geometry was available.

Accordingly, G3--G5 development runs remain short and explicitly non-benchmark. Before authorizing G6, rerun a timed
warm-up on the final source, exact benchmark prefix geometry, and fixed B=8 execution path; record throughput, projected
wall time, checkpoint storage, and rollout cost. The full 30,000-update, three-seed campaign starts only after that
resource estimate and the immutable pre-registration are accepted. Reducing updates or seeds creates a separately
labeled pilot protocol and must not be reported as the predeclared benchmark comparison.

## Staged runs

| Gate | Data/model | Pass criterion |
| --- | --- | --- |
| G0 | unit tests, no backbone | all deterministic tests pass |
| G1 | synthetic fixed batch, tiny decoder | loss falls by at least 100x and reconstructed normalized action MAE < 0.02 |
| G2 | one real sample, real backbone | cached/uncached parity and finite forward/backward |
| G3 | 32--128 real chunks, real backbone | fixed-batch loss falls by at least 20x; only declared tensors change |
| G4 | one LIBERO task, non-benchmark dev resets | nonzero success in deterministic local rollouts |
| G5 | held-out CALVIN A/B/C rollout | completes at least one annotated subtask without touching D policy scores |
| G6 | full evaluation | all suites/chains run with protocol-complete manifests |

Failure at a gate is investigated before scaling compute. A run that violates its data or evaluation contract is marked
invalid rather than compared numerically.

One earlier engineering pilot did score LIBERO Goal task 7, published initial state 0, before this boundary was fixed.
That exposure and its downstream decisions are immutable in `evaluation_contamination.json`, authenticated by
`evaluation_contamination.sha256`. The primary clean-holdout
comparison excludes that cell symmetrically from every objective/NFE/K/seed (1,999 episodes per factor cell, with 49
states for that task). A conventional all-50/all-2,000 result may additionally be shown only as non-blind full-set
comparability; it cannot be called a strict test-once estimate.

## Main comparison

The primary factors are:

- policy head: rectified flow versus direct one-step regression;
- Euler function evaluations: `N in {1, 5, 10}` for flow;
- receding-horizon execution: `K in {1, 4}` (with `K <= H=8`).

All variants share the same frozen checkpoint, LoRA placement/rank, action interface capacity, training examples,
normalization, optimizer budget, augmentations, and evaluation initial states. `N=1` flow remains a learned velocity
field evaluated at noise and is not treated as identical to direct regression.

The committed comparison configs are `configs/libero.toml` / `configs/libero_direct_regression.toml` for LIBERO and
`configs/calvin_abc_to_d.toml` / `configs/calvin_abc_to_d_direct.toml` for CALVIN. Direct uses zero action queries,
fixed `t=1`, a clean action target, and one decoder forward. Its `Wa` matrix is present for shape/declared-parameter
parity but inactive because the action input is zero. Flow serving sweeps reuse one authenticated checkpoint with
`--flow-steps {1,5,10}`; direct checkpoints reject that override. Health records the actual sampler and NFE used.

Report task success together with end-to-end policy latency. The cache ablation measures (a) prefix recomputed on every
Euler step and (b) prefix computed once per observation. Timing includes processor/device transfer and synchronization,
and reports p50 and p95 after explicitly discarded warm-up calls. Existing bring-up timings that did not discard
warm-up are diagnostics, not benchmark latency.

## Training defaults

Initial values, subject to memory smoke tests:

| Item | Value |
| --- | --- |
| horizon | 8 |
| optimizer | AdamW |
| LoRA learning rate | `1e-4` |
| interface learning rate | `1e-3` |
| LoRA | rank 16, alpha 32, dropout 0 |
| backbone precision | BF16 |
| gradient clipping | global norm 1.0 |
| AdamW betas / epsilon / weight decay | `(0.9, 0.95)` / `1e-8` / `1e-10` |
| timestep distribution | uniform `[0, 1]` |
| action noise | iid standard normal |
| sampler | uniform-step explicit Euler |
| training schedule | 30,000 updates; 1,000-step warm-up; cosine to 0.1x peak LR |
| checkpoint / validation interval | 1,000 updates |
| EMA | disabled |
| image augmentation | none in the baseline |

The global batch size is 64, implemented as a fixed physical microbatch of 8 with 8 accumulation steps. TP=2 shards
one model; it does not multiply the sample count. DiffusionGemma's stock grouped-MM MoE arithmetic couples a sample to
its batch companions. The production path therefore preserves B=8 at every encoder and decoder layer while splitting
each grouped-MM expert invocation into eight independently routed sample slices. Training and serving both install and
verify that exact hook topology. The policy server duplicates one request into all eight rows, clones one seeded flow
noise row across them, requires all normalized outputs to be bitwise identical, and returns row zero. A fixed-width,
externally pinned prefix-geometry artifact keeps every prefill at a benchmark-specific fixed width; validation contains
only complete B=8 batches. LIBERO's authenticated artifact pins `P=545`, one position beyond its maximum valid prefix,
and CALVIN's complete authenticated A/B/C plus official-D instruction inventory pins `P=538`. Both retain sentinel
padding and use the same explicit SDPA mask path. CALVIN's frozen semantic artifact SHA-256 is
`edaef86df702e34c9be6c9103e4f3c9ccc4022df8def46df7d1f00084d0831c9`.
Optimizer epsilon, weight decay, warm-up, total steps, augmentations, and data sampling weights must be
explicit in the resolved config before a benchmark-quality run.

Learning rates are indexed by the optimizer updates on which they are actually applied. For the default run, update
index 0 uses `1/1001` of peak, index 1000 uses the exact peak, and index 29,999 uses the exact 0.1x cosine endpoint.

The optimizer shape follows the official OpenPI AdamW defaults (`beta1=0.9`, `beta2=0.95`, `eps=1e-8`, effectively
zero `1e-10` weight decay, gradient norm 1.0), while retaining Duo-VLA's separately specified LoRA/interface peak
learning rates. The 30,000-update initial budget and 1,000-step warm-up are anchored to OpenPI's released LIBERO
fine-tuning configurations; they are starting hypotheses, not imported benchmark results. See
[OpenPI optimizer](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/optimizer.py) and
[OpenPI LIBERO configs](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py).

For LIBERO, each task assigns 10% of whole demonstrations (at least one and never all) to internal validation by a
seeded stable hash. CALVIN instead applies a deterministic whole-play-episode split inside each A/B/C scene and
requires every annotated task to remain on both sides. Normalization uses only the remaining training episodes.
Held-out demonstration metrics diagnose training but do not select a checkpoint in the initial fixed-final-update
protocol. They never use LIBERO benchmark initial states or CALVIN-D rollouts. The final simulator protocol runs only
after the training recipe and fixed checkpoint hash are frozen.

## Metrics

Training diagnostics:

- masked velocity MSE and direct-regression MSE;
- normalized and physical-unit action MAE by channel;
- gripper accuracy and switch-event F1;
- gradient norms by LoRA/interface group;
- timestep-binned loss (`[0,.1)`, ..., `[.9,1]`);
- examples/s, peak GPU memory, cache size, and NaN/Inf counts.

Policy metrics:

- `duovla-libero-v1` / OpenPI-style-on-`hf-libero` per-task and suite success;
- official CALVIN ABC→D sequential-task metrics;
- action-chunk inference latency and achieved control frequency;
- every training-seed result plus mean and standard deviation across the three seeds;
- paired per-episode policy differences under the common evaluation-noise set, plus the exact number of evaluated
  episodes/sequences. Wilson intervals, when shown, describe binary outcomes within one fixed checkpoint's benchmark
  set and are not presented as training-seed uncertainty.

## Sanity and intervention tests

Before reporting benchmark numbers:

1. Shuffle language within a batch; conditioned validation loss should worsen.
2. Zero each camera independently; measure the output change and rollout degradation.
3. Change only state while holding image/language/noise fixed; predictions should change.
4. Fix observation and initial noise and change language; predictions should change reproducibly.
5. Evaluate the predeclared initial-noise seed set per observation, independent of the training seed, to quantify
   stochastic action variance with common random numbers across policies.
6. Compare cache-on and cache-off outputs within an agreed numerical tolerance.

These tests diagnose whether the model uses each conditioning source; they are not substitutes for benchmark success.

## Artifact layout

```text
configs/                 committed experiment inputs
docs/                    design and benchmark contracts
runs/<run-id>/           small local metrics/manifests only (ignored by default)
<external-cache>/models  pretrained model snapshots
<external-cache>/data    LIBERO and CALVIN data
<external-cache>/venvs   locked runtime environment
```

Dataset conversion writes immutable shards plus a manifest containing source file hashes. Train/validation membership is
materialized once and stored by trajectory identifier, never regenerated from dataloader order.
