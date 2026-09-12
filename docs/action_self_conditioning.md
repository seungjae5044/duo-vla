# Experimental continuous-action self-conditioning

This is an opt-in action endpoint interface, not a reproduction of native
categorical DiffusionGemma self-conditioning and not a verified performance gain.
Legacy configurations retain their original architecture and sampling arithmetic.
The user-requested first training run starts fresh from the pinned pretrained base
for 7,500 updates, with independent encoder and decoder LoRA; it is **not** a
continuation of the 10k/15k/16k action checkpoints or a matched SC OFF/ON study.

## Signal and architecture

For the unchanged noise-to-data path `At=(1-t)*epsilon+t*A1`, the velocity target is
`A1-epsilon`. An endpoint estimate is `S=At+(1-t)*velocity`, computed in FP32.
No intermediate clipping or gripper discretization is introduced. Final normalized
action clamping and unnormalization remain at the existing execution boundary.

`ActionInterfaceConfig.self_conditioning="action_endpoint_v1"` installs a biasless,
zero-initialized `(action_dim+1) -> hidden_size` linear inside `ActionInputProjector`.
Its inputs are `[m*S, m]`, where `m` is one boolean per sample, shared across slots.
The residual is added to the original action/time/state/horizon/type embedding
before the existing valid mask. At H8/D7/hidden2816 it adds 22,528 parameters.
The adapter continues to call the native `self_conditioning(embedding, zeros)`,
retaining post-RMS normalization. The pinned native activation is GELU-tanh, not
SiLU; its frozen gated zero-input branch has zero input Jacobian. The new residual
does not use that branch. Initialization preserves the legacy CPU RNG stream too.

The absent-candidate case still executes the projection on zeros, producing a
zero gradient tensor rather than `None`. A genuine zero-action candidate has
presence one and is distinguishable from absence. Candidate gradients are detached.
Unsupported versions, invalid shapes/presence types, and nonfinite present
candidates fail closed. Padded slots cannot affect valid action keys or loss.

## Training and inference

`training_velocity` optionally runs a same-pair no-grad preliminary decoder pass,
then trains on its detached endpoint. Noise, time, observation and prefix KV are
identical in the two passes. Prefix encoding is performed once with autograd when
encoder LoRA is active; only the preliminary decoder/candidate computation is under
`no_grad`. Encoder checkpointing may recompute layers during backward. The prefix
cache remains read-only to both decoder passes.

`bootstrap_for_update(seed, update)` uses a domain-separated SHA-256 bit, without
consuming the data/noise/time RNG. The 50% choice is shared by every rank and
microbatch in an optimizer update. Production uses this stateless choice; capacity
probes explicitly alternate present/absent and never contribute to the 7,500 updates.

`sample_action_flow` is the shared opt-in sampler: FP32 uniform Euler, times `k/NFE`,
increments `velocity/NFE`, endpoint from **pre-update** action/time. History is a
local variable initialized for every call. There are exactly NFE decoder forwards,
with no candidate at the first step and the previous endpoint thereafter. NFE=1
must be identical with inference SC enabled/disabled at identical weights/noise.
Legacy generic Euler's BF16 increment behavior is deliberately not rewritten.

The generic policy, Portable Spatial sampler, native encoder-aware Spatial server,
general LIBERO server sampler, and offline objective/sampling code route opted-in
denoisers through these helpers. Legacy checkpoint protocols do not silently opt in.
To load a new SC checkpoint, the explicit encoder-aware server supports
`--action-self-conditioning --nfe N`. It authenticates the separate encoder adapter,
SC architecture/recipe and critical source hashes before inference. Existing
checkpoint loaders reject SC artifacts unless explicitly SC-aware.

## Fresh Spatial runner

`scripts/train_libero_sc_encoder.py` implements the requested separate experiment:

- Only physical GPUs 0/1; production is two full TP1 data-parallel replicas.
- Fresh decoder LoRA, encoder LoRA, action interface and AdamW state. The reference
  checkpoint supplies authenticated **data/config artifacts only**, never trainable
  initialization weights or optimizer moments.
- Frozen vision tower, multimodal projector and pretrained base weights; decoder
  rank16/alpha32/dropout0 on 115 projections and encoder rank16/alpha32 on 113
  action-reachable projections. Frozen base BF16, trainables and moments FP32.
- All groups have constant LR `5e-5`, no warmup/scheduler, betas `(0.9,0.95)`, epsilon
  `1e-8`, weight decay `1e-10`, global norm clipping at 1.0.
- Spatial's same 389 training/43 validation episodes, 10 tasks, normalization,
  P545 two-image prefix, H8/D7 action geometry and task/episode/frame-uniform stream.
- One physical forward batch per rank, no accumulation; canonical B8 chunks are
  partitioned disjointly. Gradients of globally normalized masked SSE are SUM-reduced
  with no extra world-size division. Every trainable must have a finite gradient.
- Largest measured stable batch is selected after capacity/DP qualification, not
  inferred from forward-only memory. The 72/80 widths are experimental extensions;
  qualified legacy training profiles still reject these widths.
- Every 500 updates and at the final/explicit stop boundary: the same 2,048 held-out
  anchors, separately reported no-candidate and same-time bootstrap velocity MSE,
  plus Gaussian-to-action sampled/clamped action MSE at NFE4. A final incomplete
  physical validation batch repeats its last canonical chunk for forward only;
  padded copies are excluded from **all** metric numerators/denominators.
- Permanent checkpoints at updates 2,500/5,000/7,500, plus resumable explicit stops.
  Both ranks' optimizer/RNG states, SC interface, decoder PEFT and separate encoder
  weights are authenticated. All fresh optimizer counters equal the run update.

`scripts/launch_libero_sc_encoder.py` checks GPU ownership, uses a closed environment,
and supervises only its own torchrun process. Its source argument must reference an
immutable external snapshot. The output and control directories must be explicit;
an invocation never resumes or replaces a different run automatically. Exact resume
requires the same recipe/source, optimizer inventory and committed metric prefix.
Architecture-changing legacy migration is not mislabeled as exact resume; the
requested scratch runner rejects inherited counters and initialization states.

## Qualification evidence so far

CPU tests exercise zero-init/RNG parity, presence semantics, zero/nonzero gradients,
bootstrap detach/same-pair behavior, native encoder gradient flow with/without
recomputation, read-only KV caches, padding, query reset, NFE1 behavior, deterministic
SC choice, strict serialization and optimizer identity. Actual serving-method CPU
plumbing tests compare generic/portable/encoder/server outputs, singleton padding,
batch permutation and repeated queries at NFE 1/2/4/5/10.

On the real pinned GPU model, B8 native-vs-recomputed training with SC ON/OFF/ON had
zero maximum output error and zero maximum gradient error across all 471 trainable
tensors, including updates after the SC and encoder zero-B weights changed. New SC
parameters receive nonzero gradients on candidate-present updates and zero tensors
on absent updates; encoder gradients remain connected and base weights frozen.

Single-GPU capacity probes measured B64 at 87.84 GiB peak with about 23 s/update
after the first update. B80 ran out of memory. B72 completed six updates at 92.73 GiB
peak, but with allocator mapping/retry warnings. A production width requires the
separate two-GPU qualification; single-GPU completion alone is insufficient.

Probe artifacts are external to Git under `/hdd2/hyunbin/vla/cache/reports/`:
`spatial-sc-encoder-probes-v1`, `spatial-sc-encoder-probes-v2`, and
`spatial-sc-encoder-dp2-b72-probe-v1` (with its separate `-control` directory).
No official rollout result or SC causal improvement is claimed by these checks.

The B72-per-GPU DP2 probe subsequently completed all six updates with matching
first-update gradients/weights/optimizer state, 92.73 GiB peak allocation and about
26.7 s/update after the first update. Allocator mapping/retry warnings remain, so
allocation retry counts and reserved memory are recorded in the long-run metrics.
The initial largest tested completed geometry was global batch 144; B80 could not
fit. That initial selection is superseded by the B72 long-run failure below.

The final full CPU suite passed 1,240 tests with 12 optional/CUDA/dependency skips; the
separate CUDA grouped-MM suite passed all 24 tests, including the added widths.
Each saved SC checkpoint also stores raw NFE4 predictions and their observation
inputs at training width. `scripts/verify_sc_checkpoint_runtime.py` independently
loads both adapter sets and the SC interface at the recipe's physical width, then checks raw output
agreement, repeatability, permutation and singleton padding. This is a policy I/O
qualification, not an environment rollout. Each run has a distinct recipe-bound
UUID; exact resume additionally verifies restored parameter/optimizer fingerprints
against the committed manifest.

The first real save/load test exposed a cross-width numeric difference: B72 training
versus B8 serving had raw action max/mean absolute errors of 0.0696487/0.0124107.
Processor inputs were identical; the first encoder KV value differences appeared
at low magnitude in layer zero and propagated through later layers. Reloading at
B72 matched the saved training reference exactly. At both widths independently,
repeated, permuted, singleton-padded and train/eval outputs matched exactly. These
results do not prove cross-width equivalence; no tolerance was relaxed. The SC
server now preserves the recipe's physical batch size and pads smaller queries.
This costs extra inference compute for small active batches. Lower-width serving
is a separate, unqualified numeric/optimization change. Legacy servers remain B8.
Diagnostic evidence: `spatial-sc-roundtrip-diagnostic-b8-v1` and
`spatial-sc-roundtrip-diagnostic-b72-v1` under the external reports directory.

The v2 candidate (`duo-vla-sc-encoder-train-v2` frozen workspace) passed the real
checkpoint serving test at physical B72, using eight active observations: raw
save/load, repeat, permutation and singleton errors were all exactly zero.
Evidence: `spatial-sc-encoder-roundtrip-runtime-v2/result.json`, checkpoint update 2,
manifest SHA-256 `5749b1aa9f053ad4f86dba6f3df52d4429740285cc941150bf7c4149800e3837`.
The two independently fresh qualification runs also had identical training and
validation numeric metrics at updates 1 and 2. Qualification updates are excluded
from the requested 7,500-update scratch run.

The v2 resume test restored the exact committed parameter/optimizer fingerprints,
but its next backward failed with OOM. Rank 1 had opened an additional ~718 MiB
context on GPU 0 while PEFT staged adapter weights. A separate device test confirmed
that safetensors `device="cuda"` selects CUDA 0 even with current device 1. The
checkpoint loader now explicitly stages adapter weights on CPU before loading into
the already rank-local modules; the staged values were bitwise identical. This
fix is isolated from training math. Frozen candidate `duo-vla-sc-encoder-train-v3`
subsequently passed the full real resume check: committed update 2 restored exactly,
update 3 completed with identical replica gradients/parameters/optimizer state,
validation completed and a new authenticated checkpoint was saved. The supervisor
exited with return code zero and all children exited. Each rank used only its own
GPU; peak allocated memory remained 92.73 GiB. Allocator retry warnings still occur.
Evidence: `spatial-sc-encoder-roundtrip-v3-control-resume/supervisor.json` and
`spatial-sc-encoder-roundtrip-v3/checkpoints/update-000003`, whose manifest SHA-256 is
`fba3611848bb23aa1e6940ee96e441f44c42605297c50d1f530623bad7d37df4`.
The independently loaded post-resume checkpoint also passed all four raw-output
comparisons with exactly zero error; see `spatial-sc-encoder-roundtrip-runtime-v3/result.json`.

## Authorized long run: initial B72 attempt (failed)

The fresh 7,500-update run is separate from every qualification run above:

- Frozen source: `/hdd2/hyunbin/vla/cache/workspaces/duo-vla-sc-encoder-train-v3`.
- Run: `/hdd2/hyunbin/vla/cache/runs/libero-spatial-sc-encoder-scratch-dp2-b144-lr5e-5-7500-v1`.
- Controller/log: `/hdd2/hyunbin/vla/cache/reports/spatial-sc-encoder-scratch-7500-v1-control`.
- Per-update metrics and status: `metrics.jsonl` and `progress.json` under the run.
- Permanent checkpoints: `checkpoints/update-002500`, `update-005000`, `update-007500`.

Its launch has no resume, probe or early-stop argument. Completion requires the
7,500-update checkpoint and successful terminal controller state; launch alone is
not completion. Expected elapsed time from measured short probes is roughly 57 hours,
including periodic validation, and must be refined from the real training metrics.
The run started on 2026-09-11 at 01:05:17 UTC. Its startup audit confirmed update zero,
empty optimizer state and initial trainable fingerprint
`8cb7d3b0370f40d31b18721ee2dee2c050116fdfd8081003ef1523f651ac6bf1`.
Uninterrupted production updates 1–3 exactly matched the qualification's numeric
training metrics, including update 3 after checkpoint restoration. The production
recipe differs from the qualified recipe only in its distinct run UUID.

### B72 long-run failure and B64 recovery qualification

The B144 run above **failed**, rather than completing. On 2026-09-11 at
03:16:51 UTC, both ranks raised `torch.OutOfMemoryError` during the backward for
update 289, requesting another 3.29 GiB. The last completed update was 288, with
loss 0.36485391315044163. The controller recorded `status: failed`, return code 1,
at 03:16:55 UTC; its supervisor and all training children exited. The run's
`progress.json` still says `running` because the exception interrupted its update;
the terminal controller state and actual process handles take precedence.

There are no checkpoints in this failed run: it had not reached the first
2,500-update save. Its logs and recipe are retained as failure evidence. Neither
these 288 updates nor qualification updates may be counted toward a replacement
scratch run's requested 7,500 updates. The earlier short B72 capacity, DP and
save/load checks remain valid as short tests, but do **not** establish long-run
memory stability. Repeated allocator retries were not sufficient headroom.

Recovery uses the next smaller supported canonical physical width, B64 per GPU
(global batch 128), with the same immutable v3 source, data, encoder/decoder LoRA,
SC procedure and constant LR. It requires a new B64 DP save/resume and real
checkpoint I/O qualification before a separately identified fresh long run. No
old trainable weights or optimizer state will initialize that replacement run.

B64 subsequently completed fresh updates 1–2, fixed validation, authenticated
checkpoint saving, exact optimizer/parameter restoration and update 3 with replica
consistency checks. The steady second update took 23.2132 s; peak allocation was
87.8394 GiB and allocation retries were zero in both the fresh and resumed checks.
The post-resume validation completed in 240.317 s, and the controller exited
`stopped` with return code zero. These remain short qualification measurements,
not a claim that 7,500 updates have completed or that long-run stability is proven.

Evidence under `/hdd2/hyunbin/vla/cache/reports/`:

- `spatial-sc-encoder-b64-roundtrip-v3` with separate `-control-start` and
  `-control-resume` directories.
- Update 2 manifest SHA-256:
  `f3d6637e2bbb43d641536162851bb8f9a2728bedf9ca33e511199e3f20cd5528`.
- Update 3 manifest SHA-256:
  `b2c1569ad6c5be3ccb5f65799ffdcb9aa6d59dffc73f6b4b28590cb7fd561ab8`.
- `spatial-sc-encoder-b64-roundtrip-runtime-v3/result.json`: the independent GPU 0
  loader passed at physical B64, NFE4, with eight active observations. Raw
  train/serving, repeat, permutation and singleton errors were all exactly zero.

## Replacement B64-per-GPU scratch run

The replacement run keeps the immutable v3 source and all non-batch learning
settings. It starts with fresh adapters, interface and optimizer, without loading
any qualification or failed-run trainable state. It targets the full 7,500 updates;
failed B72 and B64 qualification updates do not count toward that target.

- Run: `/hdd2/hyunbin/vla/cache/runs/libero-spatial-sc-encoder-scratch-dp2-b128-lr5e-5-7500-v1`.
- Controller/log: `/hdd2/hyunbin/vla/cache/reports/spatial-sc-encoder-scratch-b128-7500-v1-control`.
- Physical GPUs 0/1 only, B64 per GPU, global B128, constant LR `5e-5`.
- No resume, probe or early-stop launch argument.
- Validation every 500 updates; permanent checkpoints at 2,500/5,000/7,500.

Until the terminal controller and authenticated update-7,500 checkpoint establish
completion, this replacement remains an in-progress experiment. In particular,
`progress.json` alone must not override a failed controller or missing workers.

This B128 run started at 2026-09-11 03:33:26 UTC with run UUID
`e58ba1bc-f417-482e-a70a-4ed6ebd94e2e`. Startup confirmed update zero, empty optimizer
state, no loaded trainable checkpoint weights, and fresh trainable fingerprint
`8cb7d3b0370f40d31b18721ee2dee2c050116fdfd8081003ef1523f651ac6bf1`.
Its recipe matches the B64 qualification except for the distinct run UUID.
Uninterrupted production updates 1–3 exactly matched that qualification's numeric
loss/gradient metrics, including its checkpoint-restored update 3. The first six
production updates completed, with zero allocator retries and about 23.2 s/update
after warmup. This is startup evidence only; the requested long run is not complete.

## First production checkpoint and private Hub copy (2026-09-12 UTC)

The B128 production run saved `checkpoints/update-002500` at
2026-09-11 20:11:51 UTC. All 13 manifest-listed artifacts passed independent
size/SHA-256 verification. Its manifest SHA-256 is
`4d2300969babb756b644db45f46585eef18aa2ee4c1da56f1b268eca2ab64ad3`.
The original checkpoint contains 15 files totaling 791,975,405 bytes, including
encoder/decoder adapters, the SC action interface, both optimizer/RNG states,
the effective recipe, and serving-reference observations/predictions.

At update 2,500, training velocity MSE was `0.1959143144`; held-out velocity MSE
was `0.2216387659` without a candidate and `0.2147867518` with a bootstrap
candidate. Gaussian-start NFE4 sampled-action MSE was `0.1097961750`. These are
offline losses, not simulator success rates or evidence of a causal SC benefit.
The checkpoint's `complete: false` marks an intermediate update in the planned
7,500-update run, not an incomplete checkpoint write.

At the user's request, the checkpoint was copied byte-for-byte to the private
Hugging Face repository
[`seungjae5044/duo-vla-libero-spatial-sc-encoder-step2500`](https://huggingface.co/seungjae5044/duo-vla-libero-spatial-sc-encoder-step2500),
commit `8b87e841bb2126261bf7ab5ff75659233231f3f6`. Remote verification on
2026-09-12 01:15:16 UTC matched all 17 uploaded files (the 15 original files plus
a model card and upload inventory), using file sizes, LFS SHA-256 values, and
Git blob hashes. Frozen base-model weights and credentials were not uploaded.
The matching Duo-VLA SC-aware loader and pinned base model remain required;
this copy alone does not qualify cross-machine resume or rollout. Local upload
evidence is under `cache/reports/hf-uploads/spatial-sc-encoder-step2500/`, outside Git.
