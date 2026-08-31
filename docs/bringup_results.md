# Bring-up results

This file records measured development gates. These are engineering checks, not benchmark scores. Measurements use the
locked environments in `uv.lock`; only execution geometry needed to interpret the method is reported.

## Artifact and distributed-runtime checks

- Official backbone: `google/diffusiongemma-26B-A4B-it` at
  `f7f5b7f5fa82ffc52addd066915886d497f5517b`.
- The local snapshot contains all 11 BF16 safetensor shards and has no incomplete files.
- The two-GPU 64 MiB BF16 all-reduce smoke averaged 3.6785 ms over ten measured iterations.
- The explicit symmetric text/MoE TP plan passed a two-rank tiny-model prefix, continuous-suffix, LoRA, backward test.
  The frozen vision tower is replicated because the upstream Gemma4 vision TP plan is incompatible with the current
  clippable-linear implementation.

## Real-backbone forward/backward checks

The pinned 26B-A4B model was loaded in BF16 across both GPUs. A text-prefix smoke and a two-image multimodal smoke both
passed continuous action decoding and backward through decoder-attention LoRA plus the action interface.

| Check | Prefix length | Output shape | Loss | Peak allocated memory / rank |
| --- | ---: | --- | ---: | ---: |
| text prefix | 16 | `[1, 8, 7]` | 1.59861 | 24.794 GiB |
| two images + language | 532 | `[1, 8, 7]` | 1.569087 | 24.906 GiB |

The real model exposes 115 eligible decoder attention projections, producing 230 trainable LoRA tensors (A and B),
and the additive action interface contains 14 parameter tensors. The vocabulary head is not used. LoRA and interface
parameters plus Adam moments remain FP32; only the frozen backbone and forward activations use BF16 autocast.

The production parity gate subsequently ran on the update-500 flow checkpoint with the pinned real BF16 model,
TP=2, SDPA, two-image prefix length 532, and `H=8`. For two deterministic action canvases, every rank-local fresh
prefix K/V tensor matched the reusable prefix byte-for-byte, every reused-cache pointer/version/length/hash remained
unchanged, and fresh/reused outputs were bitwise identical. Native token decoding and the continuous adapter matched
bit-for-bit after self-conditioning, after each of all 30 decoder layers, after final normalization, and at the final
hidden output. Maximum absolute/relative error was zero; minimum measured cosine exceeded `0.9999998`. The prefix
cache occupied 59,924,480 bytes per rank, parity execution after process startup took 15.61 seconds, and peak allocated
memory was 24.946 GiB per rank. The checked checkpoint manifest SHA-256 was
`c657b8d64ac45e34301f7e141c9544ac2b8a00df7e331f47f23956371232d051`.
The production loader now also fail-closes unless the exact 115-target projection histogram and value-projection
layer pattern match the pinned revision. Action decoding rejects a prefix/cache length mismatch or a prefix longer
than the native 1,024-position sliding window.

A two-image/language BF16 forward/backward sweep established the initial training microbatch. Variable prompt lengths
were padded through the official processor.

| Microbatch | Peak allocated memory / rank |
| ---: | ---: |
| 1 | 24.906 GiB |
| 2 | 25.215 GiB |
| 4 | 25.708 GiB |
| 8 | 26.671 GiB |
| 16 | 28.644 GiB |
| 32 | 32.741 GiB |
| 64 | 41.404 GiB |

Batch 64 technically fits in 48 GiB, but it leaves insufficient guard space for real prompt variation, allocator
fragmentation, optimizer/metric state, and simulator co-residency. This initially led to microbatch 32 with two
gradient-accumulation steps for global batch 64.

A later real-model train/serve parity audit invalidated unmodified grouped-MM at every multi-sample physical batch.
With TP=2 and BF16, the same first sample changed substantially when only companion samples changed, even after
canonical per-sample prefix and action position IDs were supplied. This was reproduced at fixed batch widths 2 and 32;
identical rows within a batch remained bitwise identical, but mixed companion content changed the first row. A
singleton microbatch was a correctness-preserving interim fallback but was prohibitively slow. The frozen recipe now
uses physical microbatch 8 and 8 accumulation steps with sample-isolated grouped-MM expert execution in every encoder
and decoder layer. The audited wrapper reproduced singleton expert arithmetic exactly and made the same sample
bitwise invariant to its row and companions at fixed B=8. Earlier unisolated microbatch-32 runs below remain useful
engineering diagnostics but are not valid train/serve-matched benchmark evidence.

The TP-aware trainable-only checkpoint gate also passed after 200 real optimizer updates. Saving from both ranks
gathered one complete 45,989,856-byte LoRA safetensors file and wrote a separate 66,729,484-byte FP32 action-interface
file; no base-model shard was included. The adapter contains 115 replicated and 115 TP-sharded trainable tensors. The
exact 24,336-byte normalization artifact is copied into the checkpoint. Reloading all artifacts into a fresh TP-sharded
base reproduced the final loss and BF16 prediction byte-for-byte (SHA-256
`6e5738a1ddce15dd958a73d761742678621eb66d6e3acbed28c0023f53406677`). The checkpoint manifest records file sizes
and SHA-256 hashes and an interrupted write retains an `INCOMPLETE` marker.

An earlier Transformers 5.16.1 zero-initialized LoRA checkpoint was rejected as a valid gate: PEFT 0.20.0 did not
receive its expected TP hook metadata, so a zero-update output match could conceal missing adapter shards. The locked
environment now uses Transformers 5.15.0 and the adapter setup fails fast if any TP target lacks PEFT-compatible
metadata. Gradient clipping also computes one common global norm, counting replicated gradients once and all-reducing
sharded-gradient squares, so it cannot apply different scaling factors to replicated parameters on different ranks.

## Dataset and fixed-batch overfit checks

The tiny synthetic backend reduced its fixed-batch loss from `1.23233199` to `2.46027e-7` in 1,000 updates; the best
observed loss was `1.62967e-8`.

The full pinned dataset was validated as 377 physical parquet files with 273,465 contiguous global rows. Its episode
metadata contains 1,690 stale physical file pointers; resolving whole episodes by their verified global index spans
recovers all 1,693 episodes without a cross-file episode. Exact q01/q99 statistics use only the fixed task-stratified
training split: 1,525 episodes and 246,015 frames, with 168 episodes and 27,450 frames held out. Repeating the complete
scan produced the same normalization content SHA-256
`a972b5d95a8aaa8ae7582bafcbc071261979cb46c2a3515b4da7a7cf0156ac73`.

The real LIBERO G3 gate materialized 32 distinct training anchors spanning 26 tasks, decoded both real camera images,
and used the real language/state and a fixed FP32 flow pair. Terminal padding left 246 valid action positions. The
frozen prefix cache was computed once for this deliberately fixed batch. After a 10-step scaled warm-up and 200 AdamW
updates, the measured result was:

| Item | Value |
| --- | ---: |
| initial masked velocity MSE | 1.2856580 |
| final masked velocity MSE | 0.0280356 |
| best pre-update-loop MSE | 0.0213762 |
| initial/final reduction | 45.86x |
| maximum pre-clip global gradient norm | 25.12 |
| update-loop time | 103.83 s |
| peak allocated memory / rank | 32.658 GiB |
| changed LoRA / interface tensors | 230 / 14 |
| optimizer-state dtype | FP32 |

Reproduction command:

```bash
HF_HOME=/root/.cache/huggingface \
HF_HUB_DISABLE_PROGRESS_BARS=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONHASHSEED=0 \
  /root/.cache/duo-vla/venvs/train/bin/torchrun \
  --standalone --nproc-per-node=2 \
  scripts/overfit_real_libero_batch.py \
  /root/.cache/huggingface/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e \
  /root/.cache/duo-vla/data/libero/normalization-v1.json \
  --steps 200 --warmup-steps 10 \
  --checkpoint-dir /root/.cache/duo-vla/checkpoints/libero-g3-b32-20260829
```

Every declared LoRA/interface tensor changed, while frozen parameter version counters remained unchanged. The
optimizer inventory exactly matched the declared trainable inventory; replicated gradients were byte-identical on the
first two updates; the common TP-aware norm was used for clipping. The final trainable-only checkpoint and embedded
normalization file pass size/hash verification and fresh-process prediction round-trip.

## Resumable trainer and one-task rollout pilot

The full data trainer was exercised with tensor parallelism, fresh prefix encoding on every microbatch, exact masked
gradient accumulation, and per-rank AdamW/scheduler/RNG state. A three-update `turn on the stove` smoke stopped after
update 2 and resumed to update 3. `examples_seen` advanced from 8 to 12; all 244 Adam states, the scheduler, and the
trainer counter loaded at the same update boundary. A separate uninterrupted run diverged during the first CUDA
backward even though its initial forward loss was byte-identical. The current BF16 SDPA/MoE/TP training stack is
therefore numerically, not bitwise, reproducible across fresh launches. This is not evidence of a resume-state error.

The transaction-audited trainer was then rerun from scratch as
`/root/.cache/duo-vla/runs/libero-turn-on-stove-resume-smoke-v3`. It stopped at update 2 and resumed through the
relative path `checkpoints/update-000002`. The update-2 manifest SHA-256
`080898eb3e0344a0dfc5d1a0feaa2e8950a17902e49da701d76c6ceb185711da` is the exact parent recorded by the update-3
manifest and run journal; the update-3 manifest SHA-256 is
`f2096e14e8ec34e3dd8a70dd575feb4b9562201ee534b3d2ac16a9c067b4b0ea`. Progress was exactly 8→12 examples and the
final real update used the configured cosine endpoint (`LoRA=1e-5`, interface=`1e-4`). The output had three strictly
ordered metric records and no `INCOMPLETE` or rank-state staging directory.

This version also holds a single-writer OS lock, hashes the optimizer's exact ordered parameter schema, authenticates
immutable checkpoint parentage in an fsynced run journal, and saves per-rank optimizer/scheduler/Python/NumPy/Torch
RNG state. A resume reconciles metrics to the journal tip. Interrupted future staging/checkpoint paths and a partial
final metric line are preserved under `recovery_quarantine/` before replay. Unit crash fixtures cover first-checkpoint
bootstrap, incomplete and complete orphan checkpoints, partial JSON tails, ambiguous branches, and atomic-write
failures.

The direct-regression comparison path then passed the same real TP=2 stop/resume gate. It trained for two updates,
resumed from `checkpoints/update-000002`, and completed update 3 at exactly 12 examples. The update-2 manifest SHA-256
`e50cbbf623096b007e7e46e9537f6ca9ceae1c3ebc674b11b92a23257256a19f` is the exact parent recorded by the update-3
manifest, whose SHA-256 is `af02be5226ebf84b76cf75ae81be8438ebbdab085083af4ae51b4d874ed32c6e`.
The checkpoint's authenticated objective is `direct_regression`: zero action canvas, fixed `t=1`, clean normalized
action target, and one decoder forward. A fresh real policy-server load returned bitwise-identical action bytes for two
different inference seeds (SHA-256 `bafe34ce5b05556126e8c6dc3bbd44748d034ef273088125b42b4d481951ae29`), while echoing both distinct request-derived
seeds. This establishes the intended deterministic direct-serving contract, not task performance.

### Matched 500-update flow/direct development runs

Two seed-0 development runs then used the same pinned model and data, the same 45 training and five validation episodes
for `turn on the stove`, global batch 64 (microbatch 32 with two accumulation steps), optimizer, scheduler, validation
samples, and source tree SHA-256
`9aa36b0ee7076b3983d4d2032e6363638a4bedc2269d0a84113c96425112ec12`. Only the authenticated objective contract
differed. Both configurations declare 1,000 updates but deliberately stopped at update 500, so both manifests report
`complete=false`; these are development checkpoints, not completed training runs.

| Item | Rectified flow | Direct regression |
| --- | ---: | ---: |
| updates / real chunks | 500 / 32,000 | 500 / 32,000 |
| mean optimizer-update time | 20.1602 s | 20.2331 s |
| peak allocated memory / rank | 36.1553 GiB | 36.1553 GiB |
| validation loss at update 100 | 1.089120 | 0.127011 |
| validation loss at update 200 | 0.507035 | 0.122577 |
| validation loss at update 300 | 0.336308 | 0.088969 |
| validation loss at update 400 | 0.299200 | 0.080332 |
| validation loss at update 500 | 0.280972 | 0.076919 |
| final train loss | 0.303578 | 0.040508 |
| update-500 manifest SHA-256 | `c657b8d6...d051` | `a4f4c7ca...9ac0` |

The complete 50-update train-loss means were:

| Updates | Rectified flow | Direct regression |
| --- | ---: | ---: |
| 1–50 | 1.135535 | 0.130305 |
| 51–100 | 1.100724 | 0.115173 |
| 101–150 | 1.058356 | 0.092363 |
| 151–200 | 0.742747 | 0.077045 |
| 201–250 | 0.384714 | 0.076414 |
| 251–300 | 0.326525 | 0.074761 |
| 301–350 | 0.309307 | 0.070715 |
| 351–400 | 0.287980 | 0.062750 |
| 401–450 | 0.269355 | 0.056460 |
| 451–500 | 0.270798 | 0.049799 |

Each run has exactly 500 ordered metric rows, a five-link transactional manifest chain at updates 100–500, an
identical journal tip, and no incomplete, staging, or quarantine artifacts. The flow final manifest SHA-256 is
`c657b8d64ac45e34301f7e141c9544ac2b8a00df7e331f47f23956371232d051`; the direct final manifest SHA-256 is
`a4f4c7ca2b157763105b0eb8c6bf38ca1d0012f9b40a12124c5c44c3a3899ac0`.

The loss columns are **not comparable in magnitude**: flow predicts `A1 - epsilon`, while direct regression predicts
`A1`. Their within-objective decreases establish that both training paths optimize their declared targets. They do not
establish which policy controls the simulator better. That comparison requires the same uncontaminated reset bank or
sealed official episodes, common evaluation noise, and separate success/latency reporting; those gates remain open.

The strict IPC v3 migration was then exercised through both the fake server and the real TP=2 direct update-500
checkpoint. Every request requires an independent evaluation seed and repeats its official/clean-development reset
identity; v2 requests and response echo drift are rejected. The inference-seed hash excludes training seed, checkpoint,
objective, NFE, and K. With evaluation seed 123 and the same synthetic official reset, K=1 and K=4 produced the same
inference seed (`1832035156714354204`) and direct action SHA-256
`a2bcf0d2f89635e6260fbc17cbc62aecf072ddd0bf9c7375215099c9c9d16ee0`. Changing only the evaluation seed to 456
changed the inference seed to `5938211334698447304`; as required for deterministic direct regression, the action hash
did not change. Warm real-server policy calls took 0.56–0.57 seconds. This is an IPC/CRN check, not a benchmark latency
estimate or rollout result.

The flow update-500 checkpoint was checked separately at diagnostic NFE=1. Evaluation seed 123 produced inference seed
`1832035156714354204` and action SHA-256
`23437c7366a1eb2b556749e9cf10ddd67c2c0a01f4d0c31114e507e993b867a6` for both K=1 and K=4. Changing only the
evaluation seed to 456 produced inference seed `5938211334698447304` and a different action SHA-256
`8f96ccc9c3ddccbbde756428fffc236f044655039471f633dc310ef8c113266b`. This confirms that K does not perturb the
common-noise identity while the registered evaluation seed controls the flow source noise.

With that same NFE=1 flow checkpoint, a conditioning intervention held the synthetic observation identity and flow
noise fixed. Repeating the baseline request reproduced the action bytes exactly. Changing only language, zeroing only
agentview, zeroing only the wrist image, changing only state, or swapping the camera inputs changed the action output;
the maximum absolute action deltas from baseline were respectively `0.3351`, `0.5452`, `0.5586`, `0.9774`, and
`0.3353`. This demonstrates that every declared conditioning path can affect the current network output. It does not
show that the effects are semantically correct or improve task success; clean reset rollouts and negative-control
degradation remain required.

### Authenticated clean-reset LIBERO pilot

A non-benchmark development bank was generated from seeded BDDL resets for Goal task 7, `turn on the stove`, without
loading a published benchmark state. The four states are finite, nonterminal, initially unsuccessful, byte-stable
across set/get and settle replay, pairwise unique, and byte-distinct from all 50 published task states. The bank root
SHA-256 is `261dd16cb7f509e8588a284f616adf48ebc6ed9de46bc2019e0ff023cabf6745`; its task array SHA-256 is
`ac56cf22403ee4412f68bff59ef458dbd3e08180d8b9effe089290be895004eb`.

The matched update-500 development checkpoints were evaluated on all four states with evaluation seed 20260829.
Flow used its registered ten-step Euler sampler. These are four-reset engineering pilots, not benchmark estimates:

| Objective | K | Success | Successful episode steps | Policy calls | p50 policy latency | Normalized clip fraction |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| rectified flow, NFE=10 | 1 | 4 / 4 | 156, 99, 89, 77 | 421 | 1.957 s | 8.64% |
| rectified flow, NFE=10 | 4 | 1 / 4 | 75 | 244 | 1.967 s | 9.16% |
| direct regression | 1 | 0 / 4 | none | 1,200 | 0.559 s | 0.64% |
| direct regression | 4 | 4 / 4 | 82, 84, 75, 84 | 82 | 0.561 s | 5.56% |

No environment-space action was clipped in any cell. All failures exhausted the 300-step environment budget. The
opposite K dependence was the observed behavior on this tiny bank: flow succeeded only reliably with single-step
replanning, whereas direct regression succeeded only with four-action execution. Pooling K would therefore be
misleading. This closes the one-task G4 nonzero-success engineering gate, but the later physical-batch mismatch means
it cannot support a train/serve-matched model comparison or policy-quality claim.

A larger seed-0 pilot trained only the 45 training episodes for the exact LIBERO Goal task `turn on the stove`; five
held-out episodes remained internal validation. The run used microbatch 32, two accumulation steps, and a 1,000-update
contract with 100 warm-up updates, but stopped deliberately at update 100 pending a trainer transaction/schedule
audit.

| Item | Value |
| --- | ---: |
| real chunks processed | 6,400 |
| mean optimizer-update time | 19.972 s |
| initial / update-100 train loss | 1.31547 / 1.06269 |
| fixed validation loss, update 5 / 100 | 1.25953 / 1.12439 |
| validation reduction | 10.73% |
| maximum pre-clip gradient norm | 22.9271 |
| peak allocated memory / rank | 36.155 GiB |

The update-100 checkpoint manifest SHA-256 is
`1d7587dded7905e0b26ca20f12f962e14cc06fe281822850c94cc579e6812c20`. Its LoRA and interface SHA-256 values are
`ecb708b2a066d300bc3f4ad7ea9d82d54cff32875d5392eadadb7afbf43a173f` and
`9dde30c3cefc7651ac1a9cdc91e1fddf2d916148069efdcac88e666a17ea5d09` respectively.

The pinned cross-environment policy server then ran real ten-step Euler inference and the official simulator on Goal
task 7, fixed initial state 0. These are one-episode engineering pilots, not benchmark estimates:

| K | Success | Policy calls | p50 / p95 policy latency | Normalized clip fraction | Env clip fraction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0 / 1 | 75 | 1.971 / 2.007 s | 37.48% | 0% |
| 1 | 0 / 1 | 300 | 1.976 / 2.007 s | 37.30% | 0% |

Both episodes completed the 300-step policy budget without IPC, simulator, TP-rank, or timeout errors. The failure for
both execution horizons and large normalized-space clip fraction show that 100 warm-up updates are insufficient; no
control-performance claim follows from this pilot.

This pilot nevertheless exposed one published benchmark cell before the recipe was frozen and its failure influenced
the decision to continue training. It is therefore test leakage, regardless of its engineering-only intent. The exact
artifacts and downstream decisions are recorded in `evaluation_contamination.json`, whose companion SHA-256 file
authenticates the ledger; Goal task 7 / state 0 is excluded symmetrically from the primary clean-holdout result for all
future objectives, NFEs, execution horizons, and seeds.

## CALVIN authenticated data, runtime, and development gates

The official CALVIN parent repository and recursive submodules were checked out at the revisions in
`scripts/calvin/revisions.env` in a separate Python 3.8.20/PyTorch 1.13.1 CPU environment. Regenerating the official
1,000 seed-0 long-horizon sequences produced canonical SHA-256
`90191d9ac76baecb4f292ab766bbbf3ae65dbf43a83bbe5c376d99db10fd6446`.

An EGL smoke instantiated the official `calvin_scene_D_eval` source config in a headless process, rendered static
`[200,200,3]` and gripper `[84,84,3]` uint8 images, returned 15D robot and 24D scene observations, and completed one
30 Hz environment step. A nonzero action test confirmed that the pinned simulator mutates its first six action values
in place while applying the 0.02/0.05 translation/rotation scales. The policy adapter's fresh-copy boundary is therefore
required, not merely defensive.

The complete `task_ABC_D.zip` archive was authenticated at 555,309,812,705 bytes with SHA-256
`c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74`. Production uses the retained archive directly,
an exact-schema v2 member index, and projected metadata rather than extracting episode NPZ files. The deterministic
seed-1729 split assigns 132 whole A/B/C episodes to training and holds out 15 whole A/B/C episodes, with all 34
annotated tasks represented on both sides. The normalization content SHA-256 is
`56d0fd0f90aca6ffc6585223780b19aa51d0ab895fb293652088eedf9aff3808`: seven continuous state dimensions use
training-only q01/q99 statistics, the previous gripper state remains binary, and the six already scaled and clipped
CALVIN `rel_actions` channels use an identity transform.

The CALVIN trainer, strict Python-3.8/3.11 policy IPC, TP=2 policy server, and official long-horizon evaluator are now
implemented and covered by deterministic tests. The evaluator authenticates the canonical 1,000-sequence list,
requires a frozen pre-registration binding checkpoint and serving-policy hashes, and refuses fake policy health in
official-score mode.

### CALVIN fixed-anchor G3

The authenticated G3 inventory contained 32 distinct A/B/C anchors spanning 20 tasks and 1,708 valid action scalars.
After 200 fixed-inventory updates with ten warm-up updates, masked velocity MSE fell from `1.294522609` to
`0.008575519`, a `150.9556x` reduction. All 230 LoRA and 14 interface tensors changed on both ranks; all frozen
parameters and buffers and the prefix cache remained unchanged. All 11 declared criteria passed. The report-file
SHA-256 is `f2fc10a78fe680faedf71b5eb3da1d6789e876026b6f5f7449480f2c4a5cc181`. This gate used A/B/C training data only
and made no learned-policy access to D.

### CALVIN seed-0 500-update flow pilot

The pilot retained the production model, A/B/C data and split, normalization, prefix geometry, fixed physical batch,
optimizer, learning rates, and 2,048-sample validation contract. Only the declared budget changed to 500 updates, which
deterministically changed warm-up to 50 updates. It completed 500 updates and 32,000 training chunks.

| Item | Value |
| --- | ---: |
| update-1 / update-500 train loss | `1.1704777883675022` / `0.3088843029670865` |
| update-500 validation loss | `0.3035793172255102` |
| update-500 checkpoint manifest file SHA-256 | `7e6a7c2c0f759ec2dbcb2d5854ab7258414a0eb60dc41d21c97200f0b40c94aa` |
| manifest complete / metric rows | `true` / `500` |

### Held-out A/B/C development rollout

The replay-bundle root is `d04078700df0902909c39e6917569ce58c38bf03e8a9083de2c71329e48df8ec`; it covers all
15 held-out episodes, 1,944 candidate records, and 34 tasks without training-episode overlap. The reset-bank root is
`d024c26d3f7ae6f02259219014bd5551dd2afa85765821da1679699cecc5762a`; it contains 100 replay-validated scene/task
records. The preselected smoke view uses four resets per A/B/C scene.

| Objective | NFE | K | Success | Scene A / B / C | Evaluation seed | Elapsed seconds |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| rectified flow | 10 | 4 | 2/12 | 1/4, 0/4, 1/4 | `20260829` | `9026.102660089033` |

The real-policy summary has `plumbing_only=false` and `gate_passed=true`; its file SHA-256 is
`f922cc78f7e9032f719735c690e44d17a8bcc569425707d40b68590c7539198d`. This closes the G5 nonzero-success development
gate over independent held-out A/B/C subtasks. It is not an official ABC→D score and cannot be translated into the
official sequential `SR_1` through `SR_5` or `AvgLen` metrics. No learned checkpoint received a D observation, sent an
action in D, or queried a D rollout outcome.

## Current gate interpretation

- G0 (deterministic unit tests) and the synthetic overfit gate pass.
- The real-backbone finite multimodal forward/backward and checkpoint round-trip checks pass. The production G2
  fresh-prefix/reused-cache and native/adapter matrix also passes on real BF16, TP=2, SDPA, and `H=8`, with bitwise
  equality at every checked layer and no cache mutation.
- LIBERO G3 passes on 32 distinct real chunks with a 45.86x loss reduction and a frozen-state audit.
- CALVIN G3 passes with a 150.96x loss reduction and all 11 freeze, inventory, and cache criteria satisfied.
- The real LIBERO inference/IPC/simulator path passes end to end. The uncontaminated update-500 clean-reset pilot has
  nonzero success for both objectives and closes G4, while also showing that K must be reported separately.
- The real CALVIN development path passes end to end: its seed-0 update-500 flow checkpoint completed 2/12 held-out
  A/B/C subtasks at `K=4`, closing G5 without learned-policy access to D.
- No full 30,000-update, three-seed LIBERO or official CALVIN ABC→D benchmark has been completed, so no benchmark
  claim is made.
