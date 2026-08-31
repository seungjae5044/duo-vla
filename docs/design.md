# Duo-VLA implementation design

This document is the executable design contract. It resolves ambiguities in the original proposal and separates facts
about the pretrained DiffusionGemma implementation from Duo-VLA choices that must be tested experimentally.

## 1. Tensor contract

For a batch size `B`, action horizon `H`, action dimension `Da`, state dimension `Ds`, decoder width `D`, and prefix
length `P`:

| Tensor | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `clean_actions` | `[B, H, Da]` | float | normalized target chunk, `A1` |
| `action_valid_mask` | `[B, H]` | bool | true only for real (not padded) actions |
| `state` | `[B, Ds]` | float | normalized state at the first action time |
| `timesteps` | `[B]` | float | one flow time per complete chunk |
| `noise`, `noisy_actions` | `[B, H, Da]` | float | `epsilon` and `At` |
| `action_embeddings` | `[B, H, D]` | backbone dtype | continuous decoder input canvas |
| `prefix_attention_mask` | `[B, P]` | bool | valid image/language prefix positions |
| `velocity` | `[B, H, Da]` | float | predicted `dAt/dt` |

One scalar `t` is sampled for each chunk, not for each action position. Padded action values are zero-filled after
normalization, excluded from both the attention key/value set and the loss, and never executed.

## 2. Rectified-flow objective

The source is standard Gaussian and the training target is the normalized action distribution:

```text
A0 = epsilon, epsilon ~ N(0, I)
At = (1 - t) * epsilon + t * A1, t ~ Uniform(0, 1)
V* = A1 - epsilon
loss = sum(mask * (Vhat - V*)^2) / (Da * sum(mask))
```

The loss is accumulated in FP32. The velocity head is unbounded; clipping belongs only at the final policy boundary.
Euler inference integrates from `t=0` to `t=1`. It does not clip intermediate states because that changes the learned
ODE. The initial reference uses a uniform schedule and 10 function evaluations.

For the six continuous channels this is used as a continuous transport model. The LIBERO gripper target instead has
two atoms at `{-1,+1}`: the network learns a continuous gripper score with the same flow MSE, then a terminal threshold
maps it to the categorical command. Duo-VLA is therefore a hybrid continuous-score/categorical-boundary policy; it
does not claim that a regular finite-time ODE exactly creates a discrete atomic marginal.

The direct-regression control uses the same observations, state interface, action masks, train/validation split, and
decoder adaptation. It predicts `A1` in one pass and is trained with masked action MSE. This makes the comparison about
iterative flow generation rather than data or parameter-count differences.

The direct query is defined exactly: the action canvas is all zeros, every chunk receives `t=1`, and the regression
target is the clean normalized chunk `A1`. Horizon, action-type, state, image, and language conditioning remain active.
The `Wa` action-projection weight is retained so checkpoint shapes and declared parameter counts match, but its weight
gradient is identically zero in this baseline because its input is zero; its bias remains active. This inactive weight
must be disclosed rather than counted as effective direct-regression capacity.

Every resolved run has a strict canonical `policy_contract` containing the objective, training input/target/timestep,
default sampler/NFE, seed behavior, action shape, and clipping semantics. The exact contract and its SHA-256 are copied
into each checkpoint manifest and rank-state resume contract. A checkpoint with a missing, unknown, or config-mismatched
contract is not trainable or serveable. Both objectives construct one common input/target pair and then use the same
FP32 masked-SSE implementation and global valid-element denominator across all accumulated microbatches.

## 3. Action and state normalization

LIBERO continuous action and state dimensions use training-only percentile statistics, excluding padded positions:

```text
q01 = percentile(x, 1)
q99 = percentile(x, 99)
x_norm = 2 * (clip(x, q01, q99) - q01) / max(q99 - q01, 1e-6) - 1
```

The artifact algorithm is exactly `numpy.quantile(..., method="linear")` over finite float32 source values, with
float64 quantile results serialized losslessly to JSON. A dimension whose absolute q99-q01 spread is below `1e-6`
must retain the same constant/non-constant classification when converted to the runtime float32 tensors. Such a
dimension maps to zero and inversely maps to q01. Padding and every validation/test episode are excluded before
fitting; physical scan order and the selected training-row identity are recorded so the same source cannot silently
produce a different artifact.

For LIBERO, the first six source channels are the pinned OSC_POSE controller inputs already bounded to `[-1,1]`, not
measured Cartesian or rotational deltas in metres/radians. Percentile normalization is only a learned
reparameterization: inversion returns those controller-command units, followed by the benchmark adapter's native
`[-1,1]` clamp. No additional physical-unit or controller scaling is applied.

The last action dimension is the binary gripper command. It is not percentile-normalized. LIBERO maps values `>= 0`
to `+1` and values `< 0` to `-1`; CALVIN matches the pinned official wrapper exactly, so values `> 0` map to `+1`
and values `<= 0` map to `-1`. At inference, clip the final generated normalized chunk to `[-1, 1]`, invert only the six
continuous channels, then apply the same deterministic gripper threshold. Each benchmark adapter documents what its
environment means by the two signs; sign conversion happens at the adapter boundary.

CALVIN is an intentional benchmark-specific exception for actions. Its `rel_actions[:6]` are already the official
simulator-scaled, clipped command representation: translation is multiplied by 50, rotation by 20, and all six values
are in `[-1,1]`. Duo-VLA trains and serves those six channels with an identity transform; applying another percentile
map would change the command units. CALVIN state still uses q01/q99 for `robot_obs[:7]`, while `robot_obs[14]` remains
the exact previous gripper command. The normalization artifact and checkpoint policy contract record
`identity_official_scaled_rel_actions` so a LIBERO unnormalizer cannot be used accidentally.

## 4. Continuous action interface

The input at horizon position `j` is

```text
h_j = Wa At_j + ba + MLPt(phi(t)) + MLPs(s) + P_j + E_action
Vhat_j = Wout h_j_out + bout
```

`phi` is a 256-dimensional sinusoidal embedding using normalized flow time scaled by 1000 and a maximum period of
10000. Both conditioning MLPs are `Linear -> SiLU -> Linear`, with hidden/output width `D`. `P` and `E_action` are
learned. The output head is one linear projection from `D` to `Da`, initialized with a small standard deviation and no
activation. Every interface `Linear` uses a bias (`ba` and `bout` above, plus both MLP layers); these tensors are part
of the trained interface and its checkpoint inventory. No additional trainable normalization is inserted in this
interface. The native frozen DiffusionGemma zero-self-conditioning path applies its own post RMS normalization before
the first decoder layer; retaining that path avoids both bypassing pretrained behavior and normalizing the action
canvas twice.

The first implementation conditions every action slot on the same current state. Adding state history or per-step
future state is out of scope for the baseline.

## 5. DiffusionGemma integration

DiffusionGemma is an encoder-decoder model in the denoising sense, but it does not expose a conventional cross-attention
encoder output. The multimodal prompt is first processed as a prefix, producing a **layer-wise KV cache**. Action
embeddings are then processed as suffix positions against that cache.

The upstream decoder API currently requires discrete decoder token IDs. Duo-VLA therefore provides a narrow backend
adapter that starts from `action_embeddings` and executes the native decoder stack without consulting token embeddings
or the vocabulary head. The adapter must preserve native layer normalization, rotary position handling, cache layout,
sliding/full-attention pattern, and mixture-of-experts routing.

The attention relation is:

```text
                         keys/values
queries            retained valid prefix   valid action   padded action
prefix                       yes              no              no
valid action                 yes              yes             no
padded action (internal)      yes              yes             no
```

The prefix is computed before action positions and never queries them. Within a vision block, the native image mask is
preserved; the overall prompt prefill otherwise follows the pretrained model's native prefix semantics. The action
suffix receives an explicit fully bidirectional `H x H` block while retaining access to the prefix keys present in
each layer's native cache. DiffusionGemma has 25 sliding-window layers and five full-attention layers; a sliding layer
does not promise access beyond its native 1,024-position window. Across all 40 authenticated LIBERO instructions, the
measured two-image prompt ranges from 529 to 544 valid positions. The fixed physical prefix width is `P=545`, strictly
one position beyond that maximum. This mandatory sentinel padding prevents an all-valid replica batch from taking
SDPA's `mask=None` elision path while a mixed batch takes an explicit mask path. Startup must reject a contract whose
width is not strictly greater than its measured maximum, or a longer prefix until its semantics are tested.
The native two-dimensional decoder mask expresses key validity, so padded action queries are still computed internally
and may read valid prefix/action keys. Padded action positions are never keys, their final hidden states are forced to
zero, and they are excluded from both loss and execution; consequently they cannot influence any valid prediction.
Action position IDs are fixed across Euler steps and begin after each sample's number of valid prefix positions. Left
padding therefore neither shifts RoPE positions nor changes the action coordinates; the physical cache width may still
be larger because it is shared by the batch, and the prefix attention mask excludes those padded keys. This canonical
per-sample positioning deliberately tightens the upstream default, which otherwise derives positions from padded batch
width.

Two cache modes are required:

- `training`: recompute the frozen prefix under `torch.no_grad()` once for each batch. Reuse that cache for the sampled
  action timestep in the batch. Do not persist caches across optimizer steps or observations.
- `rollout`: compute the prefix cache once per new observation and language instruction, then reuse it for all Euler
  function evaluations of that action chunk.

Cache reuse is valid only if adapters do not alter prefix projections. Base parameters, token embeddings, vision tower,
and vocabulary head are frozen. LoRA is injected into decoder action-suffix attention projections only. Upstream layers
with distinct `q`, `k`, `v`, and `o` projections receive all four adapters; layers whose architecture derives values
from keys receive adapters only on projections that actually exist. Module selection is checked by fully qualified
names, and startup fails if an adapter appears in the prefix path. The current resolver verifies that every discovered
target was injected. The pinned loader additionally requires exactly 115 projections with histogram
`q=30, k=30, v=25, o=30` and the revision-specific 25-layer value-projection pattern before fresh training, resume, or
serving can proceed.

For rollout, rectified flow draws the initial Gaussian canvas from the episode-derived inference seed and evaluates the
velocity field with uniform Euler steps. Direct regression performs exactly one decoder forward and does not create or
consume a random generator; the derived inference seed is still checked and echoed so request identity remains
auditable. Serving may override flow NFE only to one of `{1, 5, 10}`. It cannot override direct regression. The
checkpoint retains its authenticated default training contract while IPC reports the actual selected serving NFE.

## 6. Images and language

Each policy query contains exactly one third-person RGB frame and one wrist RGB frame, both at the same control time,
plus the task language. The official `AutoProcessor` supplies resizing, normalization, special tokens, image ordering,
and padding. We do not hard-code a nominal image size. The serialized sample retains camera names so a silent swap is
detectable.

Language strings are the benchmark-provided instruction or annotation. Evaluation uses the same canonical instruction
mapping for every compared policy. A language-intervention smoke test pairs a fixed observation with two different
instructions and verifies that the conditioned representation or output changes.

The pinned LIBERO dataset exposes the two views under the generic fields `image` and `image2`; the adapter binds these
to third-person and wrist respectively. The dataset revision and authenticated tree pin that mapping, but the pixel
content itself does not carry a machine-verifiable camera-name label. This provenance boundary is recorded explicitly
rather than treating equal image shapes as independent proof of camera identity.

## 7. Trainable and frozen state

Trainable:

- action projection, timestep MLP, state MLP;
- horizon and action-type embeddings;
- velocity head;
- decoder-suffix attention LoRA (`rank=16`, `alpha=32`, dropout `0`).

Frozen:

- vision and text/prefix computation;
- every pretrained base tensor, including shared token embeddings;
- vocabulary output head.

The optimizer has two named parameter groups: LoRA at `1e-4` and action interface at `1e-3`. Before the first update,
the trainer prints and asserts every trainable parameter, group assignment, count, and dtype. Checkpoints contain only
trainable weights, configs, normalization statistics, exact dataset identifiers/revisions, and code revision.

## 8. Precision and distribution

The pretrained backbone and its forward activations run in BF16 autocast. Trainable LoRA and action-interface
parameters remain FP32, as do loss, percentile statistics, optimizer states, and metric accumulators. Casting the
interface module itself to BF16 is disallowed because ordinary AdamW would then create BF16 moment buffers.
The 26B checkpoint cannot be treated as a single-device baseline on 48 GiB GPUs; the initial supported setup shards the
frozen backbone across both local GPUs while keeping small interfaces colocated with the action decoder output. Exact
placement is validated with a dry forward before training.

The measured implementation currently runs the action suffix without gradient checkpointing. Prefix caches contain no
gradient-bearing tensors. Suffix-only gradient checkpointing is an optional memory optimization and must not be called
enabled until an explicit enable path plus output/gradient parity test exists; cache is disabled on any path whose
autograd history is needed. Activation memory, peak allocated GPU memory, examples/s, and policy latency are logged for
every experiment.

For TP=2, gradient clipping uses a single global norm on both ranks. Replicated LoRA/interface gradient squares are
counted once, while local squares for TP-sharded LoRA factors are summed across ranks. One shared clipping coefficient
is then applied to every local gradient. Every clipping call first requires exact cross-rank agreement on both the
replicated squared norm and the resulting total norm. Applying ordinary per-rank `clip_grad_norm_` would double-count
replicas and can produce different coefficients, causing nominally replicated trainable tensors to diverge.

## 9. Required correctness gates

No benchmark run starts until all earlier gates pass:

1. Unit tests: flow endpoints, oracle integration, masked loss, normalization, timestep embedding, interface shapes.
2. Attention tests: changing a padded action cannot change any valid prediction; changing a valid action can; prefix
   tensors are invariant to `At` and `t`; fresh-prefix and reused-cache predictions match. Tiny-model eager tests do
   not satisfy the production gate: BF16, TP=2, SDPA, the pinned real revision, measured prefix length, and `H=8` must
   also be covered.
3. Freeze tests: one optimizer step changes only the declared interface and decoder LoRA tensors.
4. Numerical tests: BF16 prediction/loss is finite at `t in {0, 0.5, 1}` and sampling is deterministic for a fixed seed.
5. Synthetic fixed-batch overfit with a small backend.
6. Fixed-batch overfit through the real DiffusionGemma adapter.
7. One task, one rollout worker, then full benchmark evaluation.

## 10. Explicit non-claims

KV reuse, bidirectional continuous suffix attention, and useful frozen-backbone features are implementation hypotheses
until their dedicated tests pass. A low training loss alone does not demonstrate conditioned control: rollout success,
language intervention, direct-regression comparison, and wall-clock latency are all required.
