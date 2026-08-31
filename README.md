# Duo-VLA

Duo-VLA adapts DiffusionGemma's bidirectional denoising decoder into a continuous robot-action policy. Its primary
objective is rectified flow, with a parameter-aligned direct one-forward regression control. The shared model core is
benchmark-agnostic; LIBERO and CALVIN ABC→D are integrated through explicit data and rollout contracts.

Given two RGB observations, a language instruction, proprioceptive state `s`, noisy action chunk `A_t`, and flow time
`t`, the policy predicts the rectified-flow velocity

```text
A_t = (1 - t) * epsilon + t * A_1
V*  = A_1 - epsilon
Vhat = f_theta(images, language, s, A_t, t)
```

DiffusionGemma performs a native multimodal prefix prefill and exposes its layer-wise KV cache to a continuous action
suffix; this is not a conventional encoder-output/cross-attention interface. The vision/language prefix and every
pretrained weight remain frozen. Training updates only rank-16 decoder action-suffix self-attention LoRA adapters (`q`,
`k`, `v`, and `o` projections that exist in the pinned model) and the continuous action interface:
action/state/timestep projections, horizon and action-type embeddings, and velocity head. The model jointly denoises
an eight-step, seven-dimensional action chunk without action tokenization. The matched direct-regression baseline uses
the same model and data, but supplies a zero action canvas at `t=1` and predicts the clean chunk in one forward pass.

The full 30,000-update optimization recipe uses a fixed physical batch of 8 with 8 gradient-accumulation steps (global
batch 64), AdamW, separate `1e-4` LoRA and `1e-3` interface learning rates, a 1,000-update linear warm-up, and cosine
decay. See [Training and reproduction](docs/training.md) for artifact preparation, exact launcher commands, the reduced
500-update CALVIN pilot, and the complete recipe.

The repository is being brought up in gated stages:

1. Mathematical core and deterministic unit tests.
2. DiffusionGemma continuous-embedding adapter and decoder-only LoRA.
3. Fixed-batch overfit and a single-task rollout.
4. Full LIBERO and CALVIN ABC→D evaluation.

Heavy model, dataset, and simulator files must live outside this repository. See `docs/` and `configs/` for the pinned
contracts as they are finalized. Measured engineering gates are recorded in `docs/bringup_results.md`; they are kept
separate from eventual LIBERO and CALVIN benchmark scores.

Current CALVIN development evidence includes a fixed-anchor G3 overfit gate and a seed-0, 500-update rectified-flow
pilot. G3 reduced masked velocity MSE from `1.294523` to `0.008576` (`150.96x`) while changing every declared
LoRA/interface tensor and preserving frozen state. The pilot completed 2 of 12 independent held-out A/B/C subtasks at
`K=4` and `NFE=10`. These are development checks, not an official CALVIN ABC→D score; no learned policy was evaluated
or scored in environment D.

Key references:

- [Executable model design](docs/design.md)
- [Environment setup](docs/environment.md)
- [LIBERO protocol](docs/libero_protocol.md)
- [CALVIN ABC→D protocol](docs/calvin_protocol.md)
- [Experiment plan and reporting boundaries](docs/experiment_plan.md)
