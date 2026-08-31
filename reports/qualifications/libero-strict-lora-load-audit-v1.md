# LIBERO strict LoRA load audit

Status: **resolved**.

## Finding and fix

The implementation design requires every production resume and serving load to authenticate the pinned
DiffusionGemma decoder-attention adapter topology before PEFT attaches it. The LIBERO call sites previously used
`load_lora_checkpoint` without enabling its strict decoder contract, even though CALVIN and the offline evaluator
already enabled it.

Both production LIBERO paths now pass:

```text
validate_decoder_contract = true
expected_rank = resolved_config.lora.rank
```

This activates exact adapter-config semantics, the pinned 115-target topology (`q=30`, `k=30`, `v=25`, `o=30`),
the 230-tensor global weight schema, warning rejection, loaded-parameter coverage equality, and finite-value checks.
A source-level regression test requires both call sites to retain these arguments.

The same audit found that the protocol-v1 trainers accepted off-design LoRA and timestep-interface values even though
the checked-in experiment contract fixes them. LIBERO and CALVIN training now reject anything other than LoRA
rank 16, alpha 32, dropout 0, timestep embedding dimension 256, timestep scale 1000, maximum period 10000, and output
head initialization standard deviation `1e-3`. Dedicated mutation tests cover every field.

## Existing pilot artifacts

The already generated update-10 flow and direct adapters were revalidated directly:

| Checkpoint | Targets | Tensors | Rank | Alpha | Dropout |
| --- | ---: | ---: | ---: | ---: | ---: |
| flow update 10 | 115 | 230 | 16 | 32 | 0.0 |
| direct update 10 | 115 | 230 | 16 | 32 | 0.0 |

No malformed existing adapter artifact was found; the defect was the missing fail-closed production load gate.

## Verification

- Focused strict-load/checkpoint/trainer/server tests: `47 passed`.
- Focused fixed-hyperparameter trainer tests: `42 passed`.
- Full suite after all audit closures: `435 passed`.
- Repository-wide Ruff check and format check: passed.
- All shell launchers: `bash -n` passed.

The canonical LIBERO training-source SHA-256 immediately after this audit changed from
`2389dfdc04acc4b805da8c69f7b0ce6a6bfbeb1042855314c9c24fa6d0050038` to
`f37a4827296cd04a0ab0e788d7bd64512b18db37f9be96861cbc794a0effdc18`.
Trainer and server independently recompute the same new value.
This hash includes the shared `configs/` tree and will therefore change once the measured CALVIN prefix SHA/width are
filled into the currently fail-closed CALVIN config. Every actual training launch must record the then-current value;
the value above identifies the post-audit, pre-CALVIN-prefix state rather than a future final checkpoint.

The old update-10 checkpoints and their qualification reports remain immutable historical pilot evidence bound to
the old source SHA. They must not be resumed or served as if they were produced by the patched source. All subsequent
training starts from the new source identity.

## Documentation clarifications

The attention table now reflects the native two-dimensional key-validity mask: padded action queries may be computed
internally, but padded action positions are never keys, their final decoder hidden states are zeroed (the linear head
may still emit its bias-only value), and they are excluded from loss and execution. The design also explicitly records
that the pinned LIBERO fields `image` and `image2` are mapped to
third-person and wrist views by authenticated dataset revision/protocol; equal pixel shapes alone do not independently
prove camera provenance. An unused legacy core seed helper that included the training seed was removed; the official
bridge remains the sole LIBERO replan-seed implementation and derives common randomness only from evaluation episode
identity.
