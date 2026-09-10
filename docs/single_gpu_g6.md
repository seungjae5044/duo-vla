# GPU 0 / TP=1 G6 execution profile

This host executes the experiment on physical GPU 0 only. The original TP=2 configs and launchers remain available;
the separate profile `duovla-single-gpu-tp1-v1` changes only `model.tensor_parallel_size` and the sealed process
topology. The physical model batch remains eight BF16 samples. Activation checkpointing is not part of this profile
and may be introduced only as a separately suffixed variant after a measured out-of-memory failure.

All mutable artifacts for this execution live below `/hdd2/hyunbin/vla`:

```bash
export DUO_VLA_CACHE_ROOT=/hdd2/hyunbin/vla/cache
export DUO_VLA_TRAIN_VENV=/hdd2/hyunbin/vla/cache/venvs/train-single-gpu
export HF_HOME=/hdd2/hyunbin/vla/huggingface
export UV_CACHE_DIR=/hdd2/hyunbin/vla/uv-cache
```

The four training configs are:

- `configs/libero_single_gpu.toml`
- `configs/libero_direct_regression_single_gpu.toml`
- `configs/calvin_abc_to_d_single_gpu.toml`
- `configs/calvin_abc_to_d_direct_single_gpu.toml`

Use `scripts/run_libero_train_single_gpu.sh` and `scripts/run_calvin_train_single_gpu.sh` for training, and the
matching `*_policy_server_single_gpu.sh` launchers for serving. These launchers use a closed environment,
`CUDA_VISIBLE_DEVICES=0`, and one torchrun process. Checkpoints and official pre-registrations bind both
`execution_profile` and `tensor_parallel_size`, so TP=1 artifacts cannot be evaluated as legacy TP=2 artifacts.

## Gate sequence

G0 and G1 write immutable reports below `${DUO_VLA_CACHE_ROOT}/reports/g6`. G2 uses the real BF16 model without a
dataset:

```bash
./scripts/run_qualify_real_fixed_b8_sample_isolation_single_gpu.sh \
  --output-json "$DUO_VLA_CACHE_ROOT/reports/g6/g2-real-fixed-b8-tp1-v1.json"
```

LIBERO G3 authenticates the dataset, normalization, model, and fixed `P=545` prefix artifact. It visits 32 distinct
anchors as four immutable physical-B=8 microbatches per optimizer update:

```bash
./scripts/run_overfit_real_libero_batch_single_gpu.sh \
  "$LIBERO_SNAPSHOT" "$LIBERO_NORMALIZATION" "$LIBERO_PREFIX" \
  --batch-size 32 --steps 200 --warmup-steps 10 --minimum-reduction 20 \
  --checkpoint-dir "$DUO_VLA_CACHE_ROOT/runs/g3-libero-tp1/checkpoint" \
  --report "$DUO_VLA_CACHE_ROOT/reports/g6/g3-libero-fixed-anchor-tp1-v1.json"
```

CALVIN G3 uses `scripts/calvin/run_fixed_anchor_overfit_single_gpu.sh` with the authenticated archive-direct dataset,
normalization artifact, and `P=538` prefix artifact. G4 and G5 use only materialized development resets; published
LIBERO fixed states and CALVIN-D policy scores remain unopened.

Provision the pinned 40-file original LIBERO HDF5 corpus for replay qualification and clean development-reset
generation with `scripts/download_libero_original_hdf5.sh`. The downloader verifies the committed 40-file inventory,
all 33,784,856,577 content bytes and SHA-256 values, and keeps mutable Hugging Face download metadata outside the
authenticated content root. Run the training-parquet binding and final qualification through
`scripts/run_bind_libero_expert_replay_single_gpu.sh` and
`scripts/run_qualify_libero_expert_replay_single_gpu.sh`; these bind the report to the TP=1 train environment while
the preserved launchers continue to bind the legacy TP=2 environment.

The official CALVIN archive endpoint is retained as the canonical identity. If that endpoint is too slow, the pinned
`scripts/calvin/download_archive_hf_mirror.sh` path downloads ten fixed-revision parts, verifies every part SHA-256,
assembles them, verifies the official 555,309,812,705-byte ZIP SHA-256, and then invokes the unchanged archive-direct
preparer. A partial official download is moved to `recovery_quarantine` only after the assembled archive passes its
full hash.

## Compute authorization boundary

Before any 30,000-update run, execute exactly ten measured updates for both objectives on both benchmarks using the
final artifacts and B=8 path. The compute report must include mean update time, peak GPU memory, checkpoint bytes,
projected time for one 30,000-update checkpoint, the twelve-run campaign, and rollout cost. No long run starts until
that report and the immutable training recipe are explicitly accepted.

After all six final checkpoints for a benchmark exist, obtain the external freeze token and create its complete
24-cell official pre-registration before the first benchmark policy call. The token is not needed for G0--G5 or the
ten-update compute qualification.

## Current staged result

The GPU-0 / TP=1 path has passed G0--G5. The real B=8/BF16 G2 probe peaked at 52.068 GiB without OOM, so no
activation-checkpointing variant was introduced. LIBERO G3 reduced fixed-anchor loss by 62.35x and round-tripped its
checkpoint exactly. The clean development G4 run trained for 500 updates and succeeded on 3/4 fixed local resets. The
independent 40-task expert-replay qualification also passed 40/40; its report SHA-256 is
`9687438e5b4de76e20d3287e8873c2477542b24a39a7056294ea79c726392300`.

CALVIN archive-direct authentication completed against the 555,309,812,705-byte official archive identity. CALVIN G3
reduced fixed-anchor loss by 43.55x. The final-source G5 checkpoint trained for 500 updates with final train loss
0.30790957, validation loss 0.31268335, and 52.654 GiB peak memory. Its predeclared twelve-reset held-out A/B/C rollout
succeeded on 3/12 resets (A 1/4, B 2/4, C 0/4), passing G5 without opening official CALVIN-D policy scores. The G5
summary SHA-256 is `6cbf87e13e6a74844c7b1c30e76e82d6c8b8906aee5bc42018923ef3a916aa7c`.

The exact four ten-update compute qualifications also completed on the final B=8/BF16/TP=1 path:

| Benchmark | Objective | Mean seconds/update | Peak GiB | Projected 30k days |
| --- | --- | ---: | ---: | ---: |
| LIBERO | rectified flow | 31.072 | 52.694 | 10.789 |
| LIBERO | direct regression | 30.590 | 52.694 | 10.621 |
| CALVIN | rectified flow | 21.673 | 52.652 | 7.525 |
| CALVIN | direct regression | 21.442 | 52.652 | 7.445 |

The resulting twelve-run, three-seed campaign projects to 109.143 serial GPU-0 days for optimizer updates alone.
Twelve final trainable checkpoints require about 3.783 GiB; retaining every permanent 5,000-update checkpoint requires
about 22.695 GiB. The immutable compute report is
`/hdd2/hyunbin/vla/cache/reports/g6/g6-compute-qualification-tp1-v1.json` with SHA-256
`f5192d82076ea61e79ce9bc4abf04a0976eff880d54123dc3cd79eb713ec6059`.

The compute gate is now at its mandatory STOP boundary. No 30,000-update run has started; explicit acceptance of the
report and immutable training recipe is still required. The external freeze token remains deferred until all six final
checkpoints for a benchmark exist and before the first official benchmark policy call.
