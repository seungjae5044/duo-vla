# Spatial checkpoint rollout

The portable runner evaluates the downloaded Spatial checkpoint at NFE=2 with
K=4 on GPUs 4/5 and K=8 on GPUs 6/7. Each setting uses all ten Spatial tasks and
published initial states 0–9: 100 episodes per setting. Evaluation seed 0,
environment seed 7, ten settling steps, and the 220-step Spatial budget are fixed.

The checkpoint is `seungjae5044/duo-vla-libero-spatial-step10000` at revision
`8f8b813f5f756cf385063e5f2352e326099be48e`. It requires the separate DiffusionGemma
base snapshot and its checkpoint-copied normalization and prefix artifacts.
Keep a materialized checkpoint directory: the checkpoint validator verifies
contained files, byte counts, and hashes, so Hugging Face cache symlinks should
be copied into a separate directory before loading.

## Running

Prepare the locked `train` and `libero-eval` environments using the bootstrap
scripts. The cache root must also contain the LIBERO simulator assets/configuration
and an `egl-nvidia.json` vendor file selecting the installed NVIDIA EGL library.
The Hugging Face cache is expected alongside the Duo-VLA cache directory.

```bash
python3 scripts/run_spatial_rollout_pair.py \
  --cache-root /data/parkce/.cache/duo-vla \
  --checkpoint /data/parkce/.cache/duo-vla/checkpoints/libero-spatial-step10000 \
  --output /path/to/new/run-directory

/data/parkce/.cache/duo-vla/venvs/libero-eval/bin/python \
  scripts/summarize_spatial_routing.py /path/to/new/run-directory
```

Each model places tied encoder/decoder layer pairs on the same GPU, retaining
the original fused-v2 expert operations and physical batch eight. Active episodes
share a batch; padded rows are excluded from diagnostics. Three predictions of
the initial batch check repeatability and invariance to batch order, followed by
an exact comparison with instrumentation enabled. These initial probes do not
establish identical trajectories across separate simulator/model processes.

## Outputs and interpretation

`k*-evaluation/episodes.jsonl` records each episode, while `summary.json` records
progress, task success counts, and confidence intervals. `k*-policy/serving.json`
records checkpoint and serving identities. `expert_routing.json` separates prefix
routing from both denoising steps, by layer and task. Final analysis includes
Markdown, JSON, PNG, and PDF artifacts.

Each token selects eight of 128 experts. Assignment share, token selection rate,
largest-gate selection rate, gate-weight share, and entropy measure different
aspects of concentration. Expert IDs are specific to each layer. These results
describe the selected 100-episode subset under the recorded portable runtime;
latencies include routing instrumentation.

## Per-policy-step decoder figures

Add `--trace-routing` to the paired launcher to save per-episode decoder routing
in `k*-policy/routing_trace/batch_*.npz`. All 30 decoder layers and each denoising
pass retain top-8 counts, highest-gate top-1 counts, gate-weight sums, and episode
identities. Warmup calls, padded rows, and prefix routing are excluded from these
traces. The usual aggregate report still includes prefix routing.

After both 100-episode runs finish:

```bash
/data/parkce/.cache/duo-vla/venvs/libero-eval/bin/python \
  scripts/plot_spatial_routing_steps.py /path/to/new/run-directory \
  --reference-run /path/to/previous/run-directory
```

The optional reference compares episode outcomes with the previous run. The
plotter requires every requested episode and real policy call exactly once and
reconciles all layer/phase/task statistics before producing 120 title-free PNGs:
`figures/k{4,8}/layer_{00..29}/figure{1_histogram,2_heatmap}.png`. An HTML gallery,
plot data, and a manifest preserve figure definitions and validation evidence.

Both NFE passes and all eight predicted action tokens contribute to each figure.
Policy step means the zero-based model invocation within an episode; actual
environment step equals policy step times K and is also saved. Heatmap columns
sum all contributing episodes; later columns can have fewer episodes after early
termination. These are raw top-8 selection frequencies, not probabilities.

To additionally pool the same expert indices across all 30 decoder layers:

```bash
/data/parkce/.cache/duo-vla/venvs/libero-eval/bin/python \
  scripts/plot_spatial_routing_all_layers.py /path/to/new/run-directory
```

This validates the saved traces and writes two title-free, 300-dpi heatmaps to
`all_layers_figures/`, with a shared color scale, plot data, and source hashes.
Expert indices are local to each layer: pooling equal indices can hide strong
concentration within individual layers. Both plotters require fresh output
directories and never need another rollout when the complete traces are present.
