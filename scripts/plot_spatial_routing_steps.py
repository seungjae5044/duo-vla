#!/usr/bin/env python3
"""Validate complete decoder traces and draw separate, title-free routing PNGs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from libero_bridge import libero_replan_seed

STATISTICS = ("counts", "top1_counts", "gate_weight_sums")
EXPERT_TICKS = [0, 16, 32, 48, 64, 80, 96, 112, 127]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_traces(run: Path, horizon: int) -> dict[str, Any]:
    """Require every real call and reconcile every layer, phase, and task count."""
    evaluation, policy = run / f"k{horizon}-evaluation", run / f"k{horizon}-policy"
    summary = read_json(evaluation / "summary.json")
    experiment = read_json(evaluation / "experiment.json")
    serving = read_json(policy / "serving.json")
    if summary["status"] != "complete" or summary["completed"] != 100 or summary["target"] != 100:
        raise ValueError(f"K={horizon}: requires exactly 100 completed episodes")
    if summary["nfe"] != 2 or summary["execution_horizon"] != horizon or not serving["decoder_trace"]["enabled"]:
        raise ValueError("unexpected inference settings or disabled decoder trace")
    if experiment["evaluation_seed"] != 0 or experiment["environment_seed"] != 7:
        raise ValueError("unexpected episode seeds")
    if experiment["max_policy_steps"] != 220 or experiment["settle_steps"] != 10:
        raise ValueError("unexpected episode step budget")
    if serving["cuda_visible_devices"] != {4: "4,5", 8: "6,7"}[horizon]:
        raise ValueError("unexpected GPU assignment")
    validation = read_json(evaluation / "inference_validation.json")
    if validation["repeat_max_abs_error"] != 0 or validation["permutation_max_abs_error"] > 1e-5:
        raise ValueError("inference determinism verification failed")
    if validation["warmup_included_in_routing"]:
        raise ValueError("warmup must be excluded from routing")
    episodes = [json.loads(line) for line in (evaluation / "episodes.jsonl").read_text().splitlines()]
    matrix = {(task, reset) for task in range(10) for reset in range(10)}
    if len(episodes) != 100 or {(row["task_id"], row["reset_id"]) for row in episodes} != matrix:
        raise ValueError("episode matrix must contain each requested episode exactly once")
    for row in episodes:
        if row["nfe"] != 2 or row["execution_horizon"] != horizon or not 0 < row["steps"] <= 220:
            raise ValueError("unexpected episode execution settings")
        if row["calls"] != (row["steps"] + horizon - 1) // horizon:
            raise ValueError("episode calls do not agree with executed actions")
    expected = {(row["task_id"], row["reset_id"], step) for row in episodes for step in range(row["calls"])}
    steps = max(row["calls"] for row in episodes)
    by_step = {
        key: np.zeros((2, 30, steps, 128), dtype=np.float64 if key == "gate_weight_sums" else np.int64)
        for key in STATISTICS
    }
    by_task = {key: np.zeros((10, 2, 30, 128), dtype=value.dtype) for key, value in by_step.items()}
    support = np.zeros(steps, dtype=np.int64)
    seen: set[tuple[int, int, int]] = set()
    sources = {}
    traces = sorted((policy / "routing_trace").glob("batch_*.npz"))
    if [path.name for path in traces] != [f"batch_{i:06d}.npz" for i in range(summary["batch_calls"])]:
        raise ValueError("missing or extra trace batches")
    for path in traces:
        sources[str(path.relative_to(run))] = digest(path)
        with np.load(path, allow_pickle=False) as archive:
            data = dict(archive)
            count = len(data["task_id"])
            if not 1 <= count <= 8:
                raise ValueError("trace has an invalid number of active rows")
            for key in ("task_id", "reset_id", "policy_step", "environment_step", "inference_seed"):
                if data[key].shape != (count,) or not np.issubdtype(data[key].dtype, np.integer):
                    raise ValueError("invalid trace identity")
            for key in STATISTICS:
                if data[key].shape != (count, 2, 30, 128) or not np.isfinite(data[key]).all() or (data[key] < 0).any():
                    raise ValueError("invalid trace statistic shape or values")
            if data["tokens"].shape != (count, 2, 30) or not (data["tokens"] == 8).all():
                raise ValueError("each decoder pass must route eight action tokens")
            if not (data["counts"].sum(axis=-1) == 64).all() or not (data["top1_counts"].sum(axis=-1) == 8).all():
                raise ValueError("top-8 or top-1 trace assignment count is incorrect")
            if (data["counts"] > 8).any() or (data["top1_counts"] > data["counts"]).any():
                raise ValueError("invalid per-token expert selection multiplicity")
            for row in range(count):
                task, reset, step = (int(data[key][row]) for key in ("task_id", "reset_id", "policy_step"))
                identity = (task, reset, step)
                if identity not in expected or identity in seen:
                    raise ValueError(f"duplicate or unexpected episode policy call: {identity}")
                if int(data["environment_step"][row]) != step * horizon:
                    raise ValueError("policy call and environment step do not agree")
                seed = libero_replan_seed(0, "libero_spatial", task, "official", reset, None, step)
                if int(data["inference_seed"][row]) != seed:
                    raise ValueError("trace inference seed differs from the episode contract")
                seen.add(identity)
                support[step] += 1
                for key in STATISTICS:
                    by_step[key][:, :, step, :] += data[key][row]
                    by_task[key][task] += data[key][row]
    if seen != expected:
        raise ValueError("missing real episode policy calls in trace")
    report = read_json(policy / "expert_routing.json")
    aggregate = {key: value.sum(axis=2) for key, value in by_step.items()}
    tasks_calls = [sum(row["calls"] for row in episodes if row["task_id"] == task) for task in range(10)]
    for records, task_scope in ((report["layers"], False), (report["per_task"], True)):
        records = [row for row in records if row["layer"].startswith("model.decoder.layers.")]
        expected_keys = {
            (phase, layer, task)
            for phase in range(2)
            for layer in range(30)
            for task in (range(10) if task_scope else [None])
        }
        checked = set()
        for record in records:
            phase, layer = int(record["phase"].split("_")[1]), int(record["layer"].rsplit(".", 1)[1])
            task = record["task_id"] if task_scope else None
            identity = (phase, layer, task)
            if identity not in expected_keys or identity in checked:
                raise ValueError("unexpected or duplicate decoder aggregate record")
            checked.add(identity)
            for key in STATISTICS:
                actual = by_task[key][task, phase, layer] if task_scope else aggregate[key][phase, layer]
                if key == "gate_weight_sums":
                    np.testing.assert_allclose(actual, record[key], rtol=1e-12, atol=1e-9)
                else:
                    np.testing.assert_array_equal(actual, record[key])
            if record["tokens"] != 8 * (tasks_calls[task] if task_scope else len(expected)):
                raise ValueError("aggregate decoder token count differs from trace")
        if checked != expected_keys:
            raise ValueError("incomplete decoder aggregate inventory")
    if summary["successes"] != sum(row["success"] for row in episodes):
        raise ValueError("summary and individual episode outcomes disagree")
    for path in [
        evaluation / name for name in ("summary.json", "episodes.jsonl", "experiment.json", "inference_validation.json")
    ] + [policy / "serving.json", policy / "expert_routing.json"]:
        sources[str(path.relative_to(run))] = digest(path)
    return {"by_step": by_step, "support": support, "summary": summary, "episodes": episodes, "sources": sources}


def render_histogram(counts: np.ndarray, output: Path, ymax: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    figure, axis = plt.subplots(figsize=(8, 3.2), layout="constrained")
    axis.bar(np.arange(128), counts, width=0.9, color="#28679B", linewidth=0)
    axis.set(xlabel="Expert", ylabel="Frequency", xlim=(-0.75, 127.75), ylim=(0, ymax))
    axis.set_xticks(EXPERT_TICKS)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=3)
    figure.savefig(
        output, dpi=300, metadata={"Description": "Top-8 expert assignment frequency; two NFE passes summed"}
    )
    plt.close(figure)


def render_heatmap(counts: np.ndarray, output: Path, vmax: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure, axis = plt.subplots(figsize=(8, 4.4), layout="constrained")
    view = axis.imshow(
        counts.T, origin="lower", interpolation="nearest", aspect="auto", cmap="Blues", vmin=0, vmax=vmax
    )
    axis.set(xlabel="Policy step", ylabel="Expert")
    axis.set_yticks(EXPERT_TICKS)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    axis.tick_params(length=3)
    colorbar = figure.colorbar(view, ax=axis, pad=0.025, fraction=0.045)
    colorbar.set_label("Frequency")
    colorbar.locator = MaxNLocator(nbins=5, integer=True)
    colorbar.update_ticks()
    figure.savefig(
        output, dpi=300, metadata={"Description": "Per-episode policy invocation index; two NFE passes summed"}
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    results = {horizon: load_traces(run, horizon) for horizon in (4, 8)}
    output = run / "figures"
    output.mkdir(exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.linewidth": 0.7,
            "savefig.facecolor": "white",
            "axes.titlepad": 0,
        }
    )
    np.savez_compressed(
        output / "plot_data.npz",
        **{
            f"k{horizon}_{key}": value
            for horizon, result in results.items()
            for key, value in {**result["by_step"], "support": result["support"]}.items()
        },
    )
    heatmap_max = max(int(result["by_step"]["counts"].sum(axis=0).max()) for result in results.values())
    manifest: dict[str, Any] = {
        "run": str(run),
        "nfe": 2,
        "layers": list(range(30)),
        "expert_ids": [0, 127],
        "png_count": 120,
        "frequency": "top-8 assignment count over all eight predicted action tokens; both denoising passes summed",
        "policy_step": "zero-based model invocation within each episode, not denoising time or environment step",
        "environment_step": "policy_step * K; original trace excludes ten settling actions",
        "heatmap_scale": {"minimum": 0, "maximum": heatmap_max, "shared_across_all_figures": True},
        "histogram_scale": "shared between K=4 and K=8 for the same layer",
        "support_caveat": "later policy steps have fewer active episodes; frequencies are unnormalized",
        "plot_data_axes": ["denoising_phase", "decoder_layer", "policy_step", "expert"],
        "plot_script_sha256": digest(Path(__file__)),
        "files": [],
        "validation": {},
    }
    concentration = []
    html = [
        "<!doctype html><meta charset='utf-8'><title>Decoder routing figures</title>",
        "<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px}"
        "img{max-width:100%;height:auto}section{border-top:1px solid #ddd;margin-top:32px}"
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:700px)"
        "{.pair{grid-template-columns:1fr}}nav a{margin-right:16px}</style>",
        "<h1>Decoder routing</h1><p>NFE=2; 100 episodes each for K=4 and K=8. "
        "Expert IDs 0-127 are local to each layer. Frequencies sum both denoising passes and all top-8 selections. "
        "Policy step is the zero-based model invocation within an episode. "
        "Later steps have fewer active episodes; counts are not normalized.</p>",
        "<nav><a href='#k4-layer20'>K=4 layer 20</a><a href='#k8-layer20'>K=8 layer 20</a>"
        "<a href='README.md'>Data definitions and validation</a></nav>",
    ]
    for horizon, result in results.items():
        manifest["validation"][f"k{horizon}"] = {
            "episodes": 100,
            "successes": result["summary"]["successes"],
            "policy_calls": int(result["support"].sum()),
            "policy_step_support": result["support"].tolist(),
            "all_calls_present_once": True,
            "all_layer_phase_task_aggregates_match": True,
            "inference_seeds_match": True,
            "source_sha256": result["sources"],
        }
        if args.reference_run is not None:
            original = [
                json.loads(line)
                for line in (args.reference_run / f"k{horizon}-evaluation/episodes.jsonl").read_text().splitlines()
            ]
            fields = (
                "task_id",
                "reset_id",
                "steps",
                "calls",
                "success",
                "action_clipped_channels",
                "execution_horizon",
                "nfe",
                "normalized_clip_fraction",
            )

            def canonical(rows: list[dict[str, Any]], fields: tuple[str, ...] = fields) -> list[tuple[Any, ...]]:
                return sorted(tuple(row[key] for key in fields) for row in rows)

            manifest["validation"][f"k{horizon}"]["reference_episode_outcomes_identical"] = canonical(
                original
            ) == canonical(result["episodes"])
            original_by_id = {(row["task_id"], row["reset_id"]): row for row in original}
            manifest["validation"][f"k{horizon}"]["reference_comparison"] = {
                "run": str(args.reference_run.resolve()),
                "reference_successes": sum(row["success"] for row in original),
                "current_successes": result["summary"]["successes"],
                "different_episode_count_by_field": {
                    field: sum(
                        row[field] != original_by_id[(row["task_id"], row["reset_id"])][field]
                        for row in result["episodes"]
                    )
                    for field in fields[2:]
                },
                "interpretation": "Figures use only this new trace run; no cross-run identity is assumed",
            }
        for layer in range(30):
            directory = output / f"k{horizon}" / f"layer_{layer:02d}"
            directory.mkdir(parents=True)
            histogram, heatmap = directory / "figure1_histogram.png", directory / "figure2_heatmap.png"
            counts = result["by_step"]["counts"][:, layer].sum(axis=0)
            top1 = result["by_step"]["top1_counts"][:, layer].sum(axis=(0, 1))
            weights = result["by_step"]["gate_weight_sums"][:, layer].sum(axis=(0, 1))
            tokens = int(result["support"].sum()) * 16
            top1_expert = int(top1.argmax())
            concentration.append(
                {
                    "k": horizon,
                    "decoder_layer": layer,
                    "top1_expert": top1_expert,
                    "top1_fraction": float(top1[top1_expert] / tokens),
                    "top1_expert_token_selection_fraction": float(counts[:, top1_expert].sum() / tokens),
                    "top1_expert_gate_weight_share": float(weights[top1_expert] / weights.sum()),
                    "active_experts": int(np.count_nonzero(counts.sum(axis=0))),
                    "routed_tokens": tokens,
                    "assignments": int(counts.sum()),
                }
            )
            ymax = (
                max(float(item["by_step"]["counts"][:, layer].sum(axis=(0, 1)).max()) for item in results.values())
                * 1.06
            )
            render_histogram(counts.sum(axis=0), histogram, ymax)
            render_heatmap(counts, heatmap, heatmap_max)
            for path in (histogram, heatmap):
                with Image.open(path) as picture:
                    picture.verify()
                with Image.open(path) as picture:
                    dimensions = list(picture.size)
                manifest["files"].append(
                    {
                        "path": str(path.relative_to(output)),
                        "sha256": digest(path),
                        "k": horizon,
                        "layer": layer,
                        "pixels": dimensions,
                    }
                )
            html.append(
                f"<section id='k{horizon}-layer{layer}'><h2>K={horizon} · Decoder layer {layer}</h2>"
                "<div class='pair'>"
                + "".join(
                    f"<a href='{path.relative_to(output)}'>"
                    f"<img loading='lazy' src='{path.relative_to(output)}' "
                    f"alt='K={horizon}, decoder layer {layer}, {path.stem}'></a>"
                    for path in (histogram, heatmap)
                )
                + "</div></section>"
            )
            print(json.dumps({"k": horizon, "layer": layer, "pngs": 2}), flush=True)
    if len(list(output.rglob("*.png"))) != 120:
        raise ValueError("expected exactly 120 individual PNG figures")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "index.html").write_text("\n".join(html) + "\n")
    with (output / "concentration.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(concentration[0]))
        writer.writeheader()
        writer.writerows(concentration)
    (output / "README.md").write_text(
        "# Decoder routing figures\n\n"
        "NFE=2; K=4 on GPUs 4/5 and K=8 on GPUs 6/7. Each setting uses all ten Spatial tasks "
        "and published initial states 0-9 (100 episodes).\n\n"
        "- `k{4,8}/layer_{00..29}/figure1_histogram.png`: expert ID versus total selection frequency.\n"
        "- `k{4,8}/layer_{00..29}/figure2_heatmap.png`: policy step versus expert ID; color is selection frequency.\n"
        "- Each PNG contains one figure, no title, at 300 dpi. Expert IDs are 0-127, local to each layer.\n"
        "- Both NFE passes (flow times 0 and 0.5) and all eight predicted action tokens are summed. "
        "Frequency counts every top-8 assignment, not just the highest-weight expert. "
        "The model predicts eight actions even when K=4 executes only four.\n"
        "- Policy step is the zero-based model invocation index within each episode. "
        "Environment step equals policy step * K, excluding ten settling actions.\n"
        "- All episodes contributing at the same policy step are summed. Early termination reduces "
        "later-step support. Raw frequencies are not probabilities or per-episode averages. "
        "K=4 normally has more model calls than K=8.\n"
        "- Heatmaps share one linear color scale across all layers and both K values. "
        "Histogram y-limits are shared between K values for each layer.\n\n"
        "`plot_data.npz` preserves counts, largest-gate top-1 counts, and gate-weight sums separately "
        "by [denoising phase, decoder layer, policy step, expert], plus contributing episode counts "
        "in `k4_support` and `k8_support`. Original per-episode traces remain in each policy directory.\n\n"
        "`concentration.csv` reports each layer's largest-gate top-1 expert, top-1 frequency, "
        "token selection fraction, and gate-weight share, pooling the two denoising passes.\n\n"
        "Validation requires all 200 episode identities and every real policy call exactly once, "
        "checks inference seeds and step mapping, excludes warmup and padding, and reconciles "
        "all 60 decoder layer/phase aggregates and all 600 layer/phase/task aggregates per K. "
        "`manifest.json` records support, source and PNG hashes, and reference-run comparisons. "
        "Matching seeds do not imply identical trajectories across separate simulator/model processes; "
        "the comparison records any observed differences without substituting previous-run counts. "
        "These are portable 100-episode subset runs, not the preregistered 500-episode official score.\n"
    )
    archive = run / "decoder_routing_figures.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output))
    with ZipFile(archive) as bundle:
        if bundle.testzip() is not None or sum(name.endswith(".png") for name in bundle.namelist()) != 120:
            raise ValueError("invalid figure archive")
    print(json.dumps({"status": "complete", "pngs": 120, "output": str(output), "zip": str(archive)}), flush=True)


if __name__ == "__main__":
    main()
