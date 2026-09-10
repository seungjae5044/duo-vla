#!/usr/bin/env python3
"""Pool decoder expert indices across layers into title-free policy-step heatmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plot_spatial_routing_steps import digest, load_traces, render_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    results = {k: load_traces(run, k) for k in (4, 8)}
    counts = {k: result["by_step"]["counts"].sum(axis=(0, 1)) for k, result in results.items()}
    for k, result in results.items():
        if counts[k].shape != (len(result["support"]), 128):
            raise ValueError("expected policy-step by expert-index frequency matrix")
        np.testing.assert_array_equal(counts[k].sum(axis=1), result["support"] * 2 * 30 * 8 * 8)
    output = run / "all_layers_figures"
    output.mkdir(exist_ok=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 0.7, "savefig.facecolor": "white"}
    )
    vmax = max(int(value.max()) for value in counts.values())
    metadata = {
        "run": str(run),
        "decoder_layers": list(range(30)),
        "nfe": 2,
        "frequency": "top-8 assignment counts summed over decoder layers, both NFE passes, and contributing episodes",
        "expert_index": "0-127; equal numeric indices are pooled across distinct layer-specific experts",
        "policy_step": "zero-based model invocation within each episode; environment step = K * policy step",
        "support_caveat": "raw frequencies; later steps can have fewer contributing episodes",
        "shared_color_scale": [0, vmax],
        "source_scripts": {
            path.name: digest(path)
            for path in (Path(__file__), Path(__file__).with_name("plot_spatial_routing_steps.py"))
        },
        "files": [],
    }
    arrays = {}
    for k, values in counts.items():
        path = output / f"k{k}_all_decoder_layers_heatmap.png"
        render_heatmap(values, path, vmax)
        with Image.open(path) as picture:
            picture.verify()
        arrays[f"k{k}_counts"] = values
        arrays[f"k{k}_support"] = results[k]["support"]
        metadata["files"].append(
            {
                "path": path.name,
                "sha256": digest(path),
                "k": k,
                "shape": list(values.shape),
                "policy_calls": int(results[k]["support"].sum()),
                "total_assignments": int(values.sum()),
                "all_trace_aggregates_verified": True,
                "source_sha256": results[k]["sources"],
            }
        )
    np.savez_compressed(output / "plot_data.npz", **arrays)
    metadata["plot_data_sha256"] = digest(output / "plot_data.npz")
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "README.md").write_text(
        "# All decoder layers: expert-index heatmaps\n\n"
        "Each PNG sums top-8 selection frequencies over decoder layers 0-29, both NFE=2 passes, "
        "all eight predicted action tokens, and contributing episodes (100 episodes per K).\n\n"
        "The x-axis is the zero-based model invocation within an episode; the y-axis is expert index 0-127. "
        "Equal numeric expert indices are pooled across layers; they do not identify a single shared expert. "
        "All figures have no title, 300 dpi, and the same linear frequency color scale. "
        "Later steps can have fewer contributing episodes after early termination.\n\n"
        "`plot_data.npz` contains [policy step, expert index] count matrices and episode support. "
        "The complete source traces were checked against all layer/phase/task aggregates before plotting. "
        "Every column sums to contributing episodes * 30 layers * 2 passes * 8 tokens * 8 experts.\n"
    )
    print(json.dumps({"status": "complete", "pngs": 2, "output": str(output), "shared_vmax": vmax}))


if __name__ == "__main__":
    main()
