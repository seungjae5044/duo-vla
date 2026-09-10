#!/usr/bin/env python3
"""Summarize paired Spatial outcomes and layer-specific expert concentration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    results = {}
    phases = ("prefix", "denoise_0", "denoise_1")
    markdown = [
        "# Spatial rollout and Expert routing",
        "",
        "NFE=2; Spatial 10 tasks x initial states 0-9; evaluation seed 0.",
        "",
        "| K | GPUs | Successes | Episodes | Success rate | 95% Wilson interval |",
        "|---|---|---:|---:|---:|---|",
    ]
    for horizon, gpus in ((4, "4,5"), (8, "6,7")):
        root = args.run / f"k{horizon}-evaluation"
        summary = json.loads((root / "summary.json").read_text())
        if summary["status"] != "complete" or summary["completed"] != 100:
            raise ValueError("analysis requires both completed 100-episode runs")
        routing = json.loads((args.run / f"k{horizon}-policy/expert_routing.json").read_text())
        episodes = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
        if {(x["task_id"], x["reset_id"]) for x in episodes} != {(t, r) for t in range(10) for r in range(10)}:
            raise ValueError("episode identities do not cover the requested matrix")
        lo, hi = summary["wilson95"]
        markdown.append(
            f"| {horizon} | {gpus} | {summary['successes']} | 100 | {summary['success_rate']:.0%} | {lo:.1%}-{hi:.1%} |"
        )
        phase_stats = {}
        for phase in phases:
            layers = [layer for layer in routing["layers"] if layer["phase"] == phase]
            if len(layers) != 30:
                raise ValueError(f"K={horizon} {phase} does not cover all 30 layers")
            expected_tokens = sum(episode["calls"] for episode in episodes) * 8
            if phase != "prefix" and any(layer["tokens"] != expected_tokens for layer in layers):
                raise ValueError("action routing token count does not match actual episode policy calls")
            worst = max(layers, key=lambda value: value["max_top1_fraction"])
            phase_stats[phase] = {
                "mean_normalized_entropy": float(np.mean([layer["normalized_entropy"] for layer in layers])),
                "min_normalized_entropy": min(layer["normalized_entropy"] for layer in layers),
                "min_effective_experts": min(layer["effective_experts"] for layer in layers),
                "max_token_selection_fraction": max(layer["max_token_selection_fraction"] for layer in layers),
                "max_top1_fraction": worst["max_top1_fraction"],
                "worst_top1_layer": worst["layer"],
                "worst_top1_expert": int(np.argmax(worst["top1_counts"])),
                "worst_top1_gate_weight_share": (
                    worst["gate_weight_sums"][int(np.argmax(worst["top1_counts"]))] / sum(worst["gate_weight_sums"])
                ),
                "max_gate_weight_share": max(layer["max_gate_weight_share"] for layer in layers),
                "layers_top1_above_50pct": sum(layer["max_top1_fraction"] > 0.5 for layer in layers),
                "min_active_experts": min(layer["active_experts"] for layer in layers),
            }
        results[str(horizon)] = {"summary": summary, "routing": phase_stats, "episodes": episodes}
    markdown.extend(["", "## Task outcomes", "", "| Task | K=4 | K=8 |", "|---|---:|---:|"])
    for task_id in range(10):
        task = next(x["task_name"] for x in results["4"]["episodes"] if x["task_id"] == task_id)
        counts = [sum(x["success"] for x in results[str(k)]["episodes"] if x["task_id"] == task_id) for k in (4, 8)]
        markdown.append(f"| {task_id}: {task} | {counts[0]}/10 | {counts[1]}/10 |")
    markdown.extend(
        [
            "",
            "## Expert concentration",
            "",
            "Expert IDs are local to each layer. Statistics exclude padding, inactive rows, and warm-up calls. "
            "At NFE=2, denoise_0 uses flow time t=0 and denoise_1 uses t=0.5. "
            "Each token selects 8 of 128 experts; uniform assignment share is 1/128 (0.78125%). "
            "A 12.5% assignment share can mean one expert is selected by every token. "
            "Top-1 denotes the selected expert with the largest gate weight; gate-weight share is also reported. "
            "Concentration alone does not establish training collapse.",
            "",
            "| K | Phase | Mean normalized entropy | Minimum effective experts | Max top-1 share | "
            "Max gate-weight share | Layers with top-1 >50% |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in (4, 8):
        for phase, stats in results[str(horizon)]["routing"].items():
            markdown.append(
                f"| {horizon} | {phase} | {stats['mean_normalized_entropy']:.3f} | "
                f"{stats['min_effective_experts']:.1f} | {stats['max_top1_fraction']:.1%} | "
                f"{stats['max_gate_weight_share']:.1%} | {stats['layers_top1_above_50pct']}/30 |"
            )
    markdown.extend(
        [
            "",
            "### Most concentrated top-1 routes",
            "",
            "Layer and expert indices below are zero-based. "
            "The gate-weight share refers to the listed expert in the listed layer.",
            "",
            "| K | Phase | Layer | Expert | Top-1 share | Gate-weight share |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for horizon in (4, 8):
        for phase, stats in results[str(horizon)]["routing"].items():
            markdown.append(
                f"| {horizon} | {phase} | {stats['worst_top1_layer']} | {stats['worst_top1_expert']} | "
                f"{stats['max_top1_fraction']:.3%} | {stats['worst_top1_gate_weight_share']:.2%} |"
            )
    metadata = json.loads((args.run / "k4-policy/serving.json").read_text())
    adapter = json.loads((Path(metadata["checkpoint"]) / "lora/adapter_config.json").read_text())
    targets = adapter["target_modules"]
    routing_targets = [name for name in targets if "router" in name or "expert" in name]
    markdown.extend(
        [
            "",
            "### Router adaptation",
            "",
            f"The checkpoint has {len(targets)} LoRA targets; router/expert targets: {len(routing_targets)}. "
            "The targets are decoder self-attention q/k/v/o projections. Frozen routers still receive hidden states "
            "affected by attention LoRA and the learned action interface. The measured concentration is therefore "
            "a routing observation, not evidence that the router parameters themselves were adapted.",
        ]
    )
    (args.run / "analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.run / "analysis.md").write_text("\n".join(markdown) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    for row, horizon in enumerate((4, 8)):
        routing = json.loads((args.run / f"k{horizon}-policy/expert_routing.json").read_text())
        for column, phase in enumerate(phases):
            layers = sorted(
                [x for x in routing["layers"] if x["phase"] == phase],
                key=lambda value: int(value["layer"].rsplit(".", 1)[1]),
            )
            shares = np.asarray([np.asarray(x["counts"]) / x["assignments"] for x in layers]) * 100
            chart = axes[row, column].imshow(shares, aspect="auto", vmin=0, vmax=12.5, cmap="magma")
            axes[row, column].set(title=f"K={horizon}, {phase}", xlabel="Expert ID", ylabel="Layer")
    figure.colorbar(chart, ax=axes, label="Share of routed assignments (%)", shrink=0.85)
    figure.suptitle("Duo-VLA Spatial step 10000: NFE=2, 100 rollouts per K")
    figure.savefig(args.run / "expert_routing.png", dpi=180)
    figure.savefig(args.run / "expert_routing.pdf")
    print(args.run / "analysis.md")


if __name__ == "__main__":
    main()
