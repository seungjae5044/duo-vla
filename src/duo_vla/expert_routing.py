"""Read-only, token-weighted MoE routing statistics for simulator rollouts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def routing_metrics(counts: np.ndarray, top1: np.ndarray, weights: np.ndarray, tokens: int) -> dict[str, Any]:
    """Distinguish assignment share from the fraction of tokens selecting an expert."""
    counts = np.asarray(counts, dtype=np.int64)
    top1 = np.asarray(top1, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    total = int(counts.sum())
    probability = counts / total if total else np.zeros_like(weights)
    positive = probability[probability > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    experts = len(counts)
    dominant = int(np.argmax(counts))
    return {
        "tokens": tokens,
        "assignments": total,
        "num_experts": experts,
        "active_experts": int(np.count_nonzero(counts)),
        "counts": counts.tolist(),
        "top1_counts": top1.tolist(),
        "gate_weight_sums": weights.tolist(),
        "dominant_expert": dominant,
        "max_assignment_share": float(probability.max()),
        "max_token_selection_fraction": float(counts.max() / tokens) if tokens else 0.0,
        "max_top1_fraction": float(top1.max() / tokens) if tokens else 0.0,
        "max_gate_weight_share": float(weights.max() / weights.sum()) if weights.sum() else 0.0,
        "max_to_uniform_ratio": float(probability.max() * experts),
        "normalized_entropy": entropy / np.log(experts) if experts > 1 else 0.0,
        "effective_experts": float(np.exp(entropy)) if total else 0.0,
    }


class RoutingAccumulator:
    """Accumulate real routes, excluding padded tokens and inactive batch rows."""

    def __init__(self, num_experts: int = 128) -> None:
        self.num_experts = num_experts
        self.groups: dict[tuple[str, str, int], dict[str, Any]] = defaultdict(self._empty)

    def _empty(self) -> dict[str, Any]:
        return {
            "counts": np.zeros(self.num_experts, dtype=np.int64),
            "top1": np.zeros(self.num_experts, dtype=np.int64),
            "weights": np.zeros(self.num_experts, dtype=np.float64),
            "tokens": 0,
            "calls": 0,
        }

    def observe(
        self,
        layer: str,
        phase: str,
        indices: np.ndarray,
        weights: np.ndarray,
        valid_mask: np.ndarray,
        task_ids: list[int],
    ) -> None:
        indices, weights = np.asarray(indices), np.asarray(weights)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if indices.ndim != 3 or weights.shape != indices.shape or valid_mask.shape != indices.shape[:2]:
            raise ValueError("expected routes [batch, tokens, top_k] and mask [batch, tokens]")
        if len(task_ids) != indices.shape[0] or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("invalid routing batch identity or route dtype")
        selected, selected_weights = indices[valid_mask], weights[valid_mask]
        if np.any(selected < 0) or np.any(selected >= self.num_experts):
            raise ValueError("selected expert is outside the model's expert inventory")
        if not np.isfinite(selected_weights).all() or np.any(selected_weights < 0):
            raise ValueError("routing weights must be finite and nonnegative")
        for task_id in sorted(set(task_ids)):
            row_mask = np.asarray(task_ids) == task_id
            task_mask = valid_mask & row_mask[:, None]
            ids, gates = indices[task_mask], weights[task_mask]
            if not len(ids):
                continue
            group = self.groups[(layer, phase, task_id)]
            group["counts"] += np.bincount(ids.ravel(), minlength=self.num_experts)
            # Highest actual gate weight, independent of top-k output ordering.
            top1_ids = ids[np.arange(len(ids)), np.argmax(gates, axis=1)]
            group["top1"] += np.bincount(top1_ids, minlength=self.num_experts)
            group["weights"] += np.bincount(ids.ravel(), weights=gates.ravel(), minlength=self.num_experts)
            group["tokens"] += len(ids)
            group["calls"] += 1

    def report(self) -> dict[str, Any]:
        by_layer: dict[tuple[str, str], dict[str, Any]] = defaultdict(self._empty)
        per_task = []
        for (layer, phase, task_id), group in sorted(self.groups.items()):
            per_task.append({"layer": layer, "phase": phase, "task_id": task_id, **self._metrics(group)})
            aggregate = by_layer[(layer, phase)]
            for key in ("counts", "top1", "weights", "tokens", "calls"):
                aggregate[key] += group[key]
        return {
            "schema": "duo-vla-rollout-expert-routing-v1",
            "scope": "valid tokens in active rollout rows; each model counted once",
            "layers": [
                {"layer": layer, "phase": phase, **self._metrics(group)}
                for (layer, phase), group in sorted(by_layer.items())
            ],
            "per_task": per_task,
        }

    @staticmethod
    def _metrics(group: dict[str, Any]) -> dict[str, Any]:
        return routing_metrics(group["counts"], group["top1"], group["weights"], group["tokens"])


class DecoderRoutingTrace:
    """Persist per-episode, per-call routes without retaining images or padded rows."""

    identity_fields = ("task_id", "reset_id", "policy_step", "environment_step", "inference_seed")

    def __init__(self, directory: Path, nfe: int, layers: int = 30, experts: int = 128) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.nfe, self.layers, self.experts = nfe, layers, experts
        self.batch_index = 0
        self.rows: list[dict[str, int]] = []
        self.observed: set[tuple[int, int]] = set()

    def begin(self, rows: list[dict[str, Any]]) -> None:
        if self.rows:
            raise RuntimeError("previous routing trace batch has not finished")
        identities = [{key: int(row[key]) for key in self.identity_fields} for row in rows]
        if not identities or any(value < 0 for row in identities for value in row.values()):
            raise ValueError("trace requires nonnegative episode and policy step identities")
        if len({(row["task_id"], row["reset_id"]) for row in identities}) != len(identities):
            raise ValueError("duplicate active episode in routing trace")
        self.rows = identities
        shape = (len(rows), self.nfe, self.layers, self.experts)
        self.counts = np.zeros(shape, dtype=np.int64)
        self.top1 = np.zeros(shape, dtype=np.int64)
        self.weights = np.zeros(shape, dtype=np.float64)
        self.tokens = np.zeros(shape[:-1], dtype=np.int64)
        self.observed = set()

    def observe(
        self, layer: int, phase: int, indices: np.ndarray, weights: np.ndarray, valid_mask: np.ndarray
    ) -> None:
        if not self.rows or not 0 <= layer < self.layers or not 0 <= phase < self.nfe:
            raise ValueError("invalid trace batch, decoder layer, or denoising phase")
        if (phase, layer) in self.observed:
            raise ValueError("duplicate decoder routing observation")
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if indices.ndim != 3 or weights.shape != indices.shape or valid_mask.shape != indices.shape[:2]:
            raise ValueError("expected routes [batch, tokens, top_k] and mask [batch, tokens]")
        if indices.shape[0] < len(self.rows) or valid_mask[len(self.rows) :].any():
            raise ValueError("trace must exclude inactive batch rows")
        selected, gates = indices[valid_mask], weights[valid_mask]
        if not np.issubdtype(indices.dtype, np.integer) or np.any(selected < 0) or np.any(selected >= self.experts):
            raise ValueError("invalid trace expert index")
        if not np.isfinite(gates).all() or np.any(gates < 0):
            raise ValueError("invalid trace routing weights")
        for row in range(len(self.rows)):
            ids, gates = indices[row, valid_mask[row]], weights[row, valid_mask[row]]
            self.counts[row, phase, layer] = np.bincount(ids.ravel(), minlength=self.experts)
            top1_ids = ids[np.arange(len(ids)), np.argmax(gates, axis=1)]
            self.top1[row, phase, layer] = np.bincount(top1_ids, minlength=self.experts)
            self.weights[row, phase, layer] = np.bincount(ids.ravel(), weights=gates.ravel(), minlength=self.experts)
            self.tokens[row, phase, layer] = len(ids)
        self.observed.add((phase, layer))

    def finish(self) -> Path:
        if not self.rows or len(self.observed) != self.nfe * self.layers:
            raise ValueError("incomplete decoder routing trace")
        path = self.directory / f"batch_{self.batch_index:06d}.npz"
        if path.exists():
            raise FileExistsError(path)
        temporary = path.with_suffix(".tmp")
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                **{key: np.asarray([row[key] for row in self.rows], dtype=np.int64) for key in self.identity_fields},
                counts=self.counts,
                top1_counts=self.top1,
                gate_weight_sums=self.weights,
                tokens=self.tokens,
            )
        temporary.replace(path)
        self.batch_index += 1
        self.rows = []
        return path
