#!/usr/bin/env python3
"""GPU check of saved SC+encoder policy at its recipe-bound training width; no simulator rollout."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rollout_libero_encoder_lora import EncoderSpatialPolicy, write_json  # noqa: E402


def main():
    import torch

    from duo_vla.runtime_determinism import configure_strict_cuda_determinism

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_strict_cuda_determinism(torch)
    policy = EncoderSpatialPolicy(args.checkpoint, args.expected_manifest_sha256, action_sc=True, nfe=4)
    artifact = policy.manifest["artifacts"]["serving_reference"]
    reference = json.loads((args.checkpoint / artifact["path"]).read_text())
    rows = reference["rows"]
    expected = torch.tensor(reference["raw_normalized_actions"])
    if len(rows) != 8 or reference["nfe"] != 4 or expected.shape != (8, 8, 7):
        raise ValueError("invalid serving reference")
    if reference["training_physical_batch_size"] != policy.physical_batch_size:
        raise ValueError("SC serving must preserve its training physical batch width")
    if policy.denoiser.action_projector.sc_projection.weight.count_nonzero().item() == 0:
        raise ValueError("qualification requires a checkpoint with learned, nonzero SC weights")

    def predict(observations):
        return torch.tensor(
            policy.predict(observations, record=False, return_normalized=True)["raw_normalized_actions"]
        )

    actual = predict(rows)
    repeated = predict(rows)
    permuted = predict(list(reversed(rows))).flip(0)
    singleton = predict(rows[:1])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
    torch.testing.assert_close(permuted, actual, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(singleton[0], actual[0], rtol=1e-5, atol=1e-5)
    result = {
        "status": "passed",
        "checkpoint_manifest_sha256": args.expected_manifest_sha256,
        "checkpoint_update": policy.manifest["update"],
        "nfe": 4,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "encoder_loaded": True,
        "self_conditioning": "action_endpoint_v1",
        "vision_frozen": True,
        "training_physical_batch_size": reference["training_physical_batch_size"],
        "serving_physical_batch_size": policy.physical_batch_size,
        "active_reference_rows": len(rows),
        "train_serving_raw_max_abs_error": float((actual - expected).abs().max()),
        "repeat_max_abs_error": float((actual - repeated).abs().max()),
        "permutation_max_abs_error": float((actual - permuted).abs().max()),
        "singleton_max_abs_error": float((actual[0] - singleton[0]).abs().max()),
    }
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
