from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_libero_offline_common.py"
    spec = importlib.util.spec_from_file_location("evaluate_libero_offline_common", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_common_metric_masks_dimensions_and_fails_zero_gripper_ties_closed() -> None:
    module = _module()
    prediction = torch.zeros(8, 8, 7)
    target = torch.zeros_like(prediction)
    target[..., 6] = 1.0
    valid = torch.zeros(8, 8, dtype=torch.bool)
    valid[:, 0] = True

    metric = module._common_action_metrics(prediction, target, valid)

    assert metric["overall"]["count"] == 8 * 7
    assert metric["continuous"]["count"] == 8 * 6
    assert metric["gripper"]["count"] == 8
    assert metric["gripper"]["mse"] == 1.0
    assert metric["gripper"]["mae"] == 1.0
    assert metric["gripper"]["sign_accuracy"] == 0.0
    assert metric["gripper"]["zero_tie_count"] == 8


def test_tensor_record_binds_shape_dtype_and_bytes() -> None:
    module = _module()
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    first = module._tensor_record(tensor)
    second = module._tensor_record(tensor.clone())
    reshaped = module._tensor_record(tensor.reshape(4, 2))

    assert first == second
    assert first["data_sha256"] == reshaped["data_sha256"]
    assert first["tensor_sha256"] != reshaped["tensor_sha256"]
