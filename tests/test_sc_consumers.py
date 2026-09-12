"""CPU sampler-plumbing parity through the actual serving entry points; no GPU/model parity claim."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_self_conditioning import RecordingDenoiser
from transformers import BatchEncoding

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rollout_libero_encoder_lora as ENCODER
import rollout_libero_spatial as PORTABLE
import serve_libero_policy as SERVER

from duo_vla.backbones import diffusion_gemma
from duo_vla.config import FlowConfig
from duo_vla.policy import RectifiedFlowPolicy


@pytest.mark.parametrize("nfe", [1, 2, 4, 5, 10])
def test_generic_portable_encoder_server_sampling_and_query_reset_agree(monkeypatch, nfe):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    prefix = SimpleNamespace(past_key_values=object(), attention_mask=torch.ones(8, 3, dtype=torch.bool))
    monkeypatch.setattr(diffusion_gemma, "encode_diffusion_gemma_prefix", lambda *args: prefix)
    monkeypatch.setattr(
        "duo_vla.prefix_geometry.apply_fixed_prefix_chat_template",
        lambda *args, **kwargs: BatchEncoding({"attention_mask": torch.ones(8, 3, dtype=torch.long)}),
    )
    normalize = SimpleNamespace(normalize=lambda value: value, unnormalize=lambda value: value)
    image = PORTABLE.encode_image(np.zeros((256, 256, 3), dtype=np.uint8))
    rows = [
        {
            "instruction": "task",
            "task_id": 0,
            "state": [0.0] * 8,
            "agentview": image,
            "wrist": image,
            "inference_seed": 7 + i,
        }
        for i in range(8)
    ]
    expected_noise = torch.cat(
        [torch.randn((1, 8, 7), generator=torch.Generator().manual_seed(7 + i)) for i in range(8)]
    )
    context = dict(
        prefix_cache=prefix.past_key_values,
        prefix_attention_mask=prefix.attention_mask,
        action_valid_mask=torch.ones(8, 8, dtype=torch.bool),
    )
    generic = RectifiedFlowPolicy(RecordingDenoiser(), FlowConfig(inference_steps=nfe))
    expected = generic.sample_normalized(torch.zeros(8, 8), initial_noise=expected_noise, **context)
    for policy_type in (PORTABLE.SpatialPolicy, ENCODER.EncoderSpatialPolicy):
        policy = policy_type.__new__(policy_type)
        policy.torch, policy.device = torch, torch.device("cpu")
        policy.model, policy.processor, policy.trace = object(), object(), None
        policy.config = {"benchmark": {"fixed_physical_prefix_width": 3}}
        policy.prefix_geometry = {"tokenization": {"padding_side": "left"}}
        policy.valid_lengths = {"task": 3}
        policy.state_norm = policy.action_norm = normalize
        policy.denoiser = RecordingDenoiser()
        policy.nfe = nfe
        policy.physical_batch_size = 8
        first = policy.predict(rows, record=False)
        second = policy.predict(rows, record=False)
        assert torch.equal(torch.tensor(first["actions"]), expected)
        assert first["actions"] == second["actions"]
        assert len(policy.denoiser.calls) == 2 * nfe
        assert policy.denoiser.calls[nfe][2] is None
        single = policy.predict(rows[:1], record=False)
        assert torch.equal(torch.tensor(single["actions"])[0], expected[0])
        permuted = policy.predict(list(reversed(rows)), record=False)
        assert torch.equal(torch.tensor(permuted["actions"]).flip(0), expected)

    server = SERVER.RealPolicy.__new__(SERVER.RealPolicy)
    server.torch, server.device = torch, torch.device("cpu")
    server.model, server.denoiser = object(), RecordingDenoiser()
    server.state_normalizer = server.action_normalizer = normalize
    server._processor_inputs = lambda request: {}
    server.encode_prefix = lambda *args: prefix
    server.policy_contract = {
        "objective": "rectified_flow",
        "nfe": nfe,
        "sampler": "euler_uniform",
        "inference_seed_behavior": "episode_identity_gaussian_noise",
    }
    monkeypatch.setattr(SERVER, "_evaluation_identity_echo", lambda request: {})
    request = {"observation": {"state": np.zeros(8, dtype=np.float32)}, "inference_seed": 7}
    first = server.predict(request)
    second = server.predict(request)
    assert torch.equal(torch.tensor(first["actions"]), expected[0])
    assert first["actions"] == second["actions"]
    assert server.denoiser.calls[nfe][2] is None


@pytest.mark.parametrize("physical_batch_size", [8, 64, 72])
def test_encoder_sc_serving_preserves_physical_width_for_singleton(monkeypatch, physical_batch_size):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    observed = []

    def process(processor, conversations, **kwargs):
        observed.append((len(conversations), kwargs["expected_batch_size"]))
        return BatchEncoding({"attention_mask": torch.ones(physical_batch_size, 3, dtype=torch.long)})

    monkeypatch.setattr("duo_vla.prefix_geometry.apply_fixed_prefix_chat_template", process)
    prefix = SimpleNamespace(
        past_key_values=object(), attention_mask=torch.ones(physical_batch_size, 3, dtype=torch.bool)
    )
    monkeypatch.setattr(diffusion_gemma, "encode_diffusion_gemma_prefix", lambda *args: prefix)
    policy = ENCODER.EncoderSpatialPolicy.__new__(ENCODER.EncoderSpatialPolicy)
    policy.torch, policy.device = torch, torch.device("cpu")
    policy.model, policy.processor = object(), object()
    policy.config = {"benchmark": {"fixed_physical_prefix_width": 3}}
    policy.prefix_geometry = {"tokenization": {"padding_side": "left"}}
    policy.valid_lengths = {"task": 3}
    policy.state_norm = policy.action_norm = SimpleNamespace(normalize=lambda x: x, unnormalize=lambda x: x)
    policy.denoiser = RecordingDenoiser()
    policy.nfe, policy.physical_batch_size = 4, physical_batch_size
    image = PORTABLE.encode_image(np.zeros((256, 256, 3), dtype=np.uint8))
    row = dict(instruction="task", state=[0.0] * 8, agentview=image, wrist=image, inference_seed=7)
    result = policy.predict([row], record=False, return_normalized=True)
    assert observed == [(physical_batch_size, physical_batch_size)]
    assert torch.tensor(result["raw_normalized_actions"]).shape == (1, 8, 7)
    assert all(call[0].shape[0] == physical_batch_size for call in policy.denoiser.calls)
    with pytest.raises(ValueError, match="observations"):
        policy.predict([row] * (physical_batch_size + 1), record=False)
