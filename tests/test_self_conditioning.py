from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from duo_vla.action_interface import ActionInputProjector, VelocityHead, self_conditioning_features
from duo_vla.config import ActionInterfaceConfig, FlowConfig
from duo_vla.modeling import DuoVLADenoiser
from duo_vla.policy import RectifiedFlowPolicy
from duo_vla.self_conditioning import (
    SC_VERSION,
    bootstrap_for_update,
    endpoint_estimate,
    sample_action_flow,
    training_velocity,
)


def test_zero_initialization_preserves_outputs_old_state_and_rng():
    config = ActionInterfaceConfig(hidden_size=16, state_dim=8, timestep_embedding_dim=8)
    torch.manual_seed(7)
    old = ActionInputProjector(config)
    old_rng = torch.get_rng_state()
    torch.manual_seed(7)
    new = ActionInputProjector(replace(config, self_conditioning=SC_VERSION))
    assert torch.equal(old_rng, torch.get_rng_state())
    assert set(new.state_dict()) - set(old.state_dict()) == {"sc_projection.weight"}
    for name, tensor in old.state_dict().items():
        assert torch.equal(tensor, new.state_dict()[name])
    actions, t, state = torch.randn(2, 8, 7), torch.rand(2), torch.randn(2, 8)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[0, 4:] = False
    original = old(actions, t, state, valid_mask=valid)
    for present in (False, True):
        output = new(actions, t, state, valid_mask=valid, sc_actions=torch.randn_like(actions), sc_present=present)
        assert torch.equal(original, output)
        assert torch.count_nonzero(output[0, 4:]) == 0


def test_projection_gradients_and_presence_bit():
    config = ActionInterfaceConfig(hidden_size=16, state_dim=8, self_conditioning=SC_VERSION)
    model = ActionInputProjector(config)
    actions, t, state = torch.randn(2, 8, 7), torch.rand(2), torch.randn(2, 8)
    candidate = torch.randn_like(actions, requires_grad=True)
    for present in (True, False):
        model.zero_grad(set_to_none=True)
        model(actions, t, state, sc_actions=candidate, sc_present=present).sum().backward()
        grad = model.sc_projection.weight.grad
        assert grad is not None and grad.isfinite().all()
        assert bool(grad.abs().sum() > 0) == present
        assert candidate.grad is None
    features = self_conditioning_features(actions, torch.zeros_like(actions), torch.tensor([True, False]))
    assert torch.equal(features[0, :, -1], torch.ones(8))
    assert torch.count_nonzero(features[1]) == 0


@pytest.mark.parametrize("present", [1, torch.ones(2), torch.ones(2, 1, dtype=torch.bool)])
def test_presence_contract_rejects_ambiguous_shapes_and_types(present):
    with pytest.raises(ValueError, match="sc_present"):
        self_conditioning_features(torch.zeros(2, 8, 7), None, present)


def test_absence_and_nonfinite_contract():
    reference = torch.zeros(2, 8, 7)
    with pytest.raises(ValueError, match="requires a candidate"):
        self_conditioning_features(reference, None, True)
    with pytest.raises(ValueError, match="finite"):
        self_conditioning_features(reference, reference + float("nan"), True)
    assert torch.count_nonzero(self_conditioning_features(reference, reference + float("nan"), False)) == 0
    legacy = ActionInputProjector(ActionInterfaceConfig(hidden_size=16, state_dim=8))
    with pytest.raises(ValueError, match="require action_endpoint_v1"):
        legacy(reference, torch.zeros(2), torch.zeros(2, 8), sc_actions=reference)


class RecordingDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_projector = SimpleNamespace(config=SimpleNamespace(self_conditioning=SC_VERSION))
        self.weight = nn.Parameter(torch.tensor(0.3))
        self.calls = []

    def forward(self, actions, timesteps, state, *, sc_actions=None, sc_present=False, **context):
        self.calls.append(
            (actions.clone(), timesteps.clone(), sc_actions, sc_present, torch.is_grad_enabled(), context)
        )
        velocity = actions * self.weight + timesteps[:, None, None]
        if sc_present:
            velocity = velocity + sc_actions * 0.2
        return velocity


def test_bootstrap_uses_same_pair_prefix_without_gradient_through_prepass():
    model = RecordingDenoiser()
    actions = torch.randn(2, 8, 7, requires_grad=True)
    time = torch.tensor([0.2, 0.7])
    cache = object()
    output = training_velocity(model, actions, time, torch.zeros(2, 8), bootstrap=True, prefix_cache=cache)
    first, main = model.calls
    assert first[4] is False and main[4] is True
    assert first[2] is None and first[3] is False
    assert main[2].grad_fn is None and not main[2].requires_grad
    assert main[3] is True and first[5]["prefix_cache"] is main[5]["prefix_cache"] is cache
    assert torch.equal(first[0], main[0]) and torch.equal(first[1], main[1])
    torch.testing.assert_close(main[2], endpoint_estimate(actions, time, actions.detach() * 0.3 + time[:, None, None]))
    output.sum().backward()
    torch.testing.assert_close(model.weight.grad, actions.detach().sum())


@pytest.mark.parametrize("nfe", [1, 2, 4, 5, 10])
def test_sampling_preupdate_endpoint_call_count_reset_and_no_clipping(nfe):
    model = RecordingDenoiser()
    initial = torch.full((2, 8, 7), 2.0)
    state = torch.zeros(2, 8)
    first = sample_action_flow(model, initial, state, num_steps=nfe)
    assert len(model.calls) == nfe and not model.calls[0][3]
    for index in range(1, nfe):
        actions, time, candidate, present, *_ = model.calls[index - 1]
        velocity = actions * 0.3 + time[:, None, None] + (candidate * 0.2 if present else 0)
        torch.testing.assert_close(model.calls[index][2], endpoint_estimate(actions, time, velocity))
    assert (first > 1).all()
    assert torch.equal(initial, torch.full_like(initial, 2.0))
    second = sample_action_flow(model, initial, state, num_steps=nfe)
    assert not model.calls[nfe][3] and model.calls[nfe][2] is None
    assert torch.equal(first, second)
    if nfe == 1:
        off = sample_action_flow(model, initial, state, num_steps=1, use_self_conditioning=False)
        assert torch.equal(first, off)


def test_selection_is_independent_reproducible_and_approximately_half():
    torch.manual_seed(4)
    state = torch.get_rng_state()
    choices = [bootstrap_for_update(0, index) for index in range(10000)]
    assert torch.equal(state, torch.get_rng_state())
    assert 0.48 < sum(choices) / len(choices) < 0.52
    assert choices == [bootstrap_for_update(0, index) for index in range(10000)]


@pytest.mark.parametrize("recompute", [False, True])
def test_native_encoder_graph_survives_bootstrap_and_cache_remains_read_only(recompute):
    pytest.importorskip("transformers")
    from test_diffusion_gemma_adapter import _tiny_model
    from test_encoder_lora import inputs

    from duo_vla.backbones.diffusion_gemma import DiffusionGemmaActionDecoder, encode_diffusion_gemma_prefix_trainable
    from duo_vla.backbones.encoder_lora import (
        encoder_adapter_parameters,
        install_checkpointed_prefix,
        install_encoder_lora,
    )

    model = _tiny_model().requires_grad_(False)
    install_encoder_lora(model, rank=2, alpha=4)
    if recompute:
        install_checkpointed_prefix(model)
    projector = ActionInputProjector(ActionInterfaceConfig(hidden_size=32, state_dim=8, self_conditioning=SC_VERSION))
    denoiser = DuoVLADenoiser(
        projector, DiffusionGemmaActionDecoder.from_block_diffusion_model(model), VelocityHead(32, 7)
    )
    with torch.no_grad():
        projector.sc_projection.weight.normal_(std=0.01)
    prefix = encode_diffusion_gemma_prefix_trainable(model, inputs())
    before = [(layer.keys.detach().clone(), layer.values.detach().clone()) for layer in prefix.past_key_values.layers]
    context = dict(
        prefix_cache=prefix.past_key_values,
        prefix_attention_mask=prefix.attention_mask,
        action_valid_mask=torch.ones(2, 8, dtype=torch.bool),
    )
    output = training_velocity(
        denoiser, torch.randn(2, 8, 7), torch.rand(2), torch.randn(2, 8), bootstrap=True, **context
    )
    output.square().mean().backward()
    for name, parameter in encoder_adapter_parameters(model):
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name
        if name.endswith("adapter_b"):
            assert parameter.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    for layer, (key, value) in zip(prefix.past_key_values.layers, before, strict=True):
        assert torch.equal(layer.keys, key) and torch.equal(layer.values, value)
    assert projector.sc_projection.weight.grad.abs().sum() > 0


def test_generic_policy_uses_shared_sampler_and_explicit_update_rng():
    model = RecordingDenoiser()
    policy = RectifiedFlowPolicy(model, FlowConfig(inference_steps=4))
    context = dict(
        prefix_cache=object(),
        prefix_attention_mask=torch.ones(2, 3, dtype=torch.bool),
        action_valid_mask=torch.ones(2, 8, dtype=torch.bool),
    )
    state, noise = torch.zeros(2, 8), torch.randn(2, 8, 7)
    expected = sample_action_flow(model, noise, state, num_steps=4, **context).clamp(-1, 1)
    actual = policy.sample_normalized(state, initial_noise=noise, **context)
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="optimizer_update"):
        policy.training_loss(noise, state, **context)
    model.calls.clear()
    policy.training_loss(noise, state, optimizer_update=3, **context).backward()
    assert len(model.calls) == 1 + int(bootstrap_for_update(0, 3))


def test_strict_interface_roundtrip_and_legacy_rejection(tmp_path):
    from safetensors.torch import save_file

    from duo_vla.checkpointing import interface_state_dict, load_interface_state_dict

    config = ActionInterfaceConfig(hidden_size=16, state_dim=8, self_conditioning=SC_VERSION)
    module = ActionInputProjector(config)
    with torch.no_grad():
        module.sc_projection.weight.normal_()
    path = tmp_path / "interface.safetensors"
    save_file(interface_state_dict({"action_projector": module}), str(path))
    restored = copy.deepcopy(module)
    restored.sc_projection.weight.data.zero_()
    load_interface_state_dict(path, {"action_projector": restored})
    for name, value in module.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    with pytest.raises(ValueError, match="tensor keys differ"):
        load_interface_state_dict(
            path, {"action_projector": ActionInputProjector(replace(config, self_conditioning="none"))}
        )
