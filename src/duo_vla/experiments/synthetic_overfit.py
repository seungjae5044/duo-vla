"""Fixed-batch overfit gate for the benchmark-independent Duo-VLA core."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from duo_vla.action_interface import ActionInputProjector, VelocityHead
from duo_vla.backbones import TinyActionDecoder
from duo_vla.config import ActionInterfaceConfig
from duo_vla.flow import make_flow_training_pair, masked_velocity_mse
from duo_vla.modeling import DuoVLADenoiser


@dataclass(frozen=True, slots=True)
class OverfitResult:
    seed: int
    device: str
    steps: int
    initial_loss: float
    final_loss: float
    best_loss: float
    reduction: float
    elapsed_seconds: float


def run_fixed_batch_overfit(
    *,
    steps: int = 1000,
    seed: int = 0,
    device: str = "cpu",
    batch_size: int = 16,
    hidden_size: int = 64,
) -> OverfitResult:
    """Overfit one immutable sampled flow batch and return deterministic diagnostics."""

    if steps <= 0 or batch_size <= 0 or hidden_size <= 0:
        raise ValueError("steps, batch_size, and hidden_size must be positive")
    if hidden_size % 4:
        raise ValueError("hidden_size must be divisible by four")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    target_device = torch.device(device)
    config = ActionInterfaceConfig(hidden_size=hidden_size, state_dim=8)
    model = DuoVLADenoiser(
        ActionInputProjector(config),
        TinyActionDecoder(hidden_size, num_heads=4, mlp_ratio=2),
        VelocityHead(hidden_size, config.action_dim),
    ).to(target_device)

    generator = torch.Generator(device=target_device).manual_seed(seed + 1)
    clean_actions = torch.empty(
        batch_size,
        config.action_horizon,
        config.action_dim,
        device=target_device,
    ).uniform_(-0.9, 0.9, generator=generator)
    pair = make_flow_training_pair(clean_actions, generator=generator)
    state = torch.randn(batch_size, config.state_dim, device=target_device, generator=generator)
    prefix_cache = torch.randn(batch_size, 6, hidden_size, device=target_device, generator=generator)
    prefix_mask = torch.ones(batch_size, 6, dtype=torch.bool, device=target_device)
    action_mask = torch.ones(
        batch_size,
        config.action_horizon,
        dtype=torch.bool,
        device=target_device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    initial_loss: float | None = None
    best_loss = float("inf")
    started = time.perf_counter()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            pair.noisy_actions,
            pair.timesteps,
            state,
            prefix_cache=prefix_cache,
            prefix_attention_mask=prefix_mask,
            action_valid_mask=action_mask,
        )
        loss = masked_velocity_mse(prediction, pair.target_velocity, action_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("synthetic overfit produced a non-finite loss")
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        best_loss = min(best_loss, float(loss.detach()))

    elapsed = time.perf_counter() - started
    assert initial_loss is not None
    with torch.no_grad():
        final_prediction = model(
            pair.noisy_actions,
            pair.timesteps,
            state,
            prefix_cache=prefix_cache,
            prefix_attention_mask=prefix_mask,
            action_valid_mask=action_mask,
        )
        final_loss = float(masked_velocity_mse(final_prediction, pair.target_velocity, action_mask))
    return OverfitResult(
        seed=seed,
        device=str(target_device),
        steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=min(best_loss, final_loss),
        reduction=initial_loss / max(final_loss, torch.finfo(torch.float32).tiny),
        elapsed_seconds=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = run_fixed_batch_overfit(steps=args.steps, seed=args.seed, device=args.device)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
