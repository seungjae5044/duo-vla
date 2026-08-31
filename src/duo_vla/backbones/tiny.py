"""Small decoder backend for interface tests and fixed-batch overfitting."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TinyActionDecoder(nn.Module):
    """A compact cross-plus-self attention backend implementing ``decode_actions``.

    ``prefix_cache`` is a tensor of shape ``[batch, prefix, hidden]``. This is deliberately not an emulation of
    DiffusionGemma internals; it exercises the benchmark-independent model and masking contract before loading a large
    checkpoint.
    """

    def __init__(self, hidden_size: int, *, num_heads: int = 4, mlp_ratio: int = 2) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        self.hidden_size = hidden_size
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.post_attention_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
        )

    def decode_actions(
        self,
        action_embeddings: Tensor,
        *,
        prefix_cache: Tensor,
        prefix_attention_mask: Tensor,
        action_valid_mask: Tensor,
    ) -> Tensor:
        batch_size, horizon, hidden_size = action_embeddings.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden_size}")
        if prefix_cache.ndim != 3 or prefix_cache.shape[0] != batch_size or prefix_cache.shape[2] != hidden_size:
            raise ValueError("prefix_cache must have shape [batch, prefix, hidden]")
        if prefix_attention_mask.shape != prefix_cache.shape[:2]:
            raise ValueError("prefix_attention_mask must match prefix_cache's first two dimensions")
        if action_valid_mask.shape != (batch_size, horizon):
            raise ValueError("action_valid_mask must have shape [batch, horizon]")
        valid = action_valid_mask.bool()
        if bool((valid.sum(dim=1) == 0).any()):
            raise ValueError("every batch item must contain at least one valid action")

        query = self.query_norm(action_embeddings)
        memory = torch.cat((prefix_cache, action_embeddings), dim=1)
        memory = self.memory_norm(memory)
        key_padding_mask = ~torch.cat((prefix_attention_mask.bool(), valid), dim=1)
        attended, _ = self.attention(
            query,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = action_embeddings + attended
        hidden = hidden + self.mlp(self.post_attention_norm(hidden))
        return hidden * valid.to(dtype=hidden.dtype)[..., None]
