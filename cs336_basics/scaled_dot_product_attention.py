from cs336_basics.softmax import Softmax
from math import sqrt
from jaxtyping import Float, Bool
import torch
from einops import einsum


class ScaledDotProductAttention(torch.nn.Module):
    def forward(
        self,
        q: Float[torch.Tensor, "... queries d_k"],
        k: Float[torch.Tensor, "... keys d_k"],
        v: Float[torch.Tensor, "... keys d_v"],
        mask: Bool[torch.Tensor, "queries keys"] | None = None,
    ) -> Float[torch.Tensor, "... queries d_v"]:
        dot_products = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys")

        if mask is not None:
            # Have to invert the mask
            dot_products.masked_fill_(~mask, -float("inf"))

        scaled_dot_products = dot_products / sqrt(q.shape[-1])
        attention_scores = Softmax()(scaled_dot_products, -1)
        return einsum(attention_scores, v, "... queries keys, ... keys d_v -> ... queries d_v")
