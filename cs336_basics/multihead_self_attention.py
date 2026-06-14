from cs336_basics.scaled_dot_product_attention import ScaledDotProductAttention
from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE
from jaxtyping import Float, Int
from torch import Tensor
import torch
from einops import einsum, rearrange


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0
        d_h = self.d_v = int(d_model / num_heads)

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

        # If we're using RoPE
        if (max_seq_len is not None) and (theta is not None):
            self.rope = RoPE(theta, d_h, max_seq_len)
        elif (max_seq_len is None) and (theta is None):
            self.rope = None
        else:
            raise Exception

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        seq_len = x.shape[-2]

        # q, k, v projections
        W_q = rearrange(self.W_q.W, "(h d_h) d_model -> h d_h d_model", h=self.num_heads)
        W_k = rearrange(self.W_k.W, "(h d_h) d_model -> h d_h d_model", h=self.num_heads)
        W_v = rearrange(self.W_v.W, "(h d_h) d_model -> h d_h d_model", h=self.num_heads)
        q = einsum(W_q, x, "h d_h d_model, ... d_model -> ... h d_h")
        k = einsum(W_k, x, "h d_h d_model, ... d_model -> ... h d_h")
        v = einsum(W_v, x, "h d_h d_model, ... d_model -> ... h d_h")
        q = rearrange(q, "... s h d_h -> ... h s d_h")
        k = rearrange(k, "... s h d_h -> ... h s d_h")
        v = rearrange(v, "... s h d_h -> ... h s d_h")

        # RoPE rotations for q and k
        if self.rope:
            if token_positions is None:
                token_positions = torch.arange(0, seq_len)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Causal self-attention
        mask = torch.tril(input=torch.fill(torch.empty((seq_len, seq_len)), True)).to(torch.bool)
        attended: Float[Tensor, "... s d_h"] = ScaledDotProductAttention()(q, k, v, mask)
        concatenated = rearrange(attended, "... h s d_h -> ... s (h d_h)")
        return self.W_o(concatenated)
