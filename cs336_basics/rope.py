from cs336_basics.linear import Linear
from einops.einops import Tensor
import einops
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, Int
import torch


class RoPE(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0

        denom = theta ** (((2 * (torch.arange(d_k / 2) + 1)) - 2) / d_k)
        cos_cache = torch.stack([torch.cos(i / denom) for i in range(max_seq_len)])
        sin_cache = torch.stack([torch.sin(i / denom) for i in range(max_seq_len)])
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

        # Use `as_tensor` to pacify ty
        self.cos_cache: Float[torch.Tensor, "s k"] = torch.as_tensor(self.cos_cache).to(device)
        self.sin_cache: Float[torch.Tensor, "s k"] = torch.as_tensor(self.sin_cache).to(device)

    def forward(
        self,
        x: Float[torch.Tensor, "... s d"],
        token_positions: Int[torch.Tensor, "... s"],
    ) -> Float[torch.Tensor, "... s d"]:
        # For sin, we need a transformed version of x
        # e.g. x =  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        #      x' = [-0.2, 0.1, -0.4, 0.3, -0.6, 0.5]
        # so that instead of doing a full matmul, we can do: output = cos_cache[i] * x + sin_cache[i] * x'
        x_rearranged = rearrange(x, " ... s (k t) -> ... s k t", t=2)
        x_permuted_and_half_negated = x_rearranged.flip(-1) * torch.tensor([-1, 1])
        x_for_sin = rearrange(x_permuted_and_half_negated, " ... s k t -> ... s (k t)")  # [... s d]

        # Effectively compute the matmul of the rotation block matrix R and vector x (for the right block matrix for the seq index)
        # First, get the right sin/cos vectors for each q or k vector
        selected_cos = self.cos_cache[token_positions]  # [... s k]
        selected_sin = self.sin_cache[token_positions]  # [... s k]
        # Second, repeat them so that they're the right size
        cos_repeated = repeat(selected_cos, "... k -> ... (k 2)")  # [... s d]
        sin_repeated = repeat(selected_sin, "... k -> ... (k 2)")  # [... s d]
        # Third, compute the equivalent of the matmul
        return x * cos_repeated + x_for_sin * sin_repeated
