from cs336_basics.silu import SiLU
from cs336_basics.linear import Linear
from einops.einops import Tensor
import einops
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Int
import torch


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_ff is None:
            d_ff = 64 * int((8 / 3 * d_model) / 64)  # Round down to nearest multiple of 64 for hardware efficiency

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        self.to(device)

    def forward(self, x: Int[torch.Tensor, "b s d"]) -> Float[torch.Tensor, "b s d"]:
        w1_up_projection = self.w1(x)
        silu = SiLU()(w1_up_projection)
        w3_up_projection = self.w3(x)
        gated_up_projection = silu * w3_up_projection
        down_projection = self.w2(gated_up_projection)
        return down_projection
