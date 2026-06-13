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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        d_ff: int = 64 * int((8 / 3 * d_model) / 64)  # Round down to nearest multiple of 64 f{or hardware efficiency
        self.w1_weight = Linear(d_model, d_ff)
        self.w2_weight = Linear(d_ff, d_model)
        self.w3_weight = Linear(d_model, d_ff)
        self.to(device)

    def forward(self, x: Int[torch.Tensor, "b s d"]) -> Float[torch.Tensor, "b s d"]:
        w1_up_projection = self.w1_weight(x)
        silu = w1_up_projection * torch.sigmoid(w1_up_projection)
        w3_up_projection = self.w3_weight(x)
        gated_up_projection = silu * w3_up_projection
        down_projection = self.w2_weight(gated_up_projection)
        return down_projection
