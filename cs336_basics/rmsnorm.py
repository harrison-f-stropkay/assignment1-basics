from einops.einops import Tensor
import einops
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Int
import torch


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.register_buffer("eps", torch.tensor(eps), persistent=False)
        self.W: Float[torch.Tensor, " d"] = torch.nn.Parameter(torch.ones(d_model))
        self.to(device)

    def forward(self, x: Int[torch.Tensor, "b s d"]) -> Float[torch.Tensor, "b s d"]:
        # "upcast your input to torch.float32 to prevent overflow when you square the input" -- instructions
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute RMSNorm
        rms: Float[torch.Tensor, "b s"] = torch.rsqrt(
            reduce(torch.square(x), "b s d -> b s", "mean") + torch.as_tensor(self.eps)
        )  # do `torch.as_tensor` to pacify ty

        result = einsum(rms, self.W, x, "b s, d, b s d  -> b s d")

        # Return the result in the original dtype
        return result.to(in_dtype)
