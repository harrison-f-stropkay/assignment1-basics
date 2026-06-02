from jaxtyping import Float
from math import sqrt
import torch
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        mean = 0
        std = sqrt(2 / (in_features + out_features))
        a = -3 * std
        b = 3 * std
        tensor = torch.empty((out_features, in_features), dtype=dtype)
        self.W: Float[torch.Tensor, "o i"] = torch.nn.Parameter(torch.nn.init.trunc_normal_(tensor, mean, std, a, b))

        self.to(device)

    def forward(self, x: Float[torch.Tensor, "... i"]) -> Float[torch.Tensor, "... o"]:
        return einsum(self.W, x, "o i, ... i -> ... o")
