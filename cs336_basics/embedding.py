from einops import rearrange
from jaxtyping import Float, Int
import torch


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        mean = 0
        std = 1
        a = -3
        b = 3
        tensor = torch.empty((num_embeddings, embedding_dim), dtype=dtype)
        self.weight: Float[torch.Tensor, "v d"] = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(tensor, mean, std, a, b)
        )
        self.to(device)

    def forward(self, token_ids: Int[torch.Tensor, "... s"]) -> Float[torch.Tensor, "... s d"]:
        return self.weight[token_ids]
