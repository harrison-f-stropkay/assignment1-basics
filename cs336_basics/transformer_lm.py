from jaxtyping import Int, Float
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from torch import Tensor
import torch


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = torch.nn.Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: Int[Tensor, "... s"]) -> Float[Tensor, "... s v"]:
        x = self.token_embeddings(token_ids)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
