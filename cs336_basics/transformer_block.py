from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.swiglu import SwiGLU
from torch import Tensor
import torch


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.rms_1 = RMSNorm(d_model)
        self.mhsa = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.rms_2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x += self.mhsa(self.rms_1(x))
        x += self.ffn(self.rms_2(x))
        return x
