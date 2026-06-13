import torch


class SiLU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid()
