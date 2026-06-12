import torch


class Softmax(torch.nn.Module):
    def forward(self, x: torch.Tensor, i: int) -> torch.Tensor:
        # Translate so no values are positive (softmax is invariant to adding some constant c to all inputs)
        translated = x - x.max(dim=i, keepdim=True).values

        exponentiated = translated.exp()

        normalized = exponentiated / exponentiated.sum(dim=i, keepdim=True)
        return normalized
