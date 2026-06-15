from jaxtyping import Float, Int
from torch import Tensor
import torch


class CrossEntropy(torch.nn.Module):
    def forward(
        self,
        logits: Float[Tensor, "... v"],
        targets: Int[Tensor, "..."],
    ) -> Float[Tensor, ""]:
        # Translate logits so no values are positive (softmax is invariant to adding some constant c to all inputs)
        logits -= logits.max(dim=-1, keepdim=True).values  # [... v]

        softmax_denominators = logits.exp().sum(dim=-1)  # [...]

        # For gather, note that we need to unsqueeze so targets have the same # of dims as probs
        gathered_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [...]

        # We never .exp() the numerator, so we don't have to .log() it
        log_probs = gathered_logits - softmax_denominators.log()  # [...]

        return -log_probs.mean()  # []
