from collections.abc import Callable
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Initialization (on first step)
                state = self.state[p]
                if "t" not in state:
                    assert "m" not in state and "v" not in state
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1  # "Note that t starts at 1"

                grad = p.grad.data
                adjusted_lr = lr * math.sqrt(1 - beta_2 ** state["t"]) / (1 - beta_1 ** state["t"])

                # weight decay in-place
                p.data *= 1 - lr * weight_decay

                # update the moment estimates in-place
                state["m"] = beta_1 * state["m"] + (1 - beta_1) * grad
                state["v"] = beta_2 * state["v"] + (1 - beta_2) * grad**2

                # update the parameters
                p.data -= adjusted_lr * state["m"] / (state["v"].sqrt() + eps)

        return loss
