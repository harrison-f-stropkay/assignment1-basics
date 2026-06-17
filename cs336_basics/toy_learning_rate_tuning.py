from collections.abc import Callable
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


for lr in [1, 1e1, 1e2, 1e3]:
    torch.manual_seed(0)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(f"lr: {lr}\tstep: {t}\tloss: {loss.cpu().item()}")
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step
    print()


"""
lr 1 is converging very slowly, lr 10 is converging less slowly, lr 100 converged quickly, lr 1000 diverged.

output:
lr: 1   step: 0 loss: 26.271406173706055
lr: 1   step: 1 loss: 25.231056213378906
lr: 1   step: 2 loss: 24.5224609375
lr: 1   step: 3 loss: 23.959409713745117
lr: 1   step: 4 loss: 23.482616424560547
lr: 1   step: 5 loss: 23.064424514770508
lr: 1   step: 6 loss: 22.689321517944336
lr: 1   step: 7 loss: 22.34758758544922
lr: 1   step: 8 loss: 22.032663345336914
lr: 1   step: 9 loss: 21.7398738861084

lr: 10.0        step: 0 loss: 26.271406173706055
lr: 10.0        step: 1 loss: 16.81369972229004
lr: 10.0        step: 2 loss: 12.394342422485352
lr: 10.0        step: 3 loss: 9.697249412536621
lr: 10.0        step: 4 loss: 7.854771614074707
lr: 10.0        step: 5 loss: 6.512505531311035
lr: 10.0        step: 6 loss: 5.492434501647949
lr: 10.0        step: 7 loss: 4.693441867828369
lr: 10.0        step: 8 loss: 4.053155899047852
lr: 10.0        step: 9 loss: 3.530749559402466

lr: 100.0       step: 0 loss: 26.271406173706055
lr: 100.0       step: 1 loss: 26.271404266357422
lr: 100.0       step: 2 loss: 4.507460117340088
lr: 100.0       step: 3 loss: 0.10787364840507507
lr: 100.0       step: 4 loss: 1.1429225992050108e-16
lr: 100.0       step: 5 loss: 1.2738581433316626e-18
lr: 100.0       step: 6 loss: 4.289528597092135e-20
lr: 100.0       step: 7 loss: 2.5553018721537346e-21
lr: 100.0       step: 8 loss: 2.192103091734976e-22
lr: 100.0       step: 9 loss: 2.4356700493370243e-23

lr: 1000.0      step: 0 loss: 26.271406173706055
lr: 1000.0      step: 1 loss: 9483.9765625
lr: 1000.0      step: 2 loss: 1638032.0
lr: 1000.0      step: 3 loss: 182213552.0
lr: 1000.0      step: 4 loss: 14759298048.0
lr: 1000.0      step: 5 loss: 931480731648.0
lr: 1000.0      step: 6 loss: 47819176017920.0
lr: 1000.0      step: 7 loss: 2057385568894976.0
lr: 1000.0      step: 8 loss: 7.583084306654822e+16
lr: 1000.0      step: 9 loss: 2.4350125357331907e+18
"""
