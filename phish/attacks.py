import torch
import torch.nn as nn


def clip(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class FastGradientSignMethod(nn.Module):
    def __init__(self, alpha=0, alpha_min=None, alpha_max=None, device="cpu"):
        super(FastGradientSignMethod, self).__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha = alpha
        self.device = device

    def forward(self, model, loss_fn, x, y):
        x.requires_grad = True

        output = model(x)

        loss = loss_fn(output, y)

        model.zero_grad()

        loss.backward()

        grad = x.grad.sign().detach()

        alpha = self.alpha
        if self.alpha_min is not None:
            alpha = (
                torch.FloatTensor(1)
                .to(self.device)
                .uniform_(self.alpha_min, self.alpha_max)[0]
            ).item()

        x = x + grad * alpha
        x = x.clamp(0, 1)
        x = x.detach()

        return x


class ProjectedGradientDecent(nn.Module):
    def __init__(
        self,
        loss_fn,
        epsilon,
        iterations,
        alpha=None,
        random_restarts=1,
        device="cpu",
    ):
        super(ProjectedGradientDecent, self).__init__()
        self.loss_fn = loss_fn
        self.alpha = epsilon / iterations if alpha is None else alpha
        self.iterations = iterations
        self.epsilon = epsilon
        self.random_restarts = random_restarts
        self.device = device

    def forward(self, x, y, model):
        max_loss = torch.tensor(0).float().to(self.device)
        max_delta = torch.zeros_like(x).to(self.device)

        for _ in range(self.random_restarts):
            delta = (
                torch.zeros_like(x)
                .uniform_(-self.epsilon, self.epsilon)
                .to(self.device)
            )

            delta = clip(delta, 0 - x, 1 - x)
            delta.requires_grad = True

            for _ in range(self.iterations):
                output = model(x + delta)

                loss = self.loss_fn(output, y, device=self.device)

                model.zero_grad()

                loss.backward()

                grad = delta.grad.detach()
                d = torch.clamp(
                    delta + self.alpha * torch.sign(grad),
                    -self.epsilon,
                    self.epsilon,
                )
                delta.data = clip(d, 0 - x, 1 - x)
                delta.grad.zero_()

            loss = self.loss_fn(model(x + delta), y, device=self.device)
            if loss > max_loss:
                max_delta = delta.detach()

        return (x + max_delta).detach()
