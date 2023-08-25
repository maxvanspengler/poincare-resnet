import torch
import torch.nn as nn

from ...manifolds import PoincareBall
from ...manifolds.math.frechet_mean import frechet_mean
from ...manifolds.math.midpoint import poincare_midpoint
from ...manifolds.math.variance import frechet_variance


class PoincareBatchNorm(nn.Module):
    """
    Basic implementation of batch normalization in the Poincare ball model.

    Based on:
        https://arxiv.org/abs/2003.00335
    """

    def __init__(
        self,
        features: int,
        ball: PoincareBall,
        use_midpoint: bool = True,
    ) -> None:
        super(PoincareBatchNorm, self).__init__()
        self.features = features
        self.ball = ball
        self.use_midpoint = use_midpoint

        self.mean = nn.Parameter(torch.zeros(features))
        self.var = nn.Parameter(torch.tensor(1.0))

        # statistics
        self.register_buffer("running_mean", torch.zeros(1, features))
        self.register_buffer("running_var", torch.tensor(1.0))
        self.updates = 0

    def forward(self, x, momentum=0.9):
        x = self.ball.expmap0(x, dim=-1)
        mean_on_ball = self.ball.expmap0(self.mean, dim=-1)
        if self.use_midpoint:
            input_mean = poincare_midpoint(x, self.ball.c, vec_dim=-1, batch_dim=0)
        else:
            input_mean = frechet_mean(x, self.ball)
        input_var = frechet_variance(x, input_mean, self.ball.c, dim=-1)

        input_logm = self.ball.transp(
            x=input_mean,
            y=mean_on_ball,
            v=self.ball.logmap(input_mean, x),
        )

        input_logm = (self.var / (input_var + 1e-6)).sqrt() * input_logm

        output = self.ball.expmap(mean_on_ball.unsqueeze(-2), input_logm)

        self.updates += 1

        if self.ball.logmap0(output, dim=-1).isnan().any():
            print("bug")
        if self.ball.logmap0(output, dim=-1).isnan().any():
            print("bn bug")
        return self.ball.logmap0(output, dim=-1)


class PoincareBatchNorm2d(nn.Module):
    """
    2D implementation of batch normalization in the Poincare ball model.

    Based on:
        https://arxiv.org/abs/2003.00335
    """

    def __init__(
        self,
        features: int,
        ball: PoincareBall,
        use_midpoint: bool = True,
    ) -> None:
        super(PoincareBatchNorm2d, self).__init__()
        self.features = features
        self.ball = ball
        self.use_midpoint = use_midpoint

        self.norm = PoincareBatchNorm(
            features=features,
            ball=ball,
            use_midpoint=use_midpoint,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input dimensions
        batch_size, height, width = x.size(0), x.size(2), x.size(3)

        # Swap batch and channel dimensions and flatten everything but channel dimension
        x = x.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)

        # Apply batchnorm
        x = self.norm(x)

        # Reshape to original dimensions
        x = x.reshape(batch_size, height, width, self.features).permute(0, 3, 1, 2)

        return x
