from typing import Tuple

import torch
import torch.nn as nn
from scipy.special import beta

from ...manifolds import PoincareBall


class PoincareConvolution2d(nn.Module):
    """Poincare 2 dimensional convolution layer"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_dims: Tuple[int, int],
        ball: PoincareBall,
        bias: bool = True,
        stride: int = 1,
        padding: int = 0,
        id_init: bool = True,
    ) -> None:
        # Store layer parameters
        super(PoincareConvolution2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_dims = kernel_dims
        self.kernel_size = kernel_dims[0] * kernel_dims[1]
        self.ball = ball
        self.stride = stride
        self.padding = padding
        self.id_init = id_init

        # Unfolding layer
        self.unfold = nn.Unfold(
            kernel_size=kernel_dims,
            padding=padding,
            stride=stride,
        )

        # Create weights
        self.has_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.weights = nn.Parameter(
            torch.empty(self.kernel_size * in_channels, out_channels)
        )

        # Initialize weights
        self.reset_parameters()

        # Create beta's for concatenating receptive field features
        self.beta_ni = beta(self.in_channels / 2, 1 / 2)
        self.beta_n = beta(self.in_channels * self.kernel_size / 2, 1 / 2)

    def reset_parameters(self):
        # Identity initialization (1/2 factor to counter 2 inside the distance formula)
        if self.id_init:
            self.weights = nn.Parameter(
                1
                / 2
                * torch.eye(self.kernel_size * self.in_channels, self.out_channels)
            )
        else:
            nn.init.normal_(
                self.weights,
                mean=0,
                std=(2 * self.in_channels * self.kernel_size * self.out_channels)
                ** -0.5,
            )
        if self.has_bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the 2 dimensional convolution layer

        Parameters
        ----------
        x : tensor (height, width, batchsize, input channels)
            contains the layer inputs

        Returns
        -------
        tensor (height, width, batchsize, output channels)
        """
        batch_size, height, width = x.size(0), x.size(2), x.size(3)
        out_height = (
            height - self.kernel_dims[0] + 1 + 2 * self.padding
        ) // self.stride
        out_width = (width - self.kernel_dims[1] + 1 + 2 * self.padding) // self.stride

        # Scalar transform for concatenation
        x = x * self.beta_n / self.beta_ni

        # Apply sliding window to input to obtain features of each frame
        x = self.unfold(x)
        x = x.transpose(1, 2)

        # Project the receptive field features back onto the Poincare ball
        x = self.ball.expmap0(x, dim=-1)

        # Apply the Poincare fully connected operation
        x = self.ball.fully_connected(
            x=x,
            z=self.weights,
            bias=self.bias,
        )

        # Convert y back to the proper shape
        x = x.transpose(1, 2).reshape(
            batch_size, self.out_channels, out_height, out_width
        )

        # return y
        return self.ball.logmap0(x, dim=1)
