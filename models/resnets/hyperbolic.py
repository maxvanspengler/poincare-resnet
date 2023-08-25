from typing import Optional

import torch
import torch.nn as nn

from ..manifolds import PoincareBall, poincareball_factory
from ..nn import PoincareBatchNorm2d, PoincareConvolution2d, PoincareLinear


def _conv3x3(
    in_channels: int,
    out_channels: int,
    ball: PoincareBall,
    stride: int = 1,
) -> PoincareConvolution2d:
    return PoincareConvolution2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_dims=(3, 3),
        ball=ball,
        bias=True,
        stride=stride,
        padding=1,
    )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        act_layer: Optional[nn.Module] = None,
        downsample: Optional[nn.Sequential] = None,
        custom_autograd: bool = True,
        learnable: bool = False,
        init_c: float = 1,
        skip_connection: str = "fr",
        bn_midpoint: bool = True,
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.act_layer = act_layer
        self.learnable = learnable
        self.init_c = init_c
        self.skip_connection = skip_connection
        self.bn_midpoint = bn_midpoint

        self.balls = {
            "ball1": poincareball_factory(
                c=init_c, custom_autograd=custom_autograd, learnable=learnable
            ),
            "ball2": poincareball_factory(
                c=init_c, custom_autograd=custom_autograd, learnable=learnable
            ),
            "bn_ball": poincareball_factory(
                c=init_c, custom_autograd=custom_autograd, learnable=learnable
            ),
            "skip_ball": poincareball_factory(
                c=init_c, custom_autograd=custom_autograd, learnable=learnable
            ),
        }

        self.conv1 = _conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            ball=self.balls["ball1"],
            stride=stride,
        )
        self.bn1 = PoincareBatchNorm2d(
            out_channels, ball=self.balls["bn_ball"], use_midpoint=self.bn_midpoint
        )
        if act_layer is not None:
            self.act1 = act_layer()
        self.conv2 = _conv3x3(
            in_channels=out_channels,
            out_channels=out_channels,
            ball=self.balls["ball2"],
            stride=1,
        )
        self.bn2 = PoincareBatchNorm2d(
            out_channels, ball=self.balls["bn_ball"], use_midpoint=self.bn_midpoint
        )
        if self.act_layer is not None:
            self.act2 = act_layer()
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        if self.act_layer is not None:
            x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        # Skip connection with Mobius addition in Poincare ball (fr: f(x) + res, rf: res + f(x)).
        x = self.balls["skip_ball"].expmap0(x, dim=-1)
        residual = self.balls["skip_ball"].expmap0(residual, dim=-1)
        if self.skip_connection == "fr":
            x = self.balls["skip_ball"].mobius_add(x, residual)
        elif self.skip_connection == "rf":
            x = self.balls["skip_ball"].mobius_add(residual, x)
        x = self.balls["skip_ball"].logmap0(x, dim=-1)

        if self.act_layer is not None:
            x = self.act2(x)

        return x


class HyperbolicResNet(nn.Module):
    """Hyperbolic Residual Networks

    Implementation of Residual Networks as described in: https://arxiv.org/pdf/1512.03385.pdf
    but with hyperbolic operations defined on the Poincare disk instead of Euclidean
    """

    def __init__(
        self,
        classes: int,
        channel_dims: list[int],
        depths: list[int],
        act_layer: Optional[nn.Module] = nn.ReLU,
        init_c: float = 0.1,
        custom_autograd: bool = True,
        learnable: bool = False,
        skip_connection: str = "fr",
        bn_midpoint: bool = True,
    ) -> None:
        super(HyperbolicResNet, self).__init__()
        self.classes = classes
        self.channel_dims = channel_dims
        self.depths = depths
        self.act_layer = act_layer
        self.init_c = init_c
        self.custom_autograd = custom_autograd
        self.learnable = learnable
        self.skip_connection = skip_connection
        self.bn_midpoint = bn_midpoint

        self.conv_ball = poincareball_factory(
            c=init_c, custom_autograd=custom_autograd, learnable=learnable
        )
        self.bn_ball = poincareball_factory(
            c=init_c, custom_autograd=custom_autograd, learnable=learnable
        )
        self.linear_ball = poincareball_factory(
            c=init_c, custom_autograd=custom_autograd, learnable=learnable
        )

        self.conv = _conv3x3(
            in_channels=3,
            out_channels=channel_dims[0],
            ball=self.conv_ball,
        )
        self.bn = PoincareBatchNorm2d(
            channel_dims[0], ball=self.bn_ball, use_midpoint=self.bn_midpoint
        )
        if act_layer is not None:
            self.act = act_layer()

        self.group1 = self._make_group(
            in_channels=channel_dims[0],
            out_channels=channel_dims[0],
            depth=depths[0],
        )

        self.group2 = self._make_group(
            in_channels=channel_dims[0],
            out_channels=channel_dims[1],
            depth=depths[1],
            stride=2,
        )

        self.group3 = self._make_group(
            in_channels=channel_dims[1],
            out_channels=channel_dims[2],
            depth=depths[2],
            stride=2,
        )

        self.avg_pool = nn.AvgPool2d(8)

        self.fc = PoincareLinear(
            in_features=channel_dims[2],
            out_features=classes,
            ball=self.linear_ball,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.act_layer is not None:
            x = self.act(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.avg_pool(x)
        x = self.fc(x.squeeze())
        return x

    def _make_group(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample_ball = poincareball_factory(
                c=self.init_c,
                custom_autograd=self.custom_autograd,
                learnable=self.learnable,
            )
            downsample = PoincareConvolution2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_dims=(1, 1),
                ball=downsample_ball,
                bias=True,
                stride=stride,
                padding=0,
            )

        layers = [
            ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                act_layer=self.act_layer,
                downsample=downsample,
                custom_autograd=self.custom_autograd,
                learnable=self.learnable,
                init_c=self.init_c,
                skip_connection=self.skip_connection,
                bn_midpoint=self.bn_midpoint,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                ResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    act_layer=self.act_layer,
                    custom_autograd=self.custom_autograd,
                    learnable=self.learnable,
                    init_c=self.init_c,
                    skip_connection=self.skip_connection,
                    bn_midpoint=self.bn_midpoint,
                )
            )

        return nn.Sequential(*layers)
