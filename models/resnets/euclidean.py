from typing import Optional

import torch
import torch.nn as nn


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True,
    )


class ResidualBlock(nn.Module):
    """The basic building block of a wide ResNet

    Note that the batch normalization and ReLU appear before the convolution instead of after
    the convolution. This makes it different from the original ResNet blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Sequential] = None,
        inplace: bool = True,
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = _conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = _conv3x3(
            in_channels=out_channels,
            out_channels=out_channels,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual
        x = self.relu(x)

        return x


class EuclideanResNet(nn.Module):
    """Residual Networks

    Implementation of Residual Networks as described in: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(
        self,
        classes: int,
        channel_dims: list[int],
        depths: list[int],
    ) -> None:
        super(EuclideanResNet, self).__init__()
        self.classes = classes
        self.channel_dims = channel_dims
        self.depths = depths

        self.relu = nn.ReLU(inplace=True)
        self.conv = _conv3x3(
            in_channels=3,
            out_channels=channel_dims[0],
        )
        self.bn = nn.BatchNorm2d(channel_dims[0])

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
        self.fc = nn.Linear(channel_dims[2], classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
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
            downsample = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=True,
            )

        layers = [
            ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                ResidualBlock(in_channels=out_channels, out_channels=out_channels)
            )

        return nn.Sequential(*layers)
