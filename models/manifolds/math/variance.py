from typing import Optional

import torch

from .diffgeom import dg_dist
from .diffgeom_autograd import ag_dist


def frechet_variance(
    x: torch.Tensor,
    mu: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1,
    w: Optional[torch.Tensor] = None,
    custom_autograd: bool = True,
) -> torch.Tensor:
    """
    Args
    ----
        x (tensor): points of shape [..., points, dim]
        mu (tensor): mean of shape [..., dim]
        w (tensor): weights of shape [..., points]

        where the ... of the three variables line up

    Returns
    -------
        tensor of shape [...]
    """
    if custom_autograd:
        distance: torch.Tensor = ag_dist(x=x, y=mu, c=c, dim=dim)
    else:
        distance: torch.Tensor = dg_dist(x=x, y=mu, c=c, dim=dim)
    distance = distance.pow(2)

    if w is None:
        return distance.mean(dim=dim)
    else:
        return (distance * w).sum(dim=dim)
