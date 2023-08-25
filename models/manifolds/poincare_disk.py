from typing import Optional

import torch
import torch.nn as nn

from .math.diffgeom import *
from .math.diffgeom_autograd import *
from .math.linreg import *
from .math.variance import frechet_variance


class PoincareBallStdGrad(nn.Module):
    """
    Class representing the Poincare ball model of hyperbolic space.

    Implementation based on the geoopt implementation,
    but changed to use hyperbolic torch functions.
    """

    def __init__(self, c=1.0, learnable=True):
        super().__init__()
        c = torch.as_tensor(c, dtype=torch.float32)
        self.isp_c = nn.Parameter(c, requires_grad=learnable)
        self.learnable = learnable

    @property
    def c(self):
        return nn.functional.softplus(self.isp_c)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor, dim: int = -1):
        return dg_mobius_add(x=x, y=y, c=self.c, dim=dim)

    def project(self, x: torch.Tensor, dim: int = -1, eps: float = -1.0):
        return dg_project(x=x, c=self.c, dim=dim, eps=eps)

    def expmap0(self, v: torch.Tensor, dim: int = -1):
        return dg_expmap0(v=v, c=self.c, dim=dim)

    def logmap0(self, y: torch.Tensor, dim: int = -1):
        return dg_logmap0(y=y, c=self.c, dim=dim)

    def expmap(self, x: torch.Tensor, v: torch.Tensor, dim: int = -1):
        return dg_expmap(x=x, v=v, c=self.c, dim=dim)

    def logmap(self, x: torch.Tensor, y: torch.Tensor, dim: int = -1):
        return dg_logmap(x=x, y=y, c=self.c, dim=dim)

    def gyration(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        dim: int = -1,
    ):
        return dg_gyration(u=u, v=v, w=w, c=self.c, dim=dim)

    def transp(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
        dim: int = -1,
    ):
        return dg_transp(x=x, y=y, v=v, c=self.c, dim=dim)

    def dist(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        return dg_dist(x=x, y=y, c=self.c, dim=dim)

    def mlr(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        return poincare_mlr(x=x, z=z, r=r, c=self.c)

    def fully_connected(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        y = poincare_fully_connected(x=x, z=z, bias=bias, c=self.c)
        return self.project(y, dim=-1)

    def frechet_variance(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        dim: int = -1,
        w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return frechet_variance(
            x=x, mu=mu, c=self.c, dim=dim, w=w, custom_autograd=False
        )


class PoincareBallCustomAutograd(nn.Module):
    """
    Class representing the Poincare ball model of hyperbolic space.

    Implementation based on the geoopt implementation,
    but changed to use custom autograd functions.
    """

    def __init__(self, c=1.0, learnable=True):
        super().__init__()
        c = torch.as_tensor(c, dtype=torch.float32)
        self.isp_c = nn.Parameter(c, requires_grad=learnable)
        self.learnable = learnable

    @property
    def c(self) -> torch.Tensor:
        if self.learnable:
            return nn.functional.softplus(self.isp_c)
        else:
            return self.isp_c

    def mobius_add(
        self, x: torch.Tensor, y: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        return ag_MobiusAddition.apply(x, y, self.c, dim)

    def project(
        self, x: torch.Tensor, dim: int = -1, eps: float = -1.0
    ) -> torch.Tensor:
        return ag_Project.apply(x, self.c, dim)

    def expmap0(self, v: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return ag_expmap0(v, self.c, dim)

    def logmap0(self, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return ag_LogMap0.apply(y, self.c, dim)

    def expmap(self, x: torch.Tensor, v: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return ag_expmap(x, v, self.c, dim)

    def logmap(self, x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return ag_logmap(x, y, self.c, dim)

    def gyration(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        return ag_gyration(u, v, w, self.c, dim)

    def transp(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        return ag_transp(x, y, v, self.c, dim)

    def dist(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        return (
            2
            / self.c.sqrt()
            * (
                self.c.sqrt()
                * self.mobius_add(-x, y, dim=dim).norm(dim=dim, keepdim=True)
            ).atanh()
        )

    def mlr(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        return poincare_mlr(x=x, z=z, r=r, c=self.c)

    def fully_connected(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        y = poincare_fully_connected(x=x, z=z, bias=bias, c=self.c)
        return self.project(y, dim=-1)

    def frechet_variance(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        dim: int = -1,
        w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return frechet_variance(
            x=x, mu=mu, c=self.c, dim=dim, w=w, custom_autograd=True
        )


PoincareBall = PoincareBallStdGrad | PoincareBallCustomAutograd


def poincareball_factory(
    c: float = 1.0, custom_autograd: bool = True, learnable: bool = True
) -> PoincareBall:
    if custom_autograd:
        return PoincareBallCustomAutograd(c=c, learnable=learnable)
    else:
        return PoincareBallStdGrad(c=c, learnable=learnable)
