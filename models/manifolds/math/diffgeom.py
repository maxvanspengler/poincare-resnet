import torch


def dg_mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, dim: int = -1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c**2 * x2 * y2
    return num / denom.clamp_min(1e-15)


def dg_project(x: torch.Tensor, c: torch.Tensor, dim: int = -1, eps: float = -1.0):
    if eps < 0:
        if x.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = 1e-5
    maxnorm = (1 - eps) / ((c + 1e-15) ** 0.5)
    maxnorm = torch.where(c.gt(0), maxnorm, c.new_full((), 1e15))
    norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def dg_expmap0(v: torch.Tensor, c: torch.Tensor, dim: int = -1):
    v_norm_c_sqrt = v.norm(dim=dim, keepdim=True).clamp_min(1e-15) * c.sqrt()
    return dg_project(torch.tanh(v_norm_c_sqrt) * v / v_norm_c_sqrt, c)


def dg_logmap0(y: torch.Tensor, c: torch.Tensor, dim: int = -1):
    y_norm_c_sqrt = y.norm(dim=dim, keepdim=True).clamp_min(1e-15) * c.sqrt()
    return torch.atanh(y_norm_c_sqrt) * y / y_norm_c_sqrt


def dg_expmap(x: torch.Tensor, v: torch.Tensor, c: torch.Tensor, dim: int = -1):
    v_norm = v.norm(dim=dim, keepdim=True).clamp_min(1e-15)
    lambda_x = 2 / (1 - c * x.pow(2).sum(dim=dim, keepdim=True)).clamp_min(1e-15)
    c_sqrt = c.sqrt()
    second_term = torch.tanh(c_sqrt * lambda_x * v_norm / 2) * v / (c_sqrt * v_norm)
    return dg_project(dg_mobius_add(x, second_term, c, dim=dim), c, dim=dim)


def dg_logmap(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, dim: int = -1):
    min_x_y = dg_mobius_add(-x, y, c, dim=dim)
    min_x_y_norm = min_x_y.norm(dim=dim, keepdim=True).clamp_min(1e-15)
    lambda_x = 2 / (1 - c * x.pow(2).sum(dim=dim, keepdim=True)).clamp_min(1e-15)
    c_sqrt = c.sqrt()
    return (
        2
        / (c_sqrt * lambda_x)
        * torch.atanh(c_sqrt * min_x_y_norm)
        * min_x_y
        / min_x_y_norm
    )


def dg_gyration(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1,
):
    u2 = u.pow(2).sum(dim=dim, keepdim=True)
    v2 = v.pow(2).sum(dim=dim, keepdim=True)
    uv = (u * v).sum(dim=dim, keepdim=True)
    uw = (u * w).sum(dim=dim, keepdim=True)
    vw = (v * w).sum(dim=dim, keepdim=True)
    K2 = c**2
    a = -K2 * uw * v2 + c * vw + 2 * K2 * uv * vw
    b = -K2 * vw * u2 - c * uw
    d = 1 + 2 * c * uv + K2 * u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(1e-15)


def dg_transp(
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1,
):
    lambda_x = 2 / (1 - c * x.pow(2).sum(dim=dim, keepdim=True)).clamp_min(1e-15)
    lambda_y = 2 / (1 - c * y.pow(2).sum(dim=dim, keepdim=True)).clamp_min(1e-15)
    return dg_gyration(y, -x, v, c, dim=dim) * lambda_x / lambda_y


def dg_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    return (
        2
        / c.sqrt()
        * (
            c.sqrt() * dg_mobius_add(-x, y, c, dim=dim).norm(dim=dim, keepdim=keepdim)
        ).atanh()
    )
