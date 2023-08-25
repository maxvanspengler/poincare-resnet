import torch


def poincare_to_klein(x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return 2 / (1 + c * x.pow(2).sum(dim=dim, keepdim=True)) * x


def klein_to_poincare(x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    gamma = 1 / (1 - c * x.pow(2).sum(dim=dim, keepdim=True)).sqrt().clamp_min(1e-15)
    return gamma / (1 + gamma) * x


def klein_midpoint(
    x: torch.Tensor,
    c: torch.Tensor,
    vec_dim: int = -1,
    batch_dim: int = 0,
) -> torch.Tensor:
    gamma = 1 / (1 - c * x.pow(2).sum(dim=vec_dim, keepdim=True)).sqrt().clamp_min(
        1e-15
    )
    numerator = (gamma * x).sum(dim=batch_dim, keepdim=True)
    denominator = gamma.sum(dim=batch_dim, keepdim=True)
    return numerator / denominator


def poincare_klein_midpoint(
    x: torch.Tensor,
    c: torch.Tensor,
    vec_dim: int = -1,
    batch_dim: int = 0,
) -> torch.Tensor:
    x = poincare_to_klein(x, c, vec_dim)
    m = klein_midpoint(x, c, vec_dim, batch_dim)
    return klein_to_poincare(m, c, vec_dim)


def poincare_midpoint(
    x: torch.Tensor,
    c: torch.Tensor,
    vec_dim: int = -1,
    batch_dim: int = 0,
):
    gamma_sq = 1 / (1 - c * x.pow(2).sum(dim=vec_dim, keepdim=True)).clamp_min(1e-15)
    numerator = (gamma_sq * x).sum(dim=batch_dim, keepdim=True)
    denominator = gamma_sq.sum(dim=batch_dim, keepdim=True) - x.size(batch_dim) / 2
    m = numerator / denominator
    gamma_m = 1 / (1 - c * m.pow(2).sum(dim=vec_dim, keepdim=True)).sqrt().clamp_min(
        1e-15
    )
    return gamma_m / (1 + gamma_m) * m
