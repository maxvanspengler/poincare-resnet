import torch


class ag_MobiusAddition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        a = 1 + 2 * c * xy + c * y2
        b = 1 - c * x2
        denom = (1 + 2 * c * xy + c**2 * x2 * y2).clamp_min(1e-15)

        ctx.save_for_backward(x, y, c, x2, y2, xy, a, b, denom)
        ctx.dim = dim

        return (a * x + b * y) / denom

    @staticmethod
    def backward(ctx, grad_output):
        x, y, c, x2, y2, xy, a, b, denom = ctx.saved_tensors
        dim = ctx.dim

        denom_pow = (1 / denom).pow(2)

        utx = (grad_output * x).sum(dim=dim, keepdim=True)
        uty = (grad_output * y).sum(dim=dim, keepdim=True)
        theta = a * utx + b * uty
        k = 2 * c / denom
        theta_frac = theta / denom

        x_grad = (
            a / denom * grad_output
            - k * (theta_frac * c * y2 + uty) * x
            - k * (theta_frac - utx) * y
        )

        y_grad = (
            b / denom * grad_output
            + k * (utx - theta_frac) * x
            + k * (utx - c * x2 * theta_frac) * y
        )

        c_grad = 1 / (
            denom * ((2 * xy + y2) * utx - x2 * uty).clamp_min(1e-15)
        ) - denom_pow * 2 * (xy + c * x2 * y2) * (a * utx + b * uty)

        if x_grad.isinf().any() or y_grad.isinf().any() or c_grad.isinf().any():
            raise ValueError("Exploded gradient encountered")

        if x_grad.isnan().any() or y_grad.isnan().any() or c_grad.isnan().any():
            raise ValueError("Exploded gradient encountered")

        return (
            x_grad,
            y_grad,
            c_grad,
            None,
        )


class ag_Project(torch.autograd.Function):
    """
    Autograd implementation of Poincare project function.
    """

    @staticmethod
    def forward(ctx, x, c, dim=-1):
        if x.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = 1e-5
        maxnorm = (1 - eps) / ((c + 1e-15) ** 0.5)
        maxnorm = torch.where(c.gt(0), maxnorm, c.new_full((), 1e15))
        norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
        cond = norm > maxnorm
        projected = x / norm * maxnorm

        ctx.save_for_backward(x, c, maxnorm, norm, cond)
        ctx.dim = dim

        return torch.where(cond, projected, x)

    @staticmethod
    def backward(ctx, grad_output):
        x, c, maxnorm, norm, cond = ctx.saved_tensors
        dim = ctx.dim

        utx = (grad_output * x).sum(dim=dim, keepdim=True)

        x_grad = (
            torch.where(
                cond, maxnorm / norm, torch.as_tensor(1, dtype=x.dtype, device=x.device)
            )
            * grad_output
            - cond * utx * maxnorm / (norm.pow(3)).clamp_min(1e-15) * x
        )

        c_grad = -(cond * utx * maxnorm / (2 * (c + 1e-15) * norm))

        if x_grad.isinf().any() or c_grad.isinf().any():
            raise ValueError("Exploded gradient encountered")

        if x_grad.isnan().any() or c_grad.isnan().any():
            raise ValueError("Exploded gradient encountered")

        if c_grad.abs().gt(1e5).any():
            print(ctx.__class__.__name__, c_grad)

        return x_grad, c_grad, None


class ag_ExpMap0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, c, dim=-1):
        v_norm = v.norm(dim=dim, keepdim=True).clamp_min(1e-15)
        v_norm_c_sqrt = v_norm * c.sqrt()
        v_norm_c_sqrt_tanh = v_norm_c_sqrt.tanh()
        ctx.save_for_backward(v, c, v_norm, v_norm_c_sqrt, v_norm_c_sqrt_tanh)
        ctx.dim = dim
        return v_norm_c_sqrt_tanh * v / v_norm_c_sqrt

    @staticmethod
    def backward(ctx, grad_output):
        v, c, v_norm, v_norm_c_sqrt, v_norm_c_sqrt_tanh = ctx.saved_tensors
        dim = ctx.dim

        v_norm_c_sqrt_cosh2 = v_norm_c_sqrt.cosh().pow(2).clamp_min(1e-15)
        v_norm2 = v_norm.pow(2)

        utv = (grad_output * v).sum(dim=dim, keepdim=True)

        v_grad = (
            1 / (v_norm2 * v_norm_c_sqrt_cosh2).clamp_min(1e-15)
            - v_norm_c_sqrt_tanh / (v_norm_c_sqrt * v_norm2).clamp_min(1e-15)
        ) * utv * v + v_norm_c_sqrt_tanh / v_norm_c_sqrt * grad_output

        c_grad = (
            utv
            / (2 * c)
            * (1 / v_norm_c_sqrt_cosh2 - v_norm_c_sqrt_tanh / v_norm_c_sqrt)
        )

        if v_grad.isinf().any() or c_grad.isinf().any():
            raise ValueError("Exploded gradient encountered")

        if v_grad.isnan().any() or c_grad.isnan().any():
            raise ValueError("Exploded gradient encountered")

        if c_grad.abs().gt(1e5).any():
            print(ctx.__class__.__name__, c_grad)

        return v_grad, c_grad, None


class ag_LogMap0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, c, dim=-1):
        y_norm = y.norm(dim=dim, keepdim=True)
        y_norm_c_sqrt = y_norm.clamp_min(1e-15) * c.sqrt()
        y_norm_c_sqrt_atanh = y_norm_c_sqrt.atanh()

        ctx.save_for_backward(y, c, y_norm, y_norm_c_sqrt, y_norm_c_sqrt_atanh)
        ctx.dim = dim

        return torch.atanh(y_norm_c_sqrt) * y / y_norm_c_sqrt

    @staticmethod
    def backward(ctx, grad_output):
        y, c, y_norm, y_norm_c_sqrt, y_norm_c_sqrt_atanh = ctx.saved_tensors
        dim = ctx.dim

        y_norm2 = y_norm.pow(2)

        uty = (grad_output * y).sum(dim=dim, keepdim=True)

        y_grad = (
            1 / (y_norm2 * (1 - c * y_norm2)).clamp_min(1e-15)
            - y_norm_c_sqrt_atanh / (y_norm_c_sqrt * y_norm2).clamp_min(1e-15)
        ) * uty * y + y_norm_c_sqrt_atanh / y_norm_c_sqrt * grad_output

        c_grad = (
            uty
            / (2 * c)
            * (
                1 / (1 - c * y_norm2).clamp_min(1e-15)
                - y_norm_c_sqrt_atanh / y_norm_c_sqrt.clamp_min(1e-15)
            )
        )

        if y_grad.isinf().any() or c_grad.isinf().any():
            raise ValueError("Exploded gradient encountered")

        if y_grad.isnan().any() or c_grad.isnan().any():
            print(c, y_norm)
            raise ValueError("Exploded gradient encountered")

        if c_grad.abs().gt(1e5).any():
            print(ctx.__class__.__name__, c_grad)

        return y_grad, c_grad, None


class ag_ConfFactor(torch.autograd.Function):
    """
    Autograd implementation of the conformal factor lambda.
    """

    @staticmethod
    def forward(ctx, x, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        ctx.save_for_backward(x, c, x2)
        return 2 / (1 - c * x2).clamp_min(1e-15)

    @staticmethod
    def backward(ctx, grad_output):
        x, c, x2 = ctx.saved_tensors

        cond = c * x2 < 1

        x_grad = grad_output * cond * (4 * c / (1 - c * x2).pow(2).clamp_min(1e-15) * x)
        c_grad = grad_output * cond * (2 * x2 / (1 - c * x2).pow(2).clamp_min(1e-15))

        if x_grad.isinf().any() or c_grad.isinf().any():
            raise ValueError("Exploded gradient encountered")

        if x_grad.isnan().any() or c_grad.isnan().any():
            raise ValueError("Exploded gradient encountered")

        if c_grad.abs().gt(1e5).any():
            print(ctx.__class__.__name__, c_grad)

        return x_grad, c_grad, None


class ag_ExpSecondTerm(torch.autograd.Function):
    """
    Autograd implementation of the second term (rhs of Mobius addition) of the Exponential map.
    """

    @staticmethod
    def forward(ctx, x, v, c, dim=-1):
        lambda_denom = 1 - c * x.pow(2).sum(dim=dim, keepdim=True)
        v_norm = v.norm(dim=dim, keepdim=True).clamp_min(1e-15)
        c_sqrt_v_norm = v_norm * c.sqrt()
        prod_of_terms = c_sqrt_v_norm / lambda_denom
        tanh_term = (prod_of_terms).tanh()

        ctx.save_for_backward(
            x, v, c, lambda_denom, v_norm, c_sqrt_v_norm, prod_of_terms, tanh_term
        )
        ctx.dim = dim

        return tanh_term / c_sqrt_v_norm * v

    @staticmethod
    def backward(ctx, grad_output):
        (
            x,
            v,
            c,
            lambda_denom,
            v_norm,
            c_sqrt_v_norm,
            prod_of_terms,
            tanh_term,
        ) = ctx.saved_tensors
        dim = ctx.dim

        prod_of_terms_cosh2 = prod_of_terms.cosh().pow(2).clamp_min(1e-15)

        utv = (grad_output * v).sum(dim=dim, keepdim=True)

        x_grad = (
            2
            * utv
            * c
            / (prod_of_terms_cosh2 * lambda_denom.pow(2)).clamp_min(1e-15)
            * x
        )

        v_grad = (
            1 / (prod_of_terms_cosh2 * v_norm.pow(2) * lambda_denom).clamp_min(1e-15)
            - tanh_term / (c_sqrt_v_norm * v_norm.pow(2)).clamp_min(1e-15)
        ) * utv * v + tanh_term / c_sqrt_v_norm * grad_output

        c_grad = (
            utv
            / (2 * c)
            * (
                1
                / prod_of_terms_cosh2
                * (2 - lambda_denom)
                / lambda_denom.pow(2).clamp_min(1e-15)
                - tanh_term / c_sqrt_v_norm
            )
        )

        if x_grad.isinf().any() or v_grad.isinf().any() or c_grad.isinf().any():
            raise ValueError("Exploded gradient encountered")

        if x_grad.isnan().any() or v_grad.isnan().any() or c_grad.isnan().any():
            raise ValueError("Exploded gradient encountered")

        if c_grad.abs().gt(1e5).any():
            print(ctx.__class__.__name__, c_grad)

        return (
            x_grad,
            v_grad,
            c_grad,
            None,
        )


class ag_LogScaledTerm(torch.autograd.Function):
    """
    The scaled version of the Mobius addition that forms the output of the Logarithmic map.
    z = MobiusAddition.apply(-x, y)
    """

    @staticmethod
    def forward(ctx, x, z, c, dim=-1):
        z_norm = z.norm(dim=dim, keepdim=True).clamp_min(1e-15)
        lambda_x_denom = 1 - c * x.pow(2).sum(dim=dim, keepdim=True)
        c_sqrt_z_norm = z_norm * c.sqrt()
        frac_of_terms = lambda_x_denom / c_sqrt_z_norm
        atanh_term = c_sqrt_z_norm.atanh()

        ctx.save_for_backward(
            x, z, c, z_norm, lambda_x_denom, c_sqrt_z_norm, frac_of_terms, atanh_term
        )
        ctx.dim = dim

        return frac_of_terms * atanh_term * z

    @staticmethod
    def backward(ctx, grad_output):
        (
            x,
            z,
            c,
            z_norm,
            lambda_x_denom,
            c_sqrt_z_norm,
            frac_of_terms,
            atanh_term,
        ) = ctx.saved_tensors
        dim = ctx.dim

        z_norm2 = z_norm.pow(2)
        lambda_z_denom = 1 - c * z_norm2

        utz = (grad_output * z).sum(dim=dim, keepdim=True)

        x_grad = -2 * utz * c * atanh_term / c_sqrt_z_norm.clamp_min(1e-15) * x

        z_grad = (
            utz
            * (
                lambda_x_denom / (lambda_z_denom * z_norm2).clamp_min(1e-15)
                - atanh_term * frac_of_terms / z_norm2.clamp_min(1e-15)
            )
            * z
            + frac_of_terms * atanh_term * grad_output
        )

        c_grad = (
            utz
            / (2 * c)
            * (
                lambda_x_denom / lambda_z_denom.clamp_min(1e-15)
                - atanh_term * (2 - lambda_x_denom) / c_sqrt_z_norm.clamp_min(1e-15)
            )
        )

        if x_grad.isinf().any() or z_grad.isinf().any() or c_grad.isinf().any():
            raise ValueError("Exploded gradient encountered")

        if x_grad.isnan().any() or z_grad.isnan().any() or c_grad.isnan().any():
            raise ValueError("Exploded gradient encountered")

        if c_grad.abs().gt(1e5).any():
            print(ctx.__class__.__name__, c_grad)

        return (
            x_grad,
            z_grad,
            c_grad,
            None,
        )


def ag_expmap0(v, c, dim):
    x = ag_ExpMap0.apply(v, c, dim)
    return ag_Project.apply(x, c, dim)


def ag_expmap(x, v, c, dim):
    y = ag_ExpSecondTerm.apply(x, v, c, dim)
    return ag_Project.apply(ag_MobiusAddition.apply(x, y, c, dim), c, dim)


def ag_logmap(x, y, c, dim):
    z = ag_MobiusAddition.apply(-x, y, c, dim)
    return ag_LogScaledTerm.apply(x, z, c, dim)


def ag_gyration(u, v, w, c, dim):
    uv = ag_MobiusAddition.apply(u, v, c, dim)
    vw = ag_MobiusAddition.apply(v, w, c, dim)
    uvw = ag_MobiusAddition.apply(u, vw, c, dim)

    return ag_MobiusAddition.apply(-uv, uvw, c, dim)


def ag_transp(x, y, v, c, dim):
    lambda_x = ag_ConfFactor.apply(x, c, dim)
    lambda_y = ag_ConfFactor.apply(y, c, dim)
    return ag_gyration(y, -x, v, c, dim) * lambda_x / lambda_y


def ag_dist(
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
            c.sqrt()
            * ag_MobiusAddition.apply(-x, y, c, dim).norm(dim=dim, keepdim=keepdim)
        ).atanh()
    )
