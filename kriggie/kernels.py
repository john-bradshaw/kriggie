
import abc

import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F

from . import transformed_params
from . import custom_distributions
from . import utils


def sq_distance(x_points, x_other):
    """
    Calculates the squared distance between x_points x_other

    we do this via the expansion (x1-x2)^t @ (x1-x2) = x1^t @ x1 + x2^t @ x2 - 2x1^t @ x2
    ^ nb note above x1 and x2 are col vectors.

    nb some discussion at https://github.com/pytorch/pytorch/issues/9406 here about doing this in an alternative way.

    :param x_points: [N, d]
    :param x_other:  [M, d]
    :return: [N, M]
    """
    x_points_sq = torch.sum(x_points**2, dim=1, keepdim=True)  # [N, 1]
    x_other_sq = torch.sum(x_other**2, dim=1, keepdim=False)[None, :]  # [1, M]
    interdot_products = x_points @ torch.transpose(x_other, 0, 1)
    return -2 * interdot_products + x_points_sq + x_other_sq


class BaseKernel(nn.Module, metaclass=abc.ABCMeta):
    def forward(self, x_points, x_other=None):
        return self.kernel(x_points, x_other)

    def kernel(self, x_points, x_other=None) -> torch.Tensor:
        if x_other is None or (x_points is x_other):
            try:
                return self._kernel_xx(x_points)
            except NotImplementedError:
                x_other = x_points
        return self._kernel_xp(x_points, x_other)

    def kernel_diag(self, x_points):
        try:
            return self._kernel_diag(x_points)
        except NotImplementedError as ex:
            return torch.diag(self.kernel(x_points))

    def _kernel_diag(self, x_points):
        raise NotImplementedError

    def _kernel_xx(self, x_points):
        raise NotImplementedError

    @abc.abstractmethod
    def _kernel_xp(self,  x_points, x_other):
        raise NotImplementedError


class RBF(BaseKernel):
    def __init__(self, ell_dimensionality):
        super().__init__()

        sigma_transfrom = custom_distributions.SoftplusTransform()
        self.sigma_sq = transformed_params.TransformedParam(torch.empty(1, dtype=utils.TORCH_FLOAT_TYPE),
                                                            transform=sigma_transfrom)

        ell_transfrom = custom_distributions.SoftplusTransform()
        self.ell = transformed_params.TransformedParam(torch.empty(ell_dimensionality, dtype=utils.TORCH_FLOAT_TYPE),
                                                         transform=ell_transfrom)
        self.reset_params()

    def reset_params(self):
        self.sigma_sq.constrained_init_set(torch.tensor(1.))
        self.ell.constrained_init_set(torch.ones_like(self.ell.unconstrained))

    def _kernel_xp(self,  x_points, x_other):
        sq_dists = sq_distance(x_points/self.ell.constrained, x_other/self.ell.constrained)
        return self.sigma_sq.constrained * torch.exp(-0.5 * sq_dists)

    def _kernel_diag(self, x_points):
        return self.sigma_sq.constrained * torch.ones(x_points.shape[0], dtype=x_points.dtype, device=x_points.device)

    def _kernel_xx(self, x_points):
        sq_dists = sq_distance(x_points/self.ell.constrained, x_points/self.ell.constrained)
        # nb there is a pdist function in PyTorch and if a squareform one becomes available may be faster to use that.
        return self.sigma_sq.constrained * torch.exp(-0.5 * sq_dists)





