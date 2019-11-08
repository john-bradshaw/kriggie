
# -*- coding: utf-8 -*-

import torch
from torch import distributions
from torch.distributions import constraints

from torch.nn import functional as F


class Delta(distributions.Distribution):
    """
    Delta distribution representing a single point.

    """
    has_rsample = True
    arg_constraints = {'v': constraints.real, 'log_density': constraints.real}
    support = constraints.real

    def __init__(self, v: torch.Tensor, validate_args=None):
        self.v = v
        batch_shape, event_shape = self.v.shape[:-1], self.v.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=None)

    def expand(self, batch_shape, _instance=None):
        v = self.v.expand(batch_shape + self.event_shape)
        return Delta(v)

    def rsample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.v.shape
        return self.v.expand(shape)

    def log_prob(self, x):
        log_prob = x.new_tensor(x == self.v).log()
        log_prob = torch.sum(log_prob, dim=-1)
        return log_prob

    @property
    def mean(self):
        return self.v

    @property
    def variance(self):
        return torch.zeros_like(self.v)


class SoftplusTransform(distributions.Transform):
    """
    Softplus(x)= (1/β) * log(1+exp(β * x))

    The use of this transform for keeping parameters positive comes from GPFlow as well as some of the numerical
    stability techniques below (eg use of lower).
    See the original at:
    https://github.com/GPflow/GPflow/blob/develop/gpflow/transforms.py

    We also bring aboard ideas from the PyTorch Softplus function in particular beta (β above) and threshold.
    Threshold means that we use a linear function for stability above this value.
    See https://pytorch.org/docs/stable/nn.html?highlight=softplus#torch.nn.Softplus for further details.

    """
    domain = constraints.real
    codomain = constraints.positive  # nb actually more restrictive than this if set lower
    bijective = True
    sign = +1

    def __init__(self, lower=0.0, beta=1, threshold=20, cache_size=0):
        super().__init__(cache_size)
        self._lower = lower
        self._beta = beta
        self._threshold = threshold

    def __eq__(self, other):
        return isinstance(other, SoftplusTransform)

    def _call(self, x):
        return F.softplus(x, self._beta, self._threshold) + self._lower

    def _inverse(self, y):
        # Note we follow PyTorch's route to avoid overflow by simply treating the function as linear above threshold
        # rather than GPflows approach which uses the logsumexp trick
        y = y - self._lower
        out = torch.zeros_like(y)

        beta_y = self._beta * y

        # Over threshold
        o_threshold = beta_y > self._threshold
        out[o_threshold] = y[o_threshold]

        # Under threshold
        out[~o_threshold] = torch.log(torch.expm1(beta_y[~o_threshold])) / self._beta

        return out

    def log_abs_det_jacobian(self, x, y):
        return -F.softplus(-self._beta*x)


def kl_mvn_and_std_norm(mvn: distributions.MultivariateNormal) -> torch.Tensor:
    """
    Takes KL between multivairate normal and standard normal.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    Allows us to skip doing some calcs which would be needed in the full multivariate case.

    :return: batch sized tensor of the KLs
    """
    assert len(mvn.event_shape) == 1, "only implemented for one dimensional means"

    # Term 1 Trace term: tr(Σ)
    term1 = torch.einsum("bii->b", mvn.covariance_matrix)

    # term 2, Mahalanobis term: μᵀμ
    term2 = torch.einsum("bi,bi->b",  mvn.mean,  mvn.mean)

    # term 3, -k
    term3 = -mvn.event_shape[0]  # this is an int but will broadcast at the end.

    # Term 4, -ln(det(Σ))
    # trick: The determinant of Cholesky to the power of two is the determinant of the covariance matrix.
    # This is useful as the determinant of the Cholesky equals the product of diagonal due to it being a
    # lower traingular matrix.
    term4 = -2* torch.sum(torch.log(torch.diagonal(mvn.scale_tril, dim1=1, dim2=2)), dim=1)

    # Assemble!
    kl = 0.5 * (term1 + term2 + term3 + term4)  # [b]
    return kl
