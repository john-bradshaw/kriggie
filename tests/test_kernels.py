
import numpy as np
import torch
from torch import nn
from scipy.spatial import distance

from kriggie import kernels
from kriggie import utils


def test_rbf_diag():
    torch.manual_seed(48574)

    x_points = torch.randn(10, 5, dtype=utils.TORCH_FLOAT_TYPE)

    rbf_kernel = kernels.RBF(5)
    rbf_kernel.sigma_sq.constrained = torch.tensor(5.43, dtype=utils.TORCH_FLOAT_TYPE)

    computed_vals = rbf_kernel.kernel_diag(x_points)
    expected_vals = 5.43 * np.ones(10)

    np.testing.assert_array_almost_equal(computed_vals.detach().numpy(), expected_vals)


def test_rbf():
    torch.manual_seed(48574)
    rng = np.random.RandomState(51)

    x_points = torch.randn(10, 5, dtype=utils.TORCH_FLOAT_TYPE)
    x_other = torch.randn(18, 5, dtype=utils.TORCH_FLOAT_TYPE)

    sigma = 5.4256
    ell = (10 * rng.randn(5).astype(utils.NP_FLOAT_TYPE)) ** 2   # squared to keep +ve

    rbf_kernel = kernels.RBF(5)
    rbf_kernel.sigma_sq.constrained = torch.tensor(sigma, dtype=utils.TORCH_FLOAT_TYPE)
    rbf_kernel.ell.constrained = torch.tensor(ell, dtype=utils.TORCH_FLOAT_TYPE)
    computed_vals = rbf_kernel(x_points, x_other).detach().numpy()

    expected_via_np = _rbf_reference_impl_np(x_points.numpy(), x_other.numpy(), ell,
                                             sigma)

    np.testing.assert_array_almost_equal(computed_vals, expected_via_np)



def test_rbf_x_and_itself():
    torch.manual_seed(48574)
    rng = np.random.RandomState(51)

    x_points = torch.randn(10, 5, dtype=utils.TORCH_FLOAT_TYPE)

    sigma = 5.4256
    ell = (10 * rng.randn(5).astype(utils.NP_FLOAT_TYPE)) ** 2  # squared to keep +ve

    rbf_kernel = kernels.RBF(5)
    rbf_kernel.sigma_sq.constrained = torch.tensor(sigma, dtype=utils.TORCH_FLOAT_TYPE)
    rbf_kernel.ell.constrained = torch.tensor(ell, dtype=utils.TORCH_FLOAT_TYPE)
    computed_vals = rbf_kernel(x_points).detach().numpy()

    expected_via_np = _rbf_reference_impl_np(x_points.numpy(), x_points.numpy(), ell,
                                             sigma)

    np.testing.assert_array_almost_equal(computed_vals, expected_via_np)


def test_rbf_not_ard():
    torch.manual_seed(48574)

    x_points = torch.randn(10, 5, dtype=utils.TORCH_FLOAT_TYPE)
    x_other = torch.randn(18, 5, dtype=utils.TORCH_FLOAT_TYPE)

    sigma = 5.4256
    ell = 6.45

    rbf_kernel = kernels.RBF(5)
    rbf_kernel.sigma_sq.constrained = torch.tensor(sigma, dtype=utils.TORCH_FLOAT_TYPE)
    rbf_kernel.ell.constrained = torch.tensor(ell, dtype=utils.TORCH_FLOAT_TYPE)
    computed_vals = rbf_kernel(x_points, x_other).detach().numpy()

    expected_via_np = _rbf_reference_impl_np(x_points.numpy(), x_other.numpy(), ell * np.ones(5),
                                             sigma)

    np.testing.assert_array_almost_equal(computed_vals, expected_via_np)


def _rbf_reference_impl_np(x1, x2, lengthscales, sigma):
    assert lengthscales.shape[0] == x1.shape[1]

    dist_mat = distance.cdist(x1/lengthscales[None, :], x2/lengthscales[None, :], 'sqeuclidean')
    return sigma * np.exp(-0.5*dist_mat)
