
import torch
from torch import distributions
import numpy as np
from scipy import stats

from kriggie import custom_distributions
from kriggie import utils

utils.TORCH_FLOAT_TYPE = torch.float64
utils.NP_FLOAT_TYPE = np.float64


@torch.no_grad()
def test_softplus_forward_backward():
    torch.manual_seed(512)

    transform = custom_distributions.SoftplusTransform()

    x = 5 * torch.randn(50,100,86,32)

    y = transform(x)
    x_tilde = transform.inv(y)

    np.testing.assert_array_almost_equal(x.numpy(), x_tilde.numpy())


@torch.no_grad()
def test_softplus_positive():
    torch.manual_seed(512)

    transform = custom_distributions.SoftplusTransform()

    x = 5 * torch.randn(50,100,86,32)

    y = transform(x)

    assert (y.numpy() >= 0.).all()


def test_delta_log_prob():
    v = torch.tensor([[3.67], [8.91], [-76.213]])
    delta_dist = custom_distributions.Delta(v)

    v2 = v.clone()
    lp = delta_dist.log_prob(v2)
    assert np.all(lp.detach().numpy() == 0.)

    v3 = v.clone()
    v3[1] = 8.45
    lp = delta_dist.log_prob(v3)
    assert np.all(lp.detach().numpy() == np.array([0., -np.inf, 0.]))


def test_kl_between_mvn_and_std():
    """
    Check that our custom implementation of KL divergence for MVN against std MVN matches KL divergence from
    PyTorch's distribution module.
    """
    import time
    torch.manual_seed(512)
    rng = np.random.RandomState(51)
    batch_size = 12
    num_points_per_batch = 57

    # Create a distribution to check
    loc = torch.randn(batch_size, num_points_per_batch, dtype=utils.TORCH_FLOAT_TYPE)
    wishart_ = stats.wishart(seed=rng, df=num_points_per_batch, scale=np.eye(num_points_per_batch))
    cov_samples = np.stack([wishart_.rvs() for _ in range(batch_size)])
    cov_samples = torch.tensor(cov_samples, dtype=utils.TORCH_FLOAT_TYPE)
    dist_1 = distributions.MultivariateNormal(loc, covariance_matrix=cov_samples)

    # Create a std normal
    mn = torch.zeros(batch_size, num_points_per_batch, dtype=utils.TORCH_FLOAT_TYPE)
    cov = torch.stack([torch.eye(num_points_per_batch, dtype=utils.TORCH_FLOAT_TYPE) for _ in range(batch_size)])
    std_norm = distributions.MultivariateNormal(mn, covariance_matrix=cov)

    # Do the computed/expected
    time_s = time.time()
    computed_kl = custom_distributions.kl_mvn_and_std_norm(dist_1).detach().numpy()
    time_mid = time.time()
    expected_kl = distributions.kl_divergence(dist_1, std_norm).detach().numpy()
    time_end = time.time()

    print(f"Pytorch impl: {time_end-time_mid}s;  Custom imp: {time_mid-time_s}s")
    # ^ not very scientific but sanity check to make sure that worth avoiding the PyTorch implementation in the
    # against std normal case.

    # Test!
    np.testing.assert_array_almost_equal(computed_kl, expected_kl)


