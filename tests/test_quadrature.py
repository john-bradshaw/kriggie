
import torch
from torch import distributions
import numpy as np

from kriggie import quadrature
from kriggie import utils

utils.TORCH_FLOAT_TYPE = torch.float64
utils.NP_FLOAT_TYPE = np.float64


def test_quadrature_by_ensuring_normalizes_to_one():

    locs = torch.tensor([[[-0.6484,  0.6291],
         [ 0.3347, -0.6964],
         [ 0.7208,  1.3262],
         [ 0.7325,  0.8045]],

        [[ 0.6336,  0.0190],
         [ 0.4499, -1.0523],
         [ 1.0438,  0.2835],
         [ 0.4851, -0.9447]],

        [[ 1.3656,  1.0986],
         [ 1.0191, -0.7491],
         [-0.2781,  1.2328],
         [ 0.1406, -1.2100]]], dtype=utils.TORCH_FLOAT_TYPE)

    stds = torch.tensor([[[0.3915, 0.1297],
         [0.0151, 0.1328],
         [0.9023, 0.2621],
         [0.3819, 0.3920]],

        [[1.3302, 0.0269],
         [0.0842, 1.3834],
         [0.0530, 2.5361],
         [0.0345, 0.2558]],

        [[2.2114, 1.1675],
         [0.3666, 0.0100],
         [2.8550, 2.2998],
         [1.8505, 3.1392]]], dtype=utils.TORCH_FLOAT_TYPE)

    gaussian = distributions.Normal(loc=locs, scale=stds)
    func = lambda x: torch.ones_like(x)

    out = quadrature.gauss_quadrature(gaussian, func)
    out_np = out.detach().numpy()

    np.testing.assert_array_almost_equal(out_np, np.ones_like(out_np))



def test_quadrature_by_getting_mean_and_var():

    locs = torch.tensor([[[-0.6484,  0.6291],
         [ 0.3347, -0.6964],
         [ 0.7208,  1.3262],
         [ 0.7325,  0.8045]],

        [[ 0.6336,  0.0190],
         [ 0.4499, -1.0523],
         [ 1.0438,  0.2835],
         [ 0.4851, -0.9447]],

        [[ 1.3656,  1.0986],
         [ 1.0191, -0.7491],
         [-0.2781,  1.2328],
         [ 0.1406, -1.2100]]], dtype=utils.TORCH_FLOAT_TYPE)

    stds = torch.tensor([[[0.3915, 0.1297],
         [0.0151, 0.1328],
         [0.9023, 0.2621],
         [0.3819, 0.3920]],

        [[1.3302, 0.0269],
         [0.0842, 1.3834],
         [0.0530, 2.5361],
         [0.0345, 0.2558]],

        [[2.2114, 1.1675],
         [0.3666, 0.0100],
         [2.8550, 2.2998],
         [1.8505, 3.1392]]], dtype=utils.TORCH_FLOAT_TYPE)

    gaussian = distributions.Normal(loc=locs, scale=stds)
    func1 = lambda x: x

    predicted_mean = quadrature.gauss_quadrature(gaussian, func1)
    predicted_mean_np = predicted_mean.detach().numpy()

    np.testing.assert_array_almost_equal(predicted_mean_np, locs.detach().numpy())

    func2 = lambda x: x**2
    expected_x_squared = quadrature.gauss_quadrature(gaussian, func2)
    predicted_std = np.sqrt(expected_x_squared.detach().numpy() - predicted_mean_np**2)
    np.testing.assert_array_almost_equal(predicted_std, stds.detach().numpy())

