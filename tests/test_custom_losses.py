

import torch
from torch import distributions
import numpy as np

from kriggie import custom_losses
from kriggie import utils


@torch.no_grad()
def test_nll_independent_gaussian():

    means = torch.tensor([[[-0.6484,  0.6291],
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

    var = torch.tensor([[[ 0.3547,  1.7990],
         [ 1.6203,  6.3797],
         [ 2.5644,  0.0555],
         [ 0.6644,  1.0074]],

        [[ 4.3134,  0.5900],
         [ 0.9509,  0.1731],
         [ 9.5454,  0.7519],
         [ 0.1243,  0.9408]],

        [[ 8.2599, 13.0994],
         [ 2.6101,  1.6198],
         [ 7.1086,  2.3575],
         [ 0.0305,  0.4462]]], dtype=utils.TORCH_FLOAT_TYPE)

    targets = torch.tensor([[[-0.6125,  0.5968],
         [ 0.345, -0.715],
         [ 0.84,  1.4262],
         [ 0.4321,  0.9165]],

        [[ 0.123,  -0.036],
         [ 0.8999, -1.1565],
         [ -1.56,  0.345],
         [ 0.3564, -0.89414]],

        [[ 1.56541,  1.21456261],
         [ 1.15641, -0.75645],
         [-0.35491,  1.15242],
         [ 0.1456, -1.29141]]], dtype=utils.TORCH_FLOAT_TYPE)


    via_func = custom_losses.nll_independent_gaussian(means, var, targets, small_const=0.)

    norm = distributions.Normal(loc=means, scale=torch.sqrt(var))
    via_t = -norm.log_prob(targets)

    np.testing.assert_array_almost_equal(via_func.numpy(), via_t.numpy())
