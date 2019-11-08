
# -*- coding: utf-8 -*-


import typing
import functools

import numpy as np
import torch
from torch import distributions

from . import utils


@functools.lru_cache(maxsize=10)
def hermgauss_torch(num_points, device):
    locs, weights = np.polynomial.hermite.hermgauss(num_points)
    return (torch.tensor(locs,  dtype=utils.TORCH_FLOAT_TYPE,
                         device=device), torch.tensor(weights, dtype=utils.TORCH_FLOAT_TYPE, device=device))


def gauss_quadrature(normal_in: distributions.Normal,
                     func_to_evaluate: typing.Callable[[torch.Tensor], torch.Tensor], num_points: int=25):
    """
    1-D Gaussian quadrature:
    1/ √(2πσ^2) ∫_∞^∞ exp( -0.5 * (x-  μ)^2/ σ^2) ) f(x) dx

    Do this via Herm gauss quadrature gives us the the weights and locations for ∫_∞^∞ e^{-t^{2}} f(t) dt
    ie ∫_∞^∞ e^{-t^{2}} f(t) dt ≈ Σ w_i  f(t_i)


    We use the transform y =  (x-  μ)/ (σ √2) to get:

        1/ √(2πσ^2)  ∫_∞^∞ exp(-y^2) f( y σ √2 + μ) (σ √2) dy
     =  1/ √π  ∫_∞^∞ exp(-y^2) f( y σ √2 + μ) dy
               ---------------------------------
        const.          evald. by quadrature

    See:
    * https://en.wikipedia.org/wiki/Gaussian_quadrature
    * https://gist.github.com/markvdw/f9ca12c99484cf2a881e84cb515b86c8

    :param normal_in: The Gaussian we wish to integrate against. [*batch_shape]
    :param func_to_evaluate: Takes in a Tensor  x [*batch_shape, num_points] and returns f(x) [*batch_shape, num_points]
    :param num_points: num points to use in the quadrature
    :return: [*batch_shape]
    """
    assert isinstance(normal_in, distributions.Normal)

    mu = normal_in.loc
    std = normal_in.scale
    batch_shape = normal_in.batch_shape

    locs, weights = hermgauss_torch(num_points, mu.device)
    locs = locs.expand(*batch_shape, -1)
    weights = weights.view(*(1,)*len(batch_shape), -1)  # put in correct size such that broadcasting can work later.

    new_locs = locs * std[..., None] * np.sqrt(2) + mu[..., None]
    evals = func_to_evaluate(new_locs)

    weighted_evals = weights * evals
    approximate_integral = weighted_evals.sum(dim=-1) / np.sqrt(np.pi)
    return approximate_integral



