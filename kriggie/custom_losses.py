
# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch.nn import functional as F

from . import utils


def log_gaussian_int_by_gaussian(mu_f, sigma_f_sq, y, sigma_l_sq):
    """
    ∫ N(f; μ_f, σ_f^2)  log  N(y; f, σ_l^2) df
    """
    term1 = np.log(2*np.pi)
    term2 = torch.log(sigma_l_sq)
    term3 = ((y - mu_f)**2 + sigma_f_sq)/sigma_l_sq
    return -0.5 * (term3 + term2 + term1)


def nll_independent_gaussian(means, var, target, reduction='none', small_const=utils.SMALL_CONSTANT):
    """
    -log N(target; μ, σ^2)
    """
    sq_term = (means - target)**2  #F.mse_loss(means, target, reduction='none')
    denom = (2. * var + small_const)

    var_term = 0.5 * torch.log(var + small_const)
    const_term = 0.5 * np.log(2 * np.pi)

    ret = var_term + sq_term / denom + const_term

    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret