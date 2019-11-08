
# -*- coding: utf-8 -*-


import typing

import torch
from torch import distributions

from . import custom_distributions
from . import utils


def conditional_gaussian(cov_aa: torch.Tensor, cov_ba: torch.Tensor,
                         cov_bb: torch.Tensor,
                         fvals_b: typing.Union[custom_distributions.Delta, distributions.MultivariateNormal],
                         return_full_cov_flag: bool=True, whiten: bool=False):
    """
    Returns the (Gaussian distribution's params) of points at location a given the value of the points at b and the
    covariance matrices.
    The points at b can either be known (Delta distribution) or distributed according to a multivariate Gaussian
     distribution themselves (MultivariateNormal)!

    A good reference for the conditional distributions is section 2.3.1 of Bishop'a PRML:

    (eqn i) μ_{a|b} = Σ_{ab} Σ_{bb}^−1 x_b  (assuming zero mean)  gives the mean
    (eqn ii) Σ_{a|b} = Σ_{aa} − Σ_{ab} Σ_{bb}^−1 Σ_{ba}.  gives the variance
    (eqn iii) Σ_{a|b} = Σ_{aa} − Σ_{ab} Σ_{bb}^−1 Σ_{ba} +   Σ_{ab} Σ_{bb}^−1 Σ_{x_b} Σ_{bb}^−1 Σ_{ba}  gives the
    variance when have uncertainty about x_b too


    :param cov_aa: [b1, b1] or [b1] if return_full_cov_flag is set to True
    :param cov_ba: [b2, b1]
    :param cov_bb: [b2, b2]
    :param fvals_b: [b2, d]
    :param return_full_cov_flag: if True return the full covariance matrix (between the data points).
    If False then returns only the diagonal.
    :param whiten: This  means that fvals_b represents the whitened
    distribution with the true covariance depending on cholesky(cov_bb) fvals_b.m.covariance_matrix cholesky(cov_bb)^T
    see eg p3 of MCMC for Variationally Sparse Gaussian Processes J Hensman, AG Matthews,
     M Filippone, Z Ghahramani Advances in Neural Information Processing Systems, 1639-1647

    :returns: mean [b1, d], variance [d, b1, b2] if return_full_cov_flag else [d, b2]
    """
    # == Check we have sensible i ==
    assert isinstance(fvals_b, custom_distributions.Delta) or isinstance(fvals_b, distributions.MultivariateNormal)

    expected_aa_num_dims = 2 if return_full_cov_flag else 1
    if expected_aa_num_dims != len(cov_aa.shape):
        raise RuntimeError("Incorrect dimensions for cov_aa")

    assert len(fvals_b.batch_shape) ==1

    # == Some Choleskies ane matrix multitipliers needed for mean and covariance calculation. ==
    chol_bb = utils.try_func_with_increasing_jitter(lambda x: torch.cholesky(x, upper=False), cov_bb, RuntimeError)  # [b2, b2]
    chol_bb_inv__cov_ba = torch.triangular_solve(cov_ba, chol_bb, upper=False)[0]  # [b2, b1]

    if not whiten:
        cov_bb_inv__cov_ba = torch.triangular_solve(chol_bb_inv__cov_ba, chol_bb, upper=False, transpose=True)[0]  # [b2, b1]
        #^ todo: the PyTorch docs are a bit confusing at this point. upper does not mean solve upper system of eqns
        # more that the original A you send in is upper triangular.
        proj_mat_t = cov_bb_inv__cov_ba # [b2, b1]
    else:
        proj_mat_t = chol_bb_inv__cov_ba  # [b2, b1]

    # == The mean (which we can now calculate using the prohection  ==
    fvals_b_mean = fvals_b.v if isinstance(fvals_b, custom_distributions.Delta) else fvals_b.mean  # [d, b2]
    fval_a_mean = fvals_b_mean @ proj_mat_t  # [d, b1]

    # == Compute the first part of the variance (due to conditioning) ==
    f_vals_b_batch_shape = fvals_b.batch_shape[0]
    if return_full_cov_flag:
        fval_a_var = cov_aa - (chol_bb_inv__cov_ba.t() @ chol_bb_inv__cov_ba)  # [b1, b1]
        fval_a_var = fval_a_var[None, :, :].repeat(f_vals_b_batch_shape,1,1)  # [d, b1, b1]
    else:
        fval_a_var = cov_aa - torch.sum(chol_bb_inv__cov_ba**2, dim=0)  # [b1]
        fval_a_var = fval_a_var[None, :].repeat(f_vals_b_batch_shape, 1)  # [d, b1]

    # == Compute the second part of the variance (due to the uncertainty in fvals_b) ==
    if isinstance(fvals_b, distributions.MultivariateNormal):
        chol_cov_points_b = fvals_b.scale_tril  # [d, b2, b2]
        second_term_part = torch.bmm(torch.transpose(chol_cov_points_b, 1, 2),
                                     proj_mat_t[None, :, :].repeat(f_vals_b_batch_shape, 1, 1))  # [d, b2, b1]

        if return_full_cov_flag:
            fval_a_var = fval_a_var + torch.bmm(torch.transpose(second_term_part, 1, 2), second_term_part)  # [d, b1, b1]
        else:
            fval_a_var = fval_a_var + torch.sum(second_term_part**2, dim=1)  # [d, b1]

    return fval_a_mean, fval_a_var

