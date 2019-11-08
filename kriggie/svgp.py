

import torch
from torch import nn
from torch import distributions

from . import custom_distributions
from . import kernels
from . import transformed_params
from . import utils
from . import conditionals


class SVGPLayer(nn.Module):
    """
    Sparse Variational GP (SVGP)
    Reference is:
    Hensman, J., Matthews, A. and Ghahramani, Z., 2015. Scalable variational Gaussian process classification.

    """
    def __init__(self, kernel: kernels.BaseKernel, num_latent_processes: int, num_inducing_points: int, data_dim: int,
                 whiten_flag: bool=True):
        super().__init__()
        self.__summed_over_batch_kl = 0.

        self.kernel = kernel
        self.whiten_flag = whiten_flag

        # Parameters
        self.z_locations = nn.Parameter(torch.empty(num_inducing_points, data_dim, dtype=utils.TORCH_FLOAT_TYPE))
        self.q_mu = nn.Parameter(torch.empty(num_latent_processes, num_inducing_points, dtype=utils.TORCH_FLOAT_TYPE))
        self.q_sqrt_constrained = transformed_params.TransformedParam(
            torch.empty(num_latent_processes, num_inducing_points, num_inducing_points, dtype=utils.TORCH_FLOAT_TYPE),
                                                                      transform=distributions.LowerCholeskyTransform(),
                                                                      requires_grad=True
                                                                      )
        self.reset_params()

    def reset_params(self):
        # Note you probably want to explore far better initialisations than these!
        nn.init.normal_(self.z_locations)
        nn.init.normal_(self.q_mu)
        self.q_sqrt_constrained.constrained_init_set(torch.stack(
            [torch.eye(self.num_inducing, self.num_inducing, dtype=utils.TORCH_FLOAT_TYPE)
             for _ in range(self.num_latent_processes)]))

    def forward(self, x_data, marginals_only=True, update_kl_on_forward=True, latent_dim_last=True):
        """
        :return: [num_funcs, num_data_points],
                [num_funcs, num_data_points] or [num_funcs, num_data_points, num_data_points]
        """
        cov_xx = self.kernel(x_data)
        cov_zx = self.kernel(self.z_locations, x_data)
        cov_zz = self.kernel(self.z_locations)
        q_dist = self.q_dist

        if update_kl_on_forward:
            if self.whiten_flag:
                # KL between this and a standard normal can be calculated more easily:
                # see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
                kl = custom_distributions.kl_mvn_and_std_norm(q_dist)
            else:
                chol_zz = utils.try_func_with_increasing_jitter(lambda x: torch.cholesky(x, upper=False), cov_zz, RuntimeError)
                prior_dist = distributions.MultivariateNormal(torch.zeros_like(self.q_mu),
                                                            scale_tril=chol_zz)
                kl = distributions.kl_divergence(q_dist, prior_dist)

            self.__summed_over_batch_kl += kl.sum()
            # assert self.whiten_flag
            # prior_dist = distributions.MultivariateNormal(torch.zeros_like(self.q_mu),
            #                 scale_tril=torch.stack([torch.eye(self.num_inducing, dtype=utils.TORCH_FLOAT_TYPE)
            #                                      for _ in range(self.num_latent_processes)]))
            # kl = distributions.kl_divergence(q_dist, prior_dist)
            # self.__summed_over_batch_kl += kl.sum()



        f_data_mean, f_data_var = self._predict(cov_xx, cov_zx, cov_zz, return_full_cov=not marginals_only,
                                                q_dist=q_dist)
        if latent_dim_last:
            f_data_mean = f_data_mean.t()
            f_data_var = f_data_var.permute(*range(1, len(f_data_var.shape)), 0)


        return f_data_mean, f_data_var  

    def _predict(self, cov_xx, cov_zx, cov_zz, return_full_cov, q_dist):
        if not return_full_cov:
            cov_xx = torch.diag(cov_xx)
        f_data_mean, f_data_var = conditionals.conditional_gaussian(cov_xx, cov_zx, cov_zz, q_dist, return_full_cov,
                                                                    whiten=self.whiten_flag)
        return f_data_mean, f_data_var

    @property
    def q_dist(self):
        return distributions.MultivariateNormal(loc=self.q_mu, scale_tril=self.q_sqrt_constrained.constrained)

    @property
    def summed_kl(self):
        return self.__summed_over_batch_kl

    def clear_kl(self):
        self.__summed_over_batch_kl = 0.

    @property
    def num_inducing(self):
        return self.q_mu.shape[1]

    @property
    def num_latent_processes(self):
        return self.q_mu.shape[0]

    @property
    def data_dim_in(self):
        return self.z_locations.shape[1]

