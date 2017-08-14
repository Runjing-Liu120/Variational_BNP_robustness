## library to examine functional perturbations
import autograd.numpy as np
import autograd.scipy as sp

from autograd import grad, hessian, hessian_vector_product, hessian, jacobian

import DP_normal_mixture_lib as dp
import DP_normal_mixture_opt_lib as dp_opt

import scipy as osp
from scipy import optimize

from copy import deepcopy

def log_q_pi(pi, tau):
    log_beta = sp.special.gammaln(tau[:,0]) + sp.special.gammaln(tau[:,1]) \
                    - sp.special.gammaln(tau[:,0] + tau[:,1])

    return -log_beta + (tau[:,0] - 1.0) * np.log(pi) \
                    + (tau[:,1] - 1.0) * np.log(1.0 - pi)


def dp_prior_perturbed(tau, alpha, u = lambda x : 0.0 * x, \
                            n_grid = 10000):
    # u(x) is the perturbations
    # n_grid is the number of points on our grid
    x = np.linspace(0.0, 1.0, n_grid)
    x = x[1:-1] # ignore 0.0 and 1.0
    delta = x[1] - x[0]

    pert = u(x)

    prior_tau = np.array([[1, alpha]])
    prior_density = np.exp(log_q_pi(x, prior_tau))
    assert len(prior_density) == len(x)

    # the broadcasting here bothers me ... but I guess it works
    variational_density = np.exp(log_q_pi(x[:, None], tau))
    assert np.shape(variational_density) == (len(x), np.shape(tau)[0])

    integrand = np.log(prior_density[:, None] + pert[:, None]) \
                    * variational_density
    assert np.shape(integrand) == np.shape(variational_density)

    return np.sum(integrand * delta)


def e_loglik_full_perturbed(x, mu, mu2, tau, e_log_v, e_log_1mv, e_z,
                    e_info_x, e_logdet_info_x, prior_mu,
                    prior_inv_wishart_scale, kappa, prior_dof, alpha, \
                    u = lambda x : 0.0 * x, n_grid = 10000):
    # combining the pieces, compute the full expected log likelihood

    prior = dp_prior_perturbed(tau, alpha, u = u, n_grid = n_grid) \
            + dp.normal_prior(mu, mu2, prior_mu, e_info_x, e_logdet_info_x, kappa)\
            + dp.wishart_prior(e_info_x, e_logdet_info_x, \
                        prior_inv_wishart_scale, prior_dof)

    return dp.loglik_obs(e_z, mu, mu2, x, e_info_x, e_logdet_info_x) \
                + dp.loglik_ind(e_z, e_log_v, e_log_1mv) + prior

def compute_elbo_perturbed(x, mu, mu2, info_mu, tau, e_log_v, e_log_1mv, e_z,
                    e_info_x, e_logdet_info_x, dof, prior_mu,
                    prior_inv_wishart_scale, kappa, prior_dof, alpha,
                    u = lambda x : 0.0 * x, n_grid = 10000):

    # entropy terms
    entropy = dp.mu_entropy(info_mu) + dp.beta_entropy(tau) \
                + dp.multinom_entropy(e_z) \
                + dp.wishart_entropy(e_logdet_info_x, e_info_x, dof)

    return e_loglik_full_perturbed(x, mu, mu2, tau, e_log_v, e_log_1mv, e_z,
                        e_info_x, e_logdet_info_x, prior_mu,
                        prior_inv_wishart_scale, kappa, prior_dof, alpha, \
                        u = u, n_grid = n_grid) + entropy


class PerturbedKL(object):
    def __init__(self, x, vb_params, prior_params, u):
        self.x = x
        self.vb_params = deepcopy(vb_params)
        self.u = u

        self.prior_mu, self.prior_dof, self.prior_inv_wishart_scale, \
            self.alpha, self.kappa = dp.get_prior_params(prior_params)

    def set_optimal_z(self):
        e_log_v, e_log_1mv, e_z, mu, mu2, info_mu, tau,\
                    e_info_x, e_logdet_info_x, dof = \
                    dp.get_vb_params(self.vb_params)

        # optimize z
        e_z_opt = dp_opt.z_update(mu, mu2, self.x, e_info_x, e_log_v, e_log_1mv)
        self.vb_params['local']['e_z'].set(e_z_opt)

    def perturbed_kl(self):
        e_log_v, e_log_1mv, _, mu, mu2, info_mu, tau,\
                    e_info_x, e_logdet_info_x, dof = \
                    dp.get_vb_params(self.vb_params)

        # optimize z
        e_z = dp_opt.z_update(mu, mu2, self.x, e_info_x, e_log_v, e_log_1mv)

        elbo = compute_elbo_perturbed(self.x, mu, mu2, info_mu, tau, e_log_v,
                            e_log_1mv, e_z, e_info_x, e_logdet_info_x, dof,
                            self.prior_mu, self.prior_inv_wishart_scale,
                            self.kappa, self.prior_dof, self.alpha,
                            u = self.u, n_grid = 10000)

        return - 1.0 * elbo
