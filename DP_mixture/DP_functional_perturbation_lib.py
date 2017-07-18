## library to examine functional perturbations
import autograd.numpy as np
import autograd.scipy as sp

from autograd import grad, hessian, hessian_vector_product, hessian, jacobian

import DP_normal_mixture_lib as dp

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
    x = x[1:-1]
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
                    prior_mu, prior_info, info_x, alpha, \
                    u = lambda x : 0.0 * x, n_grid = 10000):
    # combining the pieces, compute the full expected log likelihood

    prior = dp_prior_perturbed(tau, alpha, u = u, n_grid = n_grid) \
                + dp.normal_prior(mu, mu2, prior_mu, prior_info)

    return dp.loglik_obs(e_z, mu, mu2, x, info_x) \
                + dp.loglik_ind(e_z, e_log_v, e_log_1mv) + prior

def compute_elbo_perturbed(x, mu, mu2, info, tau, e_log_v, e_log_1mv, e_z,
                    prior_mu, prior_info, info_x, alpha, \
                    u = lambda x : 0.0 * x, n_grid = 10000):

    # entropy terms
    entropy = dp.mu_entropy(info) + dp.beta_entropy(tau) \
                    + dp.multinom_entropy(e_z)

    return e_loglik_full_perturbed(x, mu, mu2, tau, e_log_v, e_log_1mv, e_z,
                        prior_mu, prior_info, info_x, alpha, \
                        u = u, n_grid = n_grid) + entropy
















class FunctionalPerturbation(object):
    def __init__(self, x, vb_model, hyper_params, u = lambda x : 0.0 * x):
        self.x = x
        self.vb_model = deepcopy(vb_model)
        self.hyper_params = deepcopy(hyper_params)
        self.u = u

        self.alpha = hyper_params['alpha'].get()
        self.sigmas = {'A': hyper_params['var_a'].get(),
                            'eps': hyper_params['var_eps'].get()}
        self.k_approx = np.shape(vb_model['phi'].e())[0]

        #self.get_kl_grad =  grad(self.wrapped_kl, 0)
        #self.get_kl_hvp = hessian_vector_product(self.wrapped_kl, 0)
        #self.get_kl_hessian = hessian(self.wrapped_kl, 0)

        #self.get_kl_grad =  grad(
        #    lambda params: self.wrapped_kl(params, tracing=False))
        #self.get_kl_hvp = hessian_vector_product(
        #    lambda params: self.wrapped_kl(params, tracing=False))
        #self.get_kl_hessian = hessian(
        #    lambda params: self.wrapped_kl(params, tracing=False))

        self.trace = lrvb.OptimzationTrace()

    def unpack_params(self, vb_model):
        phi_mu = vb_model['phi'].mean.get()
        phi_var = 1 / vb_model['phi'].info.get()
        nu = vb_model['nu'].get()
        tau = vb_model['pi'].alpha.get()

        return tau, phi_mu.T, phi_var, nu

    def wrapped_kl(self, free_vb_params, n_grid, tracing=True):
        self.vb_model.set_free(free_vb_params)
        elbo = compute_elbo_perturbed(self.x, self.vb_model, \
                self.hyper_params, self.u, n_grid)

        if tracing:
            self.trace.update(free_vb_params, -1 * elbo)
        return -1 * elbo

    def run_newton_tr(self, params_init, n_grid = 10**6, maxiter=200, gtol=1e-6):
        get_kl_grad =  grad(
            lambda params: self.wrapped_kl(params, n_grid, tracing=False))
        get_kl_hvp = hessian_vector_product(
            lambda params: self.wrapped_kl(params, n_grid, tracing=False))
        get_kl_hessian = hessian(
            lambda params: self.wrapped_kl(params, n_grid, tracing=False))

        self.trace.reset()
        self.tr_opt = optimize.minimize(
            lambda params: self.wrapped_kl(params, n_grid, tracing=True),
            params_init, method='trust-ncg',
            jac = get_kl_grad,
            hessp = get_kl_hvp,
            tol=1e-6, options={'maxiter': maxiter, 'disp': True, 'gtol': gtol })

        print('Done with Newton trust region.')
        return self.tr_opt
