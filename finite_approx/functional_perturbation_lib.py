## library to examine functional perturbations
import autograd.numpy as np
import autograd.scipy as sp

from autograd import grad, hessian, hessian_vector_product, hessian, jacobian


import scipy as osp
from scipy import optimize

from copy import deepcopy

import finite_approx.LRVB_lib as lrvb
import finite_approx.valez_finite_VI_lib as vi

def log_q_pi(pi, tau):
    log_beta = sp.special.gammaln(tau[:,0]) + sp.special.gammaln(tau[:,1]) \
                    - sp.special.gammaln(tau[:,0] + tau[:,1])

    return -log_beta + (tau[:,0] - 1.0) * np.log(pi) \
                    + (tau[:,1] - 1.0) * np.log(1.0 - pi)


def compute_e_pi_prior_perturbed(tau, alpha, k_approx, u = lambda x : 0.0 * x, \
                            n_grid = 10000):
    # u(x) is the perturbations
    # n_grid is the number of points on our grid
    x = np.linspace(0.001, 0.999, n_grid)
    delta = x[1] - x[0]

    # p0 = osp.stats.beta.logpdf(x, 1, alpha/k_approx)
    pert = u(x)

    prior_tau = np.array([[ alpha / k_approx, 1]])
    prior_density = np.exp(log_q_pi(x, prior_tau))
    assert len(prior_density) == n_grid

    # the broadcasting here bothers me ... but I guess it works
    variational_density = np.exp(log_q_pi(x[:, None], tau))
    assert np.shape(variational_density) == (n_grid, np.shape(tau)[0])

    integrand = np.log(prior_density[:, None] + pert[:, None]) \
                    * variational_density
    assert np.shape(integrand) == np.shape(variational_density)

    e_beta_prior_perturbed = np.sum(integrand * delta, 0)

    return e_beta_prior_perturbed


def exp_log_likelihood_perturbed(nu_moment, phi_moment1, phi_moment2, \
                    e_beta_prior_perturbed, e_log_pi1, e_log_pi2,\
                    sigma_a, sigma_eps, X, alpha):
    x_d = X.shape[1]
    x_n = X.shape[0]
    k_approx = nu_moment.shape[1]

    # Compute the beta, bernoulli, and A terms.
    # beta_lh = (alpha / float(k_approx) - 1.) * np.sum(e_log_pi1)
    beta_lh = np.sum(e_beta_prior_perturbed)
    #print('beta_lh', beta_lh)

    bern_lh = np.sum(nu_moment * (e_log_pi1 - e_log_pi2)) + \
              x_n * np.sum(e_log_pi2)
    norm_a_term = -0.5 * np.sum(phi_moment2) / sigma_a

    # Compute the data likelihood term
    phi_moment1_outer = np.matmul(phi_moment1.T, phi_moment1)
    phi_moment1_outer = phi_moment1_outer - np.diag(np.diag(phi_moment1_outer))
    norm_x_nu_quadratic = \
        np.einsum('ni,nj,ij', nu_moment, nu_moment, phi_moment1_outer)
    norm_x_nu_linear = \
        np.sum(nu_moment * (-2. * np.matmul(X, phi_moment1) +
                            np.sum(phi_moment2, 0)))
    norm_x_term = -0.5 * (norm_x_nu_linear + norm_x_nu_quadratic) / sigma_eps

    return beta_lh + bern_lh + norm_a_term + norm_x_term

def compute_elbo_perturbed(x, vb_model, hyper_params, u = lambda x : 0.0 * x, \
                                    n_grid = 10**6):
    # get moments
    e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment =\
                    vi.get_moments_VB(vb_model)
    tau = vb_model['pi'].alpha.get()

    # unpack hyper_params
    alpha = hyper_params['alpha'].get()
    sigma_a = hyper_params['var_a'].get()
    sigma_eps = hyper_params['var_eps'].get()
    k_approx = np.shape(tau)[0]

    e_beta_prior_perturbed = compute_e_pi_prior_perturbed(\
                    tau, alpha, k_approx, u, n_grid)

    e_log_lik = exp_log_likelihood_perturbed(nu_moment, phi_moment1, phi_moment2, \
                        e_beta_prior_perturbed, e_log_pi1, e_log_pi2,\
                        sigma_a, sigma_eps, x, alpha)

    entropy = vi.nu_entropy(nu_moment) \
            + vi.phi_entropy(1/vb_model['phi'].info.get(), x.shape[1]) \
            + vi.pi_entropy(vb_model['pi'].alpha.get())

    return e_log_lik + entropy

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
