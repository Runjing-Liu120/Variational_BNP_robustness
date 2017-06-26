## library to examine functional perturbations
import autograd.numpy as np
import autograd.scipy as sp

import scipy as osp

import finite_approx.LRVB_lib as lrvb

def log_q_pi(pi, tau):
    log_beta = sp.special.gammaln(tau[:,0]) + sp.special.gammaln(tau[:,1]) \
                    - sp.special.gammaln(tau[:,0] + tau[:,1])

    return -log_beta + (tau[:,0] - 1.0) * np.log(pi) \
                    + (tau[:,1] - 1.0) * np.log(1.0 - pi)


def exp_pi_prior_perturbed(tau, alpha, k_approx, u, n_grid):
    # u(x) is the perturbations
    # n_grid is the number of points on our grid
    x = np.linspace(0.001, 0.999, n_grid)
    delta = x[1] - x[0]

    # p0 = osp.stats.beta.logpdf(x, 1, alpha/k_approx)
    pert = u(x)

    prior_density = np.exp(log_q_pi(x, np.array([[1,alpha/k_approx]])))
    assert len(prior_density) == n_grid

    # the broadcasting here bothers me ... but I guess it works
    variational_density = np.exp(log_q_pi(x[:, None],tau))
    assert np.shape(variational_density) == (n_grid, np.shape(tau)[0])

    integrand = (prior_density[:,None] + pert[:,None]) * variational_density
    assert np.shape(integrand) == np.shape(variational_density)

    exp_beta_lh_perturbed = np.sum(integrand * delta, 0)

    return exp_beta_lh_perturbed


def exp_log_pi_perturbed():
    pass


def exp_log_likelihood_perturb_pi(nu_moment, phi_moment1, phi_moment2, \
                       e_log_pi1, e_log_pi2, sigma_a, sigma_eps, X, alpha):
    D = X.shape[1]
    N = X.shape[0]
    K = nu_moment.shape[1]

    # Compute the beta, bernoulli, and A terms.
    beta_lh = (alpha / float(K) - 1.) * np.sum(e_log_pi1)
    bern_lh = np.sum(nu_moment * (e_log_pi1 - e_log_pi2)) + \
              N * np.sum(e_log_pi2)
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
