"""
@author: Runjing Liu
"""

import autograd.numpy as np
import autograd.scipy as sp
from copy import deepcopy
import math
import scipy as osp
from scipy.special import expit


#######################################
# CAVI updates.

def phi_updates(nu, phi_mu, phi_var, X, sigmas, k):
    s_eps = sigmas['eps']
    s_A = sigmas['A']
    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(phi_mu)[1]

    phi_var[k] = (1/s_A + np.sum(nu[:, k]) / s_eps)**(-1)

    phi_summation = 0
    for n in range(N):
        phi_dum1 = X[n, :] - np.dot(phi_mu, nu[n, :]) + nu[n, k] * phi_mu[:, k]
        phi_summation += nu[n,k]*phi_dum1

    phi_mu[:,k] = (1 / s_eps) * phi_summation\
        * (1 / s_A + np.sum(nu[:, k]) / s_eps)**(-1)


def nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, n, k, digamma_tau):
    s_eps = sigmas['eps']
    s_A = sigmas['A']
    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(phi_mu)[1]

    nu_term1 = digamma_tau[k,0] - digamma_tau[k,1]

    nu_term2 = (1. / (2. * s_eps)) * (phi_var[k] * D + \
                np.dot(phi_mu[:,k], phi_mu[:,k]))

    nu_term3 = (1./s_eps) * np.dot(phi_mu[:, k], X[n, :] - \
               np.dot(phi_mu, nu[n, :]) + nu[n,k] * phi_mu[:, k])

    script_V = nu_term1 - nu_term2 + nu_term3

    nu[n,k] = expit(script_V)


def tau_updates(tau, nu, alpha):
    N = np.shape(nu)[0]
    K = np.shape(nu)[1]

    tau[:, 0] = alpha / K + np.sum(nu, 0)
    tau[:, 1] = N  + 1 - np.sum(nu, 0)


def cavi_updates(tau, nu, phi_mu, phi_var, X, alpha, sigmas):
    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(phi_mu)[1]

    assert np.shape(X)[0] == np.shape(nu)[0]
    assert np.shape(X)[1] == np.shape(phi_mu)[0]
    assert np.shape(nu)[1] == np.shape(phi_mu)[1]

    digamma_tau = sp.special.digamma(tau)

    for n in range(N):
        for k in range(K):
            nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, n, k, digamma_tau)

    for k in range(K):
        phi_updates(nu, phi_mu, phi_var, X, sigmas, k)

    tau_updates(tau, nu, alpha)


#######################################
# Log likelihood given parameters.

def log_p_x_conditional(x, z, a, sigma_eps):
    x_centered = x - np.matmul(z, a.T)
    var_eps = sigma_eps
    return -0.5 * np.sum(x_centered ** 2) / var_eps


def log_p_z(z, pi):
    return np.sum(z * np.log(pi) + (1 - z) * np.log(1 - pi))


def log_p_a(a, sigma_a):
    var_a = sigma_a
    return -0.5 * np.sum(a ** 2) / var_a


def log_p_pi(pi, alpha, k_approx):
    param = alpha / float(k_approx)
    return np.sum((param - 1) * np.log(pi))


def log_lik(x, z, a, pi, sigma_eps, sigma_a, alpha, k_approx):
    return \
        log_p_x_conditional(x, z, a, sigma_eps) + \
        log_p_pi(pi, alpha, k_approx) + log_p_z(z, pi) + log_p_a(a, sigma_a)


#######################################
# Variational entropies.  Note that we cannot use the scipy ones because
# they are not yet supported by autograd.

def nu_entropy(nu):
    log_1mnu = np.log(1 - nu)
    log_nu = np.log(nu)
    return -1 * np.sum(nu * log_nu + (1 - nu) * log_1mnu)


def phi_entropy(phi_var, D):
    return 0.5 * D * np.sum(np.log(2. * np.pi * phi_var) + 1)


def pi_entropy(tau):
    digamma_tau0 = sp.special.digamma(tau[:, 0])
    digamma_tau1 = sp.special.digamma(tau[:, 1])
    digamma_tausum = sp.special.digamma(np.sum(tau, 1))

    lgamma_tau0 = sp.special.gammaln(tau[:, 0])
    lgamma_tau1 = sp.special.gammaln(tau[:, 1])
    lgamma_tausum = sp.special.gammaln(np.sum(tau, 1))

    lbeta = lgamma_tau0 + lgamma_tau1 - lgamma_tausum

    return np.sum(
        lbeta - \
        (tau[:, 0] - 1.) * digamma_tau0 - \
        (tau[:, 1] - 1.) * digamma_tau1 + \
        (tau[:, 0] + tau[:, 1] - 2) * digamma_tausum)


##############################################
# Variational ELBO and helper functions.

def get_moments(tau, nu, phi_mu, phi_var):
    digamma_tausum = sp.special.digamma(np.sum(tau, 1))
    e_log_pi1 = sp.special.digamma(tau[:, 0]) - digamma_tausum
    e_log_pi2 = sp.special.digamma(tau[:, 1]) - digamma_tausum

    nu_moment = nu

    phi_moment1 = phi_mu
    phi_moment2 = phi_mu ** 2 + phi_var

    return e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment


def compute_elbo(tau, nu, phi_mu, phi_var, X, sigmas, alpha):
    # Check for appropriate shapes.
    assert X.shape[1] == phi_mu.shape[0]
    assert nu.shape[0] == X.shape[0]
    assert nu.shape[1] == phi_mu.shape[1]
    assert len(phi_var) == phi_mu.shape[1]
    assert tau.shape[0] == nu.shape[1]
    assert tau.shape[1] == 2

    e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment = \
        get_moments(tau, nu, phi_mu, phi_var)

    e_log_lik = exp_log_likelihood(
        nu_moment, phi_moment1, phi_moment2,  e_log_pi1, e_log_pi2, \
        sigmas['A'], sigmas['eps'], X, alpha)

    D = X.shape[1]
    entropy = nu_entropy(nu) + phi_entropy(phi_var, D) + pi_entropy(tau)

    return e_log_lik + entropy


def exp_log_likelihood(nu_moment, phi_moment1, phi_moment2, \
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


# Draw from the variational (or conditional) disribution.
def draw_z(nu, num_samples):
    return np.random.binomial(1, nu, size=(num_samples, nu.shape[0], nu.shape[1]))


def draw_a(phi_mu, phi_var_expanded, num_samples):
    # phi_var_expanded is a version of phi_var with the same shape as phi_mu
    return np.random.normal(
        phi_mu, phi_var_expanded,
        (num_samples, phi_mu.shape[0], phi_mu.shape[1]))


def draw_pi(tau, num_samples):
    # The numpy beta draws seem to actually hit zero and one, unlike scipy.
    return osp.stats.beta.rvs(tau[:, 0], tau[:, 1],
                              size=(num_samples, tau.shape[0]))


def generate_parameter_draws(nu, phi_mu, phi_var_expanded, tau, n_test_samples):
    z_sample = draw_z(nu, n_test_samples)
    a_sample = draw_a(phi_mu, phi_var_expanded, n_test_samples)
    pi_sample = draw_pi(tau, n_test_samples)

    return z_sample, a_sample, pi_sample


####################################
# Initialization and generation of data sets.

def initialize_parameters(num_samples, D, k_approx):
    # tau1, tau2 -- beta parameters for v
    tau = np.random.uniform(0.5, 2.0, [k_approx, 2])

    # Bernoulli parameter for z_nk
    nu =  np.random.uniform(0.01, 0.99, [num_samples, k_approx])

    # kth mean (D dim vector) in kth column
    phi_mu = np.random.normal(0, 1, [D, k_approx])
    phi_var = np.ones(k_approx)

    return tau, nu, phi_mu, phi_var


def generate_data(num_samples, D, k_inf, sigma_a, sigma_eps, alpha):
    pi = np.ones(k_inf) * .8

    Z = np.zeros([num_samples, k_inf])

    # Parameters to draw A from MVN
    mu = np.zeros(D)

    # Draw Z from truncated stick breaking process
    Z = np.random.binomial(1, pi, [ num_samples, k_inf ])

    # Draw A from multivariate normal
    A = np.random.normal(0, np.sqrt(sigma_a), (k_inf, D))

    # draw noise
    epsilon = np.random.normal(0, np.sqrt(sigma_eps), (num_samples, D))

    # the observed data
    X = np.matmul(Z, A) + epsilon

    return pi, Z, mu, A, X
