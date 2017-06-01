"""
@author: Runjing Liu
"""

import autograd.numpy as np
import autograd.scipy as sp

from scipy.special import expit

import matplotlib.pyplot as plt
from copy import deepcopy
import math


# data_shape: D,N,K

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

    nu_term2 = (1. / (2. * s_eps)) * (phi_var[k]*D + np.dot(phi_mu[:,k], phi_mu[:,k]))

    nu_term3 = (1./s_eps) * np.dot(phi_mu[:, k], X[n, :] - \
               np.dot(phi_mu, nu[n, :]) + nu[n,k] * phi_mu[:, k])

    script_V = nu_term1 - nu_term2 + nu_term3

    #nu[n,k] = 1./(1.+np.exp(-script_V))
    nu[n,k] = expit(script_V)


def tau_updates(tau, nu, alpha):
    N = np.shape(nu)[0]
    K = np.shape(nu)[1]

    tau[:,0] = alpha/K + np.sum(nu,0)
    tau[:,1] = N  + 1 - np.sum(nu,0)


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


def exp_log_likelihood_old(nu_moment, phi_moment1, phi_moment2, \
                           E_log_pi1, E_log_pi2, sigmas, X, alpha):

    sigma_eps = sigmas['eps']
    sigma_A = sigmas['A']
    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(phi_moment1)[1]

    beta_lh = (alpha/K - 1.)*np.sum(E_log_pi1)
    bern_lh = np.sum(np.dot(nu_moment[n,:], E_log_pi1) \
                            + np.dot(1.-nu_moment[n,:], E_log_pi2) for n in range(N))
    Normal_A = -1/(2.*sigma_A) * np.sum(phi_moment2)

    Normal_X_sum = 0
    ## compute the data likelihood term
    for n in range(N):
        dum1 = 2.*np.sum(np.sum(nu_moment[n,i] * nu_moment[n,j] * \
                                np.dot(phi_moment1[:,i],phi_moment1[:,j]) \
                                for i in range(j)) for j in range(K))
        dum2 = np.dot(nu_moment[n,:] , phi_moment2 )

        dum3 = -2. * np.dot(X[n,:], np.dot(phi_moment1, nu_moment[n,:]))

        # dum4 = np.dot(X[n,:], X[n,:])
        Normal_X_sum += dum1 + dum2 + dum3

    Normal_X = -1/(2*sigma_eps)*Normal_X_sum

    y = beta_lh + bern_lh + Normal_A + Normal_X
    return(y)


def compute_elbo_old(tau, nu, phi_mu, phi_var, X, sigmas, alpha):

    sigma_eps = sigmas['eps']
    sigma_a = sigmas['A']
    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(phi_mu)[1]

    digamma_tau = sp.special.digamma(tau)
    digamma_sum_tau = sp.special.digamma(tau[:,0] + tau[:,1])

    # bernoulli terms
    elbo_term1 = (alpha/K - 1) * np.sum( digamma_tau[:,0] - digamma_sum_tau[:] )

    elbo_term2 = np.sum(np.dot(nu, digamma_tau[:,0]) + np.dot(1-nu, digamma_tau[:,1])) \
                - N * np.sum(digamma_sum_tau)

    # elbo_term3 = \
    #     -K*D/2.*np.log(2.*np.pi*sigma_a) - 1./(2.*sigma_a) *\
    #     (np.sum(phi_var)*D + np.trace(np.dot(phi_mu.T , phi_mu)))

    elbo_term3 = \
        - 1./(2.*sigma_a) * (np.sum(phi_var)*D + np.trace(np.dot(phi_mu.T , phi_mu)))

    # elbo_term4 = \
    #     -N * D/2. * np.log(2. * np.pi * sigma_eps)\
    #     -1/(2. * sigma_eps) * np.trace(np.dot(X.T, X)) \
    #     +1/(sigma_eps) * np.trace(np.dot(np.dot(nu, phi_mu.T), X.T))\
    #     -1/(2. * sigma_eps) * np.sum(np.dot(nu, D*phi_var + \
    #         np.diag(np.dot(phi_mu.T, phi_mu)))) \
    #     -1/(2.*sigma_eps) * np.trace(np.dot(np.dot( \
    #         nu, np.dot(phi_mu.T, phi_mu) - np.identity(K) \
    #         * np.diag(np.dot(phi_mu.T, phi_mu))), nu.T))

    elbo_term4 = \
        +1/(sigma_eps) * np.trace(np.dot(np.dot(nu, phi_mu.T), X.T))\
        -1/(2. * sigma_eps) * np.sum(np.dot(nu, D*phi_var + \
            np.diag(np.dot(phi_mu.T, phi_mu)))) \
        -1/(2.*sigma_eps) * np.trace(np.dot(np.dot( \
            nu, np.dot(phi_mu.T, phi_mu) - np.identity(K) \
            * np.diag(np.dot(phi_mu.T, phi_mu))), nu.T))

    # The log beta function is not in autograd's scipy.
    log_beta = sp.special.gammaln(tau[:,0]) + sp.special.gammaln(tau[:,1]) \
        - sp.special.gammaln(tau[:,0] + tau[:,1])

    elbo_term5 = np.sum(log_beta - \
        (tau[:,0] - 1) * digamma_tau[:,0] - \
        (tau[:,1] - 1) * digamma_tau[:,1] + \
        (tau[:,0] + tau[:,1] -2.) *  digamma_sum_tau[:])

    elbo_term6 = np.sum(1. / 2. * np.log((2. * np.pi * np.exp(1.) * \
        phi_var)**D))

    elbo_term7 = np.sum(np.sum( -np.log(nu ** nu) - np.log((1.-nu) ** (1.-nu)) ))

    elbo = elbo_term1 + elbo_term2 + elbo_term3 + elbo_term4 + elbo_term5 + \
           elbo_term6 + elbo_term7

    return(elbo, elbo_term1, elbo_term2, elbo_term3, elbo_term4, \
          elbo_term5, elbo_term6, elbo_term7)


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


def get_moments(tau, nu, phi_mu, phi_var):
    digamma_tausum = sp.special.digamma(np.sum(tau, 1))
    e_log_pi1 = sp.special.digamma(tau[:, 0]) - digamma_tausum
    e_log_pi2 = sp.special.digamma(tau[:, 1]) - digamma_tausum

    nu_moment = nu

    D = phi_mu.shape[0]
    phi_moment1 = phi_mu
    phi_moment2 = np.diag(np.dot(phi_mu.T, phi_mu) + D * phi_var)

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
        sigmas, X, alpha)

    D = X.shape[1]
    entropy = nu_entropy(nu) + phi_entropy(phi_var, D) + pi_entropy(tau)

    return e_log_lik + entropy


def exp_log_likelihood(nu_moment, phi_moment1, phi_moment2, \
                       e_log_pi1, e_log_pi2, sigmas, X, alpha):

    sigma_eps = sigmas['eps']
    sigma_a = sigmas['A']

    D = X.shape[1]
    N = X.shape[0]
    K = nu_moment.shape[1]

    # Compute the beta, bernoulli, and A terms.
    beta_lh = (alpha / float(K) - 1.) * np.sum(e_log_pi1)
    bern_lh = np.sum(nu_moment * (e_log_pi1 - e_log_pi2)) + N * np.sum(e_log_pi2)
    norm_a_term = -0.5 * np.sum(phi_moment2) / sigma_a
    #norm_a_term = -0.5 * np.sum(phi_moment1 ** 2) / sigma_a

    # Compute the data likelihood term
    phi_moment1_outer = np.matmul(phi_moment1.T, phi_moment1)
    phi_moment1_outer = phi_moment1_outer - np.diag(np.diag(phi_moment1_outer))
    norm_x_nu_quadratic = \
        np.einsum('ni,nj,ij', nu_moment, nu_moment, phi_moment1_outer)
    norm_x_nu_linear = \
        np.sum(nu_moment * (-2. * np.matmul(X, phi_moment1) + phi_moment2))
    norm_x_term = -0.5 * (norm_x_nu_linear + norm_x_nu_quadratic) / sigma_eps

    return beta_lh + bern_lh + norm_a_term + norm_x_term


def initialize_parameters(Num_samples, D, K_approx):
    # tau1, tau2 -- beta parameters for v
    tau = np.random.uniform(0.5, 2.0, [K_approx, 2])

    # Bernoulli parameter for z_nk
    nu =  np.random.uniform(0.01, 0.99, [Num_samples, K_approx])

    # kth mean (D dim vector) in kth column
    phi_mu = np.random.normal(0, 1, [D, K_approx])
    phi_var = np.ones(K_approx)

    return tau, nu, phi_mu, phi_var


def generate_data(Num_samples, D, K_inf, sigma_a, sigma_eps, alpha):
    Pi = np.ones(K_inf) * .8

    Z = np.zeros([Num_samples, K_inf])

    # Parameters to draw A from MVN
    mu = np.zeros(D)

    # Draw Z from truncated stick breaking process
    Z = np.random.binomial(1, Pi, [ Num_samples, K_inf ])

    # Draw A from multivariate normal
    A = np.random.normal(0, np.sqrt(sigma_a), (K_inf, D))

    # draw noise
    epsilon = np.random.normal(0, np.sqrt(sigma_eps), (Num_samples, D))

    # the observed data
    X = np.matmul(Z, A) + epsilon

    return Pi, Z, mu, A, X
