#!/usr/bin/python3

# Alarmingly, this fails with python2!

# First unit test-- check CAVI updates

import unittest
import valez_finite_VI_lib as finite_lib
from autograd import grad
import autograd.numpy as np
import autograd.scipy as sp
from copy import deepcopy

import scipy as osp
from scipy import stats


# draw data
num_samples = 10
D = 2
K_inf = 3
sigma_A = 1.2
sigma_eps  = 2
alpha = 10

Pi, Z, mu, A, X = \
    finite_lib.generate_data(num_samples, D, K_inf, sigma_A, sigma_eps, alpha)

# initialize variational parameters
K_approx = K_inf
# tau, nu, phi_mu, phi_var = \
#     finite_lib.initialize_parameters(num_samples, D, K_approx)

tau = np.full((K_approx, 2), 5.)
tau[:, 1] *= 2.

# Bernoulli parameter for z_nk
nu =  np.random.uniform(0.01, 0.99, (num_samples, K_approx))

# kth mean (D dim vector) in kth column
phi_mu = np.random.normal(0, 1, [D, K_approx])
phi_var = np.ones(K_approx)


# compute elbo
sigmas = {'eps': sigma_eps, 'A': sigma_A}
elbo = finite_lib.compute_elbo(tau, nu, phi_mu, phi_var, X, sigmas, alpha)

# generate samples to compute means
n_test_samples = 10**5

z_sample = np.random.binomial(1, nu, size=(n_test_samples, nu.shape[0], nu.shape[1]))

# A version of phi_var with the same shape as phi_mu
phi_var_expanded = np.array([ phi_var for d in range(D)])
a_sample = np.random.normal(phi_mu, phi_var_expanded,
                            (n_test_samples, phi_mu.shape[0], phi_mu.shape[1]))

# The numpy beta draws seem to actually hit zero and one, in contrast to scipy.
pi_sample = osp.stats.beta.rvs(tau[:, 0], tau[:, 1], size=(n_test_samples, tau.shape[0]))

tol_scale = 1. / np.sqrt(n_test_samples)
print('1 / sqrt(num test draws) = %0.6f' % tol_scale)

class TestElboComputation(unittest.TestCase):
    def assert_allclose(self, x, y, tol=1e-12):
        self.assertTrue(np.allclose(x, y, tol))

    def test_moments(self):
        e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment = \
            finite_lib.get_moments(tau, nu, phi_mu, phi_var)

        # Pi (tau)
        self.assert_allclose(np.mean(pi_sample, 0),
                             tau[:,0] / (tau[:,0] + tau[:,1]), 10 * tol_scale)
        log_pi_sample_mean = np.mean(np.log(pi_sample), 0)
        log_1mpi_sample_mean = np.mean(np.log(1 - pi_sample), 0)
        self.assert_allclose(e_log_pi1, log_pi_sample_mean,  10 * tol_scale)
        self.assert_allclose(e_log_pi2, log_1mpi_sample_mean,  10 * tol_scale)

        # Z (nu)
        z_sample_mean = np.mean(z_sample, 0)
        self.assert_allclose(z_sample_mean, nu_moment, 20 * tol_scale)

        # Mu (phi)
        a_sample_mean = np.mean(a_sample, 0)
        a2_sample = np.sum(a_sample ** 2, 1)
        a2_sample_mean = np.mean(a2_sample, 0)
        self.assert_allclose(a_sample_mean, phi_moment1, 30 * tol_scale)
        self.assert_allclose(a2_sample_mean, phi_moment2, 30 * tol_scale)

    def test_entropy(self):
        # Autograd has not implemented certain entropy functions, so test our own.
        self.assert_allclose(
            np.sum(osp.stats.beta.entropy(tau[:, 0], tau[:, 1])),
            finite_lib.pi_entropy(tau), tol=1e-12)

        self.assert_allclose(
            np.sum(osp.stats.norm.entropy(phi_mu, np.sqrt(phi_var_expanded))),
            finite_lib.phi_entropy(phi_var, D), tol=1e-12)

        self.assert_allclose(
            np.sum(osp.stats.bernoulli.entropy(nu)),
            finite_lib.nu_entropy(nu), tol=1e-12)

    def test_e_log_lik(self):
        e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment = \
            finite_lib.get_moments(tau, nu, phi_mu, phi_var)

        ell_old = finite_lib.exp_log_likelihood_old(
            nu_moment, phi_moment1, phi_moment2, e_log_pi1, e_log_pi2,
            sigmas, X, alpha)

        ell = finite_lib.exp_log_likelihood(
            nu_moment, phi_moment1, phi_moment2, e_log_pi1, e_log_pi2,
            sigmas, X, alpha)

        print('New vs old log likelihood:')
        print(ell_old)
        print(ell)


    def test_old_elbo(self):
        print('New vs old elbo:')
        print(finite_lib.compute_elbo_old(tau, nu, phi_mu, phi_var, X, sigmas, alpha)[0])
        print(finite_lib.compute_elbo(tau, nu, phi_mu, phi_var, X, sigmas, alpha))
        print('--------')

    # def test_term1(self):
    #     term1_sample = (alpha/K_approx - 1) * np.sum(np.mean(np.log(pi_sample), 1))
    #     self.assertAlmostEqual(term1_sample, elbo_term1, places = 1)
    #
    # def test_term3(self):
    #     term3_sample = -np.trace(np.einsum('jkl,jil->ki', a_sample, a_sample)\
    #                     /n_test_samples)
    #     self.assertAlmostEqual(elbo_term3, \
    #         -K_approx*D/2.*np.log(2.*np.pi*sigma_A) + term3_sample/(2*sigma_A), \
    #         places = 1)
    #
    # def test_term4(self):
    #     term4_sample = 0
    #     for n in range(num_samples):
    #         tmp = X[n,:][:,np.newaxis] \
    #                 - np.einsum('il,ikl->kl', z_sample[n,:,:], a_sample)
    #
    #         term4_sample += np.einsum('jk,jk', tmp, tmp)/n_test_samples
    #
    #     self.assertAlmostEqual(elbo_term4, \
    #         -num_samples*D/2.*np.log(2.*np.pi*sigma_eps) - term4_sample/(2.*sigma_eps), \
    #         places = 1)


if __name__ == '__main__':
    unittest.main()
