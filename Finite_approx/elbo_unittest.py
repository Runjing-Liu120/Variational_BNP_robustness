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

# Set a seed to avoid flaky tests.
np.random.seed(42)

# draw data
num_samples = 10
x_dim = 2
k_inf = 3
sigma_a = 1.2
sigma_eps  = 2
alpha = 10

pi, Z, mu, A, x = \
    finite_lib.generate_data(num_samples, x_dim, k_inf, sigma_a, sigma_eps, alpha)

# initialize variational parameters
k_approx = k_inf

tau = np.full((k_approx, 2), 5.)
tau[:, 1] *= 2.

# Bernoulli parameter for z_nk
nu =  np.random.uniform(0.01, 0.99, (num_samples, k_approx))

# kth mean (x_dim dim vector) in kth column
phi_mu = np.random.normal(0, 1, [x_dim, k_approx])
phi_var = np.ones(k_approx)
phi_var_expanded = np.array([ phi_var for d in range(x_dim)])

# compute elbo
sigmas = {'eps': sigma_eps, 'A': sigma_a}
# elbo = finite_lib.compute_elbo(tau, nu, phi_mu, phi_var, x, sigmas, alpha)

# generate samples to compute means
def generate_parameter_draws(nu, phi_mu, phi_var_expanded, tau, n_test_samples):
    z_sample = np.random.binomial(
        1, nu, size=(n_test_samples, nu.shape[0], nu.shape[1]))

    # A version of phi_var with the same shape as phi_mu
    a_sample = np.random.normal(
        phi_mu, phi_var_expanded,
        (n_test_samples, phi_mu.shape[0], phi_mu.shape[1]))

    # The numpy beta draws seem to actually hit zero and one, unlike scipy.
    pi_sample = osp.stats.beta.rvs(tau[:, 0], tau[:, 1],
                                   size=(n_test_samples, tau.shape[0]))

    return z_sample, a_sample, pi_sample


n_test_samples = 10**5
tol_scale = 1. / np.sqrt(n_test_samples)
z_sample, a_sample, pi_sample = \
    generate_parameter_draws(nu, phi_mu, phi_var_expanded, tau, n_test_samples)

print('1 / sqrt(num test draws) = %0.6f' % tol_scale)


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


class TestElboComputation(unittest.TestCase):
    def assert_allclose(self, x, y, tol=1e-12):
        self.assertTrue(np.allclose(x, y, tol))

    def test_moments(self):
        e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment = \
            finite_lib.get_moments(tau, nu, phi_mu, phi_var)

        # pi (tau)
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
            finite_lib.phi_entropy(phi_var, x_dim), tol=1e-12)

        self.assert_allclose(
            np.sum(osp.stats.bernoulli.entropy(nu)),
            finite_lib.nu_entropy(nu), tol=1e-12)

    def test_e_log_lik_old(self):
        e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment = \
            finite_lib.get_moments(tau, nu, phi_mu, phi_var)

        ell_old = finite_lib.exp_log_likelihood_old(
            nu_moment, phi_moment1, phi_moment2, e_log_pi1, e_log_pi2,
            sigmas, x, alpha)

        ell = finite_lib.exp_log_likelihood(
            nu_moment, phi_moment1, phi_moment2, e_log_pi1, e_log_pi2,
            sigmas, x, alpha)

        print('New vs old log likelihood:')
        print(ell_old)
        print(ell)


    def test_e_log_lik(self):
        n_test_samples = 10000

        for i in range(5):
            tau, nu, phi_mu, phi_var = \
                finite_lib.initialize_parameters(num_samples, x_dim, k_approx)
            phi_var_expanded = np.array([ phi_var for d in range(x_dim)])

            z_sample, a_sample, pi_sample = \
                generate_parameter_draws(nu, phi_mu, phi_var_expanded, \
                                         tau, n_test_samples)

            sample_e_log_lik = [
                log_lik(x, z_sample[n, :, :], a_sample[n, :, :], pi_sample[n, :],
                        sigma_eps, sigma_a, alpha, k_approx) \
                for n in range(n_test_samples) ]

            e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment = \
                finite_lib.get_moments(tau, nu, phi_mu, phi_var)

            ell = finite_lib.exp_log_likelihood(
                nu_moment, phi_moment1, phi_moment2, e_log_pi1, e_log_pi2,
                sigmas, x, alpha)

            print('---------')
            print(np.mean(sample_e_log_lik) - ell)
            print(np.std(sample_e_log_lik) / np.sqrt(n_test_samples))

    def test_old_elbo(self):
        print('New vs old elbo:')
        print(finite_lib.compute_elbo_old(tau, nu, phi_mu, phi_var, x, sigmas, alpha)[0])
        print(finite_lib.compute_elbo(tau, nu, phi_mu, phi_var, x, sigmas, alpha))
        print('--------')

    # def test_term1(self):
    #     term1_sample = (alpha/k_approx - 1) * np.sum(np.mean(np.log(pi_sample), 1))
    #     self.assertAlmostEqual(term1_sample, elbo_term1, places = 1)
    #
    # def test_term3(self):
    #     term3_sample = -np.trace(np.einsum('jkl,jil->ki', a_sample, a_sample)\
    #                     /n_test_samples)
    #     self.assertAlmostEqual(elbo_term3, \
    #         -k_approx*x_dim/2.*np.log(2.*np.pi*sigma_a) + term3_sample/(2*sigma_a), \
    #         places = 1)
    #
    # def test_term4(self):
    #     term4_sample = 0
    #     for n in range(num_samples):
    #         tmp = x[n,:][:,np.newaxis] \
    #                 - np.einsum('il,ikl->kl', z_sample[n,:,:], a_sample)
    #
    #         term4_sample += np.einsum('jk,jk', tmp, tmp)/n_test_samples
    #
    #     self.assertAlmostEqual(elbo_term4, \
    #         -num_samples*x_dim/2.*np.log(2.*np.pi*sigma_eps) - term4_sample/(2.*sigma_eps), \
    #         places = 1)


if __name__ == '__main__':
    unittest.main()
