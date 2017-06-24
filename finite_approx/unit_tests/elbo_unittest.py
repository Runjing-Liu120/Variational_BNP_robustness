#!/usr/bin/python3

import unittest
import finite_approx.valez_finite_VI_lib as vi
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
x_dim = 4
k_inf = 3
sigma_a = 1.2
sigma_eps  = 2
alpha = 10

pi, Z, A, x = \
    vi.generate_data(num_samples, x_dim, k_inf, sigma_a, sigma_eps, alpha)

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


class TestElboComputation(unittest.TestCase):
    def assert_allclose(self, x, y, tol=1e-12, msg=''):
        self.assertTrue(np.allclose(x, y, tol),
                        msg='{}\nx !~ y where\nx = {}\ny = {}\ntol = {}'.format(
                        msg, x, y, tol))

    def test_moments(self):
        n_test_samples = 10**5
        tol_scale = 1. / np.sqrt(n_test_samples)
        z_sample, a_sample, pi_sample = vi.generate_parameter_draws(
            nu, phi_mu, phi_var_expanded, tau, n_test_samples)

        print('1 / sqrt(num test draws) = %0.6f' % tol_scale)

        e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment = \
            vi.get_moments(tau, nu, phi_mu, phi_var)

        # pi (tau)
        self.assert_allclose(np.mean(pi_sample, 0),
                             tau[:,0] / (tau[:,0] + tau[:,1]), 10 * tol_scale)
        log_pi_sample_mean = np.mean(np.log(pi_sample), 0)
        log_1mpi_sample_mean = np.mean(np.log(1 - pi_sample), 0)
        self.assert_allclose(e_log_pi1, log_pi_sample_mean,  10 * tol_scale)
        self.assert_allclose(e_log_pi2, log_1mpi_sample_mean,  10 * tol_scale)

        # Z (nu)
        z_sample_mean = np.mean(z_sample, 0)
        self.assert_allclose(z_sample_mean, nu_moment, 20 * tol_scale, 'z')

        # Mu (phi)
        a_sample_mean = np.mean(a_sample, 0)
        #a2_sample = np.sum(a_sample ** 2, 1)
        #a2_sample_mean = np.mean(a2_sample, 0)
        a2_sample_mean = np.mean(a_sample ** 2, 0)
        self.assert_allclose(a_sample_mean, phi_moment1, 40 * tol_scale, 'a')
        self.assert_allclose(a2_sample_mean, phi_moment2, 40 * tol_scale, 'a2')

    def test_entropy(self):
        # Autograd has not implemented certain entropy functions, so test our own.
        self.assert_allclose(
            np.sum(osp.stats.beta.entropy(tau[:, 0], tau[:, 1])),
            vi.pi_entropy(tau), tol=1e-12)

        self.assert_allclose(
            np.sum(osp.stats.norm.entropy(phi_mu, np.sqrt(phi_var_expanded))),
            vi.phi_entropy(phi_var, x_dim), tol=1e-12)

        self.assert_allclose(
            np.sum(osp.stats.bernoulli.entropy(nu)),
            vi.nu_entropy(nu), tol=1e-12)

    def test_e_log_lik(self):
        n_test_samples = 10000

        # Our expected log likelihood should only differ from a sample average
        # of the generated log likelihood by a constant as the parameters
        # vary.  Check this using num_param different random parameters.
        num_params = 5
        ell_by_param = np.full(num_params, float('nan'))
        sample_ell_by_param = np.full(num_params, float('nan'))
        standard_error = 0.
        for i in range(num_params):
            tau, nu, phi_mu, phi_var = \
                vi.initialize_parameters(num_samples, x_dim, k_approx)
            phi_var_expanded = np.array([ phi_var for d in range(x_dim)])

            z_sample, a_sample, pi_sample = \
                vi.generate_parameter_draws(nu, phi_mu, phi_var_expanded, \
                                            tau, n_test_samples)

            sample_e_log_lik = [
                vi.log_lik(x, z_sample[n, :, :], a_sample[n, :, :],
                           pi_sample[n, :], sigma_eps, sigma_a, alpha,
                           k_approx) \
                for n in range(n_test_samples) ]

            sample_ell_by_param[i] = np.mean(sample_e_log_lik)
            standard_error = \
                np.max([ standard_error,
                         np.std(sample_e_log_lik) / np.sqrt(n_test_samples) ])

            e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment = \
                vi.get_moments(tau, nu, phi_mu, phi_var)

            ell_by_param[i] = vi.exp_log_likelihood(
                nu_moment, phi_moment1, phi_moment2, e_log_pi1, e_log_pi2,
                sigma_a, sigma_eps, x, alpha)

        print('Mean log likelihood standard error: %0.5f' % standard_error)
        self.assertTrue(np.std(ell_by_param - sample_ell_by_param) < \
                        3. * standard_error)

"""## testing the VB library
## define variational parameters
vb_params = ModelParamsDict(name = 'vb_params')
# stick lengths
vb_params.push_param(DirichletParamArray(name='pi', shape=(k_inf, 2)))
# variational means
vb_params.push_param(MVNArray(name='phi', shape=(k_inf, x_dim)))
# responsibilities
vb_params.push_param(ArrayParam(name = 'nu', shape = (num_samples, k_inf), \
                        lb = 0.0, ub = 1.0))

# initialize
vb_params['phi'].set_vector(np.hstack([np.ravel(phi_mu.T), phi_var]))
vb_params['pi'].set_vector(np.ravel(tau))
vb_params['nu'].set_vector(np.ravel(nu))
"""


"""class TestElboComputation_VB_lib(unittest.TestCase):
    def assert_allclose(self, x, y, tol=1e-12, msg=''):
        self.assertTrue(np.allclose(x, y, tol),
                        msg='{}\nx !~ y where\nx = {}\ny = {}\ntol = {}'.format(
                        msg, x, y, tol))

    def test_moments(self):
        n_test_samples = 10**5
        tol_scale = 1. / np.sqrt(n_test_samples)
        z_sample, a_sample, pi_sample = vi.generate_parameter_draws(
            nu, phi_mu, phi_var_expanded, tau, n_test_samples)

        print('1 / sqrt(num test draws) = %0.6f' % tol_scale)

        e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment = \
            vi.get_moments(tau, nu, phi_mu, phi_var)

        # pi (tau)
        self.assert_allclose(np.mean(pi_sample, 0),
                             tau[:,0] / (tau[:,0] + tau[:,1]), 10 * tol_scale)
        log_pi_sample_mean = np.mean(np.log(pi_sample), 0)
        log_1mpi_sample_mean = np.mean(np.log(1 - pi_sample), 0)
        self.assert_allclose(e_log_pi1, log_pi_sample_mean,  10 * tol_scale)
        self.assert_allclose(e_log_pi2, log_1mpi_sample_mean,  10 * tol_scale)

        # Z (nu)
        z_sample_mean = np.mean(z_sample, 0)
        self.assert_allclose(z_sample_mean, nu_moment, 20 * tol_scale, 'z')

        # Mu (phi)
        a_sample_mean = np.mean(a_sample, 0)
        #a2_sample = np.sum(a_sample ** 2, 1)
        #a2_sample_mean = np.mean(a2_sample, 0)
        a2_sample_mean = np.mean(a_sample ** 2, 0)
        self.assert_allclose(a_sample_mean, phi_moment1, 40 * tol_scale, 'a')
        self.assert_allclose(a2_sample_mean, phi_moment2, 40 * tol_scale, 'a2')

    def test_entropy(self):
        # Autograd has not implemented certain entropy functions, so test our own.
        self.assert_allclose(
            np.sum(osp.stats.beta.entropy(tau[:, 0], tau[:, 1])),
            vi.pi_entropy(tau), tol=1e-12)

        self.assert_allclose(
            np.sum(osp.stats.norm.entropy(phi_mu, np.sqrt(phi_var_expanded))),
            vi.phi_entropy(phi_var, x_dim), tol=1e-12)

        self.assert_allclose(
            np.sum(osp.stats.bernoulli.entropy(nu)),
            vi.nu_entropy(nu), tol=1e-12)

    def test_e_log_lik(self):
        n_test_samples = 10000

        # Our expected log likelihood should only differ from a sample average
        # of the generated log likelihood by a constant as the parameters
        # vary.  Check this using num_param different random parameters.
        num_params = 5
        ell_by_param = np.full(num_params, float('nan'))
        sample_ell_by_param = np.full(num_params, float('nan'))
        standard_error = 0.
        for i in range(num_params):
            tau, nu, phi_mu, phi_var = \
                vi.initialize_parameters(num_samples, x_dim, k_approx)
            phi_var_expanded = np.array([ phi_var for d in range(x_dim)])

            z_sample, a_sample, pi_sample = \
                vi.generate_parameter_draws(nu, phi_mu, phi_var_expanded, \
                                            tau, n_test_samples)

            sample_e_log_lik = [
                vi.log_lik(x, z_sample[n, :, :], a_sample[n, :, :],
                           pi_sample[n, :], sigma_eps, sigma_a, alpha,
                           k_approx) \
                for n in range(n_test_samples) ]

            sample_ell_by_param[i] = np.mean(sample_e_log_lik)
            standard_error = \
                np.max([ standard_error,
                         np.std(sample_e_log_lik) / np.sqrt(n_test_samples) ])

            e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment = \
                vi.get_moments(tau, nu, phi_mu, phi_var)

            ell_by_param[i] = vi.exp_log_likelihood(
                nu_moment, phi_moment1, phi_moment2, e_log_pi1, e_log_pi2,
                sigma_a, sigma_eps, x, alpha)

        print('Mean log likelihood standard error: %0.5f' % standard_error)
        self.assertTrue(np.std(ell_by_param - sample_ell_by_param) < \
                        3. * standard_error)
"""

if __name__ == '__main__':
    unittest.main()
