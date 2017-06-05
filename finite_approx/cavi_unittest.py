#!/usr/bin/python3

# Alarmingly, this fails with python2!

# First unit test-- check CAVI updates

import unittest
import finite_approx.valez_finite_VI_lib as vi
from autograd import grad
import autograd.numpy as np
import autograd.scipy as sp
from copy import deepcopy

from scipy.stats import bernoulli


# Draw Data
num_samples = 5 # sample size
D = 4 # dimension
# so X will be a n\times D matrix

k_inf = 3 # take to be large for a good approximation to the IBP
k_approx = deepcopy(k_inf)

alpha = 2 # IBP parameter

# Parameters to draw A from MVN
sigma_a = 100
sigma_eps = 1 # variance of noise

Pi, Z, mu, A, X = vi.generate_data(num_samples, D, k_inf, sigma_a, sigma_eps, alpha)

Data_shape = {'D':D, 'N': num_samples , 'K':k_approx}
sigmas = {'eps': sigma_eps, 'A': sigma_a}


class TestCaviUpdates(unittest.TestCase):
    def assert_allclose(self, x, y, tol=1e-12, msg=''):
        self.assertTrue(np.allclose(x, y, tol),
                        msg='{}\nx !~ y where\nx = {}\ny = {}\ntol = {}'.format(
                        msg, x, y, tol))

    def test_nu_updates(self):
        # initialization for cavi updates
        tau = np.random.uniform(10,100,[k_approx,2])
        digamma_tau = sp.special.digamma(tau)
        nu = np.random.uniform(0,1,[num_samples,k_approx])

        phi_mu = np.random.normal(0,1,[D,k_approx])
        phi_var = np.ones(k_approx)

        # autodiff
        d_exp_log_LH = grad(vi.exp_log_likelihood, 0)

        # compute required moments
        e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment = \
            vi.get_moments(deepcopy(tau), deepcopy(nu),
                           deepcopy(phi_mu), deepcopy(phi_var))

        for n in range(num_samples):
            for k in range(k_approx):

                nu_moment = deepcopy(nu)
                script_V_AG = d_exp_log_LH(nu_moment, phi_moment1, phi_moment2, \
                               e_log_pi1, e_log_pi2, sigma_a, sigma_eps, X, alpha)
                nu_AG = 1 / (1 + np.exp(-script_V_AG))

                vi.nu_updates(tau, nu, phi_mu, phi_var, X, \
                              sigmas, n, k, digamma_tau)

                self.assert_allclose(nu[n, k], nu_AG[n, k])

    def test_tau_updates(self):
        # initialization for cavi updates
        tau = np.random.uniform(10,100,[k_approx,2])
        nu = np.random.uniform(0,1,[num_samples,k_approx])

        phi_mu = np.random.normal(0,1,[D,k_approx])
        phi_var = np.ones(k_approx)

        # calling autodiff
        d_tau1 = grad(vi.exp_log_likelihood, 3)
        d_tau2 = grad(vi.exp_log_likelihood, 4)

        # computing moments
        e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment = \
            vi.get_moments(deepcopy(tau), deepcopy(nu),
                           deepcopy(phi_mu), deepcopy(phi_var))

        # computing updates
        tau1_AG = d_tau1(nu_moment, phi_moment1, phi_moment2, \
                         e_log_pi1, e_log_pi2, sigma_a, sigma_eps, X, alpha) + 1
        tau2_AG = d_tau2(nu_moment, phi_moment1, phi_moment2, \
                         e_log_pi1, e_log_pi2, sigma_a, sigma_eps, X, alpha) + 1
        tau_AG = np.array([tau1_AG, tau2_AG])

        vi.tau_updates(tau, nu, alpha)

        assert np.shape(tau) == np.shape(tau_AG.T)

        self.assert_allclose(tau, tau_AG.T)

    def test_phi_updates(self):
        # initialization for cavi updates
        tau = np.random.uniform(10, 100, [k_approx, 2])
        nu = np.random.uniform(0, 1, [num_samples, k_approx])

        phi_mu = np.random.normal(0,1,[D,k_approx])
        phi_var = np.ones(k_approx)

        # calling autodiff
        d_phi1  = grad(vi.exp_log_likelihood, 1)
        d_phi2 = grad(vi.exp_log_likelihood, 2)

        # compute moments
        e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment = \
            vi.get_moments(deepcopy(tau), deepcopy(nu),
                           deepcopy(phi_mu), deepcopy(phi_var))

        phi_mu_original = deepcopy(phi_mu)
        phi_var_original = deepcopy(phi_var)
        for k in range(k_approx):
            phi_mu = deepcopy(phi_mu_original)
            phi_var = deepcopy(phi_var_original)

            # compute autograd updates
            phi1_AG = d_phi1(nu_moment, phi_moment1, phi_moment2, \
                               e_log_pi1, e_log_pi2, sigma_a, sigma_eps, X, alpha)
            phi2_AG = d_phi2(nu_moment, phi_moment1, phi_moment2, \
                               e_log_pi1, e_log_pi2, sigma_a, sigma_eps, X, alpha)

            # convert to standard parametrization
            phi_var_AG = -1 / (2. * phi2_AG)
            phi_mu_AG = phi1_AG * phi_var_AG

            vi.phi_updates(nu, phi_mu, phi_var, X, sigmas, k)

            self.assert_allclose(phi_mu_AG[:, k], phi_mu[:, k])
            for d in range(D):
                self.assert_allclose(phi_var_AG[d, k], phi_var[k])


class TestElboComputation(unittest.TestCase):
    # initialize variational parameters
    tau = np.random.uniform(10,100,[k_approx, 2])
    nu = np.random.uniform(0,1,[num_samples, k_approx])

    phi_mu = np.random.normal(0,1,[D,k_approx])
    phi_var = np.ones(k_approx)

    z_test = np.zeros((num_samples, k_approx, 10**4))
    for n in range(num_samples):
        for k in range(k_approx):
            z_test[n, k, :] = bernoulli.rvs(nu[n, k], 10**4)

    A_test = np.zeros((k_approx, D, 10**4))
    for k in range(k_approx):
        for d in range(D):
            A_test[k, d, :] = np.random.normal(phi_mu[d, k], sigma_a, 10**4)

    Pi = 0

    elbo = vi.compute_elbo(tau, nu, phi_mu, phi_var, X, sigmas, alpha)





if __name__ == '__main__':
    unittest.main()
