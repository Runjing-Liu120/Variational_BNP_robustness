#!/usr/bin/python

import unittest
from valez_finite_VI_lib import initialize_parameters, get_moments
import generic_optimization_lib as packing
from copy import deepcopy

import numpy as np


class TestParameterPacking(unittest.TestCase):
    def assert_allclose(self, x, y):
        self.assertTrue(np.allclose(x, y))

    def test_parameter_packing(self):
        num_samples = 10
        d = 4
        k_approx = 3

        tau, nu, phi_mu, phi_var = \
            initialize_parameters(num_samples, d, k_approx)

        self.assert_allclose(packing.unpack_tau(pack_tau(tau), k_approx), tau)
        self.assert_allclose(packing.unpack_phi_mu(
            pack_phi_mu(phi_mu), k_approx, d), phi_mu)
        self.assert_allclose(
            packing.unpack_phi_var(packing.pack_phi_var(phi_var)), phi_var)
        self.assert_allclose(
            packing.unpack_nu(packing.pack_nu(nu), num_samples, k_approx), nu)

        params = packing.pack_params(
            deepcopy(tau), deepcopy(phi_mu), deepcopy(phi_var), deepcopy(nu))
        tau0, phi_mu0, phi_var0, nu0 = packing.unpack_params(
            params, k_approx, d, num_samples)

        self.assert_allclose(tau0, tau)
        self.assert_allclose(phi_mu0, phi_mu)
        self.assert_allclose(phi_var0, phi_var)
        self.assert_allclose(nu0, nu)

    def test_hyperparameter_packing(self):
        alpha = 2.5
        sigma_A = 10
        sigma_eps = 0.5
        hyper_params = packing.pack_hyperparameters(alpha, sigma_A, sigma_eps)
        alpha0, sigma_A0, sigma_eps0 = \
            packing.unpack_hyperparameters(hyper_params)
        self.assertAlmostEqual(alpha0, alpha)
        self.assertAlmostEqual(sigma_A0, sigma_A)
        self.assertAlmostEqual(sigma_eps0, sigma_eps)

    def test_parameter_packing(self):
        num_samples = 10
        x_dim = 4
        k_approx = 3

        tau, nu, phi_mu, phi_var = \
            initialize_parameters(num_samples, x_dim, k_approx)
        params = packing.flatten_params(tau, nu, phi_mu, phi_var)

        tau0, phi_mu0, phi_var0, nu0 = \
            packing.unflatten_params(params, k_approx, x_dim, num_samples)

        self.assert_allclose(tau, tau0)
        self.assert_allclose(nu, nu0)
        self.assert_allclose(phi_mu, phi_mu0)
        self.assert_allclose(phi_var, phi_var0)

    def test_moment_packing(self):
        x_dim = 4
        k_approx = 3

        tau, nu, phi_mu, phi_var = \
            initialize_parameters(5, x_dim, k_approx)
        e_log_pi, e_log_pi2, e_mu, phi_moment2, nu_moment = \
            get_moments(tau, nu, phi_mu, phi_var)

        param = packing.pack_moments(e_log_pi, e_mu)
        e_log_pi0, e_mu0 = packing.unpack_moments(param, k_approx, x_dim)

        self.assert_allclose(e_log_pi, e_log_pi0)
        self.assert_allclose(e_mu, e_mu0)


if __name__ == '__main__':
    unittest.main()
