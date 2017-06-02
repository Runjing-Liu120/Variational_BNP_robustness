#!/usr/bin/python

import unittest
from valez_finite_VI_lib import initialize_parameters
from generic_optimization_lib import \
    unpack_tau, unpack_phi_mu, unpack_phi_var, unpack_nu, \
    pack_tau, pack_phi_mu, pack_phi_var, pack_nu, \
    unpack_params, pack_params
from copy import deepcopy

import numpy as np


class TestParameterPacking(unittest.TestCase):
    def assert_allclose(self, x, y):
        self.assertTrue(np.allclose(x, y))

    def test_parameter_packing(self):
        num_samples = 10
        d = 2
        k_approx = 3

        tau, nu, phi_mu, phi_var = \
            initialize_parameters(num_samples, d, k_approx)

        self.assert_allclose(unpack_tau(pack_tau(tau), k_approx, d), tau)
        self.assert_allclose(unpack_phi_mu(
            pack_phi_mu(phi_mu), k_approx, d), phi_mu)
        self.assert_allclose(unpack_phi_var(pack_phi_var(phi_var)), phi_var)
        self.assert_allclose(unpack_nu(pack_nu(nu), num_samples, k_approx), nu)

        params = pack_params(
            deepcopy(tau), deepcopy(phi_mu), deepcopy(phi_var), deepcopy(nu))
        tau0, phi_mu0, phi_var0, nu0 = unpack_params(
            params, k_approx, d, num_samples)

        self.assert_allclose(tau0, tau)
        self.assert_allclose(phi_mu0, phi_mu)
        self.assert_allclose(phi_var0, phi_var)
        self.assert_allclose(nu0, nu)

    def test_hyperparameter_packing(self):
        alpha = 2.5
        sigma_A = 10
        sigma_eps = 0.5
        hyper_params = pack_hyperparameters(alpha, sigma_A, sigma_eps)
        alpha0, sigma_A0, sigma_eps0 = unpack_hyperparameters(hyper_params)
        self.assertAlmostEqual(alpha0, alpha)
        self.assertAlmostEqual(sigma_A0, sigma_A)
        self.assertAlmostEqual(sigma_eps0, sigma_eps)



if __name__ == '__main__':
    unittest.main()
