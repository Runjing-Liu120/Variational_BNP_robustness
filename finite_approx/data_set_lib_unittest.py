#!/usr/bin/python3

import unittest
from finite_approx.data_set_lib import DataSet
import finite_approx.valez_finite_VI_lib as vi
import finite_approx.generic_optimization_lib as packing
from copy import deepcopy
import numpy as np


class TestDataSet(unittest.TestCase):
    def assert_allclose(self, x, y, tol=1e-12):
        self.assertTrue(np.allclose(x, y, tol))

    def test_basics(self):
        alpha = 10
        num_samples = 5
        x_dim = 3
        sigma_a = 3.0 ** 2
        sigma_eps = 1.0 ** 2
        k_inf = 4

        pi, Z, mu, A, X = vi.generate_data(
            num_samples, x_dim, k_inf, sigma_a, sigma_eps, alpha)

        k_approx = k_inf # variational truncation

        tau_init, nu_init, phi_mu_init, phi_var_init = \
            vi.initialize_parameters(num_samples, x_dim, k_approx)

        params_init = packing.pack_params(
            deepcopy(tau_init), deepcopy(phi_mu_init),
            deepcopy(phi_var_init), deepcopy(nu_init))

        hyper_params = packing.pack_hyperparameters(alpha, sigma_a, sigma_eps)

        data_set = DataSet(X, k_approx, alpha, sigma_eps, sigma_a)
        tau, phi_mu, phi_var, nu = data_set.unpack_params(params_init)
        self.assert_allclose(tau, tau_init)
        self.assert_allclose(nu, nu_init)
        self.assert_allclose(phi_mu, phi_mu_init)
        self.assert_allclose(phi_var, phi_var_init)

        # Just check that these run.
        kl = data_set.wrapped_kl(params_init)
        grad = data_set.get_kl_grad(params_init)
        hess = data_set.get_kl_hessian(params_init)
        hvp = data_set.get_kl_hvp(params_init, grad)
        self.assert_allclose(hvp, np.matmul(hess, grad))
        kl_sens_hess = data_set.get_kl_sens_hess(params_init, hyper_params)

        # Just check that these run.
        data_set.run_cavi(tau, nu, phi_mu, phi_var, max_iter=2)
        data_set.run_newton_tr(params_init, maxiter=2)
        data_set.get_prediction(params_init)


if __name__ == '__main__':
    unittest.main()
