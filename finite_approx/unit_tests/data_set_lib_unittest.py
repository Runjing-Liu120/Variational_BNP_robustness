#!/usr/bin/python3

import unittest
from finite_approx.LRVB_lib import DataSet, log_q_a, log_q_z, log_q_pi
import finite_approx.valez_finite_VI_lib as vi
import finite_approx.generic_optimization_lib as packing
from copy import deepcopy
import numpy as np
import scipy as sp

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

class TestVariationalLh(unittest.TestCase):
    def test_normal_lh(self):
        k = 3
        d = 2
        a = np.random.normal(0,1,(k,d))
        phi_mu = np.random.normal(0,1,(k,d))
        phi_var = np.random.uniform(0,1,size = k)

        true_norm_pdf = 0
        for i in range(k):
            true_norm_pdf += sp.stats.multivariate_normal.logpdf\
                    (a[i,:], mean = phi_mu[i,:], cov = phi_var[i] * np.identity(d))
        test_norm_pdf = log_q_a(a, phi_mu, phi_var)

        self.assertTrue( np.abs(true_norm_pdf - test_norm_pdf) <= 10**(-8))

    def test_beta_lh(self):
        k = 3
        pi = np.random.uniform(0,1,size = k)
        tau = np.random.uniform(0,10, size = (k, 2))

        true_beta_pdf = 0
        for i in range(k):
            true_beta_pdf += sp.stats.beta.logpdf(pi[i], tau[i,0], tau[i,1])

        test_beta_pdf = log_q_pi(pi, tau)

        self.assertTrue( np.abs(true_beta_pdf - test_beta_pdf) <= 10**(-8))

if __name__ == '__main__':
    unittest.main()
