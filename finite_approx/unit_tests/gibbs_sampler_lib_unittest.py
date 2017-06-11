#!/usr/bin/python3

import unittest
from finite_approx.gibbs_sampler_lib import GibbsSampler, update_inv_var
import finite_approx.valez_finite_VI_lib as vi
from copy import deepcopy
import numpy as np


class TestGibbsSampler(unittest.TestCase):
    def assert_allclose(self, x, y, tol=1e-12, msg=''):
        self.assertTrue(np.allclose(x, y, tol),
                        msg='{}\nx !~ y where\nx = {}\ny = {}\ntol = {}'.format(
                        msg, x, y, tol))

    def test_basics(self):
        alpha = 10
        num_samples = 10
        x_dim = 3
        sigma_a = 3.0 ** 2
        sigma_eps = 0.2 ** 2
        k_approx = 5
        pi_true, z_true, mu_true, a_true, x =  vi.generate_data(
            num_samples, x_dim, k_approx, sigma_a, sigma_eps, alpha)

        # Just test that it runs.
        gibbs_sampler = GibbsSampler(x, k_approx, alpha, sigma_eps, sigma_a)
        self.assertEqual(len(gibbs_sampler.a_draws), 0)
        self.assertEqual(len(gibbs_sampler.pi_draws), 0)
        self.assertEqual(len(gibbs_sampler.z_draws), 0)

        burnin = 5
        n_draws = 10
        gibbs_sampler.sample(burnin, n_draws)

        self.assertEqual(len(gibbs_sampler.a_draws), n_draws)
        self.assertEqual(len(gibbs_sampler.pi_draws), n_draws)
        self.assertEqual(len(gibbs_sampler.z_draws), n_draws)

class TestCollapsedGIbbsSampler(unittest.TestCase):
    def test_rank_1_update(self):
        sigma_eps = 0.2
        sigma_A = 0.3
        Z = np.random.binomial(1, 0.5, (5,3))
        K = np.shape(Z)[1]
        N = np.shape(Z)[0]

        for n in range(N):
            for k in range(K):
                Z_flip = deepcopy(Z)
                Z_flip[n,k] = 1 - Z[n,k]

                var_inv = np.linalg.inv(np.dot(Z.T, Z) \
                        + sigma_eps/sigma_A * np.eye(K))
                var_flip = np.dot(Z_flip.T, Z_flip) \
                        + sigma_eps/sigma_A * np.eye(K)

                inv_var_flip \
                    = update_inv_var(Z, var_inv, sigma_eps, sigma_A, n, k)

                self.assertTrue(np.allclose(inv_var_flip, np.linalg.inv(var_flip)))



if __name__ == '__main__':
    unittest.main()
