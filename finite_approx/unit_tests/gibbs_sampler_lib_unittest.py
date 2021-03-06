#!/usr/bin/python3

import unittest
from finite_approx.gibbs_sampler_lib import GibbsSampler, update_inv_var, \
                                            flip_z2, flip_zx
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
        pi_true, z_true, a_true, x =  vi.generate_data(
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
        sigma_a = 0.3
        z = np.random.binomial(1, 0.5, (5,3))
        x = np.random.normal(0,1,(5,2))
        k_approx = np.shape(z)[1]
        x_n = np.shape(z)[0]

        for n in range(x_n):
            for k in range(k_approx):
                z_flip = deepcopy(z)
                z_flip[n,k] = 1 - z[n,k]

                var_inv = np.linalg.inv(np.dot(z.T, z) \
                        + sigma_eps/sigma_a * np.eye(k_approx))
                logdet_var = -np.log(np.linalg.det(var_inv))

                var_flip = np.dot(z_flip.T, z_flip) \
                        + sigma_eps/sigma_a * np.eye(k_approx)

                inv_var_flip, logdet_var_flip = update_inv_var(\
                                z_flip, var_inv, logdet_var, sigma_eps, sigma_a, n, k)

                # checking rank-1 update of inverse
                self.assertTrue(np.allclose(inv_var_flip, np.linalg.inv(var_flip)))
                self.assertTrue(np.allclose(logdet_var_flip, -np.log(np.linalg.det(inv_var_flip))))

                test_z2 = np.dot(z.T, z)
                flip_z2(z_flip, test_z2, n, k)
                # checking rank-1 update of Z.T * Z
                assert(np.allclose(test_z2, np.dot(z_flip.T, z_flip)))

                test_zx = np.dot(z.T, x)
                flip_zx(z_flip, x, test_zx, n, k)
                assert(np.allclose(test_zx, np.dot(z_flip.T, x)))



if __name__ == '__main__':
    unittest.main()
