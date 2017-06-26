# Unittest for

import numpy as np
import unittest
import finite_approx.functional_perturbation_lib as fun_pert

from scipy.stats import beta


class TestFunctionalPerturbation(unittest.TestCase):
    def u(self, x):
        return(3 * x)

    def test_integration(self):
        k_approx = 3
        alpha = 10
        tau = np.random.uniform(0.5, 2.0, [k_approx, 2])

        n_grid = 10000
        test = fun_pert.exp_pi_prior_perturbed(tau, alpha, k_approx, self.u, n_grid)

        x = np.linspace(0.001, 0.999, n_grid)
        delta = x[1] - x[0]

        prior_density = beta.pdf(x, 1, alpha/k_approx)

        for k in range(k_approx):
            variational_density = beta.pdf(x, tau[k,0], tau[k,1])

            truth = np.sum(variational_density * (self.u(x) + prior_density)\
                                * delta, 0)
            self.assertTrue(np.abs(test[k] - truth) <= 10**(-8))
