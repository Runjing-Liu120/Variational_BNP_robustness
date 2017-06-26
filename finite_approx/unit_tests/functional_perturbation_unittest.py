# Unittest for

import numpy as np
import unittest
import finite_approx.functional_perturbation_lib as fun_pert

from scipy.special import digamma
from scipy.stats import beta
from scipy import integrate

np.random.seed(534)

k_approx = 3
alpha = 10
tau = np.random.uniform(20, 40, [k_approx, 2])

n_grid = 10000

class TestFunctionalPerturbation(unittest.TestCase):
    def u(self, x):
        return(3.0 * x)

    def test_integration(self):

        test = fun_pert.compute_e_pi_prior_perturbed\
                            (tau, alpha, k_approx, self.u, n_grid)

        x = np.linspace(0.001, 0.999, n_grid)
        delta = x[1] - x[0]

        prior_density = beta.pdf(x, 1, alpha/k_approx)

        for k in range(k_approx):
            integrand = lambda x : beta.pdf(x, tau[k,0], tau[k,1]) * \
                        np.log(self.u(x) + beta.pdf(x, alpha/k_approx, 1))

            truth = integrate.quad(integrand, 0,1)
            self.assertTrue(np.abs(test[k] - truth[0]) <= 10**(-8))

    def test_moment(self):
        # test against moment
        true_moment = (alpha/k_approx - 1) * digamma(tau[:,0]) - \
                        digamma(tau[:,0] + tau[:,1])
        test_moment = fun_pert.compute_e_pi_prior_perturbed\
                            (tau, alpha, k_approx, n_grid = n_grid)
        print('look here')
        print(true_moment)
        print(test_moment)
