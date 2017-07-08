# Unittest for

import numpy as np
import unittest
import finite_approx.functional_perturbation_lib as fun_pert
import finite_approx.valez_finite_VI_lib as vi

from scipy.special import digamma, betaln
from scipy.stats import beta
from scipy import integrate

import sys
sys.path.append('../../LinearResponseVariationalBayes.py')
from VariationalBayes.ParameterDictionary import ModelParamsDict
from VariationalBayes.Parameters import ScalarParam


np.random.seed(534)

num_samples = 50 # sample size
x_d = 2 # dimension

alpha = 10
sigma_a = 3.0 ** 2
sigma_eps = 1.0 ** 2 # variance of noise
k_approx = 3

# generate data
_, _, _, x = vi.generate_data\
                    (num_samples, x_d, k_approx, sigma_a, sigma_eps, alpha)
# VI paramters
tau_init, nu_init, phi_mu_init, phi_var_init = \
    vi.initialize_parameters(num_samples, x_d, k_approx)

vb_model = vi.set_ibp_vb_model(num_samples, x_d, k_approx)

# set parameters
vb_model['phi'].set_vector(np.hstack([np.ravel(phi_mu_init.T), phi_var_init]))
# the integration apparently is better for larger tau
vb_model['pi'].set_vector(np.ravel(np.random.uniform(20, 40, (k_approx, 2)))) #
vb_model['nu'].set_vector(np.ravel(nu_init))

# consolidate hyper parameters
hyper_params = ModelParamsDict('hyper_params')
hyper_params.push_param(ScalarParam('alpha', lb = 0.0))
hyper_params.push_param(ScalarParam('var_a', lb = 0.0))
hyper_params.push_param(ScalarParam('var_eps', lb = 0.0))

hyper_params['alpha'].set(alpha)
hyper_params['var_a'].set(sigma_a)
hyper_params['var_eps'].set(sigma_eps)


class TestFunctionalPerturbation(unittest.TestCase):
    def u(self, x):
        return(3.0 * x)

    def test_integration(self):
        tau = vb_model['pi'].alpha.get()
        test = fun_pert.compute_e_pi_prior_perturbed\
                            (tau, alpha, k_approx, self.u, n_grid = 10**6)

        for k in range(k_approx):
            integrand = lambda x : beta.pdf(x, tau[k,0], tau[k,1]) * \
                        np.log(self.u(x) + beta.pdf(x, alpha/k_approx, 1))

            truth = integrate.quad(integrand, 0,1)
            self.assertTrue(np.abs(test[k] - truth[0]) <= 10**(-8))

    def test_moment(self):
        tau = vb_model['pi'].alpha.get()
        # test against moment (without pertubation)
        true_moment = (alpha/k_approx - 1) * (digamma(tau[:,0]) - \
                        digamma(tau[:,0] + tau[:,1])) -\
                        betaln(alpha/k_approx, 1)
        test_moment = fun_pert.compute_e_pi_prior_perturbed\
                            (tau, alpha, k_approx, n_grid = 10**6)
                            # here, recall that the default perturbation is 0
        self.assertTrue(np.all(np.abs(true_moment - test_moment) <= 10**(-8)))

    def test_elbo(self):
        # without perturbation, check against old elbo computations
        test_elbo = fun_pert.compute_elbo_perturbed(x, vb_model, hyper_params)
        true_elbo = vi.compute_elboII(x, vb_model, hyper_params) - \
                    k_approx * betaln(alpha/k_approx, 1)
        print(test_elbo)
        print(true_elbo)
        self.assertTrue(np.all(np.abs(test_elbo - true_elbo) <= 10**(-8)))
