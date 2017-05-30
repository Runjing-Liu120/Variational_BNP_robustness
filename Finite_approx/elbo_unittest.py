#!/usr/bin/python3

# Alarmingly, this fails with python2!

# First unit test-- check CAVI updates

import unittest
import valez_finite_VI_lib as finite_lib
from autograd import grad
import autograd.numpy as np
import autograd.scipy as sp
from copy import deepcopy


# draw data
Num_samples = 10
D = 2
K_inf = 3
sigma_A = 1.2
sigma_eps  = 2
alpha = 10

Pi, Z, mu, A, X = \
    finite_lib.generate_data(Num_samples, D, K_inf, sigma_A, sigma_eps)

# initialize variational parameters
K_approx = deepcopy(K_inf)
tau, nu, phi_mu, phi_var = \
    finite_lib.initialize_parameters(Num_samples, D, K_approx)

# compute elbo
sigmas = {'eps': sigma_eps, 'A': sigma_A}
elbo, elbo_term1, elbo_term2, elbo_term3, elbo_term4, elbo_term5, elbo_term6, elbo_term7 = \
    finite_lib.compute_elbo(tau, nu, phi_mu, phi_var, X, sigmas, alpha)

# generate samples to compute means
n_test_samples = 10**6

z_sample = np.zeros((Num_samples, K_approx, n_test_samples))
for n in range(Num_samples):
    for k in range(K_approx):
        z_sample[n,k,:] = np.random.binomial(1, nu[n,k], size = n_test_samples)

A_sample = np.zeros((K_approx, D,n_test_samples))
for k in range(K_approx):
    for d in range(D):
        A_sample[k,d,:] = np.random.normal(phi_mu[d,k], phi_var[k], n_test_samples)

Pi_sample = np.zeros((K_approx, n_test_samples))

for k in range(K_approx):
    Pi_sample[k,:] = np.random.beta(tau[k,0], tau[k,1], n_test_samples)

class TestElboComputation(unittest.TestCase):
    def test_term1(self):
        term1_sample = (alpha/K_approx - 1) * np.sum(np.mean(np.log(Pi_sample), 1))
        self.assertAlmostEqual(term1_sample, elbo_term1, places = 1)

    def test_term3(self):
        term3_sample = -np.trace(np.einsum('jkl,jil->ki', A_sample, A_sample)\
                        /n_test_samples)
        self.assertAlmostEqual(elbo_term3, \
            -K_approx*D/2.*np.log(2.*np.pi*sigma_A) + term3_sample/(2*sigma_A), \
            places = 1)
            
    def test_term4(self):
        term4_sample = 0
        for n in range(Num_samples):
            tmp = X[n,:][:,np.newaxis] \
                    - np.einsum('il,ikl->kl', z_sample[n,:,:], A_sample)

            term4_sample += np.einsum('jk,jk', tmp, tmp)/n_test_samples

        self.assertAlmostEqual(elbo_term4, \
            -Num_samples*D/2.*np.log(2.*np.pi*sigma_eps) - term4_sample/(2.*sigma_eps), \
            places = 1)


if __name__ == '__main__':
    unittest.main()
