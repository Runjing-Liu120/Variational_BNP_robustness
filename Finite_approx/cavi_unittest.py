#!/usr/bin/python3

# Alarmingly, this fails with python2!

# First unit test-- check CAVI updates

import unittest
from valez_finite_VI_lib import *
from autograd import grad
import autograd.numpy as np
import autograd.scipy as sp
from copy import deepcopy

from scipy.stats import bernoulli

# Draw Data
Num_samples = 5 # sample size
D = 2 # dimension
# so X will be a n\times D matrix

K_inf = 3 # take to be large for a good approximation to the IBP
K_approx = deepcopy(K_inf)

alpha = 2 # IBP parameter

# Parameters to draw A from MVN
sigma_A = 100
sigma_eps = 1 # variance of noise

Pi, Z, mu, A, X = generate_data(Num_samples, D, K_inf, sigma_A, sigma_eps)

Data_shape = {'D':D, 'N': Num_samples , 'K':K_approx}
sigmas = {'eps': sigma_eps, 'A': sigma_A}


class TestCaviUpdates(unittest.TestCase):
    def test_silly(self):
        self.assertEqual(2,2)
        self.assertFalse(2==3)

        for i in range(5):
            self.assertEqual(i,i)

    def test_nu_updates(self):
        # initialization for cavi updates
        tau = np.random.uniform(10,100,[K_approx,2])
        digamma_tau = sp.special.digamma(tau)
        nu = np.random.uniform(0,1,[Num_samples,K_approx])

        phi_mu = np.random.normal(0,1,[D,K_approx])
        phi_var = np.ones(K_approx)

        # autodiff
        d_exp_log_LH = grad(exp_log_likelihood, 0)

        # compute required moments
        phi_moment1 = deepcopy(phi_mu)
        phi_moment2 = np.diag(np.dot(phi_mu.T, phi_mu) + D * phi_var)
        E_log_pi1 = sp.special.digamma(tau[:,0]) - sp.special.digamma(tau[:,0] + tau[:,1])
        E_log_pi2 = sp.special.digamma(tau[:,1]) - sp.special.digamma(tau[:,0] + tau[:,1])


        for n in range(Num_samples):
            for k in range(K_approx):

                nu_moment = deepcopy(nu)
                script_V_AG = d_exp_log_LH(nu_moment, phi_moment1, phi_moment2, \
                               E_log_pi1, E_log_pi2, Data_shape, sigmas, X, alpha)
                nu_AG = 1/(1 + np.exp(-script_V_AG))

                nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, n, k, digamma_tau)


                # print(np.abs(nu[n,k] - nu_AG[n,k]))
                self.assertAlmostEqual(nu[n,k] , nu_AG[n,k])

    def test_tau_updates(self):
        # initialization for cavi updates
        tau = np.random.uniform(10,100,[K_approx,2])
        nu = np.random.uniform(0,1,[Num_samples,K_approx])

        phi_mu = np.random.normal(0,1,[D,K_approx])
        phi_var = np.ones(K_approx)

        # calling autodiff
        d_tau1 = grad(exp_log_likelihood, 3)
        d_tau2 = grad(exp_log_likelihood, 4)

        # computing moments
        nu_moment = deepcopy(nu)
        phi_moment1 = deepcopy(phi_mu)
        phi_moment2 = np.diag(np.dot(phi_mu.T, phi_mu) + D * phi_var)
        E_log_pi1 = sp.special.digamma(tau[:,0]) - sp.special.digamma(tau[:,0] + tau[:,1])
        E_log_pi2 = sp.special.digamma(tau[:,1]) - sp.special.digamma(tau[:,0] + tau[:,1])

        # computing updates
        tau1_AG = d_tau1(nu_moment, phi_moment1, phi_moment2, \
                               E_log_pi1, E_log_pi2, Data_shape, sigmas, X, alpha) + 1
        tau2_AG = d_tau2(nu_moment, phi_moment1, phi_moment2, \
                               E_log_pi1, E_log_pi2, Data_shape, sigmas, X, alpha) + 1
        tau_AG = np.array([tau1_AG, tau2_AG])

        tau_updates(tau, nu, alpha)

        assert np.shape(tau)==np.shape(tau_AG.T)

        #print('results from cavi update: \n', tau.T)
        #print('results from autograd: ')
        #print(tau1_AG)
        #print(tau2_AG)

        self.assertTrue(np.allclose(tau, tau_AG.T))

    def test_phi_updates(self):
        # initialization for cavi updates
        tau = np.random.uniform(10,100,[K_approx,2])
        nu = np.random.uniform(0,1,[Num_samples,K_approx])

        phi_mu = np.random.normal(0,1,[D,K_approx])
        phi_var = np.ones(K_approx)

        # calling autodiff
        d_phi1  = grad(exp_log_likelihood, 1)
        d_phi2 = grad(exp_log_likelihood, 2)

        # compute moments

        E_log_pi1 = sp.special.digamma(tau[:,0]) - sp.special.digamma(tau[:,0] + tau[:,1])
        E_log_pi2 = sp.special.digamma(tau[:,1]) - sp.special.digamma(tau[:,0] + tau[:,1])

        for k in range(K_approx):
            nu_moment = deepcopy(nu)
            phi_moment1 = deepcopy(phi_mu)
            phi_moment2 = np.diag(np.dot(phi_mu.T, phi_mu) + D * phi_var)

            # compute autograd updates
            phi1_AG = d_phi1(nu_moment, phi_moment1, phi_moment2, \
                               E_log_pi1, E_log_pi2, Data_shape, sigmas, X, alpha)
            phi2_AG = d_phi2(nu_moment, phi_moment1, phi_moment2, \
                               E_log_pi1, E_log_pi2, Data_shape, sigmas, X, alpha)

            # convert to standard parametrization
            phi_var_AG = -1/(2.*phi2_AG)
            phi_mu_AG = np.dot(phi1_AG, np.diag(phi_var_AG))


            phi_updates(nu, phi_mu, phi_var, X, sigmas, k) # cavi updates


            #print('mean computed by autodiff: \n', phi_mu_AG[:,k])
            #print('mean computed by cavi: \n', phi_mu[:,k])
            #print('variance computed by autodiff: ', phi_var_AG[k])
            #print('variance computed by cavi    : ', phi_var[k])
            #print('\n')

            self.assertTrue(np.allclose(phi_mu_AG[:,k], phi_mu[:,k]))
            self.assertTrue(np.allclose(phi_var_AG[k], phi_var[k]))

class TestElboComputation(unittest.TestCase):
    # initialize variational parameters
    tau = np.random.uniform(10,100,[K_approx,2])
    nu = np.random.uniform(0,1,[Num_samples,K_approx])

    phi_mu = np.random.normal(0,1,[D,K_approx])
    phi_var = np.ones(K_approx)

    z_test = np.zeros((Num_samples, K_approx, 10**4))
    for n in range(Num_samples):
        for k in range(K_approx):
            z_test[n,k,:] = bernoulli.rvs(nu[n,k], 10**4)

    A_test = np.zeros((K_approx, 2,10**4))
    for k in range(K_approx):
        for d in range(D):
            A_test[k,d,:] = np.random.normal(phi_mu[d,k], sigma_A, 10**4)

    Pi = 0

    print(np.shape(A_test))
    print(np.shape(z_test))

    # compute elbo
    [elbo,elbo_Term1,elbo_Term2,elbo_Term3,elbo_Term4,elbo_Term5,elbo_Term6,\
         elbo_Term7] = compute_elbo(tau, nu, phi_mu, phi_var, X, sigmas, alpha)





if __name__ == '__main__':
    unittest.main()
