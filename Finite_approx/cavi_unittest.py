# First unit test-- check CAVI updates

import unittest
from valez_finite_VI_lib import *
from autograd import grad
import autograd.numpy as np
import autograd.scipy as sp
from copy import deepcopy

def exp_log_likelihood(nu_moment, phi_moment1, phi_moment2, \
                       E_log_pi1, E_log_pi2, Data_shape, sigmas, X, alpha):

    sigma_eps = sigmas['eps']
    sigma_A = sigmas['A']
    D = Data_shape['D']
    N = Data_shape['N']
    K = Data_shape['K']

    beta_lh = (alpha/K - 1.)*np.sum(E_log_pi1)
    bern_lh = np.sum(np.dot(nu_moment[n,:], E_log_pi1) \
                            + np.dot(1.-nu_moment[n,:], E_log_pi2) for n in range(N))
    Normal_A = -1/(2.*sigma_A) * np.sum(phi_moment2)

    Normal_X_sum = 0
    ## compute the data likelihood term
    for n in range(N):
        dum1 = 2.*np.sum(np.sum(nu_moment[n,i] * nu_moment[n,j] * np.dot(phi_moment1[:,i],phi_moment1[:,j]) \
                                for i in range(j)) for j in range(K))
        dum2 = np.dot(nu_moment[n,:] , phi_moment2 )

        dum3 = -2. * np.dot(X[n,:], np.dot(phi_moment1, nu_moment[n,:]))

        # dum4 = np.dot(X[n,:], X[n,:])
        Normal_X_sum += dum1 + dum2 + dum3

    Normal_X = -1/(2*sigma_eps)*Normal_X_sum

    y = beta_lh + bern_lh + Normal_A + Normal_X
    return(y)

# Draw Data
Num_samples = 10 # sample size
D = 2 # dimension
# so X will be a n\times D matrix

K_inf = 3 # take to be large for a good approximation to the IBP
K_approx = deepcopy(K_inf)

alpha = 2 # IBP parameter
Pi = np.zeros(K_inf)
Z = np.zeros([Num_samples,K_inf])

# Parameters to draw A from MVN
mu = np.zeros(D)
sigma_A = 100

sigma_eps = 1 # variance of noise

# Draw Z from truncated stick breaking process
for k in range(K_inf):
    Pi[k] = np.random.beta(alpha/K_inf,1)
    for n in range(Num_samples):
        Z[n,k] = np.random.binomial(1,Pi[k])

# Draw A from multivariate normal
# A = np.random.multivariate_normal(mu, sigma_A*np.identity(D), K_approx)
A = np.random.normal(0, np.sqrt(sigma_A), (K_approx,D))

# draw noise
# epsilon = np.random.multivariate_normal(np.zeros(D), sigma_eps*np.identity(D), Num_samples)
epsilon = np.random.normal(0, np.sqrt(sigma_eps), (Num_samples, D))

# the observed data
X = np.dot(Z,A) + epsilon


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

                nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, n, k)

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


            print('mean computed by autodiff: \n', phi_mu_AG[:,k])
            print('mean computed by cavi: \n', phi_mu[:,k])
            print('variance computed by autodiff: ', phi_var_AG[k])
            print('variance computed by cavi    : ', phi_var[k])
            print('\n')

            self.assertTrue(np.allclose(phi_mu_AG[:,k], phi_mu[:,k]))
            self.assertTrue(np.allclose(phi_var_AG[k], phi_var[k]))


if __name__ == '__main__':
    unittest.main()
