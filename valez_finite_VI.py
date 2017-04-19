# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:00:06 2017

@author: Haiying Liang
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from copy import deepcopy

from valez_finite_VI_lib import *

np.random.seed(501)


N = 500 # sample size
D = 2 # dimension
# so X will be a n\times D matrix

K_inf = 5 # parameter in IBP model


alpha = 5 # IBP parameter
Pi = np.zeros(K_inf)
Z = np.zeros([N,K_inf])

# Parameters to draw A from MVN
mu = np.zeros(D)
sigma_A = 1

sigma_eps = 0.1 # variance of noise

# Draw Z from truncated stick breaking process
for k in range(K_inf):
    Pi[k] = np.random.beta(alpha/K_inf,1)
    for n in range(N):
        Z[n,k] = np.random.binomial(1,Pi[k])

print(Z[0:10,:])
input('pause' )
# Draw A from multivariate normal
A = np.random.multivariate_normal(mu, sigma_A*np.identity(D), K_inf)

# draw noise
epsilon = np.random.multivariate_normal(np.zeros(D), sigma_eps*np.identity(D), N)

# the observed data
X = np.dot(Z,A) + epsilon

K_approx = 5 # variational truncation


# Variational parameters
tau = np.random.uniform(10,100,[K_approx,2]) # tau1, tau2 -- beta parameters for v
# nu = np.random.uniform(0,1,[N,K_approx]) # Bernoulli parameter for z_nk
nu = deepcopy(Z)
phi_mu = np.random.normal(0,1,[D,K_approx]) # kth mean (D dim vector) in kth column
#phi_mu = deepcopy(A).T
phi_var = {k: np.identity(D) for k in range(K_approx)}



iterations = 100
elbo = np.zeros(iterations)
Term1 = np.zeros(iterations)
Term2 = np.zeros(iterations)
Term3 = np.zeros(iterations)
Term4 = np.zeros(iterations)
Term5 = np.zeros(iterations)
Term6 = np.zeros(iterations)
Term7 = np.zeros(iterations)

Data_shape = {'D':D, 'N': N , 'K':K_approx}
sigmas = {'eps': sigma_eps, 'A': sigma_A}

for i in range(iterations):
    for k in range(K_approx): 
        [phi_var, phi_mu] = \
            phi_updates(nu, phi_mu, phi_var, X, sigmas, Data_shape, n, k)
            
    for k in range(K_approx): 
        tau = tau_updates(tau, nu, alpha, Data_shape, n, k)    
    
    for n in range(N): 
        for k in range(K_approx):
            nu = nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape, n, k)

    round_nu = np.round(nu*(nu>=0.9) + nu*(nu<=0.1)) + nu*(nu>=0.1)*(nu<=0.9)
    
    [elbo[i],Term1[i],Term2[i],Term3[i],Term4[i],Term5[i],Term6[i],Term7[i]] \
        = Elbo(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape, alpha)
        
    print('elbo: ', elbo[i])
    print('iteration: ', i)
    print(round_nu[0:10,:])

plt.plot(elbo)
plt.xlabel('iteration')
plt.ylabel('elbo')