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



np.random.seed(41526)


Num_samples = 50 # sample size
D = 2 # dimension
# so X will be a n\times D matrix

K_inf = 5 # parameter in IBP model


alpha = 2 # IBP parameter
Pi = np.zeros(K_inf)
Z = np.zeros([Num_samples,K_inf])

# Parameters to draw A from MVN
mu = np.zeros(D)
sigma_A = 25

sigma_eps = 1 # variance of noise

# Draw Z from truncated stick breaking process
for k in range(K_inf):
    Pi[k] = np.random.beta(alpha/K_inf,1)
    for n in range(Num_samples):
        Z[n,k] = np.random.binomial(1,Pi[k])

#print(Z[0:10,:])
#input('pause' )

# Draw A from multivariate normal
A = np.random.multivariate_normal(mu, sigma_A*np.identity(D), K_inf)
#A = np.array([[100, 0], [0, 100]])

# draw noise
epsilon = np.random.multivariate_normal(np.zeros(D), sigma_eps*np.identity(D), Num_samples)

# the observed data
X = np.dot(Z,A) + epsilon

K_approx = deepcopy(K_inf) # variational truncation


# Variational parameters
#tau = np.random.uniform(0,1,[K_approx,2]) # tau1, tau2 -- beta parameters for v
tau = np.ones([K_approx,2])*1000
tau[:,1] = (tau[:,0] - Pi*tau[:,0])/Pi

nu = np.random.uniform(0,1,[Num_samples,K_approx]) # Bernoulli parameter for z_nk
#nu = deepcopy(Z)
#phi_mu = np.random.normal(0,1,[D,K_approx]) # kth mean (D dim vector) in kth column
phi_mu = deepcopy(A).T
phi_var = {k: .01*np.identity(D) for k in range(K_approx)}

#nu_init = deepcopy(nu)

iterations = 30
elbo = np.zeros(iterations)
elbo_Term1 = np.zeros(iterations)
elbo_Term2 = np.zeros(iterations)
elbo_Term3 = np.zeros(iterations)
elbo_Term4 = np.zeros(iterations)
elbo_Term5 = np.zeros(iterations)
elbo_Term6 = np.zeros(iterations)
elbo_Term7 = np.zeros(iterations)

Data_shape = {'D':D, 'N': Num_samples , 'K':K_approx}
sigmas = {'eps': sigma_eps, 'A': sigma_A}

print(nu)
for i in range(iterations):
    
    #[phi_var, phi_mu] = \
    #    phi_updates(nu, phi_mu, phi_var, X, sigmas, Data_shape)
            
    #tau = tau_updates(tau, nu, alpha, Data_shape)    

#    for k in range(K_approx): 
#        for n in range(N):
            #nu = nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape, n, k)
    
    nu = nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape)
    
    round_nu = np.round(nu*(nu>=0.9) + nu*(nu<=0.1)) + nu*(nu>=0.1)*(nu<=0.9)
    
    [elbo[i],elbo_Term1[i],elbo_Term2[i],elbo_Term3[i],elbo_Term4[i],elbo_Term5[i],elbo_Term6[i],elbo_Term7[i]] \
        = Elbo(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape, alpha)
        
    #print('elbo: ', elbo[i])
    print('iteration: ', i)
    #print(round_nu[0:10,:])
    #print(nu[0:10,:])
    print('l1 error: ', np.sum(abs(Z-nu))/np.size(Z[:]) )
    #if np.abs(elbo[i]-elbo[i-1]) <= 10^(-5):
    #    break
# tau[:,0]/(tau[:,0] + tau[:,1])

#print(Z[0:10,:])
plt.clf()
plt.plot(elbo)
plt.xlabel('iteration')
plt.ylabel('elbo')