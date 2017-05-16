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


'''''''''
DRAW DATA
'''''''''
# np.random.seed(4535)
# np.random.seed(24)
# np.random.seed(12321) # this is interesting


Num_samples = 500 # sample size
D = 2 # dimension
# so X will be a n\times D matrix

K_inf = 3 # take to be large for a good approximation to the IBP


alpha = 10 # IBP parameter
Pi = np.ones(K_inf) * .8
Z = np.zeros([Num_samples,K_inf])

# Parameters to draw A from MVN
mu = np.zeros(D)
sigma_A = 100

sigma_eps = .1 # variance of noise

# Draw Z from truncated stick breaking process
for k in range(K_inf):
    # Pi[k] = np.random.beta(alpha/K_inf,1)
    for n in range(Num_samples):
        Z[n,k] = np.random.binomial(1,Pi[k])

        
print(Z[0:10,:])

# Draw A from multivariate normal
A = np.random.multivariate_normal(mu, sigma_A*np.identity(D), K_inf)
# A = np.array([[10,10], [-10,10]])

# draw noise
epsilon = np.random.multivariate_normal(np.zeros(D), sigma_eps*np.identity(D), Num_samples)

# the observed data
X = np.dot(Z,A) + epsilon


'''
SET VARIATIONAL PARAMETERS
'''
K_approx = deepcopy(K_inf) # variational truncation

# don't set seed gives good init. 

#tau = np.random.uniform(0,1,[K_approx,2]) # tau1, tau2 -- beta parameters for v
tau = np.ones([K_approx,2])*1000
tau[:,1] = (tau[:,0] - Pi*tau[:,0])/Pi

#nu = np.ones([Num_samples, K_approx]) *0.0
nu =  np.random.uniform(0,1,[Num_samples,K_approx]) # Bernoulli parameter for z_nk
#nu = deepcopy(Z)

#phi_mu = np.random.normal(0,1,[D,K_approx]) # kth mean (D dim vector) in kth column
phi_mu = deepcopy(A).T
phi_var = np.ones(K_approx)

nu_init = np.round(nu*(nu>=0.9) + nu*(nu<=0.1)) + nu*(nu>=0.1)*(nu<=0.9)

iterations = 1000
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

# print(nu)
for i in range(iterations):
    
    nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape)

    phi_updates(nu, phi_mu, phi_var, X, sigmas, Data_shape)
        
    tau_updates(tau, nu, alpha, Data_shape)    
    
    
    
    round_nu = np.round(nu*(nu>=0.9) + nu*(nu<=0.1)) + nu*(nu>=0.1)*(nu<=0.9)
    
    [elbo[i],elbo_Term1[i],elbo_Term2[i],elbo_Term3[i],elbo_Term4[i],elbo_Term5[i],elbo_Term6[i],elbo_Term7[i]] \
        = Elbo(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape, alpha)
        
    
    print('iteration: ', i)
    print('elbo: ', elbo[i])
    #print(round_nu[0:10,:])
    #print(nu[0:10,:])
    print('l1 error: ', np.sum(abs(Z-nu))/np.size(Z[:]) )
    
    if (i>0) & (elbo[i] < elbo[i-1]): 
        print('eblo decreased!')
        break
    
    if np.abs(elbo[i]-elbo[i-1]) <= 10**(-8):
        break

Pi_computed = tau[:,0]/(tau[:,0] + tau[:,1])
print('Z \n', Z[0:10,:])
print('round_nu \n', round_nu[0:10,:])


#index = np.arange(Num_samples)
#index[(np.abs(nu[:,2]- Z[:,2])>=10**(-16))]

## match correct rows. 

#sort_index = np.ones(K_approx) + K_approx
#for k in range(K_approx):
#    best_index = np.argsort(abs(Pi_computed - Pi[k]))[0]
#
#    i = 1
#    while any(best_index == sort_index): 
#        best_index = np.argsort(abs(Pi_computed - Pi[k]))[i]
#        i += 1
#        
#    sort_index[k] = best_index
#
#sort_index = sort_index.astype(np.int64)
#
#Pi_computed = Pi_computed[sort_index]
#nu = nu[:, sort_index]
#round_nu = round_nu[:, sort_index]
#phi_mu = phi_mu[:, sort_index]
#
#
#print('l1 error: ', np.sum(abs(Z-nu))/np.size(Z[:]) )

print(np.sum(np.abs(Z[:,0] - nu[:,0]))/Num_samples)
print(np.sum(np.abs(Z[:,1] - nu[:,1]))/Num_samples)
print(np.sum(np.abs(Z[:,2] - nu[:,2]))/Num_samples)


plt.clf()
plt.plot(elbo)
plt.xlabel('iteration')
plt.ylabel('elbo')