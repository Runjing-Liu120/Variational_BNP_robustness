# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:54:08 2017

@author: Haiying Liang
"""
from autograd import grad
import autograd.numpy as np
import autograd.scipy as sp
from copy import deepcopy

# np.random.seed(4535)

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
        dum1 = 2.*np.sum(np.sum(nu_moment[n,i] * nu_moment[n,j] * np.dot(phi_moment1[:,i],phi_moment1[:,j]) for i in range(j)) for j in range(K))
        dum2 = np.dot(nu_moment[n,:] , phi_moment2 )
        
        dum3 = -2. * np.dot(X[n,:], np.dot(phi_moment1, nu_moment[n,:]))
        
        # dum4 = np.dot(X[n,:], X[n,:])
        Normal_X_sum += dum1 + dum2 + dum3
        
    Normal_X = -1/(2*sigma_eps)*Normal_X_sum
    
    y = beta_lh + bern_lh + Normal_A + Normal_X
    return(y)

def nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape): 
    
    s_eps = sigmas['eps']
    K = Data_shape['K']
    N = Data_shape['N']
    D = Data_shape['D']
    script_V = np.zeros([N, K])
    
    for n in range(N):
        for k in range(K):
                            
            nu_term1 = sp.special.digamma(tau[k,0]) - sp.special.digamma(tau[k,1])  
            
            nu_term2 = (1. / (2. * s_eps)) * (phi_var[k]*D + np.dot(phi_mu[:,k], phi_mu[:,k]))
            
            
            nu_term3 = (1./s_eps) * np.dot(phi_mu[:, k], X[n, :] - np.dot(phi_mu, nu[n, :]) + nu[n,k] * phi_mu[:, k])
            
            #if k==4 and n==3:
            #    print(nu_term2,nu_term3)

            #explit calculation of Term3
            dum = 0
            for l in range(K):
                if (l != k):
                    dum += nu[n,l] * phi_mu[:,l]
        
            nu_term3_alt = (1 / s_eps) * np.dot(phi_mu[:,k], X[n,:] - dum)
            
            if np.abs(nu_term3 - nu_term3_alt)>10**(-10):
                print(nu_term3-nu_term3_alt)
                print('calculation of nu_term3 is off')
                input('paused')
                
                
            #if k==0 and n==0:
            #    print(nu_term1, nu_term2, nu_term3)
                
            script_V[n,k] = nu_term1 - nu_term2 + nu_term3
    
    return(script_V)
    
def tau_updates(tau, nu, alpha, Data_shape): 
    K = Data_shape['K']
    N = Data_shape['N']

    tau[:,0] = alpha/K + np.sum(nu,0)
    tau[:,1] = N  + 1 - np.sum(nu,0)
    
    return(tau)

def phi_updates(nu, phi_mu, phi_var, X, sigmas, Data_shape):
    
    s_eps = sigmas['eps']
    s_A = sigmas['A']
    D = Data_shape['D']
    N = Data_shape['N']
    K = Data_shape['K']
    
    phi_mu_copy = deepcopy(phi_mu)
    phi_var_copy = deepcopy(phi_var)
    
    for k in range(K):
        phi_var[k] = (1/s_A + np.sum(nu[:, k]) / s_eps)**(-1) 
               
        phi_summation = 0
        phi_summation_alt = 0
        for n in range(N):
            phi_dum1 = X[n, :] - np.dot(phi_mu_copy, nu[n, :]) + nu[n, k] * phi_mu_copy[:, k]
            phi_summation += nu[n,k]*phi_dum1
        
            dum1 = 0
            for l in range(K):
                if (l != k):
                    dum1 += nu[n,l] * phi_mu_copy[:,l]
            phi_summation_alt += nu[n,k] * (X[n,:] - dum1)
            
            if np.linalg.norm(phi_summation - phi_summation_alt)>=10**(-10):
                print('error in phi_mu updates')
            
        phi_mu[:,k] = (1 / s_eps) * phi_summation * (1/s_A + np.sum(nu[:, k]) / s_eps)**(-1)
                 
    return(phi_mu, phi_var)
'''''''''
DRAW DATA
'''''''''
# np.random.seed(4535)
# np.random.seed(24)
#np.random.seed(12321)


Num_samples = 500 # sample size
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
A = np.random.multivariate_normal(mu, sigma_A*np.identity(D), K_inf)
# A = np.array([[10,10], [-10,10]])

# draw noise
epsilon = np.random.multivariate_normal(np.zeros(D), sigma_eps*np.identity(D), Num_samples)

# the observed data
X = np.dot(Z,A) + epsilon



Data_shape = {'D':D, 'N': Num_samples , 'K':K_approx}
sigmas = {'eps': sigma_eps, 'A': sigma_A}

# initialization for cavi updates
#tau = np.random.uniform(10,100,[K_approx,2])
tau = np.ones([K_approx,2])*1000
tau[:,1] = (tau[:,0] - Pi*tau[:,0])/Pi

#nu = np.random.uniform(0,1,[Num_samples,K_approx])
nu = deepcopy(Z)

phi_mu = deepcopy(A.T)
#phi_mu = np.random.normal(0,1,[D,K_approx])
phi_var = np.ones(K_approx)

"""
# testing nu updates
d_exp_log_LH = grad(exp_log_likelihood, 0)

nu_moment = deepcopy(nu)
phi_moment1 = deepcopy(phi_mu)
phi_moment2 = np.diag(np.dot(phi_mu.T, phi_mu) + D * phi_var)
E_log_pi1 = sp.special.digamma(tau[:,0]) - sp.special.digamma(tau[:,0] + tau[:,1]) 
E_log_pi2 = sp.special.digamma(tau[:,1]) - sp.special.digamma(tau[:,0] + tau[:,1]) 

script_V_AG = d_exp_log_LH(nu_moment, phi_moment1, phi_moment2, \
                       E_log_pi1, E_log_pi2, Data_shape, sigmas, X, alpha)
script_V = nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape)

print(script_V[0:5,:])
print(script_V_AG[0:5,:])
print(np.sum(np.abs(script_V - script_V_AG)))
"""
"""
# testing tau updates
d_tau1 = grad(exp_log_likelihood, 3)
d_tau2 = grad(exp_log_likelihood, 4)

nu_moment = deepcopy(nu)
phi_moment1 = deepcopy(phi_mu)
phi_moment2 = np.diag(np.dot(phi_mu.T, phi_mu) + D * phi_var)
E_log_pi1 = sp.special.digamma(tau[:,0]) - sp.special.digamma(tau[:,0] + tau[:,1]) 
E_log_pi2 = sp.special.digamma(tau[:,1]) - sp.special.digamma(tau[:,0] + tau[:,1]) 

tau1_AG = d_tau1(nu_moment, phi_moment1, phi_moment2, \
                       E_log_pi1, E_log_pi2, Data_shape, sigmas, X, alpha) + 1
tau2_AG = d_tau2(nu_moment, phi_moment1, phi_moment2, \
                       E_log_pi1, E_log_pi2, Data_shape, sigmas, X, alpha) + 1

tau_cavi = tau_updates(tau, nu, alpha, Data_shape)

print(tau_cavi.T)
print(tau1_AG)
print(tau2_AG)
"""
# testing phi updates
d_phi1  = grad(exp_log_likelihood, 1)
d_phi2 = grad(exp_log_likelihood, 2)

nu_moment = deepcopy(nu)

phi_moment1 = deepcopy(phi_mu)
phi_moment2 = np.diag(np.dot(phi_mu.T, phi_mu) + D * phi_var)

E_log_pi1 = sp.special.digamma(tau[:,0]) - sp.special.digamma(tau[:,0] + tau[:,1]) 
E_log_pi2 = sp.special.digamma(tau[:,1]) - sp.special.digamma(tau[:,0] + tau[:,1]) 

phi1_AG = d_phi1(nu_moment, phi_moment1, phi_moment2, \
                       E_log_pi1, E_log_pi2, Data_shape, sigmas, X, alpha) 
phi2_AG = d_phi2(nu_moment, phi_moment1, phi_moment2, \
                       E_log_pi1, E_log_pi2, Data_shape, sigmas, X, alpha) 
                 
phi_var_AG = -1/(2.*phi2_AG)
phi_mu_AG = np.dot(phi1_AG, np.diag(phi_var_AG))

[phi_mu_cavi, phi_var_cavi] = phi_updates(nu, phi_mu, phi_var, X, sigmas, Data_shape)

print(phi_mu_AG)
print(phi_mu_cavi)
print(phi_var_AG)
print(phi_var_cavi)
