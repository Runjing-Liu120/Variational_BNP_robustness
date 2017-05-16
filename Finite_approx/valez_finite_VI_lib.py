# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:10:41 2017

@author: Haiying Liang
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from copy import deepcopy
import math
from beta_process_vb_lib import *

# Data_shape: D,N,K

def phi_updates(nu, phi_mu, phi_var, X, sigmas, Data_shape):
    
    s_eps = sigmas['eps']
    s_A = sigmas['A']
    D = Data_shape['D']
    N = Data_shape['N']
    K = Data_shape['K']
    
    for k in range(K):
        phi_var[k] = (1/s_A + np.sum(nu[:, k]) / s_eps)**(-1) 
               
        phi_summation = 0
        for n in range(N):
            phi_dum1 = X[n, :] - np.dot(phi_mu, nu[n, :]) + nu[n, k] * phi_mu[:, k]
            phi_summation += nu[n,k]*phi_dum1
        
        #    dum1 = 0
        #    for l in range(K):
        #        if (l != k):
        #            dum1 += nu[n,l] * phi_mu[:,l]
    
        #    phi_summation += nu[n,k] * (X[n,:] - dum1)
    
        phi_mu[:,k] = (1 / s_eps) * phi_summation\
                 * (1 / s_A + np.sum(nu[:, k]) / s_eps)**(-1)
    
def nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape): 
    
    s_eps = sigmas['eps']
    K = Data_shape['K']
    N = Data_shape['N']
    D = Data_shape['D']
    
    
    for n in range(N):
        for k in range(K):
                            
            nu_term1 = sp.special.digamma(tau[k,0]) - sp.special.digamma(tau[k,1])  
            
            nu_term2 = (1. / (2. * s_eps)) * (phi_var[k]*D + np.dot(phi_mu[:,k], phi_mu[:,k]))
            
            
            nu_term3 = (1./s_eps) * np.dot(phi_mu[:, k], X[n, :] - np.dot(phi_mu, nu[n, :]) + nu[n,k] * phi_mu[:, k])
            
            #if k==4 and n==3:
            #    print(nu_term2,nu_term3)

            #explit calculation of Term3
#            dum = 0
#            for l in range(K):
#                if (l != k):
#                    dum += nu[n,l] * phi_mu[:,l]
#        
#            nu_term3_alt = (1 / s_eps) * np.dot(phi_mu[:,k], X[n,:] - dum)
#            
#            if np.abs(nu_term3 - nu_term3_alt)>10**(-10):
#                print(nu_term3-nu_term3_alt)
#                print('calculation of nu_term3 is off')
#                input('paused')
                
                
            #if k==0 and n==0:
            #    print(nu_term1, nu_term2, nu_term3)
                
            script_V = nu_term1 - nu_term2 + nu_term3
            
            nu[n,k] = 1./(1.+np.exp(-script_V))
    
    
    
def tau_updates(tau, nu, alpha, Data_shape): 
    K = Data_shape['K']
    N = Data_shape['N']

    tau[:,0] = alpha/K + np.sum(nu,0)
    tau[:,1] = N  + 1 - np.sum(nu,0)
    

def Elbo(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape, alpha):
    
    sigma_eps = sigmas['eps']
    sigma_A = sigmas['A']
    D = Data_shape['D']
    N = Data_shape['N']
    K = Data_shape['K']

    # bernoulli terms 
    eblo_term1 = np.sum( np.log(alpha/K) + (alpha/K - 1)*(sp.special.digamma(tau[k,0]) \
                  - sp.special.digamma(tau[k,0] + tau[k, 1])) for k in range(K)) 
    
    eblo_term2 = 0
    for k in range(K):
        for n in range(N): 
            eblo_term2 += nu[n,k] * sp.special.digamma(tau[k,0]) + (1-nu[n,k])*\
                    sp.special.digamma(tau[k,1]) \
                    - sp.special.digamma(tau[k,0]+tau[k,1])
    
    eblo_term3 = np.sum(-D/2*np.log(2*np.pi*sigma_A) - 1/(2*sigma_A) *\
        (phi_var[k]*D + \
        np.dot(phi_mu[:, k] , phi_mu[:, k])) for k in range(K) )
    
    eblo_term4 = 0
    for n in range(N):
        summ1 = np.sum(nu[n,k] * np.dot(phi_mu[:,k], X[n,:]) for k in range(K))
        summ2 = np.sum(
            np.sum(
                nu[n,k1] * nu[n,k2] * np.dot(phi_mu[:,k1], phi_mu[:,k2]) \
                for k1 in range(k2)) for k2 in range(K))
        summ3 = np.sum(nu[n,k] * (D*phi_var[k] + \
            np.dot(phi_mu[:,k], phi_mu[:,k])) for k in range(K))

        eblo_term4 += - D / 2 * np.log(2 * np.pi * sigma_eps) - \
            1 / (2 * sigma_eps) * (
                np.dot(X[n,:], X[n,:]) - 2 * summ1 + 2 * summ2 + summ3)
    
    eblo_term5 = np.sum(sp.special.betaln(tau[:,0],tau[:,1]) - \
        (tau[:,0] - 1) * sp.special.digamma(tau[:,0]) - \
        (tau[:,1] - 1) * sp.special.digamma(tau[:,1]) + \
        (tau[:,0] + tau[:,1] -2) *  sp.special.digamma(tau[:,0] + tau[:,1]))

    eblo_term6 = np.sum(1 / 2 * np.log((2 * np.pi * np.exp(1)) ** D * \
        phi_var[k]**D) for k in range(K))

    eblo_term7 = np.sum(np.sum( -np.log(nu ** nu) - np.log((1-nu) ** (1-nu)) ))

    elbo = eblo_term1 + eblo_term2 + eblo_term3 + eblo_term4 + eblo_term5 + eblo_term6 + eblo_term7

    return(elbo, eblo_term1, eblo_term2, eblo_term3, eblo_term4, eblo_term5, eblo_term6, eblo_term7)
