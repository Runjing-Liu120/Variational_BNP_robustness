# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:10:41 2017

@author: Haiying Liang
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from copy import deepcopy

from beta_process_vb_lib import *

# Data_shape: D,N,K

def phi_updates(nu, phi_mu, phi_var, X, sigmas, Data_shape, n, k):
    
    sigma_eps = sigmas['eps']
    sigma_A = sigmas['A']
    D = Data_shape['D']
    N = Data_shape['N']
    K = Data_shape['K']

    phi_var[k] = 1/(1/sigma_A + np.sum(nu[:, k]) / sigma_eps) * np.identity(D)
    
    
    Summation = 0
    for n in range(N):
        dum1 = X[n, :] - np.dot(phi_mu, nu[n, :]) + nu[n, k] * phi_mu[:, k]
        Summation += nu[n,k]*dum1
    
    #    dum1 = 0
    #    for l in range(K):
    #        if (l != k):
    #            dum1 += nu[n,l] * phi_mu[:,l]

    #    Summation += nu[n,k] * (X[n,:] - dum1)

    phi_mu[:,k] = 1 / sigma_eps * Summation\
             * 1 / (1 / sigma_A + np.sum(nu[:, k]) / sigma_eps)
    
    return(phi_var, phi_mu)

def nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape, n, k): 
    sigma_eps = sigmas['eps']
    K = Data_shape['K']

    Term1 = sp.special.digamma(tau[k,0]) - sp.special.digamma(tau[k,1])  
    
    Term2 = 1 / (2 * sigma_eps) * (np.trace(phi_var[k]) + \
        np.dot(phi_mu[:, k], phi_mu[:, k]))
    
    Term3 = 1/sigma_eps * np.dot(phi_mu[:, k], X[n, :] - \
                    np.dot(phi_mu, nu[n, :]) + nu[n, k] * phi_mu[:, k])
    
    #explit calculation of Term3
    #dum = 0
    #for l in range(K):
    #    if (l != k):
    #        dum += nu[n,l] * phi_mu[:,l]

    #Term3 = 1 / sigma_eps * np.dot( phi_mu[:, k], X[n,:] - dum)
    
    script_V = Term1 - Term2 + Term3
    
    nu[n,k] = 1/(1+np.exp(-script_V))
    
    return(nu)
    
    
def tau_updates(tau, nu, alpha, Data_shape, n, k): 
    K = Data_shape['K']
    N = Data_shape['N']

    tau[k,0] = alpha/K + np.sum(nu[:,k])
    tau[k,1] = N + 1  - np.sum(nu[:,k])
    
    return(tau)

def Elbo(tau, nu, phi_mu, phi_var, X, sigmas, Data_shape, alpha):
    
    sigma_eps = sigmas['eps']
    sigma_A = sigmas['A']
    D = Data_shape['D']
    N = Data_shape['N']
    K = Data_shape['K']

    # bernoulli terms 
    Term1 = np.sum( np.log(alpha/K) + (alpha/K - 1)*(sp.special.digamma(tau[k,0]) \
                  - sp.special.digamma(tau[k,0] + tau[k, 1])) for k in range(K)) 
    
    Term2 = 0
    for k in range(K):
        for n in range(N): 
            Term2 += nu[n,k] * sp.special.digamma(tau[k,0]) + (1-nu[n,k])*\
                    sp.special.digamma(tau[k,1]) \
                    - sp.special.digamma(tau[k,0]+tau[k,1])
    
    Term3 = np.sum(-D/2*np.log(2*np.pi*sigma_A) - 1/(2*sigma_A) *\
        (np.trace(phi_var[k]) + \
        np.dot(phi_mu[:, k] , phi_mu[:, k])) for k in range(K) )
    
    Term4 = 0
    for n in range(N):
        summ1 = np.sum(nu[n,k] * np.dot(phi_mu[:,k], X[n,:]) for k in range(K))
        summ2 = np.sum(
            np.sum(
                nu[n,k1] * nu[n,k2] * np.dot(phi_mu[:,k1], phi_mu[:,k2]) \
                for k1 in range(k2)) for k2 in range(K))
        summ3 = np.sum(nu[n,k] * (np.trace(phi_var[k]) + \
            np.dot(phi_mu[:,k], phi_mu[:,k])) for k in range(K))

        Term4 += - D / 2 * np.log(2 * np.pi * sigma_eps) - \
            1 / (2 * sigma_eps) * (
                np.dot(X[n,:], X[n,:]) - 2 * summ1 + 2 * summ2 + summ3)
    
    Term5 = np.sum(sp.special.betaln(tau[:,0],tau[:,1]) - \
        (tau[:,0] - 1) * sp.special.digamma(tau[:,0]) - \
        (tau[:,1] - 1) * sp.special.digamma(tau[:,1]) + \
        (tau[:,0] + tau[:,1] -2) *  sp.special.digamma(tau[:,0] + tau[:,1]))

    Term6 = np.sum(1 / 2 * np.log((2 * np.pi * np.exp(1)) ** D * \
        np.linalg.det(phi_var[k])) for k in range(K))

    Term7 = np.sum(np.sum( -np.log(nu ** nu) - np.log((1-nu) ** (1-nu)) ))

    elbo = Term1 + Term2 + Term3 + Term4 + Term5 + Term6 + Term7

    return(elbo, Term1, Term2, Term3, Term4, Term5, Term6, Term7)
