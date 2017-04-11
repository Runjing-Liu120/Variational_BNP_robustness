# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:56:55 2017

@author: Runjing Liu
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from copy import deepcopy

# np.random.seed(50)


N = 500 # sample size
D = 2 # dimension
# so X will be a n\times D matrix

K_inf = 5 # truncation parameter for sampling

alpha = 5 # IBP parameter
v = np.zeros(K_inf) # Beta sticks
Pi = np.zeros(K_inf) 
Z = np.zeros([N,K_inf])

# Parameters to draw A from MVN
mu = np.zeros(D)
sigma_A = 1

sigma_eps = 0.5 # variance of noise

# Draw Z from truncated stick breaking process
for k in range(K_inf):
    v[k] = np.random.beta(alpha,1)
    if k != 0:
        Pi[k] = Pi[k-1] * v[k]
    else: 
        Pi[k] = v[k]
    
    for n in range(N):
        Z[n,k] = np.random.binomial(1,Pi[k])

print(Z[0:10,:])

# Draw A from multivariate normal
A = np.random.multivariate_normal(mu, sigma_A*np.identity(D), K_inf)

# draw noise
epsilon = np.random.multivariate_normal(np.zeros(D), sigma_eps*np.identity(D), N)

# the observed data
X = np.dot(Z,A) + epsilon

K = 5 # variational truncation


# Variational parameters
tau = np.random.uniform(0,1,[K,2]) # tau1, tau2 -- beta parameters for v
nu = np.random.uniform(0,1,[N,K]) # Bernoulli parameter for z_nk
#nu = deepcopy(Z)
phi_mu = np.random.normal(0,1,[D,K]) # kth mean (D dim vector) in kth column
#phi_mu = deepcopy(A).T
phi_var = {k: np.identity(D) for k in range(K)}

# Update functions

def Phi_updates(nu,phi_mu,phi_var,X,sigma_A,sigma_eps,D,N,K,n,k):
    phi_var[k] = 1/(1/sigma_A + np.sum(nu[:,k])/sigma_eps) * np.identity(D)
    Summation = 0                        
    for n in range(N):
        dum1 = X[n,:] - np.dot(phi_mu, nu[n,:]) + nu[n,k]*phi_mu[:,k]
        Summation = nu[n,k] * dum1 + Summation
    
    phi_mu[:,k] = 1/sigma_eps * Summation * 1/(1/sigma_A + np.sum(nu[:,k])/sigma_eps)
    
    return(phi_var, phi_mu)


# [a,b] = Phi_updates(tau,nu,phi_mu, phi_var,X,sigma_A,sigma_eps,2)

# Multinomial lower bound computation
def multi_q(tau, k):
    q = np.zeros(k+1)
    log_q = np.zeros(k+1)
    for i in range(k+1):
        dum2 = np.sum(sp.special.digamma(tau[0:i,0])) 
        dum3 = np.sum(sp.special.digamma(tau[0:i+1,0] + tau[0:i+1,1]))
        q[i] = np.exp(sp.special.digamma(tau[i,1]) + dum2 - dum3)
        log_q[i] = sp.special.digamma(tau[i,1]) + dum2 - dum3
        
        
    q = q/np.sum(q) # probability of mutlinomial atoms
    log_q = log_q - sp.misc.logsumexp(log_q) # log probability 

    q_upper = [np.sum(q[m:]) for m in range(k+1)]
    return(q, q_upper, log_q)
    
def Exp_lowerbound(tau,k): # lower bound of expectation using multinomial
    [q, q_upper, log_q] = multi_q(tau, k)
    exp1 = np.dot(q,sp.special.digamma(tau[0:k+1,1]))
    
    exp2 = np.sum(q_upper[m+1]*sp.special.digamma(tau[m,0]) for m in range(k)) 
    
    exp3 = np.dot(q_upper,sp.special.digamma(tau[0:k+1,0]+tau[0:k+1,1]))
    
    exp4 = np.dot(q,log_q)
    
    return(exp1 + exp2 - exp3 - exp4)

def Exp_true(tau,k): # true expectation using MC estimate
    beta = np.zeros(k+1)
    B = np.zeros(10000)

    for i in range(10000):
        for r in range(k+1):
            beta[r] = np.random.beta(tau[r,0], tau[r,1])
        
        B[i] = np.log(1-np.prod(beta))

    Exp_true = np.average(B)
    return(Exp_true)
    
    
def Nu_updates(Expectation_k,tau,nu,phi_mu,phi_var,X,D,N,K,n,k):
    term1 = np.sum(sp.special.digamma(tau[0:k+1,0]) - sp.special.digamma(tau[0:k+1,0]+tau[0:k+1,1]) )
    
    # term2 = Exp_lowerbound(tau,k)
    term2 = Expectation_k
    
    term3 = 1/(2*sigma_eps) * (np.trace(phi_var[k]) + np.dot(phi_mu[:,k], phi_mu[:,k]))
    
    term4 = 1/sigma_eps* np.dot( phi_mu[:,k],X[n,:] - np.dot(phi_mu, nu[n,:]) + nu[n,k]*phi_mu[:,k])
    
    #dummy = 0
    #for l in range(K):
    #    if (l != k):
    #        dummy += nu[n,l] * phi_mu[:,l]

    #term4 = 1/sigma_eps* np.dot(phi_mu[:,k],X[n,:] - dummy)
    
    
    script_V = term1 - term2 - term3 + term4
    
    nu[n,k] = 1/(1+np.exp(-script_V))
    
    return(nu)

# a = Nu_updates(tau,nu,phi_mu,phi_var,X,sigma_A,sigma_eps,n,2)
    
    
def Tau_updates(tau,nu,alpha,D,N,K,n,k):
    dummy4 = 0 
    for m in range(k+1,K):
        [q,q_upper,log_q] = multi_q(tau,m)
        dummy4 = dummy4 + (N - np.sum(nu[:,m]))*q_upper[k+1]
    
    tau[k,0]= alpha + np.sum(np.sum(nu[:,m]) for m in np.arange(k, K))\
            + dummy4
    
    dummy5 = 1
    for m in range(k,K):
        [q,q_upper,log_q] = multi_q(tau,m)
        dummy5 = dummy5 + (N - np.sum(nu[:,m]))*q[k]
    
    tau[k,1] = dummy5

    return(tau)

    
def Elbo(tau,nu,phi_mu,phi_var,X,sigma_A,sigma_eps,alpha,D,K,N):
    Term1 = np.sum(np.log(alpha) + (alpha-1)*(sp.special.digamma(tau[:,0]) \
                   - sp.special.digamma(tau[:,0]+tau[:,1])) )
    
    Term2 = 0
    for k in range(K):
        Expectation = Exp_true(tau,k)
        for n in range(N): 
    
            asdf = nu[n,k]*np.sum(sp.special.digamma(tau[0:k+1,1]) - sp.special.digamma(tau[0:k+1,0]+tau[0:k+1,1]))  \
                        + (1-nu[n,k]) * Expectation
                
            Term2 = Term2 + asdf
        
    Term3 = np.sum(-D/2*np.log(2*np.pi*sigma_A) - 1/(2*sigma_A)\
                   *np.array([np.trace(phi_var[k]) + np.dot(phi_mu[:,k],phi_mu[:,k]) for k in range(K)]))
    
    Term4 = 0
    for n in range(N):
        summ1 = np.sum([nu[n,k]*np.dot(phi_mu[:,k],X[n,:]) for k in range(K)])
        summ2 = np.sum([np.sum([nu[n,k1]*nu[n,k2]*np.dot(phi_mu[:,k1],phi_mu[:,k2]) for k1 in range(k2)]) for k2 in range(K)])
        summ3 = np.sum([nu[n,k]*(np.trace(phi_var[k]) + np.dot(phi_mu[:,k],phi_mu[:,k])) for k in range(K)])
        
        Term4 = Term4 - D/2*np.log(2*np.pi*sigma_eps) - 1/(2*sigma_eps)*(np.dot(X[n,:],X[n,:]) - 2*summ1 + 2*summ2 + summ3)
        
    Term5 = np.sum(sp.special.betaln(tau[:,0],tau[:,1]) \
                  - (tau[:,0] - 1) * sp.special.digamma(tau[:,0]) - (tau[:,1] - 1) * sp.special.digamma(tau[:,1])   \
                  + (tau[:,0] + tau[:,1] -2) *  sp.special.digamma(tau[:,0] + tau[:,1]))
                   
    
    Term6 = np.sum([1/2*np.log((2*np.pi*np.exp(1))**D * np.linalg.det(phi_var[k])) for k in range(K)])               
    
    Term7 = np.sum(np.sum(-nu*np.log(nu) - (1-nu)*np.log(1-nu)))
    
    elbo = Term1 + Term2 + Term3 + Term4 + Term5 + Term6 + Term7
    
    return(elbo, Term1, Term2, Term3, Term4, Term5, Term6, Term7)
    #return(Term7)
    
    # Term7 and term 5 give NaNs. Term7 bc nu is close to 1; term 5 bc tau's are large!?
iterations = 100
elbo = np.zeros(iterations)
Term1 = np.zeros(iterations)
Term2 = np.zeros(iterations)
Term3 = np.zeros(iterations)
Term4 = np.zeros(iterations)
Term5 = np.zeros(iterations)
Term6 = np.zeros(iterations)
Term7 = np.zeros(iterations)

#print(A)
#input('pause')

for i in np.arange(iterations):    
        
            
    for k in np.arange(K): 
        tau = Tau_updates(tau,nu,alpha,D,N,K,n,k)
    
    for k in np.arange(K):
        Expectation_k = Exp_true(tau,k)
        for n in np.arange(N):
            nu = Nu_updates(Expectation_k,tau,nu,phi_mu,phi_var,X,D,N,K,n,k)
    
    for k in np.arange(K):
        [phi_var, phi_mu] = Phi_updates(nu,phi_mu,phi_var,X,sigma_A,sigma_eps,D,N,K,n,k) 
            
    #print('true')
    #print(A)
        
    #print('estimate')
    #print(phi_mu.T)
    #input('pause')
            
    [elbo[i],Term1[i],Term2[i],Term3[i],Term4[i],Term5[i],Term6[i],Term7[i]] = Elbo(tau,nu,phi_mu,phi_var,X,sigma_A,sigma_eps,alpha,D,K,N)
    print('iteration: ', i)
    print('ELbo')
    print(elbo[i])
    
    #print('Z')
    #print(Z[0:10,:])
    
    #print('nu')
    round_nu = np.round(nu*(nu>=0.9) + nu*(nu<=0.1)) + nu*(nu>=0.1)*(nu<=0.9)
    
    print(round_nu[0:10,:])
    
    # input("paused")
    
    if np.abs(elbo[i]-elbo[i-1]) <= 10^(-5):
        break
    
    
plt.figure(1)
plt.clf()
plt.plot(np.arange(iterations), elbo, linewidth=2.0)
plt.plot(np.arange(iterations), Term1)
plt.plot(np.arange(iterations), Term2)
plt.plot(np.arange(iterations), Term3)
plt.plot(np.arange(iterations), Term4) 
plt.plot(np.arange(iterations), Term5) 
plt.plot(np.arange(iterations), Term6)
plt.plot(np.arange(iterations), Term7)
plt.xlabel('iteration')
plt.ylabel('elbo')
plt.legend(['elbo', 'Term1', 'Term2', 'Term3', 'Term4', 'Term5', 'Term6', 'Term7'])




