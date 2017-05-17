import numpy as np
import scipy as sp
from scipy import special
import matplotlib.pyplot as plt
from copy import deepcopy

# Update functions
def phi_updates(nu, phi_mu, phi_var, X, sigma_A, sigma_eps, D, N, K, n, k):
    phi_var[k] = 1 / (1 / sigma_A + np.sum(nu[:, k]) / sigma_eps) * np.identity(D)
    Summation = 0
    for n in range(N):
        dum1 = X[n, :] - np.dot(phi_mu, nu[n, :]) + nu[n, k] * phi_mu[:, k]
        summation = nu[n,k] * dum1 + summation

    phi_mu[:,k] = \
        1 / sigma_eps * summation * 1 / (1 / sigma_A + np.sum(nu[:, k]) / sigma_eps)

    return(phi_var, phi_mu)


# [a,b] = Phi_updates(tau,nu,phi_mu, phi_var,X,sigma_A,sigma_eps,2)

# Multinomial lower bound computation.  Tau is a (k + 1) by 2 matrix of beta
# coefficients, with alpha in the first column and beta in the second.
def multi_q(tau, k):
    assert tau.shape[0] >= k + 1, \
        "tau has wrong number of rows %d, k = %d" % (tau.shape[0], k)
    assert tau.shape[1] == 2, \
        "tau has wrong number of columns, %d" % tau.shape[1]
    q = np.zeros(k + 1)
    log_q = np.zeros(k + 1)
    for i in range(k + 1):
        # Note: re-computing the digamma function over and over again is
        # very computationally wasteful.  Better to compute the digamma of
        # all the taus once and then express these terms as cumulative sums.
        dum2 = np.sum(sp.special.digamma(tau[0:i, 0]))
        dum3 = np.sum(sp.special.digamma(tau[0:i + 1, 0] + tau[0:i + 1, 1]))

        # Note: may want to re-normalize q before exponentiating to
        # avoid numeric errors.  That is, accumulate log_q, then subtract the
        # max, then exponentiate and normalize.
        
        # q[i] = np.exp(sp.special.digamma(tau[i, 1]) + dum2 - dum3)
        log_q[i] = sp.special.digamma(tau[i, 1]) + dum2 - dum3

    # q = q / np.sum(q) # probability of mutlinomial atoms
    log_q = log_q - sp.misc.logsumexp(log_q) # log probability
    q = np.exp(log_q) 
    
    q_upper = [np.sum(q[m:]) for m in range(k + 1)]
    return(q, q_upper, log_q)


def Exp_lowerbound(tau, k): # lower bound of expectation using multinomial
    [q, q_upper, log_q] = multi_q(tau, k)

    # This is E_y[digamma(\tau_{y2})]
    exp1 = np.dot(q, sp.special.digamma(tau[0:k + 1, 1]))

    # This is E_y[\sum_{m=1}^{y-1} digamma(\tau_{m1})].  Note that in the paper
    # the empty product in section 3.3 is taken to be one and the empty sum
    # is taken to be zero.
    exp2 = np.sum(q_upper[m + 1] * sp.special.digamma(tau[m, 0]) for m in range(k))

    # This is E_y[\sum_{m=1}^{y} digamma(\tau_{m1} + \tau_{m2})].
    exp3 = np.dot(q_upper, sp.special.digamma(tau[0:k + 1, 0] + tau[0:k + 1, 1]))

    # This is negative the entropy of q.
    exp4 = np.dot(q, log_q)

    return(exp1 + exp2 - exp3 - exp4)

def Exp_true(tau, k): # true expectation using MC estimate
    beta = np.zeros(k + 1)
    B = np.zeros(10000)

    for i in range(10000):
        for r in range(k + 1):
            beta[r] = np.random.beta(tau[r, 0], tau[r, 1])

        B[i] = np.log(1 - np.prod(beta))

    Exp_true = np.average(B)
    return(Exp_true)


def Nu_updates(Expectation_k, tau, nu, phi_mu, phi_var, sigma_eps, X, D, N, K, n, k):
    term1 = np.sum(sp.special.digamma(tau[0:k + 1,0]) - \
        sp.special.digamma(tau[0:k + 1, 0] + tau[0:k + 1,1]) )

    # term2 = Exp_lowerbound(tau,k)
    term2 = Expectation_k

    term3 = 1 / (2 * sigma_eps) * (np.trace(phi_var[k]) + \
        np.dot(phi_mu[:, k], phi_mu[:, k]))

    term4 = 1 / sigma_eps * np.dot( phi_mu[:, k], X[n, :] - \
        np.dot(phi_mu, nu[n, :]) + nu[n, k] * phi_mu[:, k])

    #dummy = 0
    #for l in range(K):
    #    if (l != k):
    #        dummy += nu[n,l] * phi_mu[:,l]

    #term4 = 1/sigma_eps* np.dot(phi_mu[:,k],X[n,:] - dummy)

    script_V = term1 - term2 - term3 + term4

    nu[n,k] = 1 / (1 + np.exp(-script_V))

    return(nu)


def Tau_updates(tau, nu, alpha, D, N, K, n, k):
    dummy4 = alpha
    for m in range(k + 1, K):
        [q,q_upper, log_q] = multi_q(tau,m)
        dummy4 += (N - np.sum(nu[:,m])) * q_upper[k + 1]

    tau[k, 0]= np.sum(np.sum(nu[:, m]) for m in np.arange(k, K)) + dummy4

    dummy5 = 1
    for m in range(k,K):
        [q,q_upper,log_q] = multi_q(tau,m)
        dummy5 += (N - np.sum(nu[:,m]))*q[k]

    tau[k,1] = dummy5

    return(tau)


def Elbo(tau, nu, phi_mu, phi_var, X, sigma_A, sigma_eps, alpha, D, K, N):
    Term1 = np.sum(np.log(alpha) + (alpha - 1) * (sp.special.digamma(tau[:, 0]) \
                   - sp.special.digamma(tau[:, 0] + tau[:, 1])) )

    Term2 = 0
    for k in range(K):
        Expectation = Exp_lowerbound(tau,k)
        for n in range(N):

            Term2 += nu[n, k] * np.sum(sp.special.digamma(tau[0:k + 1, 1]) - \
                sp.special.digamma(tau[0:k + 1, 0] + tau[0:k + 1, 1])) + \
                (1 - nu[n,k]) * Expectation
            #Term2 = (1 - nu[n,k]) * Expectation
    Term3 = np.sum(-D/2*np.log(2*np.pi*sigma_A) - 1/(2*sigma_A) *\
        np.array([np.trace(phi_var[k]) + \
        np.dot(phi_mu[:, k] , phi_mu[:, k]) for k in range(K)]))

    Term4 = 0
    for n in range(N):
        summ1 = np.sum([nu[n,k] * np.dot(phi_mu[:,k], X[n,:]) for k in range(K)])
        summ2 = np.sum(
            [np.sum(
                [nu[n,k1] * nu[n,k2] * np.dot(phi_mu[:,k1], phi_mu[:,k2]) \
                for k1 in range(k2)]) for k2 in range(K)])
        summ3 = np.sum([nu[n,k] * (np.trace(phi_var[k]) + \
            np.dot(phi_mu[:,k], phi_mu[:,k])) for k in range(K)])

        Term4 = Term4 - D / 2 * np.log(2 * np.pi * sigma_eps) - \
            1 / (2 * sigma_eps) * (
                np.dot(X[n,:], X[n,:]) - 2 * summ1 + 2 * summ2 + summ3)

    Term5 = np.sum(sp.special.betaln(tau[:,0],tau[:,1]) - \
        (tau[:,0] - 1) * sp.special.digamma(tau[:,0]) - \
        (tau[:,1] - 1) * sp.special.digamma(tau[:,1]) + \
        (tau[:,0] + tau[:,1] -2) *  sp.special.digamma(tau[:,0] + tau[:,1]))


    Term6 = np.sum([1 / 2 * np.log((2 * np.pi * np.exp(1)) ** D * \
        np.linalg.det(phi_var[k])) for k in range(K)])

    Term7 = np.sum(np.sum( -np.log(nu**nu) - np.log((1-nu)**(1-nu)) ))

    elbo = Term1 + Term2 + Term3 + Term4 + Term5 + Term6 + Term7

    return elbo, Term1, Term2, Term3, Term4, Term5, Term6, Term7
    #return(Term7)
