import autograd.numpy as np
import autograd.scipy as sp

from scipy.special import expit

import matplotlib.pyplot as plt
from copy import deepcopy
import math


def likelihood_X(X, Z_Gibbs, sigma_eps, sigma_A):
# likelihood p(X|Z)-- equation (8) in Griffiths and Ghahramani
# http://mlg.eng.cam.ac.uk/zoubin/papers/ibp-nips05.pdf

    assert np.shape(X)[0] == np.shape(Z_Gibbs)[0]

    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(Z_Gibbs)[1]

    var = np.dot(Z_Gibbs.T, Z_Gibbs) + sigma_eps/sigma_A * np.eye(K)

    const = np.linalg.det(var)**(D/2)

    #const = (2*np.pi)**(N*D/2) * sigma_eps**((N-K)*D/2) * sigma_A**(K*D/2) * \
    #    np.linalg.det(var)**(D/2)

    mean_A = np.dot(np.linalg.solve(var, Z_Gibbs.T), X)

    log_likelihood = -1/(2*sigma_eps) * \
            np.trace(np.dot(X.T, X - np.dot(Z_Gibbs, mean_A)) )

    return log_likelihood - np.log(const), mean_A

def draw_Znk(X, Z_Gibbs, sigma_eps, sigma_A, alpha, n,k):

    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(Z_Gibbs)[1]

    #p(z_nk = 1 | Z_{-nk}): equation (6) in iffiths and Ghahramani
    P_znk1 = (np.sum(Z_Gibbs[:,k]) - Z_Gibbs[n,k] + alpha/K)/(N + alpha/K)
    assert (P_znk1 >= 0) & (P_znk1 <= 1)

    P_znk0 = 1 - P_znk1

    Z_Gibbs[n,k] = 1
    [log_likelihood1, _] = likelihood_X(X, Z_Gibbs, sigma_eps, sigma_A)

    Z_Gibbs[n,k] = 0
    [log_likelihood0, _] = likelihood_X(X, Z_Gibbs, sigma_eps, sigma_A)

    log_P1 = log_likelihood1 + np.log(P_znk1) \
        - sp.misc.logsumexp(\
        [log_likelihood1 + np.log(P_znk1) , log_likelihood0 + np.log(P_znk0)])

    assert (np.exp(log_P1) >= 0) & (np.exp(log_P1) <= 1)

    Z_Gibbs[n,k] = np.random.binomial(1, np.exp(log_P1))


def display_results_Gibbs(X, Z, Z_Gibbs, mean_A, A, manual_perm = None):

    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(Z_Gibbs)[1]

    print('Z (unpermuted): \n', Z[0:10])

    # Find the minimizing permutation.
    accuracy_mat = [[ np.sum(np.abs(Z[:, i] - Z_Gibbs[:, j]))/N for i in range(K) ]
                      for j in range(K) ]
    perm_tmp = np.argmin(accuracy_mat, 1)

    # check that we have a true permuation
    if len(perm_tmp) == len(set(perm_tmp)):
        perm = perm_tmp
    else:
        print('** procedure did not give a true permutation')
        if manual_perm == None:
            perm = np.arange(K)
        else:
            perm = manual_perm

    print('permutation: ', perm)

    # print Z (permuted) and nu
    print('Z (permuted) \n', Z[0:10, perm])
    print('round_nu \n', Z_Gibbs[0:10,:])

    print('l1 error (after permutation): ', \
        [ np.sum(np.abs(Z[:, perm[i]] - Z_Gibbs[:, i]))/N for i in range(K) ])

    # examine phi_mu
    print('\n')
    print('true A (permuted): \n', A[perm, :])
    print('poster mean A: \n', mean_A)

    # plot posterior predictive
    pred_x = np.dot(Z_Gibbs, mean_A)
    for col in range(D):
        plt.clf()
        plt.plot(pred_x[:, col], X[:, col], 'ko')
        diag = np.linspace(np.min(pred_x[:,col]),np.max(pred_x[:,col]))
        plt.plot(diag,diag)
        plt.title('Posterior predictive, column' + str(col))
        plt.xlabel('predicted X')
        plt.ylabel('true X')
        plt.show()
