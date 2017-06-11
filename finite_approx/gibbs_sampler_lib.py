import autograd.numpy as np
import autograd.scipy as sp
from autograd.scipy import special
from autograd import grad

from copy import deepcopy
import progressbar

import matplotlib.pyplot as plt

import finite_approx.valez_finite_VI_lib as vi

class GibbsSampler(object):
    def __init__(self, x, k_approx, alpha, sigma_eps, sigma_a):
        self.x = x
        self.x_n = x.shape[0]
        self.x_d = x.shape[1]
        self.k_approx = k_approx
        self.alpha = alpha
        self.sigma_eps = sigma_eps
        self.sigma_a = sigma_a

        self.get_z_cond_params = grad(self.z_lp, argnum=0)
        self.get_a_cond_params = grad(self.a_lp, argnum=0)
        self.get_a2_cond_params = grad(self.a_lp, argnum=1)
        self.get_pi1_cond_params = grad(self.pi_lp, argnum=0)
        self.get_pi2_cond_params = grad(self.pi_lp, argnum=1)

        self.initialize_sampler()

    def z_lp(self, z):
        return vi.exp_log_likelihood(
            z, self.a, self.a**2, np.log(self.pi), np.log(1 - self.pi),
            self.sigma_a, self.sigma_eps, self.x, self.alpha)

    def pi_lp(self, logpi, log1mpi):
        return vi.exp_log_likelihood(
            self.z, self.a, self.a**2, logpi, log1mpi,
            self.sigma_a, self.sigma_eps, self.x, self.alpha)

    def a_lp(self, a, a2):
        return vi.exp_log_likelihood(
            self.z, a, a2, np.log(self.pi), np.log(1 - self.pi),
            self.sigma_a, self.sigma_eps, self.x, self.alpha)

    def initialize_sampler(self):
        # Initial values for draws
        self.pi = np.ones(self.k_approx) * .8
        self.z = np.random.binomial(1, self.pi, [ self.x_n, self.k_approx ])
        self.z = self.z.astype(float)
        self.a = np.random.normal(
            0, np.sqrt(self.sigma_a), (self.x_d, self.k_approx))

        self.pi_draws = []
        self.z_draws = []
        self.a_draws = []

    def draw_z_column(self, k):
        # Because the z columns are inter-dependent, only draw one column at a time.
        assert k < self.k_approx
        z_cond_params = self.get_z_cond_params(self.z)
        z_logsumexp = sp.misc.logsumexp(z_cond_params, 1)
        z_logsumexp = np.broadcast_to(z_logsumexp, (self.k_approx, self.x_n)).T
        z_means = np.exp(z_cond_params - z_logsumexp)
        self.z[:, k] = vi.draw_z(z_means, 1)[0, :, k].astype(float)

    def draw_a(self):
        a_cond_params = self.get_a_cond_params(self.a, self.a**2)
        a2_cond_params = self.get_a2_cond_params(self.a, self.a**2)

        a_var = -0.5 / a2_cond_params
        a_mean = a_var * a_cond_params
        assert np.all(a_var > 0)

        self.a = vi.draw_a(a_mean, a_var, 1)[0, :, :]

    def draw_pi(self):
        pi1_cond_params = self.get_pi1_cond_params(np.log(self.pi), np.log(1 - self.pi))
        pi2_cond_params = self.get_pi2_cond_params(np.log(self.pi), np.log(1 - self.pi))

        # Note -- add one to get the beta distribution parameters from the gradients.
        pi_params = np.vstack([pi1_cond_params, pi2_cond_params]).T + 1
        self.pi = vi.draw_pi(pi_params, 1)[0, :]

    def draw(self, keep_draw=False):
        # Draw the z columns in a random order.
        self.draw_a()
        self.draw_pi()
        for z_col in np.random.permutation(self.k_approx):
            self.draw_z_column(z_col)

        if keep_draw:
            self.a_draws.append(self.a)
            self.pi_draws.append(self.pi)
            self.z_draws.append(self.z)

    def sample(self, burnin, num_gibbs_draws):
        self.initialize_sampler()

        print('Sampling:')
        bar = progressbar.ProgressBar(max_value=num_gibbs_draws + burnin)
        for n in bar(range(num_gibbs_draws + burnin)):
            self.draw(keep_draw = n >= burnin)

        print('Holy cow, done sampling!')

class CollapsedGibbsSampler(object):
    def __init__(self, x, k_approx, alpha, sigma_eps, sigma_a):
        self.x = x
        self.x_n = x.shape[0]
        self.x_d = x.shape[1]
        self.k_approx = k_approx
        self.alpha = alpha
        self.sigma_eps = sigma_eps
        self.sigma_a = sigma_a

        self.initialize_sampler()

    def initialize_sampler(self):
        # Initial values for draws
        self.z = np.random.binomial(1, 0.5, [self.x_n, self.k_approx ])
        self.z = self.z.astype(float)

        self.z_draws = []

    def draw_z(self, keep_draw=False):

        # compute the inverse to get started
        # after this, inverses are computed via rank 1 updates
        var = np.dot(self.z.T, self.z)\
                + self.sigma_eps/self.sigma_a * np.eye(self.k_approx)
        var_inv = np.linalg.inv(var)

        const = np.linalg.det(var)**(self.x_d/2)
        mean_A = np.dot(np.linalg.solve(var, self.z.T), self.x)
        log_lh_X_stay = -1/(2*self.sigma_eps) * \
                np.trace(np.dot(self.x.T, self.x - np.dot(self.z, mean_A)) )

        for n in range(self.x_n):
            for k in range(self.k_approx):
                # first, compute p(z_nk = 1 | Z_{-nk})
                # This is equation (6) in Griffiths and Ghahramani
                # http://mlg.eng.cam.ac.uk/zoubin/papers/ibp-nips05.pdf
                p_z_nk1_cond_Z = (np.sum(self.z[:, k]) - self.z[n, k] + \
                                self.alpha/self.k_approx)\
                                /(self.x_n + self.alpha/self.k_approx)
                assert (p_z_nk1_cond_Z >= 0) & (p_z_nk1_cond_Z <= 1)

                if self.z[n,k] == 1:
                    p_znk_cond_z_stay = p_z_nk1_cond_Z
                    p_znk_cond_z_update = 1 - p_z_nk1_cond_Z
                else:
                    p_znk_cond_z_update = p_z_nk1_cond_Z
                    p_znk_cond_z_stay = 1 - p_z_nk1_cond_Z

                # next compute p(X|Z_new)
                # equation (8) in Griffiths and Ghahramani
                [log_lh_X_update, var_inv_update] = update_x_lp_cond_z(\
                    self.x, self.z, var_inv, self.sigma_eps, self.sigma_a, \
                    self.k_approx, n, k)


                log_p_zn_update = log_lh_X_update + np.log(p_znk_cond_z_update) \
                    - sp.misc.logsumexp([log_lh_X_update + np.log(p_znk_cond_z_update),\
                        log_lh_X_stay+ np.log(p_znk_cond_z_stay)])

                assert (np.exp(log_p_zn_update) >= 0) & (np.exp(log_p_zn_update) <= 1)

                choice = np.random.binomial(1, np.exp(log_p_zn_update))

                # update Z and prep log likelihood for next round:
                if choice == 1: # flip Z_nk
                    log_lh_X_stay = deepcopy(log_lh_X_update)
                    var_inv = deepcopy(var_inv_update)
                    self.z[n,k] = 1 - self.z[n,k]
                else:
                    #log_lh_X_stay = deepcopy(log_lh_X_stay)
                    #var_inv = deepcopy(var_inv)
                    #self.z[n,k] = self.z[n,k]
                    pass

        if keep_draw:
            self.z_draws.append(self.z)

    def sample(self, burnin, num_gibbs_draws):

        print('Sampling:')
        bar = progressbar.ProgressBar(max_value=num_gibbs_draws + burnin)
        for n in bar(range(num_gibbs_draws + burnin)):
            self.draw_z(keep_draw = n >= burnin)

        print('Done sampling :)')


def update_x_lp_cond_z(X, Z, Var_inv, sigma_eps, sigma_A, K_approx, n,k):
# likelihood p(X|Z)-- equation (8) in Griffiths and Ghahramani
# http://mlg.eng.cam.ac.uk/zoubin/papers/ibp-nips05.pdf
# outputs the likelihood at Z  when Z has component n,k flipped
# Var_inv refers to inv(Z^T * Z + sigma_eps/sigma_A I), the previous inverse

    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(Z)[1]

    assert np.shape(X)[0] == np.shape(Z)[0]

    Z_update = deepcopy(Z)
    Z_update[n,k] = 1 - Z[n,k]

    inv_var_flip = update_inv_var(Z, Var_inv, sigma_eps, sigma_A, n, k)

    mean_A = np.dot(np.dot(inv_var_flip, Z_update.T), X)

    log_likelihood = -1/(2*sigma_eps) * \
            np.trace(np.dot(X.T, X - np.dot(Z_update, mean_A)) )

    const = np.linalg.det(inv_var_flip)**(-D/2)

    return log_likelihood - np.log(const), inv_var_flip

def update_inv_var(Z, Var_inv, sigma_eps, sigma_A, n, k):

    K = np.shape(Z)[1]

    Var = np.dot(Z.T, Z) + sigma_eps/sigma_A * np.eye(K)

    Z_flip_nk = deepcopy(Z)
    Z_flip_nk[n,k] = 1 - Z[n,k]

    var_flip = np.dot(Z_flip_nk.T, Z_flip_nk) + sigma_eps/sigma_A * np.eye(K)

    # set up pieces to compute inverse of var_flip
    u1 = Z_flip_nk[n,:]
    v1 = np.zeros(K)
    v1[k] = Z_flip_nk[n,k] - Z[n,k]
    assert len(u1) == len(v1)

    u2 = Z[n,:]
    v2 = deepcopy(v1)
    assert len(u2) == len(v2)

    # some intermediate quantities: inv1 = inv(var + u1 * v1^T)
    outer1 = np.outer(u1, v1)
    assert np.shape(outer1)[0] == K
    assert np.shape(outer1)[1] == K
    denom = 1 + np.dot(np.dot(v1, Var_inv), u1)
    inv1 = Var_inv - np.dot(np.dot(Var_inv, outer1), Var_inv) / denom


    #print(inv1)
    #print(np.linalg.inv(Var + outer1))
    # assert np.allclose(inv1, np.linalg.inv(Var + outer1))

    # compute inverse of var_flip
    outer2 = np.outer(v2, u2)
    assert np.shape(outer2)[0] == K
    assert np.shape(outer2)[1] == K
    denom2 = 1 + np.dot(np.dot(u2, inv1), v2)
    inv_var_flip = inv1 - np.dot(np.dot(inv1, outer2), inv1) / denom2

    #print(var_flip)
    #print(np.dot(Z.T, Z) + sigma_eps/sigma_A * np.eye(K) + outer1 + outer2)
    assert np.allclose(var_flip, Var + outer1 + outer2)

    #print(inv_var_flip)
    #print(np.linalg.inv(var_flip))
    return(inv_var_flip)

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
