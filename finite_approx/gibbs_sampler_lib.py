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
        self.x2 = np.dot(self.x.T, self.x)
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
        self.z2 = np.dot(self.z.T, self.z)
        self.zx = np.dot(self.z.T, self.x)

        self.lh_quantities()

        self.z_draws = []

    def lh_quantities(self):
        # intermediate quantities to compute likelihoods
        # should only call this once to initialize,
        # since it will compute the matrix inverse
        self.var = np.dot(self.z.T, self.z)\
                + self.sigma_eps/self.sigma_a * np.eye(self.k_approx)
        self.var_inv = np.linalg.inv(self.var)

        [_, logconst] = np.linalg.slogdet(self.var)
        mean_a = np.dot(np.linalg.solve(self.var, self.z.T), self.x)
        self.log_lh_X = -1/(2*self.sigma_eps) * \
                np.trace(np.dot(self.x.T, self.x - np.dot(self.z, mean_a)) )\
                - (self.x_d/2.0) * logconst

    def update_z(self, n,k):
        # flip the (n,k)th component of z, and update some necessary quantities

        # flip z[n,k],
        self.z[n,k] = 1 - self.z[n,k]

        #update z.T * z (efficiently)
        flip_z2(self.z, self.z2, n, k)

        # update z.T * x (effieciently)
        flip_zx(self.z, self.x, self.zx, n, k)


    def draw_z(self, keep_draw=False):
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

                # flip z[n,k],
                self.update_z(n,k)

                [log_lh_X_update, var_inv_update] = update_x_lp_cond_z(\
                    self.x, self.x2, self.z, self.z2, self.zx, \
                    self.var_inv, self.sigma_eps, self.sigma_a, self.k_approx, n, k)

                log_p_znk_cond_z_update = np.log(p_znk_cond_z_update)
                log_p_zn_update = log_lh_X_update + log_p_znk_cond_z_update \
                    - sp.misc.logsumexp([log_lh_X_update + log_p_znk_cond_z_update,\
                        self.log_lh_X+ np.log(p_znk_cond_z_stay)])

                #assert (np.exp(log_p_zn_update) >= 0) & (np.exp(log_p_zn_update) <= 1)

                choice = np.random.binomial(1, np.exp(log_p_zn_update))

                # update Z and prep log likelihood for next round:
                if choice == 1: # update the necessary quantities
                    self.log_lh_X = deepcopy(log_lh_X_update)
                    self.var_inv = deepcopy(var_inv_update)
                else:
                    # flip back to the orginal value of z[n,k]
                    self.update_z(n,k)


        if keep_draw:
            self.z_draws.append(self.z)

    def sample(self, burnin, num_gibbs_draws):

        print('Sampling:')
        bar = progressbar.ProgressBar(max_value=num_gibbs_draws + burnin)
        for n in bar(range(num_gibbs_draws + burnin)):
            self.draw_z(keep_draw = n >= burnin)

        print('Done sampling :)')


def update_x_lp_cond_z(x, x2, z_update, z_update2, zx, \
                        var_inv, sigma_eps, sigma_a, K_approx, n,k):
    # likelihood p(X|Z)-- equation (8) in Griffiths and Ghahramani
    # http://mlg.eng.cam.ac.uk/zoubin/papers/ibp-nips05.pdf
    # outputs the likelihood at Z  when Z has component n,k flipped
    # var_inv refers to inv(Z^T * Z + sigma_eps/sigma_a I), the previous inverse
    # z_update2 is z_update.T * z_update
    x_d = np.shape(x)[1]
    x_n = np.shape(x)[0]
    k_approx = np.shape(z_update)[1]

    assert np.shape(x)[0] == np.shape(z_update)[0]

    inv_var_flip = \
        update_inv_var(z_update, z_update2, var_inv, sigma_eps, sigma_a, n, k)

    #mean_a = np.dot(np.dot(inv_var_flip, z_update.T), x)
    mean_a = np.dot(inv_var_flip, zx)

    #log_likelihood = -1/(2*sigma_eps) * \
    #        np.trace(x2 - np.dot(x.T, np.dot(z_update, mean_a)) )
    log_likelihood = -1/(2*sigma_eps) * \
            np.trace(x2 -np.dot(zx.T, mean_a) )

    [_, logconst] = np.linalg.slogdet(inv_var_flip)

    return log_likelihood - (-x_d/2) * logconst, inv_var_flip

def update_inv_var(z_update, z_update2, var_inv, sigma_eps, sigma_a, n, k):
    # compute the inverse of Z^T * Z + sigma_eps/sigma_a * I when Z has
    # element n,k flipped.
    # var_inv refers to the previous computation, before Z had its
    # (n,k) component flipped
    # z_update2 is z_update.T * z_update

    k_approx = np.shape(z_update)[1]

    # var_flip = z_update2 + sigma_eps/sigma_a * np.eye(k_approx)

    # set up pieces to compute inverse of var_flip
    # u1 = deepcopy(z[n,:])
    v1 = np.zeros(k_approx)
    v1[k] = (z_update[n,k] == 0) * -1 + (z_update[n,k] == 1) * 1
    # assert len(u1) == len(v1)

    u2 = deepcopy(z_update[n,:])
    u2[k] = 1 - u2[k]
    assert len(u2) == len(v1)

    # some intermediate quantities: inv1 = inv(var + u1 * v1^T)
    outer1 = np.outer(z_update[n,:], v1)
    assert np.shape(outer1)[0] == k_approx
    assert np.shape(outer1)[1] == k_approx
    denom = 1 + np.dot(np.dot(v1, var_inv), z_update[n,:])
    #inv1 = var_inv - np.dot(np.dot(var_inv, outer1), var_inv) / denom
    inv1 = var_inv - np.outer(np.dot(var_inv, z_update[n,:]),\
                                np.dot(v1.T, var_inv)) / denom

    # print(inv1)
    # print(np.linalg.inv(Var + outer1))
    # assert np.allclose(inv1, np.linalg.inv(Var + outer1))

    # compute inverse of var_flip
    outer2 = np.outer(v1, u2)
    assert np.shape(outer2)[0] == k_approx
    assert np.shape(outer2)[1] == k_approx
    denom2 = 1 + np.dot(np.dot(u2, inv1), v1)
    #inv_var_flip = inv1 - np.dot(np.dot(inv1, outer2), inv1) / denom2
    inv_var_flip = inv1 - np.outer(np.dot(inv1, v1), np.dot(u2.T, inv1)) / denom2
    # assert np.allclose(var_flip, var + outer1 + outer2)

    # print(inv_var_flip)
    # print(np.linalg.inv(var_flip))
    return(inv_var_flip)


def flip_z2(z_update, z2, n, k):
    # this updates Z.T * Z when z has its (n,k)th commponent flipped

    # update Z.T * Z
    tmp = z2[k,k]
    #z2[k,:] = z2[k,:] - (2*z[n,k] - 1) * z[n,:]
    #z2[:,k] = z2[:,k] - (2*z[n,k] - 1) * z[n,:]
    z2[k,:] = z2[k,:] - (1 - 2*z_update[n,k]) * z_update[n,:]
    z2[:,k] = z2[:,k] - (1 - 2*z_update[n,k]) * z_update[n,:]
    z2[k,k] = tmp - (1 - 2*z_update[n,k])


def flip_zx(z_update, x, zx, n, k):
    # this updates X.T * Z, and Z.T * X when Z has its (n,k)th component flipped
    zx[k,:] = zx[k,:] + (2 * z_update[n,k] - 1) * x[n,:]

def display_results_Gibbs(X, Z, Z_Gibbs, mean_a, A, manual_perm = None):

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
    print('poster mean A: \n', mean_a)

    # plot posterior predictive
    pred_x = np.dot(Z_Gibbs, mean_a)
    for col in range(D):
        plt.clf()
        plt.plot(pred_x[:, col], X[:, col], 'ko')
        diag = np.linspace(np.min(pred_x[:,col]),np.max(pred_x[:,col]))
        plt.plot(diag,diag)
        plt.title('Posterior predictive, column' + str(col))
        plt.xlabel('predicted X')
        plt.ylabel('true X')
        plt.show()
