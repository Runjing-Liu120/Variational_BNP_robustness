import autograd.numpy as np
import autograd.scipy as sp
from autograd.scipy import special
from autograd import grad

from copy import deepcopy
import progressbar

import valez_finite_VI_lib as vi

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
        for n in range(self.x_n):

            for k in range(self.k_approx):
                #p(z_nk = 1 | Z_{-nk}): equation (6) in Griffiths and Ghahramani
                p_znk1 = (np.sum(self.z[:, k]) - self.z[n, k] + \
                                    self.alpha/self.k_approx)\
                                    /(self.x_n + self.alpha/self.k_approx)

                assert (p_znk1 >= 0) & (p_znk1 <= 1)

                p_znk0 = 1 - p_znk1

                z_tmp = deepcopy(self.z)
                z_tmp[n,k] = 1
                [log_likelihood1, _] = \
                            x_lp_cond_z(self.x, z_tmp, self.sigma_eps, self.sigma_a, self.k_approx)

                z_tmp[n,k] = 0
                [log_likelihood0, _] = \
                            x_lp_cond_z(self.x, z_tmp, self.sigma_eps, self.sigma_a, self.k_approx)

                log_p1 = log_likelihood1 + np.log(p_znk1) \
                    - sp.misc.logsumexp([log_likelihood1 + np.log(p_znk1),\
                        log_likelihood0 + np.log(p_znk0)])

                assert (np.exp(log_p1) >= 0) & (np.exp(log_p1) <= 1)

                self.z[n,k] = np.random.binomial(1, np.exp(log_p1))

        if keep_draw:
            self.z_draws.append(self.z)

    def sample(self, burnin, num_gibbs_draws):

        print('Sampling:')
        bar = progressbar.ProgressBar(max_value=num_gibbs_draws + burnin)
        for n in bar(range(num_gibbs_draws + burnin)):
            self.draw_z(keep_draw = n >= burnin)

        print('Holy cow, done sampling!')


def x_lp_cond_z(X, Z, sigma_eps, sigma_A, K_approx):
# likelihood p(X|Z)-- equation (8) in Griffiths and Ghahramani
# http://mlg.eng.cam.ac.uk/zoubin/papers/ibp-nips05.pdf

    assert np.shape(X)[0] == np.shape(Z)[0]

    D = np.shape(X)[1]
    N = np.shape(X)[0]
    K = np.shape(Z)[1]

    var = np.dot(Z.T, Z) + sigma_eps/sigma_A * np.eye(K_approx)

    const = np.linalg.det(var)**(D/2)

    #const = (2*np.pi)**(N*D/2) * sigma_eps**((N-K)*D/2) * sigma_A**(K*D/2) * \
    #    np.linalg.det(var)**(D/2)

    mean_A = np.dot(np.linalg.solve(var, Z.T), X)

    log_likelihood = -1/(2*sigma_eps) * \
            np.trace(np.dot(X.T, X - np.dot(Z, mean_A)) )

    return log_likelihood - np.log(const), mean_A

"""def compute_mean_a(X, Z, sigma_eps, sigma_A, K_approx, inv_Mk, k, truth = False):
    if truth: # compute true inverse
        M = np.linalg.inv(np.dot(Z.T, Z) + sigma_eps/sigma_A * np.eye(K_approx))
    else:
        Mk =

        mean_A = np.dot(np.linalg.solve(var, Z.T), X)

    return(mean_A)
"""
