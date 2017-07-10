## library containing the necessary pieces to compute the elbo for a
## DP normal mixture, culminating in a DPNormalMixture object
## which can then be passed for optimization

import autograd.numpy as np
import autograd.scipy as sp
from autograd import grad

import scipy as osp

from copy import deepcopy

import matplotlib.pyplot as plt

################
# define entropies
def mu_entropy(mu_info):
    return - 0.5 * np.sum(np.linalg.slogdet(mu_info)[1])

def beta_entropy(tau):
    digamma_tau0 = sp.special.digamma(tau[:, 0])
    digamma_tau1 = sp.special.digamma(tau[:, 1])
    digamma_tausum = sp.special.digamma(np.sum(tau, 1))

    lgamma_tau0 = sp.special.gammaln(tau[:, 0])
    lgamma_tau1 = sp.special.gammaln(tau[:, 1])
    lgamma_tausum = sp.special.gammaln(np.sum(tau, 1))

    lbeta = lgamma_tau0 + lgamma_tau1 - lgamma_tausum

    return np.sum(
        lbeta - \
        (tau[:, 0] - 1.) * digamma_tau0 - \
        (tau[:, 1] - 1.) * digamma_tau1 + \
        (tau[:, 0] + tau[:, 1] - 2) * digamma_tausum)

def multinom_entropy(e_z):
    return -1 * np.sum(np.log(e_z ** e_z))


################
# define priors

def dp_prior(alpha, e_log_1mv):
    return (alpha - 1) * np.sum(e_log_1mv)

def normal_prior(mu, info, prior_mu, prior_info):
    k_approx = np.shape(mu)[0]
    mu_centered = mu - prior_mu
    return - 0.5 * np.sum(\
            np.einsum('ki, ij, kj -> k', mu_centered, prior_info, mu_centered) +\
            np.array([np.dot(np.diag(np.linalg.inv(info[k])), np.diag(prior_info)) \
                        for k in range(k_approx)]))

##############
# likelihoods

def loglik_ind_by_k(e_log_v, e_log_1mv):
    # expected log likelihood of z_n belonging to the kth stick
    k_approx = np.shape(e_log_v)[0] + 1
    e_log_stick_remain = np.array([np.sum(e_log_1mv[0:k]) \
                                for k in range(k_approx)])
    e_log_new_stick = np.concatenate((e_log_v, np.array([0])))

    return e_log_stick_remain + e_log_new_stick
    # return np.sum(np.dot(e_z, e_log_stick_remain + e_log_new_stick))

def loglik_ind(e_z, e_log_v, e_log_1mv):
    # expected log likelihood of all indicators for all n observations
    return np.sum(np.dot(e_z, loglik_ind_by_k(e_log_v, e_log_1mv)))

def loglik_obs_by_nk(mu, info, x, info_x):
    # expected log likelihood of nth observation when it belongs to component k
    k_approx = np.shape(mu)[0]
    return np.einsum('ni, ij, kj -> nk', x, info_x, mu) + \
            - 0.5 * np.einsum('ki, ij, kj -> k', mu, info_x, mu) + \
            - 0.5 * np.array([np.dot(np.diag(np.linalg.inv(info[k])), np.diag(info_x)) \
                for k in range(k_approx)])
            # - 0.5 * np.einsum('kii, ii -> k', info, info_x) # autograd doesn't like this

def loglik_obs(e_z, mu, info, x, info_x):
    # expected log likelihood of all observations
    return np.sum(e_z * loglik_obs_by_nk(mu, info, x, info_x))

def e_loglik_full(x, mu, info, tau, e_log_v, e_log_1mv, e_z,
                    prior_mu, prior_info, info_x, alpha):
    # combining the pieces, compute the full expected log likelihood

    prior = dp_prior(alpha, e_log_1mv) \
                + normal_prior(mu, info, prior_mu, prior_info)

    return loglik_obs(e_z, mu, info, x, info_x) \
                + loglik_ind(e_z, e_log_v, e_log_1mv) + prior

############
# and finally, the elbo

def compute_elbo(x, mu, info, tau, e_log_v, e_log_1mv, e_z,
                    prior_mu, prior_info, info_x, alpha):

    # entropy terms
    entropy = mu_entropy(info) + beta_entropy(tau) + multinom_entropy(e_z)

    return e_loglik_full(x, mu, info, tau, e_log_v, e_log_1mv, e_z,
                        prior_mu, prior_info, info_x, alpha) + entropy

############
# CAVI updates
def soft_thresh(e_z, ub, lb):
    constrain = (ub - lb) * e_z + lb
    renorm = np.sum(constrain, axis = 1)
    return constrain / renorm[:, None]

def z_update(mu, info, x, info_x, e_log_v, e_log_1mv, fudge_factor = 0.0):
    log_propto = loglik_obs_by_nk(mu, info, x, info_x) + \
                    loglik_ind_by_k(e_log_v, e_log_1mv)
    log_denom = sp.misc.logsumexp(log_propto, axis = 1)

    e_z_true = np.exp(log_propto - log_denom[:, None])
    return soft_thresh(e_z_true, 1 - fudge_factor, fudge_factor)

def mu_update(x, e_z, prior_mu, prior_info, info_x):
    k_approx = np.shape(e_z)[1]
    info_update = np.array([prior_info + np.sum(e_z[:, i]) * info_x \
                            for i in range(k_approx)])
    nat_param = np.dot(prior_mu, prior_info) + np.dot(e_z.T, np.dot(x, info_x))
    # print(nat_param)

    #mu_update = np.array([np.dot(nat_param[k,:], info_update[k])\
    #                        for k in range(k_approx)])
    mu_update = np.array([np.linalg.solve(info_update[k], nat_param[k,:])\
                            for k in range(k_approx)])

    return mu_update, info_update

def tau_update(e_z, alpha):
    k_approx = np.shape(e_z)[1]
    sum_e_z = np.sum(e_z, axis = 0)
    sum_e_z_upper = np.cumsum(sum_e_z[::-1])[::-1]

    #cum_sum_z = np.concatenate(([0.0], np.cumsum(sum_e_z)[:-2]))

    tau_update = np.zeros((k_approx - 1, 2))
    tau_update[:, 0] = sum_e_z[:-1] + 1
    tau_update[:, 1] = alpha + sum_e_z_upper[1:]

    return tau_update
############
# the object
class DPNormalMixture(object):
    def __init__(self, x, vb_params, prior_params):
        self.x = x
        self.vb_params = deepcopy(vb_params)
        self.prior_params = deepcopy(prior_params)

        self.prior_mu, self.prior_info, self.info_x, self.alpha \
                    = get_prior_params(self.prior_params)

        # self.weights = np.full((x.shape[0], 1), 1.0)
        # self.get_moment_jacobian = \
        #     autograd.jacobian(self.get_interesting_moments)

    def get_vb_params(self):
        e_log_v, e_log_1mv, e_z, mu, info, tau = get_vb_params(self.vb_params)

        return e_log_v, e_log_1mv, e_z, mu, info, tau

    def set_optimal_z(self):
        # note this isn't actually called in the kl method below
        # since we don't want to compute e_log_v twice (it has digammas)

        e_log_v, e_log_1mv, e_z, mu, info, tau = self.get_vb_params()

        # optimize z
        e_z_opt = z_update(mu, info, self.x, self.info_x, e_log_v, e_log_1mv)
        return e_z_opt
        # self.vb_params['local']['e_z'].set(e_z_opt)

    def get_kl(self, verbose = False):
        e_log_v, e_log_1mv, e_z, mu, info, tau = self.get_vb_params()

        elbo = compute_elbo(self.x, mu, info, tau, e_log_v, e_log_1mv, e_z,
                        self.prior_mu, self.prior_info, self.info_x, self.alpha)
        if verbose:
            print('kl:\t', -1 * elbo)

        return -1 * elbo

    def kl_optimize_z(self, verbose=False):
        # this is the function that will be passed to the Newton
        # optimization
        e_log_v, e_log_1mv, e_z, mu, info, tau = self.get_vb_params()

        # optimize z
        e_z = z_update(mu, info, self.x, self.info_x, e_log_v, e_log_1mv, \
                                        fudge_factor = 10**(-10))
        # self.vb_params['local']['e_z'].set(e_z)

        elbo = compute_elbo(self.x, mu, info, tau, e_log_v, e_log_1mv, e_z,
                        self.prior_mu, self.prior_info, self.info_x, self.alpha)
        if verbose:
            print('kl:\t', -1 * elbo)

        return -1 * elbo



################
# other functions

def draw_data(alpha, mu_prior, mu_prior_info, info_x, x_dim, k_approx, num_obs):

    # true means
    mu_spacing = np.linspace(-10, 10, k_approx)
    true_mu = np.array([ mu_spacing, mu_spacing]).T
    # true_mu = np.random.multivariate_normal(np.zeros(x_dim), \
    #                                        100 * np.eye(x_dim), k_approx)

    # draw beta sticks
    true_v = np.zeros(k_approx)
    true_pi = np.zeros(k_approx)
    stick_remain = np.zeros(k_approx)

    true_v[0] = np.random.beta(1, alpha)
    true_pi[0] = true_v[0]
    stick_remain[0] = 1 - true_v[0]

    for i in range(1, k_approx):
        if i == k_approx - 1: # the last stick
            true_v[i] = 1.0
        else:
            true_v[i] = np.random.beta(1, alpha)

        true_pi[i] = stick_remain[i - 1] * true_v[i]
        stick_remain[i] = stick_remain[i - 1] * (1 - true_v[i])

    # draw group indicators
    true_z_ind = np.random.choice(range(k_approx), p = true_pi, size = num_obs)
    true_z = np.zeros((num_obs, k_approx))
    for i in range(num_obs):
        true_z[i, true_z_ind[i]] = 1.0

    # draw observations
    x = np.array([ np.random.multivariate_normal(
                true_mu[true_z_ind[n]], np.linalg.inv(info_x)) \
               for n in range(num_obs) ])
    return x, true_mu, true_z, true_z_ind, true_v, true_pi

def get_vb_params(vb_params):
    e_log_v = vb_params['global']['v_sticks'].e_log()[:,0] # E[log v]
    e_log_1mv = vb_params['global']['v_sticks'].e_log()[:,1] # E[log 1 - v]
    e_z = vb_params['local']['e_z'].get()
    mu = vb_params['global']['mu'].get()
    info = vb_params['global']['info'].get()
    tau = vb_params['global']['v_sticks'].alpha.get()

    return e_log_v, e_log_1mv, e_z, mu, info, tau

def get_prior_params(prior_params):
    prior_mu = prior_params['prior_mu'].get()
    prior_info = prior_params['prior_info'].get()
    info_x = prior_params['info_x'].get()
    alpha = prior_params['alpha'].get()

    return prior_mu, prior_info, info_x, alpha



# draw samples from the variational distribution (used in unittesting)
def variational_samples(mu, info, tau, e_z, n_samples):
    # draw v sticks
    v_samples = osp.stats.beta.rvs(tau[:,0], tau[:,1], size = (n_samples, np.shape(tau)[0]))
    v_samples[v_samples == 1.] = 1 - 10**(-10)
    v_samples[v_samples == 0.] = 10 ** (-10)

    # compute multinomial probabilities from v sticks
    pi_samples = np.zeros((n_samples, np.shape(v_samples)[1] + 1))
    for n in range(n_samples):
        stick_remain = np.ones(np.shape(v_samples)[1])
        for i in range(0, np.shape(v_samples)[1]):
            pi_samples[n, i] = stick_remain[i - 1] * v_samples[n, i]
            stick_remain[i] = stick_remain[i - 1] * (1 - v_samples[n, i])

        pi_samples[n, -1] = stick_remain[-1]

    assert np.all((np.sum(pi_samples, axis = 1) - 1.) < 10**(-15))

    # draw normal means
    mu_samples = np.zeros((n_samples, mu.shape[0], mu.shape[1]))
    for k in range(np.shape(mu)[0]):
        mu_samples[:,k,:] = np.random.multivariate_normal(mu[k,:], \
                            np.linalg.inv(info[k]), size = n_samples)

    # draw z
    z_samples = np.zeros((n_samples, e_z.shape[0], e_z.shape[1]))
    for n in range(e_z.shape[0]):
        z_samples[:, n, :] = np.random.multinomial(1, e_z[n, :], size = n_samples)

    return v_samples, pi_samples, mu_samples, z_samples

###################
# function to examine from vb
def display_results(x, true_z, true_mu, e_z, mu, true_v, tau, manual_perm = None):

    x_dim = np.shape(x)[1]
    n_obs = np.shape(x)[0]
    k_approx = np.shape(true_z)[1]

    # print('true_z (unpermuted): \n', true_z[0:10])

    # Find the minimizing permutation.
    accuracy_mat = [[ np.linalg.norm(true_mu[i,:] - mu[j,:]) \
                for i in range(k_approx)] for j in range(k_approx) ]
    perm_tmp = np.argmin(accuracy_mat, 1)

    # check that we have a true permuation
    if len(perm_tmp) == len(set(perm_tmp)):
        perm = perm_tmp
    else:
        print('** procedure did not give a true permutation')
        if manual_perm == None:
            perm = np.arange(k_approx)
        else:
            perm = manual_perm

    print('permutation: ', perm)

    # print Z (permuted) and nu
    e_z_rounded = deepcopy(e_z)
    e_z_rounded[e_z > 0.8] = 1.0
    e_z_rounded[e_z < 0.2] = 0.0
    print('true Z (permuted) \n', true_z[0:10, perm])
    print('e_z (rounded) \n', e_z_rounded[0:10,:])

    print('l1 error (after permutation): ', \
        [ np.sum(np.abs(true_z[:, perm[i]] - e_z[:, i]))/n_obs
            for i in range(k_approx) ])

    # examine phi_mu
    print('\n')
    print('true A (permuted): \n', true_mu[perm, :])
    print('poster mean A: \n', mu)

    # plot posterior predictive
    pred_x = np.dot(e_z, mu)
    for col in range(x_dim):
        plt.clf()
        plt.plot(pred_x[:, col], x[:, col], 'ko')
        diag = np.linspace(np.min(pred_x[:,col]),np.max(pred_x[:,col]))
        plt.plot(diag,diag)
        plt.title('Posterior predictive, column' + str(col))
        plt.xlabel('predicted X')
        plt.ylabel('true X')
        plt.show()

    # check stick lengths
    print('true stick lengths', true_v)
    print('variational stick lengths: ', tau[:,0] / np.sum(tau, axis = 1))
