## library containing the necessary pieces to compute the elbo for a
## DP normal mixture, culminating in a DPNormalMixture object
## which can then be passed for optimization

import autograd.numpy as np
import autograd.scipy as sp

import scipy as osp

from copy import deepcopy

from DP_normal_mixture_opt_lib import z_update

################
# define entropies
def mu_entropy(info_mu):
    return - 0.5 * np.sum(np.linalg.slogdet(info_mu)[1])

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

def wishart_entropy(e_logdet_info_x, e_info_x, dof):
    dim = np.shape(e_info_x)[0]
    assert dim == np.shape(e_info_x)[1]
    assert dof > dim

    const = 0.5 * dof * np.linalg.slogdet(e_info_x / dof)[1] \
            + 0.5 * dof * dim * np.log(2) \
            + sp.special.multigammaln(dof/2, dim)
    return const - 0.5 * (dof - dim - 1) * e_logdet_info_x + dof * dim / 2

################
# define priors

def dp_prior(alpha, e_log_1mv):
    return (alpha - 1) * np.sum(e_log_1mv)

def wishart_prior(e_info_x, e_logdet_info_x, inv_wishart_scale, dof):
    dim = e_info_x.shape[0]
    assert dim == e_info_x.shape[1]
    assert dof > dim - 1

    return 0.5 * (dof - dim - 1) * e_logdet_info_x \
        - 0.5 * np.trace(inv_wishart_scale * e_info_x)

def normal_prior(mu, mu2, prior_mu, e_info_x, kappa):
    k_approx = np.shape(mu)[0]
    prior_mu2 = np.outer(prior_mu, prior_mu)

    return np.sum(np.dot(mu, np.dot(e_info_x * kappa, prior_mu)) \
            - 0.5 * np.einsum('kij, ji -> k', mu2, e_info_x * kappa))\
            - k_approx * 0.5 * np.einsum('ij, ji', prior_mu2, e_info_x * kappa)


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

def loglik_obs_by_nk(mu, mu2, x, e_info_x):
    # expected log likelihood of nth observation when it belongs to component k
    k_approx = np.shape(mu)[0]
    return np.einsum('ni, ij, kj -> nk', x, e_info_x, mu) \
            - 0.5 * np.einsum('kij, ji -> k', mu2, e_info_x)[None, :]\
            - 0.5 * np.einsum('ni, ij, nj -> n', x, e_info_x, x)[:, None]

def loglik_obs(e_z, mu, mu2, x, e_info_x):
    # expected log likelihood of all observations
    return np.sum(e_z * loglik_obs_by_nk(mu, mu2, x, e_info_x))

def e_loglik_full(x, mu, mu2, tau, e_log_v, e_log_1mv, e_z,
                    e_info_x, e_logdet_info_x, prior_mu,
                    prior_inv_wishart_scale, kappa, prior_dof, alpha):
    # combining the pieces, compute the full expected log likelihood

    prior = dp_prior(alpha, e_log_1mv) \
                + normal_prior(mu, mu2, prior_mu, e_info_x, kappa)\
                + wishart_prior(e_info_x, e_logdet_info_x, \
                    prior_inv_wishart_scale, prior_dof)

    return loglik_obs(e_z, mu, mu2, x, e_info_x) \
                + loglik_ind(e_z, e_log_v, e_log_1mv) + prior

############
# and finally, the elbo

def compute_elbo(x, mu, mu2, info_mu, tau, e_log_v, e_log_1mv, e_z,
                    e_info_x, e_logdet_info_x, dof, prior_mu,
                    prior_inv_wishart_scale, kappa, prior_dof, alpha):

    # entropy terms
    entropy = mu_entropy(info_mu) + beta_entropy(tau) + multinom_entropy(e_z)\
                + wishart_entropy(e_logdet_info_x, e_info_x, dof)

    return e_loglik_full(x, mu, mu2, tau, e_log_v, e_log_1mv, e_z,
                        e_info_x, e_logdet_info_x, prior_mu,
                        prior_inv_wishart_scale, kappa, prior_dof, alpha) \
                        + entropy

############
# the object
class DPNormalMixture(object):
    def __init__(self, x, vb_params, prior_params):
        self.x = x
        self.vb_params = deepcopy(vb_params)

        self.prior_mu, self.prior_dof, self.prior_inv_wishart_scale, \
            self.alpha, self.kappa = get_prior_params(prior_params)

    def get_vb_params(self):
        e_log_v, e_log_1mv, e_z, mu, mu2, info_mu, tau,\
                    e_info_x, e_logdet_info_x, dof = get_vb_params(self.vb_params)

        return e_log_v, e_log_1mv, e_z, mu, mu2, info_mu, tau,\
                    e_info_x, e_logdet_info_x, dof

    """def set_optimal_z(self):
        # note this isn't actually called in the kl method below
        # since we don't want to compute e_log_v twice (it has digammas)

        e_log_v, e_log_1mv, e_z, mu, mu2, info, tau = self.get_vb_params()

        # optimize z
        e_z_opt = z_update(mu, mu2, self.x, self.info_x, e_log_v, e_log_1mv)
        self.vb_params['local']['e_z'].set(e_z_opt)
    """
    def get_kl(self, verbose = False):
        # get the kl without optimizing z
        e_log_v, e_log_1mv, e_z, mu, mu2, info_mu, tau,\
                    e_info_x, e_logdet_info_x, dof = self.get_vb_params()

        elbo = compute_elbo(self.x, mu, mu2, info_mu, tau, e_log_v, e_log_1mv, \
                            e_z, e_info_x, e_logdet_info_x, dof, self.prior_mu,\
                            self.prior_inv_wishart_scale, self.kappa,\
                            self.prior_dof, self.alpha)
        if verbose:
            print('kl:\t', -1 * elbo)

        return -1 * elbo
    """
    def kl_optimize_z(self, verbose=False):
        # here we are optimizing z before computing the kl
        # this is the function that will be passed to the Newton method

        e_log_v, e_log_1mv, _, mu, mu2, info, tau = self.get_vb_params()

        # optimize z
        e_z = z_update(mu, mu2, self.x, self.info_x, e_log_v, e_log_1mv, \
                                        fudge_factor = 10**(-10))

        elbo = compute_elbo(self.x, mu, mu2, info, tau, e_log_v, e_log_1mv, e_z,
                        self.prior_mu, self.prior_info, self.info_x, self.alpha)

        if verbose:
            print('kl:\t', -1 * elbo)

        return -1 * elbo
    """

################
# other functions

def draw_data(info_x, x_dim, k_truth, num_obs):

    # true means
    mu_spacing = np.arange(k_truth) * 5
    true_mu = np.array([ mu_spacing, mu_spacing]).T

    # mixing proportions
    true_pi = np.ones(k_truth)
    true_pi = true_pi / k_truth

    # draw group indicators
    true_z_ind = np.random.choice(range(k_truth), p = true_pi, size = num_obs)
    true_z = np.zeros((num_obs, k_truth))
    for i in range(num_obs):
        true_z[i, true_z_ind[i]] = 1.0

    # draw observations
    x = np.array([ np.random.multivariate_normal(
                true_mu[true_z_ind[n]], np.linalg.inv(info_x)) \
               for n in range(num_obs) ])

    return x, true_mu, true_z, true_pi

def get_vb_params(vb_params):
    e_log_v = vb_params['global']['v_sticks'].e_log()[:,0] # E[log v]
    e_log_1mv = vb_params['global']['v_sticks'].e_log()[:,1] # E[log 1 - v]
    e_z = vb_params['local']['e_z'].get()
    mu = vb_params['global']['mu'].get()
    info_mu = vb_params['global']['info_mu'].get()

    mu2 = np.array([np.linalg.inv(info_mu[k]) + np.outer(mu[k,:], mu[k,:]) \
                        for k in range(np.shape(mu)[0])])

    dof = vb_params['global']['wishart_dof'].get()[0]
    wishart_scale = vb_params['global']['wishart_scale'].get()
    e_info_x = dof * wishart_scale

    # TODO: you should compute a range of digammas at the start and then
    # just call them here ...
    dim = np.shape(wishart_scale)[0]
    multi_digamma = np.sum([sp.special.digamma(dof - 0.5 * i )\
                                for i in range(dim)])
    e_logdet_info_x = multi_digamma + dim * np.log(2) \
                        + np.linalg.slogdet(wishart_scale)[0]

    tau = vb_params['global']['v_sticks'].alpha.get()

    return e_log_v, e_log_1mv, e_z, mu, mu2, info_mu, tau,\
                e_info_x, e_logdet_info_x, dof

def get_prior_params(prior_params):
    prior_mu = prior_params['prior_mu'].get()

    prior_dof = prior_params['wishart_dof'].get()
    prior_inv_wishart_scale = prior_params['inv_wishart_scale'].get()

    alpha = prior_params['alpha'].get()
    kappa = prior_params['kappa'].get()
    return prior_mu, prior_dof, prior_inv_wishart_scale, alpha, kappa



# draw samples from the variational distribution (used in unittesting)
def variational_samples(mu, info_mu, tau, e_z, wishart_scale, dof, n_samples):
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

    # draw wishart informations
    info_samples = osp.stats.wishart.rvs(dof, wishart_scale, size = n_samples)

    # draw normal means
    mu_samples = np.zeros((n_samples, mu.shape[0], mu.shape[1]))
    for k in range(np.shape(mu)[0]):
        mu_samples[:,k,:] = np.random.multivariate_normal(mu[k,:], \
                            np.linalg.inv(info_mu[k]), size = n_samples)

    # draw z
    z_samples = np.zeros((n_samples, e_z.shape[0], e_z.shape[1]))
    for n in range(e_z.shape[0]):
        z_samples[:, n, :] = np.random.multinomial(1, e_z[n, :], size = n_samples)

    return v_samples, pi_samples, mu_samples, z_samples, info_samples
