### optimization algorithms for the DP normal mixture
import autograd.numpy as np
import autograd.scipy as sp
from autograd import grad

import matplotlib.pyplot as plt

import DP_normal_mixture_lib as dp

import matplotlib.pyplot as plt


############
# CAVI updates
def soft_thresh(e_z, ub, lb):
    constrain = (ub - lb) * e_z + lb
    renorm = np.sum(constrain, axis = 1)
    return constrain / renorm[:, None]

def z_update(mu, mu2, x, e_info_x, e_log_v, e_log_1mv, fudge_factor = 0.0):
    log_propto = dp.loglik_obs_by_nk(mu, mu2, x, e_info_x) + \
                    dp.loglik_ind_by_k(e_log_v, e_log_1mv)
    log_denom = sp.misc.logsumexp(log_propto, axis = 1)

    e_z_true = np.exp(log_propto - log_denom[:, None])
    return soft_thresh(e_z_true, 1 - fudge_factor, fudge_factor)

def mu_update(x, e_z, prior_mu, e_info_x, kappa):
    k_approx = np.shape(e_z)[1]
    info_mu_update = np.array([e_info_x * kappa + np.sum(e_z[:, i]) * e_info_x \
                            for i in range(k_approx)])
    nat_param = np.dot(prior_mu, e_info_x * kappa) \
                    + np.dot(e_z.T, np.dot(x, e_info_x))
    # print(nat_param)

    #mu_update = np.array([np.dot(nat_param[k,:], info_update[k])\
    #                        for k in range(k_approx)])
    mu_update = np.array([np.linalg.solve(info_mu_update[k], nat_param[k,:])\
                            for k in range(k_approx)])

    return mu_update, info_mu_update

def tau_update(e_z, alpha):
    k_approx = np.shape(e_z)[1]
    sum_e_z = np.sum(e_z, axis = 0)
    sum_e_z_upper = np.cumsum(sum_e_z[::-1])[::-1]

    #cum_sum_z = np.concatenate(([0.0], np.cumsum(sum_e_z)[:-2]))

    tau_update = np.zeros((k_approx - 1, 2))
    tau_update[:, 0] = sum_e_z[:-1] + 1
    tau_update[:, 1] = alpha + sum_e_z_upper[1:]

    return tau_update

def wishart_updates(x, mu, mu2, e_z, prior_mu, prior_inv_wishart_scale, prior_wishart_dof, kappa):
    k_approx = np.shape(mu)[0]

    prior_mu_tile = np.tile(prior_mu, (k_approx, 1))
    cross_term1 = np.dot(mu.T, prior_mu_tile)
    prior_normal_term = kappa * (np.sum(mu2, axis = 0)\
                                - cross_term1 - cross_term1.T \
                                + k_approx * np.outer(prior_mu, prior_mu))

    predictive = np.dot(e_z, mu)
    outer_mu = np.array([np.outer(mu[i,:], mu[i,:]) for i in range(k_approx)])
    z_sum = np.sum(e_z, axis = 0)

    cross_term2 = np.dot(x.T, predictive)
    data_lh_term = np.dot(x.T, x) - cross_term2 - cross_term2.T\
                    + np.einsum('kij, k -> ij', outer_mu, z_sum)

    inv_scale_update = prior_inv_wishart_scale + prior_normal_term + data_lh_term
    scale_update = np.linalg.inv(inv_scale_update)

    dof_update = prior_wishart_dof + np.shape(x)[0] + k_approx

    # there's some numerical issue in which the update is not exactly symmetric
    if np.any(np.abs(scale_update - scale_update.T) >= 1e-10):
        print('wishart scale not symmetric?')
        print(scale_update - scale_update.T)

    scale_update = 0.5 * (scale_update + scale_update.T)

    return scale_update, np.array([dof_update])


def run_cavi(model, init_par_vec, max_iter = 100, tol = 1e-8, disp = True):

    # the data
    x = model.x

    # the prior parameters
    prior_mu = model.prior_mu
    prior_wishart_dof = model.prior_dof
    prior_inv_wishart_scale = model.prior_inv_wishart_scale
    kappa = model.kappa
    alpha = model.alpha

    # initialize
    model.vb_params.set_free(init_par_vec)

    kl = np.zeros(max_iter)
    diff = -10

    e_info_x = model.vb_params['global']['wishart'].e()

    for i in range(max_iter):

        # tau update
        e_z = model.vb_params['local']['e_z'].get()
        tau_new = tau_update(e_z, alpha)
        model.vb_params['global']['v_sticks'].alpha.set(tau_new)

        # mu update
        #e_info_x = np.linalg.inv(model.vb_params['global']['inv_wishart_scale'].get())\
        #            * model.vb_params['global']['wishart_dof'].get()

        mu_new, info_new = mu_update(x, e_z, prior_mu, e_info_x, kappa)
        model.vb_params['global']['mu'].set(mu_new)
        model.vb_params['global']['info_mu'].set(info_new)

        # wishart update
        mu = model.vb_params['global']['mu'].get()
        mu2 = np.array([np.linalg.inv(info_new[k]) + np.outer(mu[k,:], mu[k,:]) \
                            for k in range(np.shape(mu)[0])])
        wishart_scale_new, wishart_dof_new = \
            wishart_updates(x, mu, mu2, e_z, prior_mu, \
                    prior_inv_wishart_scale, prior_wishart_dof, kappa)

        model.vb_params['global']['wishart'].params['v'].set(wishart_scale_new)
        model.vb_params['global']['wishart'].params['df'].set(wishart_dof_new)

        # z update
        e_log_v = model.vb_params['global']['v_sticks'].e_log()[:,0] # E[log v]
        e_log_1mv = model.vb_params['global']['v_sticks'].e_log()[:,1] # E[log 1 - v]
        e_info_x = model.vb_params['global']['wishart'].e()

        e_z_new = z_update(mu, mu2, x, e_info_x, e_log_v, e_log_1mv)
        model.vb_params['local']['e_z'].set(e_z_new)

        # evaluate elbo
        kl[i] = model.get_kl()

        if i > 0:
            diff = kl[i] - kl[i-1]

        if disp:
            print('iteration: ', i, 'kl: ', kl[i])

        if np.isnan(kl[i]):
            print('NAN ELBO.')
            break

        if (i > 0) & (diff > 0):
            print('kl increased!  Difference: ', diff)
            break

        if (np.abs(diff) < tol) & (i < max_iter):
            print('CAVI terminated successfully :)')
            print('iterations ran: ', i)
            break

    if i == max_iter - 1:
        print('max iteration reached')

    plt.plot(kl[:i])
    plt.xlabel('iteration')
    plt.ylabel('kl')
