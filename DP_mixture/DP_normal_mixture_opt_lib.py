### optimization algorithms for the DP normal mixture
import autograd.numpy as np
import autograd.scipy as sp
from autograd import grad

import matplotlib.pyplot as plt

import DP_normal_mixture_lib as dp

import matplotlib.pyplot as plt

def run_cavi(model, init_par_vec, max_iter = 100, tol = 1e-8, disp = True):

    x = model.x
    prior_mu = model.prior_mu
    prior_info = model.prior_info
    info_x = model.info_x
    alpha = model.alpha

    # initialize
    model.vb_params.set_free(init_par_vec)

    kl = np.zeros(max_iter)
    diff = -10

    for i in range(max_iter):

        # tau update
        e_z = model.vb_params['local']['e_z'].get()

        tau_new = dp.tau_update(e_z, alpha)
        model.vb_params['global']['v_sticks'].alpha.set(tau_new)

        # mu update
        mu_new, info_new = dp.mu_update(x, e_z, prior_mu, prior_info, info_x)
        model.vb_params['global']['mu'].set(mu_new)
        model.vb_params['global']['info'].set(info_new)

        # z update
        e_log_v = model.vb_params['global']['v_sticks'].e_log()[:,0] # E[log v]
        e_log_1mv = model.vb_params['global']['v_sticks'].e_log()[:,1] # E[log 1 - v]
        mu = model.vb_params['global']['mu'].get()
        info = model.vb_params['global']['info'].get()

        e_z_new = dp.z_update(mu, info, x, info_x, e_log_v, e_log_1mv)
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
