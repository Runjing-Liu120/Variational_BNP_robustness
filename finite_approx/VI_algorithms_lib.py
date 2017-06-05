
import numpy as np
import scipy as sp

import valez_finite_VI_lib as vi
from copy import deepcopy


def cavi_anneal_update(tau, nu, phi_mu, phi_var, X, alpha, sigmas,
    anneal_temp, step_size, stochastic = False, random_seed = None):

        # have not implemented stochastic = True here
        # np.random.seed(None)

        D = np.shape(X)[1]
        N = np.shape(X)[0]
        K = np.shape(phi_mu)[1]

        #if stochastic:
        #    [tau_r, nu_r, phi_mu_r, phi_var_r] = initialize_parameters(N, D, K)

        assert np.shape(X)[0] == np.shape(nu)[0]
        assert np.shape(X)[1] == np.shape(phi_mu)[0]
        assert np.shape(nu)[1] == np.shape(phi_mu)[1]

        digamma_tau = sp.special.digamma(tau)

        for n in range(N):
            for k in range(K):
                vi.nu_updates(tau, nu, phi_mu, phi_var, X, sigmas, n, k, \
                    digamma_tau, anneal_temp = anneal_temp)

        for k in range(K):
            vi.phi_updates(nu, phi_mu, phi_var, X, sigmas, k)
            phi_var[k] = phi_var[k] / 1.0

        vi.tau_updates(tau, nu, alpha)
        tau[:] = tau[:] / anneal_temp



class VI_algorithms(object):
    def __init__(self, x, k_approx, alpha, sigma_eps, sigma_a):
        self.x = x
        self.k_approx = k_approx
        self.alpha = alpha
        self.data_shape = {'D': x.shape[1], 'N': x.shape[0] , 'K':k_approx}
        self.sigmas = {'eps': sigma_eps, 'A': sigma_a}

        self.tau, self.nu, self.phi_mu, self.phi_var = \
                vi.initialize_parameters(x.shape[0], x.shape[1], k_approx)

    def re_init(self, random_seed = None):
        np.random.seed(random_seed)

        self.tau, self.nu, self.phi_mu, self.phi_var = \
                vi.initialize_parameters(\
                self.data_shape['N'], self.data_shape['D'], self.data_shape['K'])

    def get_elbo(self):
        return(vi.compute_elbo(self.tau, self.nu, self.phi_mu, self.phi_var, \
                self.x, self.sigmas, self.alpha))

    def run_cavi(self, max_iter=200, tol=1e-6, verbose = True):

        stepnum = 0
        diff = 100
        elbo = np.array([])
        err = False

        while diff > tol and stepnum < max_iter:
            vi.cavi_updates(self.tau, self.nu, self.phi_mu, self.phi_var, \
                self.x, self.alpha, self.sigmas)
            elbo = np.append(elbo, vi.compute_elbo(\
                self.tau, self.nu, self.phi_mu, self.phi_var, \
                self.x, self.sigmas, self.alpha))

            if stepnum>0:
                diff = elbo[stepnum]-elbo[stepnum-1]

            if verbose:
                print('iteration: ', stepnum, 'elbo: ', elbo[stepnum])

            if np.isnan(elbo[stepnum]):
                print('NAN ELBO.')
                break

            if (stepnum>0) & (diff < 0):
                print('elbo decreased!  Difference: ', diff, 'elbo: ', elbo[-1])
                err = True
                break

            stepnum += 1

        if not(err):
            if max_iter <= stepnum:
                print('Max iteration reached; elbo: ', elbo[-1])
            else:
                print('elbo converged: ', elbo[-1])

        return self.tau, self.nu, self.phi_mu, self.phi_var, elbo


    def run_stochastic_VI(self, batch_size, decay_param = 1, \
                max_iter=200, tol=1e-6, verbose = True):

        elbo = np.array([])

        assert batch_size % 1 == 0  # make sure this is an integer
        batch_size = int(batch_size)

        diff = 100
        stepnum = 0
        while diff > tol and stepnum < max_iter:
            # sample data
            indices = np.random.choice(self.data_shape['N'], size = batch_size,\
                replace = False)


            # update local parameters: in this case, nu
            digamma_tau = sp.special.digamma(self.tau)
            for n in indices:
                for k in range(self.data_shape['K']):
                    vi.nu_updates(self.tau, self.nu, self.phi_mu, self.phi_var, \
                    self.x, self.sigmas, n, k, digamma_tau)

            # replicate data
            replicates = self.data_shape['N']/batch_size
            # the method below will only work when Num_samples is a multiple of batch_size
            assert self.data_shape['N']%batch_size == 0

            replicates = int(replicates)

            nu_stochastic = np.zeros((self.data_shape['N'], self.data_shape['K']))
            X_stochastic = np.zeros((self.data_shape['N'], self.data_shape['D']))

            for j in range(batch_size):
                nu_stochastic[j*replicates:(j+1)*replicates , :] = self.nu[indices[j], :]
                X_stochastic[j*replicates:(j+1)*replicates , :] = self.x[indices[j], :]


            # update global paremeters: phi_mu, phi_var, and tau

            step_size = (stepnum+1)**(-decay_param)

            phi_mu_old = deepcopy(self.phi_mu)
            phi_var_old = deepcopy(self.phi_var)
            tau_old = deepcopy(self.tau)

            diff = 100
            """
            # iterate to find optimal
            for j in range(100):
                phi_mu_tmp = deepcopy(self.phi_mu)
                phi_var_tmp = deepcopy(self.phi_var)
                tau_tmp = deepcopy(self.tau)

                # do updates
                for k in range(self.data_shape['K']):
                    vi.phi_updates(nu_stochastic, self.phi_mu, self.phi_var, \
                        X_stochastic, self.sigmas, k)

                for k in range(self.data_shape['K']):
                    vi.tau_updates(self.tau, nu_stochastic, self.alpha)

                diff = np.sum(np.abs(self.phi_mu[:] - phi_mu_tmp[:])) + \
                    np.sum(np.abs(self.phi_var[:] - phi_var_tmp[:])) + \
                    np.sum(np.abs(self.tau[:] - tau_tmp[:]))

                if diff < 10**(-4):
                    break
            # print(j)

            self.phi_mu = phi_mu_old * (1-step_size) + self.phi_mu * step_size
            self.phi_var = phi_var_old * (1-step_size) + self.phi_var * step_size
            self.tau = tau_old * (1-step_size) + self.tau * step_size
            """



            for k in range(self.data_shape['K']):
                phi_mu_old = deepcopy(self.phi_mu)
                phi_var_old = deepcopy(self.phi_var)

                vi.phi_updates(nu_stochastic, self.phi_mu, self.phi_var, \
                    X_stochastic, self.sigmas, k)

                self.phi_mu = phi_mu_old * (1-step_size) + self.phi_mu * step_size
                self.phi_var = phi_var_old * (1-step_size) + self.phi_var * step_size

            for k in range(self.data_shape['K']):
                tau_old = deepcopy(self.tau)
                vi.tau_updates(self.tau, nu_stochastic, self.alpha)
                self.tau = tau_old * (1-step_size) + self.tau * step_size

            elbo = np.append(elbo, vi.compute_elbo(self.tau, self.nu, self.phi_mu, \
                self.phi_var, self.x, self.sigmas, self.alpha))

            if verbose:
                print('iteration: ', stepnum, 'elbo: ', elbo[stepnum])


            if stepnum>0:
                diff = np.abs(elbo[stepnum]-elbo[stepnum-1])

            stepnum = stepnum + 1

        if max_iter <= stepnum:
            print('Max iteration reached; elbo: ', elbo[-1])
        else:
            print('elbo converged', elbo[-1])

        return self.tau, self.nu, self.phi_mu, self.phi_var, elbo


    def run_cavi_anneal(self, rho = 0.9, max_iter=200, tol=1e-6, verbose = True):

        stepnum = 0
        diff = 100
        elbo = np.array([])

        step_size = 0 # only necessary if doing stochastic annealing

        while diff > tol and stepnum < max_iter:

            anneal_temp = (5 * rho**(stepnum+1) + 1)**(-1)

            cavi_anneal_update(self.tau, self.nu, self.phi_mu, self.phi_var, \
                self.x, self.alpha, self.sigmas, anneal_temp, step_size)

            elbo = np.append(elbo, vi.compute_elbo(\
                self.tau, self.nu, self.phi_mu, self.phi_var, \
                self.x, self.sigmas, self.alpha))

            if stepnum>0:
                diff = elbo[stepnum]-elbo[stepnum-1]

            if verbose:
                print('iteration: ', stepnum, 'elbo: ', elbo[stepnum])

            if np.isnan(elbo[stepnum]):
                print('NAN ELBO.')
                break

            #if (stepnum>0) & (diff < 0):
            #    print('elbo decreased!  Difference: ', diff)
            #    err = True
            #    break

            stepnum += 1

        if max_iter <= stepnum:
            print('warning: max iteration reached; elbo: ', elbo[-1])
        else:
            print('elbo converged: ', elbo[-1])

        return self.tau, self.nu, self.phi_mu, self.phi_var, elbo
