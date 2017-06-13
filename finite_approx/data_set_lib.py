import autograd.numpy as np
import autograd.scipy as sp
from autograd.scipy import special
from autograd import grad, hessian, hessian_vector_product, hessian, jacobian

from copy import deepcopy

from scipy import optimize

import valez_finite_VI_lib as vi
import generic_optimization_lib as packing


class OptimzationTrace(object):
    def __init__(self, verbose=True):
        self.reset()
        self.print_every = 10
        self.verbose = verbose

    def reset(self):
        self.stepnum = 0
        self.objective = []
        self.params = []

    def update(self, params, objective):
        self.params.append(params)
        self.objective.append(objective)
        if self.verbose and self.stepnum % self.print_every == 0:
            print('Step {} objective: {}'.format(self.stepnum, objective))
        self.stepnum += 1


class DataSet(object):
    def __init__(self, x, k_approx, alpha, sigma_eps, sigma_a):
        self.x = x
        self.k_approx = k_approx
        self.alpha = alpha
        self.data_shape = {'D': x.shape[1], 'N': x.shape[0] , 'K':k_approx}
        self.sigmas = {'eps': sigma_eps, 'A': sigma_a}

        self.get_kl_grad =  grad(
            lambda params: self.wrapped_kl(params, tracing=False))
        self.get_kl_hvp = hessian_vector_product(
            lambda params: self.wrapped_kl(params, tracing=False))
        self.get_kl_hessian = hessian(
            lambda params: self.wrapped_kl(params, tracing=False))

        # It turns out to be much faster to take the gradient wrt the
        # small vector first.
        self.get_wrapped_kl_hyperparams_hyperparamgrad = \
            grad(self.wrapped_kl_hyperparams, argnum=1)
        self.get_kl_sens_hess = \
            jacobian(self.get_wrapped_kl_hyperparams_hyperparamgrad, argnum=0)

        self.trace = OptimzationTrace()

    def unpack_params(self, params):
         return packing.unpack_params(
            params, self.data_shape['K'], self.data_shape['D'],
            self.data_shape['N'])

    def cavi_updates(self, tau, nu, phi_mu, phi_var):
        vi.cavi_updates(tau, nu, phi_mu, phi_var, \
                        self.x, self.alpha, self.sigmas)

    def wrapped_kl(self, params, tracing=True):
        tau, phi_mu, phi_var, nu = self.unpack_params(params)
        elbo = vi.compute_elbo(tau, nu, phi_mu, phi_var, \
                               self.x, self.sigmas, self.alpha)
        if tracing:
            self.trace.update(params, -1 * elbo)
        return -1 * elbo

    def wrapped_kl_hyperparams(self, params, hyper_params):
        tau, phi_mu, phi_var, nu = self.unpack_params(params)
        alpha, sigma_a, sigma_eps = packing.unpack_hyperparameters(hyper_params)
        sigmas = {'eps': sigma_eps, 'A': sigma_a}
        elbo = vi.compute_elbo(tau, nu, phi_mu, phi_var, self.x, sigmas, alpha)
        return -1 * elbo

    def get_prediction(self, params):
        tau, phi_mu, phi_var, nu = self.unpack_params(params)
        return np.matmul(nu, phi_mu.T)

    def run_cavi(self, tau, nu, phi_mu, phi_var, max_iter=200, tol=1e-6):
        params = packing.flatten_params(tau, nu, phi_mu, phi_var)

        self.trace.reset()
        diff = np.float('inf')
        while diff > tol and self.trace.stepnum < max_iter:
            self.cavi_updates(tau, nu, phi_mu, phi_var)
            new_params = packing.flatten_params(tau, nu, phi_mu, phi_var)
            diff = np.max(np.abs(new_params - params))
            self.trace.update(params, diff)
            if not np.isfinite(diff):
                print('Error: non-finite parameter difference.')
                break
            params = new_params

        if self.trace.stepnum >= max_iter:
            print('Warning: CAVI reached max_iter.')

        print('Done with CAVI.')
        return tau, nu, phi_mu, phi_var

    def run_newton_tr(self, params_init, maxiter=200, gtol=1e-6):
        self.trace.reset()
        tr_opt = optimize.minimize(
            lambda params: self.wrapped_kl(params, tracing=True),
            params_init, method='trust-ncg',
            jac=self.get_kl_grad,
            hessp=self.get_kl_hvp,
            tol=1e-6, options={'maxiter': maxiter, 'disp': True, 'gtol': gtol })

        print('Done with Newton trust region.')
        return tr_opt

    def get_variational_log_lh(self, params, a, z, pi):
        tau, phi_mu, phi_var, nu = self.unpack_params(params)
        return log_q_a(a, phi_mu.T, phi_var) + log_q_z(z, nu) + log_q_pi(pi, tau)


###################
# Variational likelihoods

# normal means
def log_q_a(a, phi_mu, phi_var):
    # note the shapes below: phi_mu should be the "correct",
    # not the transposed shape
    k_approx = np.shape(phi_mu)[0]
    x_d = np.shape(phi_mu)[1]
    assert np.shape(phi_mu)[0] == len(phi_var), 'shape of phi_var and phi_mu do not match'
    assert np.shape(a) == np.shape(phi_mu), 'shape of A and phi_mu do not match'

    log_denom = (x_d/2) * np.sum(np.log(2 * np.pi * phi_var))
    deviation = np.dot(a.T - phi_mu.T, 1 / phi_var)
    return -0.5 * np.dot(deviation.T, deviation) - log_denom

# bernoulli responsibilities
def log_q_z(z, nu):
    return np.sum(np.log(nu**z) + np.log((1 - nu)**(1-z)))

# pi stick lengths
def log_q_pi(pi, tau):
    log_beta = sp.special.gammaln(tau[:,0]) + sp.special.gammaln(tau[:,1]) \
                    - sp.special.gammaln(tau[:,0] + tau[:,1])

    return np.sum(-log_beta + (tau[:,0] - 1) * np.log(pi) \
                    + (tau[:,1] - 1) * np.log(1 - pi))

####################
# Prior llikelihoods
def log_p0_a(a, sigma_a):
    prior_mean = np.zeros(np.shape(a))
    prior_var = np.ones(np.shape(a)[0]) * sigma_a
    return log_q_a(a, prior_mean, prior_var)

# bernoulli responsibilities
def log_p0_z(z, pi):
    nu = np.tile(pi, (np.shape(z)[0], 1))
    return log_q_z(z, nu)

# pi stick lengths
def log_p0_pi(pi, alpha, k_approx):
    tau = np.array([np.full(len(pi), 1), np.full(len(pi), alpha/k_approx)]).T
    return log_q_pi(pi, tau)

def log_p0_all(a, z, pi, alpha, k_approx, sigma_a):
    return log_p0_a(a, sigma_a) + log_p0_z(z, pi) \
            + log_p0_pi(pi, alpha, k_approx)
