import autograd.numpy as np
import autograd.scipy as sp
from autograd.scipy import special
from autograd import grad, hessian, hessian_vector_product, hessian, jacobian

from copy import deepcopy

from scipy import optimize

from valez_finite_VI_lib import \
    initialize_parameters, generate_data, compute_elbo, cavi_updates
from generic_optimization_lib import \
    unpack_params, pack_params, pack_hyperparameters, unpack_hyperparameters, \
    flatten_params


class OptimzationTrace(object):
    def __init__(self, verbose=True):
        self.reset()
        self.print_every = 10
        self.verbose = verbose

    def reset(self):
        self.stepnum = 0
        self.kl = []
        self.params = []

    def update(self, params, kl):
        self.params.append(params)
        self.kl.append(kl)
        if self.verbose and self.stepnum % self.print_every == 0:
            print('Step {} objective: {}'.format(self.stepnum, kl))
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
         return unpack_params(params, self.data_shape['K'],
                              self.data_shape['D'], self.data_shape['N'])

    def cavi_updates(self, tau, nu, phi_mu, phi_var):
        cavi_updates(tau, nu, phi_mu, phi_var, \
                     self.x, self.alpha, self.sigmas)

    def wrapped_kl(self, params, tracing=True):
        tau, phi_mu, phi_var, nu = self.unpack_params(params)
        elbo = compute_elbo(tau, nu, phi_mu, phi_var, \
                            self.x, self.sigmas, self.alpha)
        if tracing:
            self.trace.update(params, -1 * elbo)
        return -1 * elbo

    def wrapped_kl_hyperparams(self, params, hyper_params):
        tau, phi_mu, phi_var, nu = self.unpack_params(params)
        alpha, sigma_a, sigma_eps = unpack_hyperparameters(hyper_params)
        sigmas = {'eps': sigma_eps, 'A': sigma_a}
        elbo = compute_elbo(tau, nu, phi_mu, phi_var, self.x, sigmas, alpha)
        return -1 * elbo

    def get_prediction(self, params):
        tau, phi_mu, phi_var, nu = self.unpack_params(params)
        return np.matmul(nu, phi_mu.T)

    def run_cavi(self, tau, nu, phi_mu, phi_var, max_iter=200, tol=1e-6):
        params = flatten_params(tau, nu, phi_mu, phi_var)

        self.trace.reset()
        diff = np.float('inf')
        while diff > tol and self.trace.stepnum < max_iter:
            self.cavi_updates(tau, nu, phi_mu, phi_var)
            new_params = flatten_params(tau, nu, phi_mu, phi_var)
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
