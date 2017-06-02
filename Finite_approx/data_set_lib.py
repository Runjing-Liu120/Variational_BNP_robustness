import autograd.numpy as np
import autograd.scipy as sp
from autograd.scipy import special
from autograd import grad, hessian, hessian_vector_product, hessian, jacobian

from copy import deepcopy

from scipy import optimize

from valez_finite_VI_lib import \
    initialize_parameters, generate_data, compute_elbo, cavi_updates
from generic_optimization_lib import \
    unpack_params, pack_params, pack_hyperparameters, unpack_hyperparameters


class DataSet(object):
    def __init__(self, x, k_approx, alpha, sigma_eps, sigma_a):
        self.x = x
        self.k_approx = k_approx
        self.alpha = alpha
        self.data_shape = {'D': x.shape[1], 'N': x.shape[0] , 'K':k_approx}
        self.sigmas = {'eps': sigma_eps, 'A': sigma_a}

        self.get_kl_grad = grad(self.wrapped_kl)
        self.get_kl_hvp = hessian_vector_product(self.wrapped_kl)
        self.get_kl_hessian = hessian(self.wrapped_kl)

        # It turns out to be much faster to take the gradient wrt the
        # small vector first.
        self.get_wrapped_kl_hyperparams_hyperparamgrad = \
            grad(self.wrapped_kl_hyperparams, argnum=1)
        self.get_kl_sens_hess = \
            jacobian(self.get_wrapped_kl_hyperparams_hyperparamgrad, argnum=0)

    def unpack_params(self, params):
         return unpack_params(params, self.data_shape['K'],
                              self.data_shape['D'], self.data_shape['N'])

    def cavi_updates(self, tau, nu, phi_mu, phi_var):
        cavi_updates(tau, nu, phi_mu, phi_var, \
                     self.x, self.alpha, self.sigmas)

    def wrapped_kl(self, params, verbose=False):
        tau, phi_mu, phi_var, nu = self.unpack_params(params)
        elbo = compute_elbo(tau, nu, phi_mu, phi_var, \
                            self.x, self.sigmas, self.alpha)
        if verbose:
            print -1 * elbo
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
